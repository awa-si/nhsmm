"""Stage 2V.8: ground-truth feature separability and fitted NHSMM emission diagnosis.

Chronological training/test partition; labels only used for oracle and diagnostics.
"""
import json
from pathlib import Path
import numpy as np
import torch
from nhsmm import NHSMM
import validate_trading_regime_usefulness as v

DATA=Path("data/reference/trading-regimes-v1/trading_regimes.npz")
OUTPUT=Path("/data/workspace/stage2/emission-identifiability.json")
NAMES=("bull","bear","range","chaotic")
def confusion(true,pred):
    mat=np.zeros((4,4),dtype=int)
    np.add.at(mat,(true,pred),1)
    return mat.tolist()

def main():
    torch.set_num_threads(1)
    with np.load(DATA) as z:
        d={k:z[k] for k in z.files}
    n=len(d["gt_regime_id"]);cut=int(n*.6)
    x=np.column_stack([d[f] for f in v.FEATURES]).astype(np.float64)
    y=d["gt_regime_id"].astype(int)
    mean=x[:cut].mean(0);std=np.maximum(x[:cut].std(0),1e-6)
    scaled=(x-mean)/std
    tr=scaled[:cut];te=scaled[cut:]
    mu=np.stack([tr[y[:cut]==k].mean(0) for k in range(4)])
    var=np.stack([np.maximum(tr[y[:cut]==k].var(0),1e-5) for k in range(4)])
    prior=np.bincount(y[:cut],minlength=4)/cut
    scores=-.5*(np.log(2*np.pi*var)[None,:,:]+(te[:,None,:]-mu[None,:,:])**2/var[None,:,:]).sum(-1)+np.log(prior)[None,:]
    oracle=scores.argmax(-1)
    oracle_conf=confusion(y[cut:],oracle)
    gap={}
    for index,key in enumerate(v.FEATURES):
        signal=float(np.var(mu[:,index],ddof=0))
        noise=float(np.mean(var[:,index]))
        gap[key]={"between_state_mean_variance":signal,"within_state_variance":noise,"signal_to_noise_ratio":signal/noise}
    # full-covariance Gaussian oracle: more expressive supervised discriminant, no labels on test.
    cov=np.stack([np.cov(tr[y[:cut]==k].T)+np.eye(len(v.FEATURES))*1e-4 for k in range(4)])
    full_scores=[]
    for k in range(4):
        sign,logdet=np.linalg.slogdet(cov[k])
        residual=te-mu[k]
        quad=np.einsum("ni,ij,nj->n",residual,np.linalg.inv(cov[k]),residual)
        full_scores.append(-.5*(logdet+quad)+np.log(prior[k]))
    full_pred=np.argmax(np.column_stack(full_scores),axis=1)
    # Entire NHSMM is unsupervised. Use existing configured window sampling for 1-seed pilot.
    train=v._windows(d,0,cut,12)
    test=v._windows(d,int(n*.8),n,4)
    train_x,test_x=v._standardize(train[0],test[0])
    torch.manual_seed(901);np.random.seed(901)
    cfg=v._config(901,context=False,hmm=False)
    cfg.max_iter=12
    model=NHSMM(cfg,device="cpu")
    model.initialize_distributions(jitter=0)
    # KMeans-initialized and optimized emissions; store separate pre/post snapshots.
    initialized={}
    original=model._initialize_run_state
    def capture(*args,**kwargs):
        original(*args,**kwargs)
        emission=model.dist.emission
        initialized["mu"]=emission.mu.detach().numpy().tolist()
        initialized["log_var"]=emission.log_var.detach().numpy().tolist()
    model._initialize_run_state=capture
    model.optimize(torch.tensor(train_x))
    model.eval()
    with torch.no_grad():
        em=model.dist.emission
        learned=em.mu.detach().numpy()
        learned_var=em.log_var.detach().exp().numpy()
        test_p=model.dist.emission.log_prob(torch.tensor(test_x))
        # Known-state assignment to emissions (ignores transition/duration).
        emission_pred=test_p.argmax(-1).numpy()
        perm,accuracy=v._best_perm(test[2].ravel(),emission_pred.ravel())
        predicted=np.asarray(perm)[emission_pred]
        gt_sample=np.stack([d[f][int(n*.8):] for f in v.FEATURES],axis=1)
    result={
        "protocol":"60/40 full-bar supervised Gaussian oracle; additional 1-seed unsupervised NHSMM on 12 x 480 train windows / 4 x 480 OOS windows, 12 iterations; labels never used in NHSMM.optimize",
        "train_rows":cut,"test_rows":n-cut,
        "diag_gaussian_oracle":{"accuracy":float((oracle==y[cut:]).mean()),"ari":v._ari(y[cut:],oracle),"confusion":oracle_conf},
        "full_cov_gaussian_oracle":{"accuracy":float((full_pred==y[cut:]).mean()),"ari":v._ari(y[cut:],full_pred),"confusion":confusion(y[cut:],full_pred)},
        "feature_signal_to_noise":gap,
        "model_emissions":{"matched_accuracy":accuracy,"ari":v._ari(test[2].ravel(),emission_pred.ravel()),"confusion_aligned":confusion(test[2].ravel(),predicted.ravel()),"kmeans_initialized":initialized,"final_mu":learned.tolist(),"final_variance":learned_var.tolist(),"state_occupancy_by_emission_argmax":np.bincount(emission_pred.ravel(),minlength=4).tolist()},
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:val for k,val in result.items() if k!="model_emissions" and k!="feature_signal_to_noise"},indent=2))
    print(json.dumps({"fitted_emission_accuracy":accuracy,"fitted_emission_ari":result["model_emissions"]["ari"],"occupancy":result["model_emissions"]["state_occupancy_by_emission_argmax"]},indent=2))

if __name__=="__main__":main()
