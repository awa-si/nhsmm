from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from nhsmm.filtering import filter_model_sequence
from diagnose_trading_freeze_2d import FrozenNHSMM,BALANCED
from validate_trading_regime_usefulness import _windows,_standardize,_best_perm,_ari,_boundary_f1,_run_medians,_ll,_config,REGIMES,K,TRAIN_WINDOWS,EVAL_WINDOWS
def score(truth,pred):
    perm,acc=_best_perm(truth.ravel(),pred.ravel())
    mapped=np.asarray(perm)[pred]
    return dict(accuracy=acc,ari=_ari(truth.ravel(),pred.ravel()),boundary_f1=_boundary_f1(truth,mapped),run_medians={REGIMES[i]:v for i,v in _run_medians(mapped).items()},permutation=list(perm),switches=int((pred[:,1:]!=pred[:,:-1]).sum()))
def run(variant,seed,path):
    torch.set_num_threads(1);np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    n=len(data["close"]);tr=_windows(data,0,int(.6*n),TRAIN_WINDOWS);te=_windows(data,int(.8*n),n,EVAL_WINDOWS)
    names=("log_return_1","log_return_3","log_return_12","trend_12","trend_48","ewma_vol_12","ewma_vol_48","volume_z_48")
    idx=[names.index(x) for x in BALANCED]
    train,test=_standardize(tr[0][:,:,idx],te[0][:,:,idx])
    cfg=_config(seed,context=False,hmm=False);cfg.n_features=len(idx)
    m=FrozenNHSMM(cfg,device="cpu");m.variant=variant;m.initialize_distributions();m.optimize(torch.tensor(train));m.eval()
    with torch.inference_mode():
        filt=filter_model_sequence(m,torch.tensor(test)).state_posterior.argmax(-1).numpy()
        vit=np.stack([p.numpy() for p in m.predict(torch.tensor(test),mode="viterbi",verbose=False)])
    return dict(variant=variant,seed=seed,filter=score(te[2],filt),viterbi=score(te[2],vit),agreement=float((filt==vit).mean()),oos_ll=_ll(m,test),note="Causal Viterbi uses the same explicit duration-tail hazard and D+ continuation semantics as causal filtering; Viterbi remains a global MAP path rather than marginal posterior argmax.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--variant",choices=["A","B","C"],required=True);p.add_argument("--seed",type=int,default=901);p.add_argument("--dataset",default="/data/workspace/nhsmm-trading-regimes-v1/trading_regimes.npz");p.add_argument("--output",required=True);a=p.parse_args()
    r=run(a.variant,a.seed,a.dataset);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r))
