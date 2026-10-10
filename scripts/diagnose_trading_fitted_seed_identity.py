"""Compare fitted model parameters and decoded paths across seeds."""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from nhsmm.filtering import filter_model_sequence
from diagnose_trading_freeze_2d import FrozenNHSMM,BALANCED
from validate_trading_regime_usefulness import _windows,_standardize,_config,TRAIN_WINDOWS,EVAL_WINDOWS,_best_perm,_ari

def sha(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()

def run(seeds,path,max_iter):
    torch.set_num_threads(1)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    n=len(data['close']);tr=_windows(data,0,int(.6*n),TRAIN_WINDOWS);te=_windows(data,int(.8*n),n,EVAL_WINDOWS)
    names=('log_return_1','log_return_3','log_return_12','trend_12','trend_48','ewma_vol_12','ewma_vol_48','volume_z_48')
    idx=[names.index(x) for x in BALANCED];train,test=_standardize(tr[0][:,:,idx],te[0][:,:,idx]);out=[];preds=[];posteriors=[]
    for seed in seeds:
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
        cfg=_config(seed,context=False,hmm=False);cfg.n_features=len(idx);cfg.max_iter=max_iter
        m=FrozenNHSMM(cfg,device='cpu');m.variant='A';m.initialize_distributions();m.optimize(torch.tensor(train));m.eval()
        with torch.inference_mode():p=filter_model_sequence(m,torch.tensor(test)).state_posterior.detach().cpu().numpy()
        pred=p.argmax(-1);preds.append(pred);posteriors.append(p)
        hashes={k:sha(v.detach().cpu().numpy()) for k,v in m.named_parameters()}
        out.append(dict(seed=seed,final_parameter_hashes=hashes,decoded_path_sha256=sha(pred),posterior_sha256=sha(p),switches=int((pred[:,1:]!=pred[:,:-1]).sum())))
    comparisons=[]
    for i in range(len(seeds)):
        for j in range(i+1,len(seeds)):
            a,b=preds[i],preds[j];perm,acc=_best_perm(a.ravel(),b.ravel());comparison=dict(seeds=[seeds[i],seeds[j]],raw_path_agreement=float(np.mean(a==b)),best_permuted_path_agreement=acc,permutation=list(perm),path_ari=_ari(a.ravel(),b.ravel()),posterior_max_abs_difference=float(np.max(np.abs(posteriors[i]-posteriors[j]))),equal_final_parameter_hashes=sum(out[i]['final_parameter_hashes'][k]==out[j]['final_parameter_hashes'][k] for k in out[i]['final_parameter_hashes']),total_parameters=len(out[i]['final_parameter_hashes']))
            comparisons.append(comparison)
    return dict(stage='fitted_seed_identity',max_iter=max_iter,models=out,comparisons=comparisons)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[905,906]);p.add_argument('--dataset',default='data/reference/trading-regimes-v1/trading_regimes.npz');p.add_argument('--max-iter',type=int,default=30);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.seeds,a.dataset,a.max_iter);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r['comparisons']),flush=True)
