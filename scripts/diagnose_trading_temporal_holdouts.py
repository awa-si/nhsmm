"""Chronological holdout replication with distinct seeds and inference-only scales."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from nhsmm.filtering import filter_model_sequence
from diagnose_trading_freeze_2d import FrozenNHSMM,BALANCED
from validate_trading_regime_usefulness import _windows,_standardize,_config,_best_perm,_ari,_boundary_f1,TRAIN_WINDOWS,EVAL_WINDOWS

FEATURE_NAMES=('log_return_1','log_return_3','log_return_12','trend_12','trend_48','ewma_vol_12','ewma_vol_48','volume_z_48')

def run(seed,path,max_iter):
    torch.set_num_threads(1);np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    n=len(data['close']);idx=[FEATURE_NAMES.index(x) for x in BALANCED]
    tr=_windows(data,0,int(.6*n),TRAIN_WINDOWS)
    tests={'middle_60_80':_windows(data,int(.6*n),int(.8*n),EVAL_WINDOWS),'late_80_100':_windows(data,int(.8*n),n,EVAL_WINDOWS)}
    train,*test_x=_standardize(tr[0][:,:,idx],*(tests[k][0][:,:,idx] for k in tests))
    cfg=_config(seed,context=False,hmm=False);cfg.n_features=len(idx);cfg.max_iter=max_iter
    model=FrozenNHSMM(cfg,device='cpu');model.variant='A';model.initialize_distributions();model.optimize(torch.tensor(train));model.eval()
    original=model._build_sequence_set;results=[]
    for (period,(_,_,truth,*_)),x in zip(tests.items(),test_x):
        for scale in (0.25,1.0):
            def scaled(*args,**kwargs):
                sequence=original(*args,**kwargs);sequence.log_probs=sequence.log_probs*scale;return sequence
            model._build_sequence_set=scaled
            with torch.inference_mode():post=filter_model_sequence(model,torch.tensor(x)).state_posterior.detach().cpu().numpy()
            pred=post.argmax(-1);perm,acc=_best_perm(truth.ravel(),pred.ravel());mapped=np.asarray(perm)[pred]
            results.append(dict(period=period,scale=scale,accuracy=acc,ari=_ari(truth.ravel(),pred.ravel()),boundary_f1_tol5=_boundary_f1(truth,mapped),switches=int((pred[:,1:]!=pred[:,:-1]).sum()),permutation=list(perm)))
    model._build_sequence_set=original
    return dict(seed=seed,max_iter=max_iter,train_range='0-60%',evaluation_ranges=['60-80%','80-100%'],results=results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[904,907]);p.add_argument('--dataset',default='data/reference/trading-regimes-v1/trading_regimes.npz');p.add_argument('--max-iter',type=int,default=30);p.add_argument('--output-dir',required=True)
    a=p.parse_args();d=Path(a.output_dir);d.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        r=run(seed,a.dataset,a.max_iter);(d/f'temporal-holdout-{seed}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
