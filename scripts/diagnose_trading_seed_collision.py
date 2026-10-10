"""Investigate seed-to-seed identity before expensive training."""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from diagnose_trading_freeze_2d import FrozenNHSMM,BALANCED
from validate_trading_regime_usefulness import _windows,_standardize,_config,TRAIN_WINDOWS,EVAL_WINDOWS

def digest(t):
    a=t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()

def run(seeds,path):
    torch.set_num_threads(1)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    n=len(data['close']);tr=_windows(data,0,int(.6*n),TRAIN_WINDOWS);te=_windows(data,int(.8*n),n,EVAL_WINDOWS)
    names=('log_return_1','log_return_3','log_return_12','trend_12','trend_48','ewma_vol_12','ewma_vol_48','volume_z_48')
    idx=[names.index(x) for x in BALANCED];train,test=_standardize(tr[0][:,:,idx],te[0][:,:,idx]);out=[]
    for seed in seeds:
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
        cfg=_config(seed,context=False,hmm=False);cfg.n_features=len(idx)
        model=FrozenNHSMM(cfg,device='cpu');model.variant='A';model.initialize_distributions()
        before={name:digest(p) for name,p in model.named_parameters()}
        # Instrument the same K-Means initialization used by optimize, without a full training run.
        centers=model._kmeans_centers(torch.tensor(train).reshape(-1,len(idx)),cfg.n_states)
        out.append(dict(seed=seed,initial_parameter_hashes=before,kmeans_centers_sha256=digest(centers),kmeans_centers=centers.tolist()))
    return dict(stage='seed_collision_diagnostic',seeds=out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[904,905,906]);p.add_argument('--dataset',default='data/reference/trading-regimes-v1/trading_regimes.npz');p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.seeds,a.dataset);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps([{'seed':x['seed'],'kmeans_sha256':x['kmeans_centers_sha256'],'initial_hash_count':len(x['initial_parameter_hashes'])} for x in r['seeds']]))
