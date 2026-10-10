"""Stage 2S: multi-seed emission-strength × boundary self-transition bias sweep."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from nhsmm.filtering import filter_model_sequence
from diagnose_trading_freeze_2d import FrozenNHSMM,BALANCED
from validate_trading_regime_usefulness import (_windows,_standardize,_best_perm,_ari,_boundary_f1,_run_medians,_config,REGIMES,TRAIN_WINDOWS,EVAL_WINDOWS)

SCALES=(0.25,1.0)
BONUSES=(0.0,0.5,1.0,2.0)

def run(seed,path,max_iter=30):
    torch.set_num_threads(1);np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    n=len(data['close']);tr=_windows(data,0,int(.6*n),TRAIN_WINDOWS);te=_windows(data,int(.8*n),n,EVAL_WINDOWS)
    names=('log_return_1','log_return_3','log_return_12','trend_12','trend_48','ewma_vol_12','ewma_vol_48','volume_z_48')
    idx=[names.index(x) for x in BALANCED]
    train,test=_standardize(tr[0][:,:,idx],te[0][:,:,idx]);truth=te[2]
    cfg=_config(seed,context=False,hmm=False);cfg.n_features=len(idx);cfg.max_iter=max_iter
    model=FrozenNHSMM(cfg,device='cpu');model.variant='A';model.initialize_distributions();model.optimize(torch.tensor(train));model.eval()
    original_sequence=model._build_sequence_set
    original_transition=model.dist.transition.log_matrix
    results=[]
    for scale in SCALES:
        def scaled_sequence(*args,**kwargs):
            sequence=original_sequence(*args,**kwargs);sequence.log_probs=sequence.log_probs*scale;return sequence
        model._build_sequence_set=scaled_sequence
        for bonus in BONUSES:
            def boosted_transition(*args,**kwargs):
                logp=original_transition(*args,**kwargs)
                if bonus==0:return logp
                eye=torch.eye(logp.shape[-1],dtype=logp.dtype,device=logp.device)
                adjusted=logp+bonus*eye.unsqueeze(-2) if logp.ndim==5 else logp+bonus*eye
                return torch.log_softmax(adjusted,dim=-1)
            model.dist.transition.log_matrix=boosted_transition
            with torch.inference_mode():posterior=filter_model_sequence(model,torch.tensor(test)).state_posterior.numpy()
            pred=posterior.argmax(-1);perm,acc=_best_perm(truth.ravel(),pred.ravel());mapped=np.asarray(perm)[pred]
            results.append(dict(scale=scale,persistence_bonus=bonus,matched_accuracy=acc,ari=_ari(truth.ravel(),pred.ravel()),boundary_f1_tol5=_boundary_f1(truth,mapped),predicted_run_medians={REGIMES[k]:v for k,v in _run_medians(mapped).items()},permutation=list(perm),switches=int((pred[:,1:]!=pred[:,:-1]).sum())))
    model._build_sequence_set=original_sequence;model.dist.transition.log_matrix=original_transition
    return dict(stage='2S',seed=seed,max_iter=max_iter,variant='A',results=results,note='Inference-only intervention: emission log-score multiplier and diagonal log-bonus to normalized episode-boundary transition matrix. Self-transition at a boundary starts a new episode; this does not directly change duration hazard. One common model fit per seed.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[904,905,906]);p.add_argument('--dataset',default='data/reference/trading-regimes-v1/trading_regimes.npz');p.add_argument('--max-iter',type=int,default=30);p.add_argument('--output-dir',required=True)
    a=p.parse_args();d=Path(a.output_dir);d.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        r=run(seed,a.dataset,a.max_iter);(d/f'stage2s-A-{seed}.json').write_text(json.dumps(r,indent=2));print(json.dumps({'seed':seed,'results':r['results']}),flush=True)
