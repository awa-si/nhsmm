from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from nhsmm import NHSMM
from nhsmm.filtering import filter_model_sequence
from validate_trading_regime_usefulness import (_windows,_standardize,_best_perm,_ari,_boundary_f1,_run_medians,_ll,_config,REGIMES,K,TRAIN_WINDOWS,EVAL_WINDOWS)
BALANCED=("log_return_12","trend_48","ewma_vol_12","ewma_vol_48","volume_z_48")
class FrozenNHSMM(NHSMM):
    variant="A"
    def _initialize_run_state(self,*args,**kwargs):
        super()._initialize_run_state(*args,**kwargs)
        with torch.no_grad():
            if self.variant=="B":
                # Long-duration prior, including explicit tail end probability.
                ages=torch.arange(1,65,dtype=self.dist.duration.logits.dtype)
                weights=torch.exp(-((ages-48.)/23.)**2/2)
                self.dist.duration.logits.copy_(weights.log()[None,:].expand(K,-1))
                self.dist.duration.tail_end_probability.fill_(-3.0)
                self.duration_logits_bias.zero_()
            elif self.variant=="C":
                self.dist.transition.logits.zero_()
        for name,p in self.named_parameters():
            freeze=(self.variant=="A" and name.startswith("dist.emission.")) or (self.variant=="B" and (name.startswith("dist.duration.") or name=="duration_logits_bias")) or (self.variant=="C" and name.startswith("dist.transition."))
            if freeze:p.requires_grad_(False)
def run(variant,seed,path):
    torch.set_num_threads(1)
    np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    n=len(data["close"])
    tr=_windows(data,0,int(.6*n),TRAIN_WINDOWS)
    te=_windows(data,int(.8*n),n,EVAL_WINDOWS)
    all_names=("log_return_1","log_return_3","log_return_12","trend_12","trend_48","ewma_vol_12","ewma_vol_48","volume_z_48")
    idx=[all_names.index(x) for x in BALANCED]
    train,test=_standardize(tr[0][:,:,idx],te[0][:,:,idx])
    cfg=_config(seed,context=False,hmm=False)
    cfg.n_features=len(idx)
    m=FrozenNHSMM(cfg,device="cpu");m.variant=variant;m.initialize_distributions()
    m.optimize(torch.tensor(train));m.eval()
    with torch.inference_mode():
        posterior=filter_model_sequence(m,torch.tensor(test)).state_posterior.detach().numpy()
    pred=posterior.argmax(-1);truth=te[2]
    perm,acc=_best_perm(truth.ravel(),pred.ravel())
    mapped=np.asarray(perm)[pred]
    result=dict(variant=variant,seed=seed,matched_accuracy=acc,ari=_ari(truth.ravel(),pred.ravel()),boundary_f1_tol5=_boundary_f1(truth,mapped),predicted_run_medians={REGIMES[k]:v for k,v in _run_medians(mapped).items()},true_run_medians={REGIMES[k]:v for k,v in _run_medians(truth).items()},occupancy=posterior.mean((0,1)).tolist(),oos_ll_per_row=_ll(m,test),permutation=list(perm),duration_tail_end=m.dist.duration.tail_end_probability.detach().tolist())
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--variant",choices=["A","B","C"],required=True);p.add_argument("--seed",type=int,default=901);p.add_argument("--dataset",default="/data/workspace/nhsmm-trading-regimes-v1/trading_regimes.npz");p.add_argument("--output",required=True)
    a=p.parse_args();r=run(a.variant,a.seed,a.dataset);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r))
