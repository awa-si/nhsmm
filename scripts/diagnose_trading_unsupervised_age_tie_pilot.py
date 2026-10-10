"""Bounded unsupervised NHSMM pilot: native age-tied versus free age-dependent transitions."""
from __future__ import annotations
import dataclasses
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
sys.path.insert(0,"scripts")
import validate_trading_regime_usefulness as v
from nhsmm import NHSMM
from nhsmm.filtering import filter_model_sequence

DATA=Path("data/reference/trading-regimes-v1/trading_regimes.npz")
OUT=Path("/data/workspace/stage2/unsupervised-age-tie-pilot.json")
SEED=901

def main():
    torch.set_num_threads(1)
    with np.load(DATA) as z:
        d={key:z[key] for key in z.files}
    n=len(d["close"])
    train=v._windows(d,0,int(n*.6),12)
    test=v._windows(d,int(n*.8),n,4)
    train_x,test_x=v._standardize(train[0],test[0])
    torch.manual_seed(SEED)
    results=[]
    for tied in (False,True):
        torch.manual_seed(SEED)
        config=v._config(SEED,context=False,hmm=False)
        if dataclasses.is_dataclass(config):
            config=dataclasses.replace(config,max_iter=12)
        else:
            config.max_iter=12
        model=NHSMM(config,device="cpu")
        model.initialize_distributions(jitter=0.0)
        handles=[]
        if tied:
            # optimize() replaces model.dist in _initialize_run_state. Attach
            # the gradient hook to the fresh transition, after initialization.
            original_initialize=model._initialize_run_state
            def initialize_tied(*args, **kwargs):
                original_initialize(*args, **kwargs)
                matrix=model.dist.transition.logits
                with torch.no_grad():
                    matrix.copy_(matrix.mean(dim=1,keepdim=True).expand_as(matrix))
                handles.append(matrix.register_hook(lambda g:g.sum(dim=1,keepdim=True).expand_as(g)))
            model._initialize_run_state=initialize_tied
        model.optimize(torch.tensor(train_x))
        model.eval()
        with torch.no_grad():
            trace=filter_model_sequence(model,torch.tensor(test_x))
            prediction=trace.state_posterior.argmax(-1).numpy()
            true=test[2]
            perm,accuracy=v._best_perm(true.reshape(-1),prediction.reshape(-1))
            mapped=np.asarray(perm)[prediction]
            nll=v._ll(model,test_x)
            probs=model.dist.transition.log_matrix()[0,0].exp()
            age_error=float((probs-probs[:,:1,:]).abs().max())
        if tied and age_error>1e-5:
            raise AssertionError(f"Age tie not retained after NHSMM.optimize(): {age_error}")
        for h in handles:h.remove()
        results.append({"age_tied":tied,"oos_loglikelihood_per_row":nll,"oos_matched_accuracy":accuracy,"oos_ari":v._ari(true.reshape(-1),prediction.reshape(-1)),"oos_boundary_f1_tol5":v._boundary_f1(true,mapped,tol=5),"age_invariance_error":age_error})
    result={"protocol":"Unsupervised NHSMM.optimize() pilot; train 12 windows x480 first60%; test4 windows x480 last20%; 12 iterations; no context; same seed; labels evaluation only","seed":SEED,"results":results}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
