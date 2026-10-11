"""Paired synthetic trading-regime test of native emission initializers.

Keep all unsupervised model settings and data fixed; only emission_init_mode changes.
Ground-truth states are used for evaluation, never for model fitting.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_trading_regime_usefulness import (
    FEATURES, _ari, _best_perm, _boundary_f1, _config,
    _ll, _standardize, _windows,
)
from nhsmm import NHSMM
from nhsmm.filtering import filter_model_sequence

MODES=("kmeans","kmeans_cluster_variance")
SEEDS=(901,902,903)
DATA=Path("data/reference/trading-regimes-v1/trading_regimes.npz")


def one(seed, mode, train_x, test_x, test_states, max_iter):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg=dataclasses.replace(
        _config(seed,context=False,hmm=False),
        emission_init_mode=mode, max_iter=max_iter,
    )
    model=NHSMM(cfg,device="cpu")
    model.initialize_distributions(jitter=0.0)
    snapshot={}
    original=model._initialize_run_state

    def init_and_capture(*args,**kwargs):
        original(*args,**kwargs)
        emission=model.dist.emission
        snapshot["initial_variances"]=emission.log_var.detach().exp().cpu().numpy().tolist()
        snapshot["initial_centers"]=emission.mu.detach().cpu().numpy().tolist()

    model._initialize_run_state=init_and_capture
    model.optimize(torch.tensor(train_x))
    model.eval()
    with torch.inference_mode():
        posterior=filter_model_sequence(model,torch.tensor(test_x)).state_posterior.detach().cpu().numpy()
        pred=posterior.argmax(-1)
    perm,accuracy=_best_perm(test_states.ravel(),pred.ravel())
    aligned=np.asarray(perm)[pred]
    emission=model.dist.emission
    return {
        "seed":seed,"mode":mode,"matched_accuracy":accuracy,
        "ari":_ari(test_states.ravel(),pred.ravel()),
        "boundary_f1_tol5":_boundary_f1(test_states,aligned,tol=5),
        "test_ll_per_row":_ll(model,test_x),
        "state_posterior_occupancy":posterior.mean(axis=(0,1)).tolist(),
        "state_argmax_occupancy":np.bincount(pred.ravel(),minlength=4).tolist(),
        "initial_variances":snapshot["initial_variances"],
        "final_variances":emission.log_var.detach().exp().cpu().numpy().tolist(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=DATA)
    p.add_argument("--train-windows",type=int,default=8)
    p.add_argument("--test-windows",type=int,default=4)
    p.add_argument("--iterations",type=int,default=10)
    p.add_argument("--output",type=Path,default=Path("/data/workspace/stage2/emission-init-ablation.json"))
    a=p.parse_args()
    torch.set_num_threads(1)
    with np.load(a.dataset) as f: data={k:f[k] for k in f.files}
    n=len(data["close"])
    train=_windows(data,0,int(n*.6),a.train_windows)
    test=_windows(data,int(n*.8),n,a.test_windows)
    train_x,test_x=_standardize(train[0],test[0])
    runs=[]
    for seed in SEEDS:
        for mode in MODES:
            result=one(seed,mode,train_x,test_x,test[2],a.iterations)
            runs.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ("initial_variances","final_variances")}),flush=True)
    grouped={}
    for mode in MODES:
        subset=[r for r in runs if r["mode"]==mode]
        grouped[mode]={key:float(np.median([r[key] for r in subset])) for key in ("matched_accuracy","ari","boundary_f1_tol5","test_ll_per_row")}
    report={
        "protocol":{"dataset":str(a.dataset),"train_windows":a.train_windows,"test_windows":a.test_windows,"iterations":a.iterations,"seeds":SEEDS,"modes":MODES,"chronological_train_fraction":.6,"chronological_test_start":.8,"labels_used_only_for_evaluation":True},
        "summary_median":grouped,"runs":runs,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("MEDIANS",json.dumps(grouped),flush=True)

if __name__=="__main__":
    main()
