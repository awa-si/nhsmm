"""Ground-truth supervised diagnostics of actual NHSMM duration and transition distributions.

This is NOT NHSMM.optimize() and is not an unsupervised recovery benchmark.
"""
import json
from pathlib import Path

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM

REGIMES = ("bull", "bear", "range", "chaotic")
K, D = 4, 64
DATA = Path("/workspace/data/reference/trading-regimes-v1/trading_regimes.npz")
OUTPUT = Path("/data/workspace/stage2/oracle-native-distributions.json")

def duration_loss(duration, states, lengths):
    # D+ bucket is P(length>=D); geometric tail for length>=D.
    logp = duration.log_matrix()[0, 0]
    tail = duration.tail_probability()
    bucket = torch.clamp(lengths-1, max=D-1)
    picked = logp[states, bucket]
    extra = torch.clamp(lengths-D, min=0)
    lp = picked + torch.where(lengths>=D, extra*torch.log1p(-tail[states]) + torch.log(tail[states]), torch.zeros_like(picked))
    return -lp.mean()

def train_duration(model, train_state, train_length, test_state, test_length):
    duration = model.dist.duration
    for p in duration.parameters():
        p.requires_grad_(True)
    torch.manual_seed(2311)
    opt = torch.optim.Adam([duration.logits,duration.tail_end_probability], lr=.08)
    initial=float(duration_loss(duration,train_state,train_length).detach())
    for step in range(350):
        opt.zero_grad()
        loss=duration_loss(duration,train_state,train_length)
        loss.backward()
        opt.step()
        with torch.no_grad():
            duration.tail_end_probability.clamp_(.001,.999)
    fitted=float(duration_loss(duration,train_state,train_length).detach())
    heldout=float(duration_loss(duration,test_state,test_length).detach())
    with torch.no_grad():
        p=duration.log_matrix()[0,0].exp().numpy()
        h=duration.tail_probability().numpy()
        # P(d>=d) mass for D+; model-implied P(d>64) = P(d>=64)*(1-h).
        tail_gt64=p[:,-1]*(1-h)
        estimate=[]
        for row,hk in zip(p,h):
            support=np.arange(1,D)
            durations=np.r_[support,np.arange(D, D+900)]
            masses=np.r_[row[:-1], row[-1]*hk*((1-hk)**np.arange(900))]
            masses=masses/masses.sum()
            estimate.append(float(durations[np.searchsorted(np.cumsum(masses),.5)]))
    gt={}
    for k,name in enumerate(REGIMES):
        tr=train_length[train_state==k].numpy()
        te=test_length[test_state==k].numpy()
        gt[name]={"train_median":float(np.median(tr)),"test_median":float(np.median(te)),"predicted_median":estimate[k],"train_gt64":float(np.mean(tr>64)),"test_gt64":float(np.mean(te>64)),"predicted_gt64":float(tail_gt64[k]),"tail_end_probability":float(h[k])}
    median_relative=float(np.median([abs(gt[name]["predicted_median"]-gt[name]["test_median"])/gt[name]["test_median"] for name in REGIMES]))
    tail_mae=float(np.mean([abs(gt[name]["predicted_gt64"]-gt[name]["test_gt64"]) for name in REGIMES]))
    return {"initial_train_nll":initial,"final_train_nll":fitted,"final_test_nll":heldout,"median_relative_error":median_relative,"tail_gt64_mae":tail_mae,"states":gt}

def trans_logprob(distribution, src, age, dst, ctx=None):
    if ctx is None:
        logits=distribution.log_matrix()[0,0]
        return logits[src,age,dst]
    logits=distribution.log_matrix(context=ctx[:,None,:])[:,0]
    return logits[torch.arange(len(src)),src,age,dst]

def train_trans(model, train_src, train_age, train_dst, train_ctx, test_src, test_age, test_dst, test_ctx, gt_probs):
    distribution=model.dist.transition
    opt=torch.optim.Adam([p for p in distribution.parameters() if p.requires_grad],lr=.012)
    before=float(-trans_logprob(distribution,train_src,train_age,train_dst,train_ctx).mean())
    for step in range(220):
        opt.zero_grad()
        lp=trans_logprob(distribution,train_src,train_age,train_dst,train_ctx)
        loss=-lp.mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        fitted=float(-trans_logprob(distribution,train_src,train_age,train_dst,train_ctx).mean())
        test_nll=float(-trans_logprob(distribution,test_src,test_age,test_dst,test_ctx).mean())
        matrices=distribution.log_matrix(context=test_ctx[:,None,:])[:,0].exp()
        pred=matrices[torch.arange(len(test_src)),test_src,test_age].numpy()
        mae=float(np.abs(pred-gt_probs).mean())
        grad_nonzero=any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in distribution.parameters() if p.requires_grad)
    return {"initial_train_nll":before,"final_train_nll":fitted,"test_nll":test_nll,"test_transition_mae":mae,"finite_nonzero_parameter_gradients":bool(grad_nonzero)}

def main():
    torch.set_num_threads(1)
    np.random.seed(2311)
    torch.manual_seed(2311)
    with np.load(DATA) as z:
        d={k:z[k] for k in z.files}
    y=d["gt_regime_id"].astype(np.int64)
    e=d["gt_episode_id"]
    n=len(y)
    cut=int(.6*n)
    starts=np.r_[0,np.flatnonzero(e[1:]!=e[:-1])+1]
    ends=np.r_[starts[1:],n]
    states=y[starts]
    durations=ends-starts
    tr=(ends<=cut)
    te=(starts>=cut)&(ends<n)
    model=NHSMM(ModelConfig(n_states=K,n_features=8,max_duration=D,duration_tail=True,causal=True,context_dim=1,use_context_encoder=False,dropout=0,seed=2311,duration_init_mode="uniform",transition_init_mode="uniform",verbose=False),device="cpu")
    model.initialize_distributions(jitter=0.0)
    tolong=lambda a:torch.tensor(a,dtype=torch.long)
    result={"method":"NHSMM native distributions supervised with true episode boundaries; no NHSMM.optimize","train_rows":cut,"test_rows":n-cut,"duration":train_duration(model,tolong(states[tr]),tolong(durations[tr]),tolong(states[te]),tolong(durations[te]))}
    # Transition at the boundary, with context from the bar preceding next episode.
    indices=np.arange(len(starts)-1)
    b=starts[1:]-1
    train=(starts[1:]<cut)
    test=(starts[1:]>=cut)
    ctx=d["macro_stress"][b].astype("float32")
    # Match model's age-dependence using episode age of source at boundary.
    ages=np.minimum(d["gt_episode_age"][b].astype("int64")-1,D-1)
    gt=np.column_stack([d["gt_p_next_"+k] for k in REGIMES])
    result["transition"]=train_trans(model,tolong(y[b[train]]),tolong(ages[train]),tolong(y[b[train]+1]),torch.tensor(ctx[train,None]),tolong(y[b[test]]),tolong(ages[test]),tolong(y[b[test]+1]),torch.tensor(ctx[test,None]),gt[b[test]])
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
