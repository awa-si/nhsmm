"""Ground-truth supervised transition ablation: age-specific NHSMM with/without context."""
import json
import numpy as np
import torch
from pathlib import Path
from nhsmm import ModelConfig,NHSMM

D=64
K=4
OUT=Path("/data/workspace/stage2/transition-ablation.json")
with np.load("/workspace/data/reference/trading-regimes-v1/trading_regimes.npz") as z:
    d={k:z[k] for k in z.files}
y=d["gt_regime_id"].astype(int)
starts=np.r_[0,np.flatnonzero(d["gt_episode_id"][1:]!=d["gt_episode_id"][:-1])+1]
boundary=starts[1:]-1
split=int(len(y)*.6)
tr=starts[1:]<split
te=starts[1:]>=split
ages=np.clip(d["gt_episode_age"][boundary].astype(int)-1,0,D-1)
context=d["macro_stress"][boundary].astype("float32")
truth=np.column_stack([d["gt_p_next_"+k] for k in ("bull","bear","range","chaotic")])
src=y[boundary]
dst=y[boundary+1]
torch.set_num_threads(1)

def fit(use_context,seed):
    torch.manual_seed(seed)
    model=NHSMM(ModelConfig(n_states=K,n_features=8,max_duration=D,duration_tail=True,causal=True,context_dim=1 if use_context else None,use_context_encoder=False,dropout=0,seed=seed,transition_init_mode="uniform",verbose=False),device="cpu")
    model.initialize_distributions(jitter=0)
    p=model.dist.transition
    opt=torch.optim.Adam([x for x in p.parameters() if x.requires_grad],lr=.012)
    src_t=torch.tensor(src[tr],dtype=torch.long)
    dst_t=torch.tensor(dst[tr],dtype=torch.long)
    age_t=torch.tensor(ages[tr],dtype=torch.long)
    ctx_t=torch.tensor(context[tr,None],dtype=torch.float32)
    def probs(src,age,ctx):
        if use_context:
            m=p.log_matrix(context=torch.tensor(ctx[:,None,None],dtype=torch.float32))[:,0].exp()
            return m[torch.arange(len(src)),torch.tensor(src),torch.tensor(age)]
        m=p.log_matrix()[0,0].exp()
        return m[torch.tensor(src),torch.tensor(age)]
    def train_lp():
        if use_context:
            m=p.log_matrix(context=ctx_t[:,None,:])[:,0]
            return m[torch.arange(len(src_t)),src_t,age_t,dst_t]
        m=p.log_matrix()[0,0]
        return m[src_t,age_t,dst_t]
    before=float((-train_lp().mean()).detach())
    for step in range(220):
        opt.zero_grad()
        loss=-train_lp().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        predicted=probs(src[te],ages[te],context[te]).numpy()
        train_nll=float((-train_lp().mean()).detach())
        test_nll=float(-np.log(np.clip(predicted[np.arange(len(predicted)),dst[te]],1e-12,1)).mean())
        mae=float(np.abs(predicted-truth[boundary[te]]).mean())
    return {"context":use_context,"seed":seed,"train_nll_initial":before,"train_nll":train_nll,"test_nll":test_nll,"test_transition_mae":mae}
runs=[fit(c,s) for c in (False,True) for s in (2311,2312)]
counts=np.zeros((K,K),dtype=float)
np.add.at(counts,(src[tr],dst[tr]),1)
pooled=counts/counts.sum(1,keepdims=True)
reference={"method":"pooled empirical supervised transition matrix","test_transition_mae":float(np.abs(pooled[src[te]]-truth[boundary[te]]).mean()),"test_nll":float(-np.log(np.clip(pooled[src[te],dst[te]],1e-12,1)).mean())}
result={"method":"Ground-truth boundary supervised transition ablation, fixed 60/40 chronological split","train_boundaries":int(tr.sum()),"test_boundaries":int(te.sum()),"runs":runs,"pooled":reference}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
