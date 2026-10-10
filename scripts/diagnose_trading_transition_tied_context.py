"""Native NHSMM transition ablation: age-tied distributions with and without external context.

Ground-truth boundary-supervised diagnostic; not NHSMM.optimize().
Both base logits and context-network output projections are age-tied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nhsmm import ModelConfig, NHSMM

K, D = 4, 64
REGIMES = ("bull", "bear", "range", "chaotic")


def tie_age_parameters(transition):
    """Exact age sharing via symmetric initialization and summed gradient hooks."""
    handles = []
    with torch.no_grad():
        transition.logits.copy_(transition.logits.mean(dim=1, keepdim=True).expand_as(transition.logits))
    handles.append(transition.logits.register_hook(lambda g: g.sum(dim=1, keepdim=True).expand_as(g)))
    if transition.context_net is not None:
        last = transition.context_net[-1]
        assert isinstance(last, torch.nn.Linear)
        hidden = last.weight.shape[1]
        with torch.no_grad():
            weights = last.weight.view(K, D, K, hidden)
            weights.copy_(weights.mean(dim=1, keepdim=True).expand_as(weights))
            biases = last.bias.view(K, D, K)
            biases.copy_(biases.mean(dim=1, keepdim=True).expand_as(biases))
        handles.append(last.weight.register_hook(lambda g: g.reshape(K,D,K,hidden).sum(dim=1,keepdim=True).expand(K,D,K,hidden).reshape_as(g)))
        handles.append(last.bias.register_hook(lambda g: g.reshape(K,D,K).sum(dim=1,keepdim=True).expand(K,D,K).reshape_as(g)))
    return handles


def run(dataset: Path, steps: int = 220):
    with np.load(dataset) as z:
        data = {name: z[name] for name in z.files}
    y = data["gt_regime_id"].astype(int)
    episodes = data["gt_episode_id"]
    starts = np.r_[0,np.flatnonzero(episodes[1:] != episodes[:-1])+1]
    boundaries = starts[1:] - 1
    split = int(len(y)*.6)
    train = starts[1:] < split
    test = starts[1:] >= split
    ages = np.clip(data["gt_episode_age"][boundaries].astype(int)-1,0,D-1)
    stress = data["macro_stress"][boundaries].astype(np.float32)
    target_probs = np.column_stack([data["gt_p_next_"+name] for name in REGIMES])
    src = y[boundaries]
    dest = y[boundaries+1]
    torch.set_num_threads(1)

    tr_src = torch.tensor(src[train],dtype=torch.long)
    tr_age = torch.tensor(ages[train],dtype=torch.long)
    tr_dst = torch.tensor(dest[train],dtype=torch.long)
    te_src = torch.tensor(src[test],dtype=torch.long)
    te_age = torch.tensor(ages[test],dtype=torch.long)
    te_dst = torch.tensor(dest[test],dtype=torch.long)
    tr_ctx = torch.tensor(stress[train,None],dtype=torch.float32)
    te_ctx = torch.tensor(stress[test,None],dtype=torch.float32)

    def fit(seed, context):
        torch.manual_seed(seed)
        config = ModelConfig(
            n_states=K,n_features=8,max_duration=D,duration_tail=True,
            causal=True,context_dim=1 if context else None,
            use_context_encoder=False,dropout=0.0,seed=seed,
            transition_init_mode="uniform",verbose=False)
        model = NHSMM(config,device="cpu")
        model.initialize_distributions(jitter=0.0)
        dist=model.dist.transition
        handles=tie_age_parameters(dist)
        params=[p for p in dist.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(params,lr=0.012)
        def logmat(x):
            if context:
                return dist.log_matrix(context=x[:,None,:])[:,0]
            return dist.log_matrix()[0,0]
        def loss():
            m=logmat(tr_ctx)
            if context:
                return -m[torch.arange(len(tr_src)),tr_src,tr_age,tr_dst].mean()
            return -m[tr_src,tr_age,tr_dst].mean()

        before=float(loss().detach())
        for _ in range(steps):
            optimizer.zero_grad()
            l=loss()
            l.backward()
            optimizer.step()
        dist.eval()
        with torch.no_grad():
            probs=logmat(te_ctx).exp()
            if context:
                selected=probs[torch.arange(len(te_src)),te_src,te_age]
                max_age=float((probs-probs[:,:,:1,:]).abs().max())
            else:
                selected=probs[te_src,te_age]
                max_age=float((probs-probs[:,:1,:]).abs().max())
            nll=float(-selected[torch.arange(len(te_dst)),te_dst].clamp_min(1e-12).log().mean())
            mae=float(np.abs(selected.numpy()-target_probs[boundaries[test]]).mean())
            train_nll=float(loss().detach())
        for h in handles:
            h.remove()
        if max_age>1e-6:
            raise AssertionError(f"Age tie broken: {max_age}")
        return {"seed":seed,"context":context,"train_nll_initial":before,"train_nll_final":train_nll,"test_nll":nll,"test_transition_mae":mae,"max_age_probability_difference":max_age}

    runs=[fit(seed,context) for context in (False,True) for seed in (2311,2312)]
    return {"protocol":"GT boundary supervision; age-tied base+context weights; 60/40 chronological split; 220 Adam steps, lr=0.012","train_boundaries":int(train.sum()),"test_boundaries":int(test.sum()),"runs":runs}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=Path("data/reference/trading-regimes-v1/trading_regimes.npz"))
    p.add_argument("--output",type=Path,default=Path("/data/workspace/stage2/age-tied-context.json"))
    args=p.parse_args()
    result=run(args.dataset)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
