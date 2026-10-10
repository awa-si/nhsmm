"""Age-tied native transition context regularization benchmark on known episode boundaries."""
from pathlib import Path
import json
import numpy as np
import torch
from nhsmm import NHSMM, ModelConfig

K,D=4,64
def main():
    with np.load("data/reference/trading-regimes-v1/trading_regimes.npz") as z:
        d={k:z[k] for k in z.files}
    y=d["gt_regime_id"].astype(int)
    episode=d["gt_episode_id"]
    starts=np.r_[0,np.flatnonzero(episode[1:]!=episode[:-1])+1]
    b=starts[1:]-1
    cut=int(len(y)*.6)
    train=starts[1:]<cut
    test=~train
    age=np.clip(d["gt_episode_age"][b].astype(int)-1,0,D-1)
    stress=d["macro_stress"][b].astype(np.float32)
    probs_gt=np.column_stack([d["gt_p_next_"+n] for n in ("bull","bear","range","chaotic")])
    src=y[b];dst=y[b+1]
    torch.set_num_threads(1)
    def tensors(mask):
        return (torch.tensor(src[mask],dtype=torch.long),torch.tensor(age[mask],dtype=torch.long),torch.tensor(dst[mask],dtype=torch.long),torch.tensor(stress[mask,None],dtype=torch.float32))
    tr=tensors(train);te=tensors(test)
    def fit(seed,context,decay):
        torch.manual_seed(seed)
        model=NHSMM(ModelConfig(n_states=K,n_features=8,max_duration=D,duration_tail=True,causal=True,context_dim=1 if context else None,use_context_encoder=False,dropout=0,seed=seed,transition_init_mode="uniform",verbose=False),device="cpu")
        model.initialize_distributions(jitter=0)
        p=model.dist.transition
        hooks=[p.logits.register_hook(lambda g:g.sum(dim=1,keepdim=True).expand_as(g))]
        if context:
            layer=p.context_net[-1]
            hidden=layer.weight.shape[-1]
            with torch.no_grad():
                a=layer.weight.view(K,D,K,hidden)
                a.copy_(a.mean(dim=1,keepdim=True).expand_as(a))
                a=layer.bias.view(K,D,K)
                a.copy_(a.mean(dim=1,keepdim=True).expand_as(a))
            hooks.append(layer.weight.register_hook(lambda g:g.view(K,D,K,hidden).sum(dim=1,keepdim=True).expand(K,D,K,hidden).reshape_as(g)))
            hooks.append(layer.bias.register_hook(lambda g:g.view(K,D,K).sum(dim=1,keepdim=True).expand(K,D,K).reshape_as(g)))
        parameters=[{"params":[x for name,x in p.named_parameters() if x.requires_grad and not name.startswith("context_net")],"weight_decay":0.0}]
        if context:
            parameters.append({"params":[x for name,x in p.named_parameters() if x.requires_grad and name.startswith("context_net")],"weight_decay":decay})
        opt=torch.optim.Adam(parameters,lr=.012)
        def selected(vals):
            s,a,_,c=vals
            if context:
                m=p.log_matrix(context=c[:,None,:])[:,0]
                return m[torch.arange(len(s)),s,a]
            return p.log_matrix()[0,0][s,a]
        for i in range(220):
            opt.zero_grad()
            loss=-selected(tr)[torch.arange(len(tr[0])),tr[2]].mean()
            loss.backward()
            opt.step()
        p.eval()
        with torch.no_grad():
            test_p=selected(te).exp()
            train_loss=float(-selected(tr)[torch.arange(len(tr[0])),tr[2]].mean())
            test_loss=float(-test_p[torch.arange(len(te[0])),te[2]].clamp_min(1e-12).log().mean())
            mae=float(np.abs(test_p.numpy()-probs_gt[b[test]]).mean())
            if context:
                matrices=p.log_matrix(context=te[3][:,None,:])[:,0].exp()
                age_diff=float((matrices-matrices[:,:,:1,:]).abs().max())
            else:
                matrices=p.log_matrix()[0,0].exp()
                age_diff=float((matrices-matrices[:,:1,:]).abs().max())
        for h in hooks:h.remove()
        if age_diff>1e-6:raise AssertionError(f"Age tying failed: {age_diff}")
        return {"seed":seed,"context":context,"context_weight_decay":decay,"train_nll":train_loss,"test_nll":test_loss,"test_mae":mae,"age_invariance_max_error":age_diff}
    runs=[fit(2311,False,0.0)]
    for decay in (0.0,0.01,0.1):
        for seed in (2311,2312):
            runs.append(fit(seed,True,decay))
    result={"method":"Native age-tied contextual transition regularization, ground-truth supervised, 60/40 chronological split","train_boundaries":int(train.sum()),"test_boundaries":int(test.sum()),"runs":runs}
    out=Path("/data/workspace/stage2/age-tied-context-regularization.json")
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
