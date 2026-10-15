"""Reproduce model-boundary findings on the frozen 2026-10-15 audit source."""

from pathlib import Path
from unittest.mock import patch
import argparse
import json
import subprocess

import torch
from nhsmm import NHSMM, ModelConfig
from nhsmm.context import SequenceSet
from nhsmm.distributions import Duration
from nhsmm.models import DistributionSet


def _sequence(emission):
    t = emission.shape[0]
    return SequenceSet(
        sequences=torch.zeros(1, t, 1, dtype=emission.dtype),
        lengths=torch.tensor([t]),
        masks=torch.ones(1, t, 1, dtype=torch.bool),
        contexts=torch.zeros(1, t, 0, dtype=emission.dtype),
        canonical=torch.zeros(1, 1, 0, dtype=emission.dtype),
        log_probs=emission.unsqueeze(0),
    )


assert subprocess.check_output(
    ["git", "hash-object", "nhsmm/models/base.py"], text=True
).strip() == "47571aca9bcaea304464a925d96e9ad0bd911df5", "audit source changed"
torch.manual_seed(20261015)
torch.set_num_threads(1)
out={"source_blob":subprocess.check_output(["git","hash-object","nhsmm/models/base.py"],text=True).strip(),"cases":{}}
def make(**kw):
    cfg=ModelConfig(n_states=1,n_features=1,max_duration=1,n_init=1,max_iter=1,
        convergence_stop=False,use_scheduler=False,verbose=False,optimizer_initialization="current",**kw)
    model=NHSMM(cfg,device="cpu")
    model.initialize_distributions()
    model.eval()
    return model
for causal in [True,False]:
    for tail in [False,True]:
        m=make(causal=causal,duration_tail=tail,duration_tail_init_probability=1.)
        with torch.no_grad():
            m.dist.emission.mu.zero_()
            m.dist.emission.log_var.zero_()
        x=torch.tensor([[[1000.]]])
        expected=m.dist.emission.log_prob(x).sum()
        observed=m.log_likelihood(x,reduce=True)
        g=torch.autograd.grad(observed,m.dist.emission.mu)[0]
        out["cases"][f"d1_density_causal_{causal}_tail_{tail}"]={"expected":float(expected.detach()),"observed":float(observed.detach()),"mean_gradient":float(g.detach())}
for chunk in [None,1]:
    m=make(causal=True,optimizer_batch_size=chunk)
    before={k:v.clone() for k,v in m.state_dict().items()}
    try:
        m.optimize(torch.empty(0,2,1))
        status={"accepted":True,"restart_scores":m._restart_scores,"history":m._optimization_history}
    except Exception as e:
        status={"accepted":False,"exception":type(e).__name__,"message":str(e)}
    status["weights_changed"]=any(not torch.equal(v,before[k]) for k,v in m.state_dict().items())
    out["cases"][f"empty_training_chunk_{chunk}"]=status
m=make(causal=True,context_dim=1,transition_refine_steps=1)
before={k:v.clone() for k,v in m.state_dict().items()}
try:
    m.optimize(torch.tensor([[[0.],[1.],[2.]]]))
    status={"accepted":True}
except Exception as e:
    status={"accepted":False,"exception":type(e).__name__,"message":str(e)}
status["weights_changed"]=any(not torch.equal(v,before[k]) for k,v in m.state_dict().items())
status["history"]=getattr(m,"_optimization_history",None)
out["cases"]["missing_refinement_context"]=status
m=make(causal=True)
a=m.predict(torch.empty(1,0,1),mode="log_likelihood",verbose=False)
b=m.log_likelihood(torch.empty(1,0,1))
out["cases"]["empty_predict_likelihood_type"]={"predict_type":type(a).__name__,"direct_type":type(b).__name__}
m=NHSMM(ModelConfig(n_states=2,n_features=1,max_duration=2,causal=True,use_context_encoder=True,n_init=1,verbose=False),device="cpu")
m.initialize_distributions()
m.double()
try:
    y=m.log_likelihood(torch.randn(1,3,1,dtype=torch.float64))
    status={"accepted":True,"dtype":str(y.dtype)}
except Exception as e:
    status={"accepted":False,"exception":type(e).__name__,"message":str(e)}
out["cases"]["double_internal_encoder"]=status
for causal in [True,False]:
    for tail in [False,True]:
        m=make(causal=causal,duration_tail=tail)
        try:
            alpha=m.forward(m._build_sequence_set(torch.empty(1,0,1)))
            status={"accepted":True,"shape":list(alpha.shape)}
        except Exception as e:
            status={"accepted":False,"exception":type(e).__name__,"message":str(e)}
        out["cases"][f"empty_forward_causal_{causal}_tail_{tail}"]=status

base_result=out
out={}
def model(**kw):
    m=NHSMM(ModelConfig(n_states=1,n_features=1,max_duration=1,n_init=1,max_iter=1,convergence_stop=False,use_scheduler=False,verbose=False,optimizer_initialization="current",**kw),device="cpu")
    m.initialize_distributions()
    m.eval()
    return m
for causal in [True,False]:
    for tail in [False,True]:
        m=model(causal=causal,duration_tail=tail,duration_tail_init_probability=1.).double()
        with torch.no_grad():
            m.dist.emission.mu.zero_()
            m.dist.emission.log_var.zero_()
        x=torch.tensor([[[1e20]]],dtype=torch.float64)
        # Use the exact coerced observations the public method itself scores.
        expected=m.dist.emission.log_prob(m._ensure_tensor(x)).sum()
        observed=m.log_likelihood(x,reduce=True)
        g=torch.autograd.grad(observed,m.dist.emission.mu)[0]
        out[f"float64_density_causal_{causal}_tail_{tail}"]={"expected":float(expected.detach()),"observed":float(observed.detach()),"mean_gradient":float(g.detach())}
m=model(causal=True)
x=torch.tensor([[0.],[1.],[2.]])
out["empty_sequence_objective"]={
    "nonempty":float(m._target_training_score([x],None,1.)),
    "with_one_empty":float(m._target_training_score([x,torch.empty(0,1)],None,1.)),
    "with_two_empty":str(float(m._target_training_score([x,torch.empty(0,1),torch.empty(0,1)],None,1.))),
}
class OnlyTwo(Duration):
    def _apply_constraints(self, logits, mask=None):
        return super()._apply_constraints(logits, mask=torch.tensor([False,True],device=logits.device))
cfg=ModelConfig(n_states=2,n_features=1,max_duration=2,causal=False,verbose=False)
m=NHSMM(cfg,device="cpu")
m.dist=DistributionSet(config=m.config,duration=OnlyTwo)
m.dist.initialize()
x=torch.zeros(1,1,1)
out["impossible_terminal_decode"]={"log_likelihood":str(float(m.log_likelihood(x).detach()[0])),"path":m.decode(x,verbose=False).tolist()}
m=model(causal=True)
seq=_sequence(torch.tensor([[-torch.inf],[0.]],dtype=torch.float64))
alpha=m.forward(seq)
out["zero_evidence_causal_forward"]={"nan_entries":int(torch.isnan(alpha).sum()),"elements":alpha.numel()}
with patch.object(m,"_build_sequence_set",return_value=seq):
    ll=m.log_likelihood(torch.zeros(1,2,1))
    out["zero_evidence_causal_forward"]["public_ll"]=float(ll.detach()[0])

supplement_result=out
assert base_result["cases"]["empty_training_chunk_1"]["accepted"]
assert base_result["cases"]["missing_refinement_context"]["weights_changed"]
assert not base_result["cases"]["double_internal_encoder"]["accepted"]
assert supplement_result["float64_density_causal_False_tail_False"]["mean_gradient"] == 0
assert supplement_result["float64_density_causal_True_tail_False"]["mean_gradient"] > 0
assert supplement_result["zero_evidence_causal_forward"]["nan_entries"] == 2
assert supplement_result["impossible_terminal_decode"]["log_likelihood"] == "-inf"
assert supplement_result["impossible_terminal_decode"]["path"] == [0]
assert supplement_result["empty_sequence_objective"]["with_one_empty"] < -1e38
assert supplement_result["empty_sequence_objective"]["with_two_empty"] == "-inf"
assert base_result["cases"]["empty_predict_likelihood_type"] == {
    "predict_type": "list", "direct_type": "Tensor"
}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
payload={"schema":"nhsmm-models-audit-reproduction-v1","base":base_result,
         "supplement":supplement_result,"cases":20,"findings_reproduced":7,
         "research_fits":0,"core_changed":False}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
print(json.dumps(payload,allow_nan=False))
