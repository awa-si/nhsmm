"""Measure generator-grounded duration and boundary transition structure."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from generate_trading_regime_dataset import REGIMES,BASE_TRANSITION

def run(path):
 with np.load(path) as z:d={k:z[k] for k in z.files}
 states=d['gt_regime_id'];episodes=d['gt_episode_id'];n=len(states)
 starts=np.r_[0,np.flatnonzero(episodes[1:]!=episodes[:-1])+1];ends=np.r_[starts[1:],n]
 counts=np.zeros((4,4),dtype=int)
 for a,b in zip(starts[:-1],starts[1:]):counts[states[a],states[b]]+=1
 summary=[]
 for k,name in enumerate(REGIMES):
  durations=(ends-starts)[states[starts]==k];full=durations.copy()
  # Exclude last censored episode from duration estimation.
  if states[starts[-1]]==k:full=full[:-1]
  summary.append(dict(state=name,episodes=int(len(durations)),complete_episodes=int(len(full)),mean_duration=float(full.mean()),median_duration=float(np.median(full)),p90_duration=float(np.percentile(full,90)),mass_duration_gt64=float(np.mean(full>64)),mass_duration_ge64=float(np.mean(full>=64)),empirical_hazard_at64=float(np.mean(full==64)/np.mean(full>=64)) if np.any(full>=64) else None))
 transitions=(counts/counts.sum(axis=1,keepdims=True)).tolist()
 return dict(rows=n,episodes=len(starts),max_duration=64,duration=summary,boundary_counts=counts.tolist(),boundary_probabilities=transitions,base_transition=BASE_TRANSITION.tolist(),diagonal_transition_count=int(np.trace(counts)),macro_stress_min=float(d['macro_stress'].min()),macro_stress_max=float(d['macro_stress'].max()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/reference/trading-regimes-v1/trading_regimes.npz');p.add_argument('--output',required=True);a=p.parse_args();result=run(a.dataset);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
