"""Small deterministic replay check; not a replacement for full Monte Carlo runs."""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np
import pandas as pd
from inference_simulation import one_fit

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('baseline_replay',ROOT/'baseline/robustness_simulation.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
stored = pd.read_csv(ROOT/'baseline/results/robustness_replicates.csv')
checks=[]
for scenario,g in stored.groupby('scenario',sort=False):
    seed=int(g.iloc[0]['seed'])
    actual=pd.DataFrame(baseline.one_replicate(seed,scenario,60,10)).set_index('method')
    expected=g[g.seed.eq(seed)].set_index('method').loc[actual.index]
    cols=['beta1_intercept','beta1_slope','beta2_intercept','beta2_slope','sigma2']
    delta=float(np.max(np.abs(actual[cols].to_numpy()-expected[cols].to_numpy())))
    passed=bool(np.allclose(actual[cols],expected[cols],atol=5e-5,rtol=0) and actual.converged.equals(expected.converged) and actual.sigma2_bound_hit.equals(expected.sigma2_bound_hit))
    checks.append(dict(kind='robustness',scenario=scenario,seed=seed,method_records=len(actual),max_abs_difference=delta,passed=passed))
    print(checks[-1],flush=True)
stored=pd.read_csv(ROOT/'results/finite_cluster_inference_replicates.csv')
for key,g in stored.groupby(['design','clusters','cluster_size','sigma2'],sort=False):
    design,n,m,v=key
    seed=int(g.iloc[0]['seed'])
    actual=pd.DataFrame(one_fit(seed,design,int(n),int(m),float(v))).set_index(['method','component'])
    expected=g[g.seed.eq(seed)].set_index(['method','component']).loc[actual.index]
    cols=['estimate','standard_error']
    delta=float(np.nanmax(np.abs(actual[cols].to_numpy()-expected[cols].to_numpy())))
    passed=bool(np.allclose(actual[cols],expected[cols],atol=5e-5,rtol=0,equal_nan=True) and actual.valid_inference.equals(expected.valid_inference))
    checks.append(dict(kind='inference',design=design,clusters=int(n),cluster_size=int(m),sigma2=float(v),seed=seed,component_records=len(actual),max_abs_difference=delta,passed=passed))
    print(checks[-1],flush=True)
(ROOT/'results/replay_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
assert all(c['passed'] for c in checks)
