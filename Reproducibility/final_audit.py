"""Reaggregate supplied formal records and generate submission tables/figures.

This program does not rerun the Monte Carlo experiments. BMI is rerun separately.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inference_simulation import summarize as inference_summary

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SOURCE = ROOT.parent / 'Manuscript_Source'
(SOURCE / 'generated').mkdir(exist_ok=True)
(SOURCE / 'figures').mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('baseline_audit', ROOT / 'baseline' / 'robustness_simulation.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)

rob = pd.read_csv(ROOT / 'baseline/results/robustness_replicates.csv')
inf = pd.read_csv(RESULTS / 'finite_cluster_inference_replicates.csv')
assert len(rob) == 6400 and len(inf) == 14400
assert rob.groupby(['scenario', 'method']).size().eq(200).all()
assert inf.groupby(['design', 'clusters', 'cluster_size', 'sigma2', 'method', 'component']).size().eq(200).all()
assert np.isfinite(rob[['beta_squared_error', 'sigma2_error']]).all().all()
rs = baseline.summarize(rob)
fs = inference_summary(inf)
for computed, stored, keys in [
    (rs, pd.read_csv(ROOT / 'baseline/results/robustness_summary.csv'), ['scenario','method']),
    (fs, pd.read_csv(RESULTS / 'finite_cluster_inference_summary.csv'), ['design','clusters','cluster_size','sigma2','method','component'])
]:
    pd.testing.assert_frame_equal(computed.sort_values(keys).reset_index(drop=True), stored.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-11, rtol=1e-10)

scenarios = ['clean','individual-label','cluster-shift','t3-random-effect']
scenario_labels = ['Clean','Individual label','Cluster shift','Scaled $t_3$']
plot_scenario_labels = ['Clean','Individual-label\ncontamination','Cluster shift','Scaled $t_3$\nrandom effect']
methods = ['CR(-0.5)','CR(-0.25)','CR(0)','CR(0.666667)','CR(1)','CP-DPD(0.3)','weighted-MDPD(0.2)','weighted-marginal-ML']
labels = [r'PM$\phi$($-1/2$)',r'PM$\phi$($-1/4$)',r'PM$\phi$($0$)',r'PM$\phi$($2/3$)',r'PM$\phi$($1$)','CP-DPD','W-DPD','ML']
metrics = [('beta_rmse','robust_rmse'),('sigma2_mae','robust_variance'),('sigma2_bound_rate','robust_boundary')]
for metric, name in metrics:
    lines=[]
    for sc, lab in zip(scenarios,scenario_labels):
        vals=[float(rs[(rs.scenario==sc)&(rs.method==m)][metric].iloc[0]) for m in methods]
        lines.append(lab+' & '+' & '.join(f'{v:.4f}' for v in vals)+r' \\')
    (SOURCE/'generated'/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

rows=[]
for keys,g in inf.groupby(['design','clusters','cluster_size','sigma2','method','component'],sort=False):
    valid=g[g.valid_inference.eq(1)]
    n=len(g); nv=len(valid)
    cov=float(valid.cover_normal.mean()) if nv else np.nan
    rej=float(valid.reject_zero.mean()) if nv else np.nan
    def se(p): return float(np.sqrt(p*(1-p)/nv)) if nv else np.nan
    def wilson(p):
        z=norm.ppf(.975); den=1+z*z/nv
        mid=(p+z*z/(2*nv))/den
        half=z*np.sqrt(p*(1-p)/nv+z*z/(4*nv*nv))/den
        return mid-half,mid+half
    rec=dict(zip(['design','clusters','cluster_size','sigma2','method','component'],keys))
    rec.update(total=n,valid_count=nv,convergence_rate=g.converged.mean(),boundary_rate=g.sigma_bound_hit.mean(),valid_rate=nv/n,coverage=cov,coverage_mcse=se(cov),rejection_zero=rej,rejection_mcse=se(rej),valid_and_covered=g.cover_normal.mean(),valid_and_rejected=g.reject_zero.mean())
    rec['coverage_wilson_low'],rec['coverage_wilson_high']=wilson(cov)
    rec['rejection_wilson_low'],rec['rejection_wilson_high']=wilson(rej)
    rows.append(rec)
full=pd.DataFrame(rows)
full.to_csv(RESULTS/'submission_inference_diagnostics.csv',index=False)
order=['CR(0)','CR(0.666667)','weighted-marginal-ML']
mlab={'CR(0)':r'PM$\phi$($0$)','CR(0.666667)':r'PM$\phi$($2/3$)','weighted-marginal-ML':'ML'}
configs=[(60,10,.1),(60,20,.5),(100,10,1.)]
for design,name,metric,mcse in [('coverage','coverage','coverage','coverage_mcse'),('null-slope','type1','rejection_zero','rejection_mcse')]:
    lines=[]
    for n,m,v in configs:
        for meth in order:
            row=full[(full.design==design)&(full.clusters==n)&(full.cluster_size==m)&(full.sigma2==v)&(full.method==meth)&(full.component=='beta2_slope')].iloc[0]
            line=f'{n} & {m} & {v:g} & {mlab[meth]} & {row.boundary_rate:.3f} & {row.valid_rate:.3f} & {row[metric]:.3f} & {row[mcse]:.3f}'
            if design=='coverage': line+=f' & {row.valid_and_covered:.3f}'
            lines.append(line+r' \\')
    (SOURCE/'generated'/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

bmi=pd.read_csv(RESULTS/'bmi_aggregate_estimates.csv')
assert len(bmi)==5 and bmi.converged.eq(1).all() and bmi.sigma_at_lower_bound.eq(1).all()
lines=[]
for row,lam in zip(bmi.itertuples(index=False),['0',r'$2/3$','1','1.5','2']):
    lines.append(lam+' & '+' & '.join(f'{getattr(row,k):.4f}' for k in ['beta11_gender','beta12_age','beta21_gender','beta22_age','sigma2','objective'])+r' \\')
(SOURCE/'generated/bmi_fit.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
method_colors=['#0072B2']*5+['#E69F00','#009E73','#666666']
method_markers=['o']*5+['^','s','D']
fig,axes=plt.subplots(2,4,figsize=(10.5,5.5),sharex='row',sharey=True,layout='constrained')
y=np.arange(len(methods))
for col,(sc,lab) in enumerate(zip(scenarios,plot_scenario_labels)):
    sub=rs[rs.scenario.eq(sc)].set_index('method').loc[methods]
    axes[0,col].set_title(lab,fontsize=11.5)
    for row,(metric,err) in enumerate([('beta_rmse','beta_rmse_mcse'),('sigma2_bound_rate','sigma2_bound_rate_mcse')]):
        ax=axes[row,col]
        for pos,(meth,color,marker) in enumerate(zip(methods,method_colors,method_markers)):
            ax.errorbar(
                float(sub.loc[meth,metric]),pos,
                xerr=1.96*float(sub.loc[meth,err]),
                color=color,marker=marker,markersize=4,linestyle='none',
                elinewidth=1,capsize=2,capthick=1
            )
        ax.set_yticks(y)
        ax.set_ylim(len(methods)-.5,-.5)
        ax.grid(axis='x',alpha=.2)
        ax.set_axisbelow(True)
    axes[0,col].set_xlabel('Fixed-effect RMSE')
    axes[1,col].set_xlabel('Variance-bound rate')
axes[1,0].set_xlim(-.025,1.025)
for row in range(2):
    axes[row,0].set_yticklabels(labels)
    axes[row,0].tick_params(axis='y',labelleft=True)
for col in range(1,4):
    axes[0,col].tick_params(axis='y',labelleft=False)
    axes[1,col].tick_params(axis='y',labelleft=False)
fig.savefig(SOURCE/'figures/robustness.pdf'); fig.savefig(SOURCE/'figures/robustness.png',dpi=180); plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(10.5,3.6),layout='constrained')
calibration_colors=['#0072B2','#D55E00','#666666']
for meth,col in zip(order,calibration_colors):
    for design,idx,key,sekey in [('coverage',0,'valid_rate',None),('coverage',1,'coverage','coverage_mcse'),('null-slope',2,'rejection_zero','rejection_mcse')]:
        sub=full[(full.design==design)&(full.method==meth)&(full.component=='beta2_slope')].set_index(['clusters','cluster_size','sigma2']).loc[configs]
        err = None
        if sekey is not None:
            prefix = 'coverage' if key == 'coverage' else 'rejection'
            err = np.maximum(0,np.vstack((sub[key]-sub[prefix+'_wilson_low'],sub[prefix+'_wilson_high']-sub[key])))
        axes[idx].errorbar(range(3),sub[key],yerr=err,color=col,marker='o',markersize=4,capsize=3,label=mlab[meth])
for ax in axes:
    ax.set_xticks(range(3),['60 / 10 / 0.1','60 / 20 / 0.5','100 / 10 / 1.0'],rotation=20,ha='right'); ax.set_xlabel(r'$n$ / $m$ / $\sigma^2$');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Valid-inference rate\n(coverage design)'); axes[0].set_ylim(0,1.06)
axes[1].set_ylabel('Conditional coverage');axes[1].axhline(.95,color='gray',ls='--',lw=1)
axes[2].set_ylabel('Conditional null rejection');axes[2].axhline(.05,color='gray',ls='--',lw=1)
axes[0].legend(fontsize=8,frameon=False)
fig.savefig(SOURCE/'figures/calibration.pdf');fig.savefig(SOURCE/'figures/calibration.png',dpi=180);plt.close(fig)

counts=pd.read_csv(ROOT/'bmi_aggregated.csv')
count_lines = []
for r in counts.itertuples(index=False):
    age = '15--64 years' if r.age_group == '15-64' else '65 years and older'
    count_lines.append(age+' & '+r.gender+' & '+' & '.join(f'{int(getattr(r,k)):,}' for k in ['acceptable','overweight','obese'])+r' \\')
(SOURCE/'generated/bmi_counts.tex').write_text('\n'.join(count_lines)+'\n',encoding='utf-8')
vals=counts[['acceptable','overweight','obese']].to_numpy()
props=100*vals/vals.sum(axis=1,keepdims=True)
fig,ax=plt.subplots(figsize=(8.5,3),layout='constrained')
left=np.zeros(4)
for k,(name,col) in enumerate(zip(['Acceptable','Overweight','Obese'],['#0072B2','#E69F00','#CC79A7'])):
    ax.barh(range(4),props[:,k],left=left,color=col,label=name,height=.62)
    for i,v in enumerate(props[:,k]): ax.text(left[i]+v/2,i,f'{v:.1f}%',ha='center',va='center',fontsize=9,color='white' if k==0 else 'black')
    left+=props[:,k]
ax.set_yticks(range(4),[f'{r.age_group}, {r.gender} (n={int(vals[i].sum()):,})' for i,r in enumerate(counts.itertuples())]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Observed within-cell percentage');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=3,frameon=False)
fig.savefig(SOURCE/'figures/bmi_descriptive.pdf');fig.savefig(SOURCE/'figures/bmi_descriptive.png',dpi=180);plt.close(fig)

report={'audit_date':'2026-09-29','scope':'Reaggregation of existing formal Monte Carlo records and display-only table/figure updates; BMI fits retained from the previous revision; no new optimization or Monte Carlo experiment in this audit.','robustness_records':len(rob),'robustness_group_size':200,'inference_component_records':len(inf),'inference_dataset_method_fits':len(inf)//4,'reaggregation':'PASS','python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform(),'source_hashes':{}}
for p in [ROOT/'baseline/results/robustness_replicates.csv',RESULTS/'finite_cluster_inference_replicates.csv',ROOT/'bmi_aggregated.csv',RESULTS/'bmi_aggregate_estimates.csv']:
    report['source_hashes'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(RESULTS/'final_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
