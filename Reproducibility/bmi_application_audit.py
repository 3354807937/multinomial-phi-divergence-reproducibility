"""Check the expanded BMI application against unchanged data and stored fits.

This checks arithmetic and objective evaluation, not statistical validity or
the uniqueness/global optimality of the fitted estimates. It runs no new tests.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / 'bmi_aggregated.csv')
fits = pd.read_csv(ROOT / 'results/bmi_aggregate_estimates.csv')
counts = data[['acceptable', 'overweight', 'obese']].to_numpy(float)
sizes = counts.sum(axis=1)
observed = counts / sizes[:, None]
design = np.column_stack((data.gender.eq('Male'), data.age_group.eq('65+'))).astype(float)
assert counts.shape == (4, 3) and sizes.sum() == 74640
assert np.array_equal(design, [[1, 0], [0, 0], [1, 1], [0, 1]])
nodes, weights = hermgauss(25)
weights = weights / np.sqrt(np.pi)
rows, checks = [], []
for fit in fits.itertuples(index=False):
    values = fit._asdict()
    lam = values['lambda'] if 'lambda' in values else float(fit[0])
    beta = np.array([[fit.beta11_gender, fit.beta12_age], [fit.beta21_gender, fit.beta22_age]])
    logits = design @ beta.T
    conditional_logits = logits[:, None, :] + np.sqrt(2 * fit.sigma2) * nodes[None, :, None]
    conditional_logits = np.concatenate((conditional_logits, np.zeros((4, 25, 1))), axis=2)
    probability = np.einsum('iqr,q->ir', softmax(conditional_logits, axis=2), weights)
    assert np.allclose(probability.sum(axis=1), 1, atol=1e-14)
    assert np.all(probability > 0)
    ratio = observed / probability
    if abs(lam) < 1e-12:
        phi = ratio * np.log(ratio) - ratio + 1
    else:
        phi = (ratio ** (lam + 1) - ratio - lam * (ratio - 1)) / (lam * (lam + 1))
    objective = float(np.sum(sizes[:, None] * probability * phi) / sizes.sum())
    delta = abs(objective - fit.objective)
    assert delta < 1e-12, (lam, delta)
    assert abs(fit.sigma2 - .02 ** 2) < 1e-12
    checks.append({'lambda': lam, 'objective_recomputed': objective, 'stored_objective': fit.objective, 'absolute_difference': delta})
    for i in range(4):
        row = {'lambda': lam, 'age_group': data.age_group[i], 'gender': data.gender[i], 'cell_size': int(sizes[i])}
        for r, category in enumerate(['acceptable', 'overweight', 'obese']):
            row['observed_' + category] = float(observed[i, r])
            row['fitted_' + category] = float(probability[i, r])
        rows.append(row)

summary = pd.read_csv(ROOT / 'baseline/results/robustness_summary.csv')
label = summary[summary.scenario.eq('individual-label')].set_index('method')
clean = summary[summary.scenario.eq('clean')].set_index('method')
comparisons = []
for method in ['CR(0.666667)', 'CR(1)']:
    for ref in ['weighted-marginal-ML', 'weighted-MDPD(0.2)']:
        own = float(label.loc[method, 'beta_rmse'])
        other = float(label.loc[ref, 'beta_rmse'])
        comparisons.append({'method': method, 'reference': ref, 'label_rmse': own, 'reference_label_rmse': other, 'relative_reduction_percent': 100 * (1 - own / other), 'clean_rmse': float(clean.loc[method, 'beta_rmse']), 'reference_clean_rmse': float(clean.loc[ref, 'beta_rmse'])})

pd.DataFrame(rows).to_csv(ROOT / 'results/bmi_observed_fitted_proportions.csv', index=False)
report = {
    'audit_date': '2026-09-29', 'status': 'PASS',
    'scope': 'Read-only numerical check of unchanged fits; no new optimization, hypothesis test, or Monte Carlo experiment.',
    'cells': 4, 'individual_total': int(sizes.sum()), 'parameter_dimension': 5,
    'cell_order_and_coding': design.astype(int).tolist(),
    'all_five_objectives_match': True, 'all_variances_at_lower_optimization_limit': True,
    'objectives': checks, 'hypothesis_test_results_computed': False,
    'likelihood_and_dpd_comparisons': comparisons,
    'source_hashes': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ['bmi_aggregated.csv', 'results/bmi_aggregate_estimates.csv', 'baseline/results/robustness_summary.csv']}
}
(ROOT / 'results/bmi_application_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
