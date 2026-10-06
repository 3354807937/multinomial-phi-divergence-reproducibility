# Reproduction and numerical verification

## Environment

Recorded environment: Windows 11, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, and Matplotlib 3.10.5. Package versions are pinned in `requirements.txt`. Simulations use 15-node Gauss-Hermite quadrature; the BMI fits use 25 nodes.

From `Reproducibility/`:

```shell
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows, or `source .venv/bin/activate` on macOS/Linux, then run:

```shell
python -m pip install -r requirements.txt
```

## Reconstruction and checks

```shell
python bmi_aggregate_fit.py
python final_audit.py
python verify_replay.py
python bmi_application_audit.py
python numerical_comparison_audit.py
```

`bmi_aggregate_fit.py` refits the five BMI tunings. `final_audit.py` reaggregates stored simulations, checks the formal summaries, and regenerates seven numerical tables and three vector figures in `../Manuscript_Source/`. It records the environment and input hashes in `results/final_audit.json`.

`verify_replay.py` reruns one stored child seed in each robustness scenario and inference configuration, without overwriting the original records. The retained pinned-environment report contains ten passing checks (32 robustness method records and 72 inference coefficient records), with maximum absolute differences at most 2.3e-16 for the checked estimates and standard errors. This is a deterministic spot check, not a full Monte Carlo rerun.

`bmi_application_audit.py` checks the aggregate counts and evaluates the objective at all five stored estimates. `numerical_comparison_audit.py` verifies the records, inference denominators, and all 16 paired fixed-effect comparisons for PMphi(2/3) and PMphi(1) against CP-DPD and W-DPD, retaining both favorable and unfavorable comparisons.

## Full simulations

These commands replace the respective stored outputs. Use a separate copy if preserving the supplied records.

```shell
python baseline/robustness_simulation.py --replicates 200 --seed 20260820 --clusters 60 --cluster-size 10 --jobs 4 --output baseline/results
python inference_simulation.py --replicates 200 --seed 20260903 --configurations 60,10,0.1 60,20,0.5 100,10,1.0 --jobs 4 --output results
python bmi_aggregate_fit.py
python final_audit.py
```

NumPy `SeedSequence` assigns an explicit child seed to each dataset. The top-level `robustness_simulation.py` provides model functions imported by the inference script; the eight-method, four-scenario simulation entry point is `baseline/robustness_simulation.py`.

## Records and method labels

| File | Rows | Contents |
| --- | ---: | --- |
| `baseline/results/robustness_replicates.csv` | 6400 | 4 scenarios x 8 methods x 200 replications |
| `baseline/results/robustness_summary.csv` | 32 | RMSE, variance errors, boundary rates, and MCSE |
| `results/finite_cluster_inference_replicates.csv` | 14400 | 6 designs x 3 methods x 200 replications x 4 coefficients |
| `results/finite_cluster_inference_summary.csv` | 72 | Inference summaries |
| `results/submission_inference_diagnostics.csv` | 72 | Explicit denominators, validity rates, MCSE, and Wilson intervals |
| `bmi_aggregated.csv` | 4 | Age-by-gender aggregate counts |
| `results/bmi_aggregate_estimates.csv` | 5 | BMI fits across five tunings |

The stored method keys `CR(...)`, `CP-DPD(0.3)`, `weighted-MDPD(0.2)`, and `weighted-marginal-ML` correspond to the display labels PMphi(lambda), CP-DPD, W-DPD, and ML. Keys are preserved for traceability.

The 14,400 inference rows represent 3,600 method fits on 1,200 independent datasets. Coefficients from the same fit are not separate Monte Carlo replications. Conditional coverage and rejection use the valid-fit denominator; boundary and validity rates use all 200 replications. `valid_and_covered` treats invalid fits as failures to deliver a covering interval.

The legacy `type1_normal` column is the complement of coverage of the generating coefficient. For the zero-null test reported in the analysis, use `design=null-slope`, `component=beta2_slope`, and `rejection_zero` in the diagnostics file. Nonzero CR tuning targets a pseudo-true parameter; structural-null calibration is assessed empirically.

The BMI data contain four aggregate cells, not individual records. Equal cell weights are used because inclusion probabilities were not supplied. The model has no fixed intercept; all five stored variance estimates equal the optimization lower bound 0.0004. These are descriptive aggregate fits.

## Numerical-environment sensitivity

The retained Python 3.10.9 / NumPy 1.23.5 / SciPy 1.10.0 replay failed seven of ten checks at the pinned-environment tolerance; see `results/replay_check_python310.json`. For individual-label seed 2003837550, CR(-0.25) gave sigma-squared 6.25 rather than 4.90946, despite an objective difference of approximately 9.09e-6. Optimization can stop at different positions along a nearly flat variance direction. Use the pinned versions and compare objective values, parameter estimates, and boundary flags; convergence alone does not establish variance identification or a unique global optimum.

## Input checksums

- `baseline/results/robustness_replicates.csv`: `a1d1052b522c3a91b534899ae773fc423fc71a30f61294c5a39aabef2cce2605`
- `results/finite_cluster_inference_replicates.csv`: `c02d9f9bc8c8c38a137970c23b17c8f397f725a653076756639965459e8be57b`
- `bmi_aggregated.csv`: `4867c4fb4d604646b4031b8de0c3d0d24b53ab3c2a94c16b4752915a0011669e`
- `results/bmi_aggregate_estimates.csv`: `7860ab1fdd2cacd6e6608f3f28fd1beb4547a904d99da098b62fc56691bc4224`

No stored numerical result has been manually adjusted for this repository. Retain the runtime report when comparing independent runs.
