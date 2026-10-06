#!/usr/bin/env python3
"""Independent numerical audit of the retained formal simulation records.

This script does not rerun a Monte Carlo experiment and does not select or
discard scenarios.  It reaggregates the formal 200-replication records used by
the manuscript, checks them against the stored summaries, computes paired
within-replication comparisons, and verifies the denominators used for
coverage and structural-null rejection.  The 30-replication pilot directory is
deliberately outside the input list and is not used in any formal result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ROBUSTNESS_REPLICATES = ROOT / "baseline" / "results" / "robustness_replicates.csv"
ROBUSTNESS_SUMMARY = ROOT / "baseline" / "results" / "robustness_summary.csv"
INFERENCE_REPLICATES = RESULTS / "finite_cluster_inference_replicates.csv"
INFERENCE_DIAGNOSTICS = RESULTS / "submission_inference_diagnostics.csv"
INFERENCE_SOURCE = ROOT / "inference_simulation.py"
ROBUSTNESS_SOURCE = ROOT / "baseline" / "robustness_simulation.py"
OUTPUT = RESULTS / "numerical_comparison_audit.json"

SCENARIO_ORDER = [
    "clean",
    "individual-label",
    "cluster-shift",
    "t3-random-effect",
]
METHOD_ORDER = [
    "CR(-0.5)",
    "CR(-0.25)",
    "CR(0)",
    "CR(0.666667)",
    "CR(1)",
    "CP-DPD(0.3)",
    "weighted-MDPD(0.2)",
    "weighted-marginal-ML",
]
PAIRS = [
    ("CR(0.666667)", "CP-DPD(0.3)"),
    ("CR(0.666667)", "weighted-MDPD(0.2)"),
    ("CR(1)", "CP-DPD(0.3)"),
    ("CR(1)", "weighted-MDPD(0.2)"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rate_mcse(rate: float, denominator: int) -> float | None:
    if denominator <= 0 or not np.isfinite(rate):
        return None
    return float(math.sqrt(rate * (1.0 - rate) / denominator))


def wilson_95(successes: int, denominator: int) -> tuple[float | None, float | None]:
    if denominator <= 0:
        return None, None
    z = 1.959963984540054
    rate = successes / denominator
    scale = 1.0 + z * z / denominator
    center = (rate + z * z / (2.0 * denominator)) / scale
    half_width = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / denominator
            + z * z / (4.0 * denominator * denominator)
        )
        / scale
    )
    return center - half_width, center + half_width


def finite_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def reaggregate_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (scenario, method), group in frame.groupby(["scenario", "method"], sort=False):
        squared_error = group["beta_squared_error"].to_numpy(dtype=float)
        sigma_error = group["sigma2_error"].to_numpy(dtype=float)
        bound = group["sigma2_bound_hit"].to_numpy(dtype=float)
        mse = float(np.mean(squared_error))
        rmse = float(math.sqrt(mse))
        mse_mcse = float(np.std(squared_error, ddof=1) / math.sqrt(len(group)))
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "replicates": int(len(group)),
                "convergence_rate": float(group["converged"].mean()),
                "beta_rmse": rmse,
                "beta_rmse_mcse": mse_mcse / (2.0 * rmse),
                "sigma2_bias": float(np.mean(sigma_error)),
                "sigma2_mae": float(np.mean(np.abs(sigma_error))),
                "sigma2_mae_mcse": float(
                    np.std(np.abs(sigma_error), ddof=1) / math.sqrt(len(group))
                ),
                "sigma2_bound_rate": float(np.mean(bound)),
                "sigma2_bound_rate_mcse": rate_mcse(float(np.mean(bound)), len(group)),
            }
        )
    return pd.DataFrame(rows)


def audit_paired_comparisons(frame: pd.DataFrame, summary: pd.DataFrame) -> list[dict]:
    pivot = frame.pivot(
        index=["scenario", "seed"], columns="method", values="beta_squared_error"
    )
    summary_index = summary.set_index(["scenario", "method"])
    rows: list[dict] = []
    for scenario in SCENARIO_ORDER:
        scenario_pivot = pivot.xs(scenario, level="scenario")
        for proposed, comparator in PAIRS:
            paired = (
                scenario_pivot[proposed] - scenario_pivot[comparator]
            ).dropna()
            assert len(paired) == 200
            mean_difference = float(paired.mean())
            mcse = float(paired.std(ddof=1) / math.sqrt(len(paired)))
            proposed_rmse = float(summary_index.loc[(scenario, proposed), "beta_rmse"])
            comparator_rmse = float(summary_index.loc[(scenario, comparator), "beta_rmse"])
            rows.append(
                {
                    "scenario": scenario,
                    "proposed": proposed,
                    "comparator": comparator,
                    "paired_replicates": int(len(paired)),
                    "proposed_beta_rmse": proposed_rmse,
                    "comparator_beta_rmse": comparator_rmse,
                    "relative_rmse_reduction": (
                        comparator_rmse - proposed_rmse
                    )
                    / comparator_rmse,
                    "mean_paired_beta_mse_difference_proposed_minus_comparator": mean_difference,
                    "paired_difference_mcse": mcse,
                    "paired_difference_normal_95_low": mean_difference - 1.96 * mcse,
                    "paired_difference_normal_95_high": mean_difference + 1.96 * mcse,
                    "paired_difference_z": mean_difference / mcse if mcse > 0 else None,
                    "proposed_has_lower_rmse": bool(proposed_rmse < comparator_rmse),
                }
            )
    return rows


def audit_inference(frame: pd.DataFrame, diagnostics: pd.DataFrame) -> tuple[list[dict], dict]:
    component = "beta2_slope"
    selected = frame[frame["component"].eq(component)].copy()
    selected_diagnostics = diagnostics[diagnostics["component"].eq(component)].copy()
    rows: list[dict] = []

    for keys, group in selected.groupby(
        ["design", "clusters", "cluster_size", "sigma2", "method"], sort=False
    ):
        design, clusters, cluster_size, sigma2, method = keys
        total = int(len(group))
        unique_seeds = int(group["seed"].nunique())
        valid = group[group["valid_inference"].eq(1)]
        valid_count = int(len(valid))
        assert total == unique_seeds == 200

        if design == "coverage":
            numerator = int(valid["cover_normal"].sum())
            event_name = "covered"
        elif design == "null-slope":
            assert np.allclose(group["truth"].to_numpy(dtype=float), 0.0)
            numerator = int(valid["reject_zero"].sum())
            event_name = "rejected_structural_null"
        else:
            raise AssertionError(f"Unexpected design: {design}")

        conditional_rate = numerator / valid_count if valid_count else math.nan
        all_replicate_event_rate = numerator / total
        conditional_wilson_low, conditional_wilson_high = wilson_95(
            numerator, valid_count
        )
        all_wilson_low, all_wilson_high = wilson_95(numerator, total)
        record = {
            "design": design,
            "clusters": int(clusters),
            "cluster_size": int(cluster_size),
            "sigma2": float(sigma2),
            "method": method,
            "component": component,
            "total_replicates": total,
            "unique_dataset_seeds": unique_seeds,
            "valid_count": valid_count,
            "valid_rate": valid_count / total,
            "variance_boundary_rate": float(group["sigma_bound_hit"].mean()),
            "event": event_name,
            "event_numerator_among_valid": numerator,
            "conditional_rate": finite_float(conditional_rate),
            "conditional_rate_mcse_using_valid_count": rate_mcse(
                conditional_rate, valid_count
            ),
            "conditional_rate_wilson_95_low": conditional_wilson_low,
            "conditional_rate_wilson_95_high": conditional_wilson_high,
            "all_replicate_event_rate": all_replicate_event_rate,
            "all_replicate_event_mcse_using_200": rate_mcse(
                all_replicate_event_rate, total
            ),
            "all_replicate_event_wilson_95_low": all_wilson_low,
            "all_replicate_event_wilson_95_high": all_wilson_high,
        }

        matched = selected_diagnostics[
            selected_diagnostics["design"].eq(design)
            & selected_diagnostics["clusters"].eq(clusters)
            & selected_diagnostics["cluster_size"].eq(cluster_size)
            & np.isclose(selected_diagnostics["sigma2"], sigma2)
            & selected_diagnostics["method"].eq(method)
        ]
        assert len(matched) == 1
        stored = matched.iloc[0]
        assert int(stored["total"]) == total
        assert int(stored["valid_count"]) == valid_count
        assert math.isclose(float(stored["valid_rate"]), valid_count / total, abs_tol=1e-12)
        if design == "coverage":
            assert math.isclose(float(stored["coverage"]), conditional_rate, abs_tol=1e-12)
            assert math.isclose(
                float(stored["coverage_mcse"]),
                rate_mcse(conditional_rate, valid_count),
                abs_tol=1e-12,
            )
            assert math.isclose(
                float(stored["valid_and_covered"]), all_replicate_event_rate, abs_tol=1e-12
            )
        else:
            assert math.isclose(
                float(stored["rejection_zero"]), conditional_rate, abs_tol=1e-12
            )
            assert math.isclose(
                float(stored["rejection_mcse"]),
                rate_mcse(conditional_rate, valid_count),
                abs_tol=1e-12,
            )
            assert math.isclose(
                float(stored["valid_and_rejected"]), all_replicate_event_rate, abs_tol=1e-12
            )
        rows.append(record)

    invalid_se_are_missing = bool(
        frame.loc[frame["valid_inference"].eq(0), "standard_error"].isna().all()
    )
    valid_se = frame.loc[frame["valid_inference"].eq(1), "standard_error"].to_numpy(
        dtype=float
    )
    valid_se_are_positive_finite = bool(
        np.isfinite(valid_se).all() and np.all(valid_se > 0.0)
    )
    checks = {
        "invalid_fits_have_missing_standard_errors": invalid_se_are_missing,
        "valid_fits_have_positive_finite_standard_errors": valid_se_are_positive_finite,
    }
    assert all(checks.values())
    return rows, checks


def audit_n_scaling(source_text: str) -> dict:
    checks = {
        "n_is_number_of_cluster_score_rows": bool(
            re.search(r"n\s*,\s*p\s*=\s*scores\.shape", source_text)
        ),
        "jacobian_uses_mean_cluster_score": "numerical_scores(objectives, upper, method).mean(axis=0)"
        in source_text,
        "meat_uses_cluster_average": bool(
            re.search(r"\(centered\.T\s*@\s*centered\)\s*/\s*n", source_text)
        ),
        "sandwich_covariance_divides_by_n": bool(
            re.search(r"covariance\s*=\s*asymptotic_covariance\s*/\s*n", source_text)
        ),
    }
    assert all(checks.values())
    return {
        **checks,
        "verified_scale": "H^{-1} G H^{-T} / n, where n is the number of independent cluster-score rows",
    }


def main() -> None:
    for path in [
        ROBUSTNESS_REPLICATES,
        ROBUSTNESS_SUMMARY,
        INFERENCE_REPLICATES,
        INFERENCE_DIAGNOSTICS,
        INFERENCE_SOURCE,
        ROBUSTNESS_SOURCE,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    robustness = pd.read_csv(ROBUSTNESS_REPLICATES)
    stored_robustness_summary = pd.read_csv(ROBUSTNESS_SUMMARY)
    inference = pd.read_csv(INFERENCE_REPLICATES)
    diagnostics = pd.read_csv(INFERENCE_DIAGNOSTICS)

    assert len(robustness) == 6400
    assert set(robustness["scenario"]) == set(SCENARIO_ORDER)
    assert set(robustness["method"]) == set(METHOD_ORDER)
    group_sizes = robustness.groupby(["scenario", "method"]).size()
    assert len(group_sizes) == 32 and group_sizes.eq(200).all()
    assert np.isfinite(
        robustness[["beta_squared_error", "sigma2_error"]].to_numpy(dtype=float)
    ).all()

    reaggregated = reaggregate_robustness(robustness)
    metrics = [
        "replicates",
        "convergence_rate",
        "beta_rmse",
        "beta_rmse_mcse",
        "sigma2_bias",
        "sigma2_mae",
        "sigma2_mae_mcse",
        "sigma2_bound_rate",
        "sigma2_bound_rate_mcse",
    ]
    left = reaggregated.sort_values(["scenario", "method"]).reset_index(drop=True)
    right = stored_robustness_summary.sort_values(["scenario", "method"]).reset_index(
        drop=True
    )
    assert left[["scenario", "method"]].equals(right[["scenario", "method"]])
    maximum_summary_difference = float(
        np.max(
            np.abs(
                left[metrics].to_numpy(dtype=float)
                - right[metrics].to_numpy(dtype=float)
            )
        )
    )
    assert maximum_summary_difference < 1e-10

    assert len(inference) == 14400
    fit_groups = inference.groupby(
        ["design", "clusters", "cluster_size", "sigma2", "method", "seed"]
    ).size()
    assert len(fit_groups) == 3600 and fit_groups.eq(4).all()
    component_groups = inference.groupby(
        ["design", "clusters", "cluster_size", "sigma2", "method", "component"]
    ).size()
    assert len(component_groups) == 72 and component_groups.eq(200).all()

    paired = audit_paired_comparisons(robustness, reaggregated)
    inference_rows, standard_error_checks = audit_inference(inference, diagnostics)
    inference_source_text = INFERENCE_SOURCE.read_text(encoding="utf-8")
    robustness_source_text = ROBUSTNESS_SOURCE.read_text(encoding="utf-8")
    n_scaling = audit_n_scaling(inference_source_text)

    label_pairs = [row for row in paired if row["scenario"] == "individual-label"]
    label_vs_mdpd = {
        row["proposed"]: row["relative_rmse_reduction"]
        for row in label_pairs
        if row["comparator"] == "weighted-MDPD(0.2)"
    }
    mdpd_better_outside_label = [
        row["scenario"]
        for row in paired
        if row["proposed"] == "CR(0.666667)"
        and row["comparator"] == "weighted-MDPD(0.2)"
        and row["scenario"] != "individual-label"
        and not row["proposed_has_lower_rmse"]
    ]

    tuning_checks = {
        "formal_cr_tunings": [-0.5, -0.25, 0.0, 2.0 / 3.0, 1.0],
        "formal_mdpd_alpha": 0.2,
        "single_mdpd_alpha_only": "weighted-MDPD(0.2)" in set(robustness["method"])
        and not any(
            method.startswith("weighted-MDPD(") and method != "weighted-MDPD(0.2)"
            for method in set(robustness["method"])
        ),
        "alpha_literal_verified_in_source": bool(
            re.search(r"marginal_density_power_divergence\(th,\s*0\.20\)", robustness_source_text)
        ),
        "scope_note_cn": (
            "W-MDPDE仅按固定alpha=0.2比较，正式记录没有alpha敏感性网格；"
            "因此结论只适用于该固定调参值，不能外推为对整个MDPDE方法族的优劣判断。"
        ),
    }
    assert tuning_checks["single_mdpd_alpha_only"]
    assert tuning_checks["alpha_literal_verified_in_source"]

    report = {
        "audit_date": "2026-09-27",
        "scope": (
            "Read-only reaggregation of retained formal records; no new large Monte Carlo run; "
            "30-replication pilot excluded."
        ),
        "inputs": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in [
                ROBUSTNESS_REPLICATES,
                ROBUSTNESS_SUMMARY,
                INFERENCE_REPLICATES,
                INFERENCE_DIAGNOSTICS,
                INFERENCE_SOURCE,
                ROBUSTNESS_SOURCE,
            ]
        },
        "pilot_20260927_used_in_formal_audit": False,
        "formal_robustness": {
            "replicate_rows": int(len(robustness)),
            "scenario_method_groups": int(len(group_sizes)),
            "replicates_per_group": 200,
            "stored_summary_maximum_absolute_difference": maximum_summary_difference,
            "all_32_scenario_method_rows": reaggregated.to_dict(orient="records"),
        },
        "paired_fixed_effect_mse_comparisons": paired,
        "label_contamination_relative_rmse_reduction_vs_fixed_alpha_mdpd": label_vs_mdpd,
        "inference": {
            "coefficient_level_rows": int(len(inference)),
            "method_fits": int(len(fit_groups)),
            "components_per_fit": 4,
            "component_groups": int(len(component_groups)),
            "coverage_and_null_audit_for_beta2_slope": inference_rows,
            "standard_error_record_checks": standard_error_checks,
            "n_scaling_source_checks": n_scaling,
        },
        "tuning_fairness_scope": tuning_checks,
        "conclusions_cn": [
            (
                "个体标签污染下，CR(2/3)相对固定alpha=0.2的W-MDPDE，"
                f"固定效应RMSE下降{100.0 * label_vs_mdpd['CR(0.666667)']:.2f}%；"
                f"CR(1)下降{100.0 * label_vs_mdpd['CR(1)']:.2f}%。"
            ),
            (
                "这一优势具有情景限定：CR(2/3)在clean、cluster-shift和"
                f"t3-random-effect情景均未优于W-MDPDE（{', '.join(mdpd_better_outside_label)}）。"
            ),
            (
                "四情景八方法共6400行均纳入汇总；不利的方差MAE和边界率比较未删除。"
            ),
            (
                "覆盖率和真实零假设拒绝率均以有效推断拟合数为条件分母，并同时保留"
                "200次全分母的有效率/事件率；标准误sandwich按独立簇数n缩放。"
            ),
            tuning_checks["scope_note_cn"],
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
