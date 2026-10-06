#!/usr/bin/env python3
"""Focused rerun of the manuscript's coverage, Type-I-error and power tables.

The experiment keeps the original multinomial random-intercept setup and
evaluates CR(0), CR(2/3), and weighted marginal ML.  For every fitted data set
it constructs a cluster sandwich covariance numerically.  Fits at a variance
boundary or with a numerically singular sensitivity matrix are reported as
invalid rather than being assigned a nominal interval.  The uncalibrated raw
divergence-difference "LRT" from the submitted manuscript is deliberately not
reproduced because the revised theory does not justify a plain chi-square law.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from robustness_simulation import (
    ModelObjectives,
    SimData,
    TRUE_BETA,
    conditional_probabilities,
    draw_sampled_covariates,
    fit_objective_internal,
)


METHODS = ("CR(0)", "CR(0.666667)", "weighted-marginal-ML")
COMPONENTS = ("beta1_intercept", "beta1_slope", "beta2_intercept", "beta2_slope")


def generate_clean_data(
    rng: np.random.Generator,
    beta: np.ndarray,
    n_clusters: int,
    cluster_size: int,
    true_sigma: float,
) -> SimData:
    x, weights = draw_sampled_covariates(rng, n_clusters)
    u = rng.normal(0.0, true_sigma, size=n_clusters)
    probabilities = conditional_probabilities(x, u, beta)
    counts = np.vstack(
        [rng.multinomial(cluster_size, probabilities[i]) for i in range(n_clusters)]
    )
    return SimData(x=x, counts=counts, weights=weights)


def cluster_contributions(
    objectives: ModelObjectives,
    theta: np.ndarray,
    method: str,
) -> np.ndarray:
    if method.startswith("CR("):
        lam = 0.0 if method == "CR(0)" else 2.0 / 3.0
        fitted = np.clip(objectives.marginal_probabilities(theta), 1e-12, 1.0)
        ratio = np.clip(objectives.observed_proportions / fitted, 1e-12, 1e12)
        if lam == 0.0:
            phi = ratio * np.log(ratio) - ratio + 1.0
        else:
            phi = (
                ratio ** (lam + 1.0) - ratio - lam * (ratio - 1.0)
            ) / (lam * (lam + 1.0))
        loss = np.sum(fitted * phi, axis=1)
        return objectives.data.weights * objectives.cluster_size * loss

    if method == "weighted-marginal-ML":
        beta = theta[:4]
        sigma = np.exp(theta[4])
        u = np.sqrt(2.0) * sigma * objectives.gh_nodes
        probabilities = conditional_probabilities(
            objectives.data.x[:, None], u[None, :], beta
        )
        log_probabilities = np.log(np.clip(probabilities, 1e-14, 1.0))
        conditional_loglik = np.sum(
            objectives.data.counts[:, None, :] * log_probabilities, axis=2
        )
        cluster_loglik = logsumexp(
            conditional_loglik + objectives.log_gh_weights[None, :], axis=1
        )
        return -objectives.data.weights * cluster_loglik

    raise ValueError(f"Unknown method: {method}")


def numerical_scores(
    objectives: ModelObjectives,
    theta: np.ndarray,
    method: str,
    relative_step: float = 2e-5,
) -> np.ndarray:
    n = objectives.data.counts.shape[0]
    p = theta.size
    scores = np.empty((n, p), dtype=float)
    for k in range(p):
        step = relative_step * max(1.0, abs(float(theta[k])))
        upper = theta.copy()
        lower = theta.copy()
        upper[k] += step
        lower[k] -= step
        scores[:, k] = (
            cluster_contributions(objectives, upper, method)
            - cluster_contributions(objectives, lower, method)
        ) / (2.0 * step)
    return scores


def sandwich_result(
    objectives: ModelObjectives,
    theta: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    scores = numerical_scores(objectives, theta, method)
    n, p = scores.shape
    jacobian = np.empty((p, p), dtype=float)
    for k in range(p):
        step = 4e-4 * max(1.0, abs(float(theta[k])))
        upper = theta.copy()
        lower = theta.copy()
        upper[k] += step
        lower[k] -= step
        score_upper = numerical_scores(objectives, upper, method).mean(axis=0)
        score_lower = numerical_scores(objectives, lower, method).mean(axis=0)
        jacobian[:, k] = (score_upper - score_lower) / (2.0 * step)

    condition_number = float(np.linalg.cond(jacobian))
    if not np.isfinite(condition_number) or condition_number > 1e10:
        raise np.linalg.LinAlgError("Ill-conditioned sensitivity matrix")

    centered = scores - scores.mean(axis=0, keepdims=True)
    meat = (n / (n - 1.0)) * (centered.T @ centered) / n
    inverse = np.linalg.inv(jacobian)
    asymptotic_covariance = inverse @ meat @ inverse.T
    covariance = asymptotic_covariance / n
    influence = -centered @ inverse.T
    return covariance, influence, condition_number


def one_fit(
    seed: int,
    design: str,
    n_clusters: int,
    cluster_size: int,
    sigma2: float,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    beta_true = TRUE_BETA.copy()
    if design == "null-slope":
        beta_true[3] = 0.0
    data = generate_clean_data(
        rng, beta_true, n_clusters, cluster_size, true_sigma=np.sqrt(sigma2)
    )
    objectives = ModelObjectives(data)
    method_objectives = {
        "CR(0)": lambda value: objectives.cressie_read(value, 0.0),
        "CR(0.666667)": lambda value: objectives.cressie_read(value, 2.0 / 3.0),
        "weighted-marginal-ML": objectives.marginal_negative_loglikelihood,
    }

    rows: list[dict] = []
    for method in METHODS:
        theta, converged, boundary = fit_objective_internal(method_objectives[method])
        base = {
            "seed": seed,
            "design": design,
            "clusters": n_clusters,
            "cluster_size": cluster_size,
            "sigma2": sigma2,
            "method": method,
            "converged": int(converged),
            "sigma_bound_hit": int(boundary),
        }
        valid = bool(converged and not boundary and np.all(np.isfinite(theta)))
        covariance = None
        influence = None
        condition_number = np.nan
        if valid:
            try:
                covariance, influence, condition_number = sandwich_result(
                    objectives, theta, method
                )
                valid = bool(
                    np.all(np.isfinite(covariance))
                    and np.all(np.diag(covariance)[:4] > 0.0)
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                valid = False

        standard_errors = np.full(4, np.nan)
        if valid and covariance is not None and influence is not None:
            standard_errors = np.sqrt(np.diag(covariance)[:4])

        normal_critical = norm.ppf(0.975)
        for k, component in enumerate(COMPONENTS):
            estimate = float(theta[k])
            truth = float(beta_true[k])
            se = float(standard_errors[k])
            row = dict(base)
            row.update(
                {
                    "component": component,
                    "truth": truth,
                    "estimate": estimate,
                    "standard_error": se,
                    "condition_number": condition_number,
                    "valid_inference": int(valid),
                    "cover_normal": int(valid and abs(estimate - truth) <= normal_critical * se),
                    "reject_zero": int(valid and abs(estimate / se) > normal_critical),
                }
            )
            rows.append(row)
    return rows


def _task(arguments: tuple[int, str, int, int, float]) -> list[dict]:
    return one_fit(*arguments)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["design", "clusters", "cluster_size", "sigma2", "method", "component"]
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        valid = group[group["valid_inference"] == 1]
        record = dict(zip(group_columns, keys))
        record.update(
            {
                "replicates": group["seed"].nunique(),
                "convergence_rate": group["converged"].mean(),
                "sigma_bound_rate": group["sigma_bound_hit"].mean(),
                "valid_inference_rate": group["valid_inference"].mean(),
                "coverage_normal": valid["cover_normal"].mean() if len(valid) else np.nan,
                "type1_normal": 1.0 - valid["cover_normal"].mean() if len(valid) else np.nan,
                "rejection_zero": valid["reject_zero"].mean() if len(valid) else np.nan,
                "mean_se": valid["standard_error"].mean() if len(valid) else np.nan,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--configurations",
        nargs="+",
        default=["60,10,0.1", "60,20,0.5", "100,10,1.0"],
        help="Entries are clusters,cluster_size,sigma2.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    configurations = []
    for value in args.configurations:
        clusters, cluster_size, sigma2 = value.split(",")
        configurations.append((int(clusters), int(cluster_size), float(sigma2)))
    designs = []
    for clusters, cluster_size, sigma2 in configurations:
        designs.append(("coverage", clusters, cluster_size, sigma2))
        designs.append(("null-slope", clusters, cluster_size, sigma2))

    sequence = np.random.SeedSequence(args.seed)
    children = sequence.spawn(args.replicates * len(designs))
    tasks = []
    index = 0
    for design, clusters, cluster_size, sigma2 in designs:
        for _ in range(args.replicates):
            task_seed = int(children[index].generate_state(1)[0])
            index += 1
            tasks.append(
                (task_seed, design, clusters, cluster_size, sigma2)
            )

    rows: list[dict] = []
    if args.jobs == 1:
        iterator = map(_task, tasks)
        for result in iterator:
            rows.extend(result)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for result in executor.map(_task, tasks, chunksize=1):
                rows.extend(result)

    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    frame.to_csv(args.output / "finite_cluster_inference_replicates.csv", index=False)
    summary.to_csv(args.output / "finite_cluster_inference_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
