#!/usr/bin/env python3
"""Contamination study for the weighted cluster-proportion estimators.

The script is self-contained and writes both replicate-level and summary CSV
files.  It compares the Cressie--Read cluster-proportion estimators, including
negative lambda values, with cluster-proportion DPD, survey-weighted marginal
likelihood and weighted marginal MDPD comparators.

Example
-------
python robustness_simulation.py --replicates 200 --seed 20260820 --output results
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp


TRUE_BETA = np.array([0.40, 0.80, -0.30, 0.50], dtype=float)
TRUE_SIGMA = 0.70
LAMBDA_VALUES = (-0.50, -0.25, 0.0, 2.0 / 3.0, 1.0)
SCENARIOS = ("clean", "individual-label", "cluster-shift", "t3-random-effect")


@dataclass(frozen=True)
class SimData:
    x: np.ndarray
    counts: np.ndarray
    weights: np.ndarray


def draw_sampled_covariates(
    rng: np.random.Generator, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw sampled clusters under informative Poisson inclusion.

    Candidate covariates follow U(-1.5, 1.5).  A candidate is retained with
    probability expit(0.30 - 0.40 x), so retained covariates follow the
    inclusion-tilted distribution.  Conditioning on the requested number of
    retained clusters keeps the Monte Carlo sample size fixed; inverse-
    inclusion weights recover the intended superpopulation moment.
    """
    accepted: list[np.ndarray] = []
    remaining = n_clusters
    while remaining > 0:
        batch_size = max(32, 3 * remaining)
        candidate = rng.uniform(-1.5, 1.5, size=batch_size)
        candidate_probability = expit(0.30 - 0.40 * candidate)
        retained = candidate[rng.random(batch_size) < candidate_probability]
        if retained.size:
            accepted.append(retained[:remaining])
            remaining -= min(remaining, retained.size)
    x = np.concatenate(accepted)[:n_clusters]
    inclusion_probability = expit(0.30 - 0.40 * x)
    return x, 1.0 / inclusion_probability


def conditional_probabilities(x: np.ndarray, u: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return P(Y=r|x,u) for r=1,2,3 with a shared random intercept."""
    eta1 = beta[0] + beta[1] * x + u
    eta2 = beta[2] + beta[3] * x + u
    baseline = np.zeros_like(eta1)
    stacked = np.stack((eta1, eta2, baseline), axis=-1)
    stacked -= np.max(stacked, axis=-1, keepdims=True)
    numer = np.exp(stacked)
    return numer / np.sum(numer, axis=-1, keepdims=True)


def generate_data(rng: np.random.Generator, scenario: str, n_clusters: int, cluster_size: int) -> SimData:
    x, weights = draw_sampled_covariates(rng, n_clusters)

    if scenario == "t3-random-effect":
        u = rng.standard_t(df=3, size=n_clusters) * TRUE_SIGMA / np.sqrt(3.0)
    else:
        u = rng.normal(0.0, TRUE_SIGMA, size=n_clusters)

    if scenario == "cluster-shift":
        contaminated = rng.random(n_clusters) < 0.10
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        u = u + contaminated * signs * (3.5 * TRUE_SIGMA)

    probabilities = conditional_probabilities(x, u, TRUE_BETA)
    counts = np.zeros((n_clusters, 3), dtype=int)
    for i in range(n_clusters):
        labels = rng.choice(3, size=cluster_size, p=probabilities[i])
        if scenario == "individual-label":
            relabel = rng.random(cluster_size) < 0.10
            labels[relabel] = 2
        counts[i] = np.bincount(labels, minlength=3)
    return SimData(x=x, counts=counts, weights=weights)


class ModelObjectives:
    def __init__(self, data: SimData, quadrature_nodes: int = 15):
        self.data = data
        nodes, weights = hermgauss(quadrature_nodes)
        self.gh_nodes = nodes
        self.gh_weights = weights / np.sqrt(np.pi)
        self.log_gh_weights = np.log(self.gh_weights)
        self.cluster_size = data.counts.sum(axis=1).astype(float)
        self.observed_proportions = data.counts / self.cluster_size[:, None]
        normalized = data.weights / np.mean(data.weights)
        self.objective_weights = normalized / np.sum(normalized)
        m = int(self.cluster_size[0])
        if not np.all(self.cluster_size == m):
            raise ValueError("The MDPD comparator currently requires a common cluster size.")
        self.count_support = np.array(
            [(a, b, m - a - b) for a in range(m + 1) for b in range(m - a + 1)],
            dtype=float,
        )
        self.support_log_coefficient = (
            gammaln(m + 1.0) - np.sum(gammaln(self.count_support + 1.0), axis=1)
        )
        self.observed_log_coefficient = (
            gammaln(self.cluster_size + 1.0) - np.sum(gammaln(self.data.counts + 1.0), axis=1)
        )

    def marginal_probabilities(self, theta: np.ndarray) -> np.ndarray:
        beta = theta[:4]
        sigma = np.exp(theta[4])
        u = np.sqrt(2.0) * sigma * self.gh_nodes
        x_grid = self.data.x[:, None]
        u_grid = u[None, :]
        probs = conditional_probabilities(x_grid, u_grid, beta)
        return np.sum(probs * self.gh_weights[None, :, None], axis=1)

    def cressie_read(self, theta: np.ndarray, lam: float) -> float:
        q = np.clip(self.marginal_probabilities(theta), 1e-12, 1.0)
        ratio = np.clip(self.observed_proportions / q, 1e-12, 1e12)
        if abs(lam) < 1e-12:
            phi = ratio * np.log(ratio) - ratio + 1.0
        elif abs(lam + 1.0) < 1e-12:
            phi = -np.log(ratio) + ratio - 1.0
        else:
            phi = (ratio ** (lam + 1.0) - ratio - lam * (ratio - 1.0)) / (lam * (lam + 1.0))
        cluster_loss = np.sum(q * phi, axis=1)
        return float(np.sum(self.objective_weights * cluster_loss))

    def cluster_proportion_dpd(self, theta: np.ndarray, alpha: float = 0.30) -> float:
        q = np.clip(self.marginal_probabilities(theta), 1e-12, 1.0)
        p = self.observed_proportions
        cluster_loss = np.sum(q ** (1.0 + alpha), axis=1)
        cluster_loss -= (1.0 + 1.0 / alpha) * np.sum(p * q**alpha, axis=1)
        return float(np.sum(self.objective_weights * cluster_loss))

    def marginal_negative_loglikelihood(self, theta: np.ndarray) -> float:
        beta = theta[:4]
        sigma = np.exp(theta[4])
        u = np.sqrt(2.0) * sigma * self.gh_nodes
        probabilities = conditional_probabilities(self.data.x[:, None], u[None, :], beta)
        log_probabilities = np.log(np.clip(probabilities, 1e-14, 1.0))
        conditional_loglik = np.sum(self.data.counts[:, None, :] * log_probabilities, axis=2)
        cluster_loglik = logsumexp(conditional_loglik + self.log_gh_weights[None, :], axis=1)
        return float(-np.sum(self.objective_weights * cluster_loglik))

    def marginal_density_power_divergence(self, theta: np.ndarray, alpha: float = 0.20) -> float:
        """Weighted MDPD for the logistic-normal multinomial cluster pmf."""
        beta = theta[:4]
        sigma = np.exp(theta[4])
        u = np.sqrt(2.0) * sigma * self.gh_nodes
        probabilities = conditional_probabilities(self.data.x[:, None], u[None, :], beta)
        log_probabilities = np.log(np.clip(probabilities, 1e-14, 1.0))

        observed_conditional = np.sum(self.data.counts[:, None, :] * log_probabilities, axis=2)
        observed_logpmf = self.observed_log_coefficient + logsumexp(
            observed_conditional + self.log_gh_weights[None, :], axis=1
        )

        # Dimensions: cluster x quadrature node x count-support point.
        support_conditional = np.einsum("nqk,sk->nqs", log_probabilities, self.count_support)
        support_logpmf = self.support_log_coefficient[None, :] + logsumexp(
            support_conditional + self.log_gh_weights[None, :, None], axis=1
        )
        integral_term = np.sum(np.exp((1.0 + alpha) * support_logpmf), axis=1)
        observed_term = np.exp(alpha * observed_logpmf)
        cluster_loss = integral_term - (1.0 + 1.0 / alpha) * observed_term
        return float(np.sum(self.objective_weights * cluster_loss))


def fit_objective_internal(fun) -> tuple[np.ndarray, bool, bool]:
    """Fit an objective and retain the internal log-standard-deviation scale."""
    starts = (
        np.array([0.0, 0.4, 0.0, 0.3, np.log(0.60)]),
        np.array([0.5, 0.8, -0.4, 0.5, np.log(0.90)]),
    )
    bounds = [(-5.0, 5.0)] * 4 + [(np.log(0.08), np.log(2.50))]
    candidates = []
    for start in starts:
        result = minimize(fun, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-10})
        if np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        return np.full(5, np.nan), False, False
    best = min(candidates, key=lambda item: item.fun)
    theta = best.x.copy()
    sigma_bound_hit = bool(
        np.isclose(theta[4], bounds[-1][0], atol=1e-6)
        or np.isclose(theta[4], bounds[-1][1], atol=1e-6)
    )
    return theta, bool(best.success), sigma_bound_hit


def fit_objective(fun) -> tuple[np.ndarray, bool, bool]:
    """Fit an objective and report the variance component on the sigma-squared scale."""
    theta, converged, sigma_bound_hit = fit_objective_internal(fun)
    if not np.all(np.isfinite(theta)):
        return theta, converged, sigma_bound_hit
    theta[4] = np.exp(2.0 * theta[4])
    return theta, converged, sigma_bound_hit


def one_replicate(seed: int, scenario: str, n_clusters: int, cluster_size: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    data = generate_data(rng, scenario, n_clusters, cluster_size)
    objectives = ModelObjectives(data)
    methods = [(f"CR({lam:g})", lambda th, value=lam: objectives.cressie_read(th, value)) for lam in LAMBDA_VALUES]
    methods += [
        ("CP-DPD(0.3)", lambda th: objectives.cluster_proportion_dpd(th, 0.30)),
        ("weighted-MDPD(0.2)", lambda th: objectives.marginal_density_power_divergence(th, 0.20)),
        ("weighted-marginal-ML", objectives.marginal_negative_loglikelihood),
    ]
    rows = []
    for method, objective in methods:
        estimate, converged, sigma_bound_hit = fit_objective(objective)
        beta_error = estimate[:4] - TRUE_BETA
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "method": method,
                "converged": int(converged),
                "beta1_intercept": estimate[0],
                "beta1_slope": estimate[1],
                "beta2_intercept": estimate[2],
                "beta2_slope": estimate[3],
                "sigma2": estimate[4],
                "beta_squared_error": float(np.mean(beta_error**2)),
                "sigma2_error": float(estimate[4] - TRUE_SIGMA**2),
                "sigma2_bound_hit": int(sigma_bound_hit),
            }
        )
    return rows


def _beta_rmse_mcse(value: pd.Series) -> float:
    squared_error = value.to_numpy(dtype=float)
    mean_squared_error = float(np.mean(squared_error))
    if squared_error.size < 2 or mean_squared_error <= 0.0:
        return 0.0
    mse_mcse = float(np.std(squared_error, ddof=1) / np.sqrt(squared_error.size))
    return mse_mcse / (2.0 * np.sqrt(mean_squared_error))


def _mae_mcse(value: pd.Series) -> float:
    absolute_error = np.abs(value.to_numpy(dtype=float))
    if absolute_error.size < 2:
        return 0.0
    return float(np.std(absolute_error, ddof=1) / np.sqrt(absolute_error.size))


def _rate_mcse(value: pd.Series) -> float:
    rate = float(np.mean(value))
    return float(np.sqrt(rate * (1.0 - rate) / value.size))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    valid = results.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta_squared_error", "sigma2_error"])
    grouped = valid.groupby(["scenario", "method"], sort=False)
    summary = grouped.agg(
        replicates=("seed", "size"),
        convergence_rate=("converged", "mean"),
        beta_rmse=("beta_squared_error", lambda value: float(np.sqrt(np.mean(value)))),
        beta_rmse_mcse=("beta_squared_error", _beta_rmse_mcse),
        sigma2_bias=("sigma2_error", "mean"),
        sigma2_mae=("sigma2_error", lambda value: float(np.mean(np.abs(value)))),
        sigma2_mae_mcse=("sigma2_error", _mae_mcse),
        sigma2_bound_rate=("sigma2_bound_hit", "mean"),
        sigma2_bound_rate_mcse=("sigma2_bound_hit", _rate_mcse),
    )
    return summary.reset_index()


def _run_task(task: tuple[int, str, int, int]) -> list[dict]:
    seed, scenario, n_clusters, cluster_size = task
    return one_replicate(seed, scenario, n_clusters, cluster_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--clusters", type=int, default=60)
    parser.add_argument("--cluster-size", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    seed_sequence = np.random.SeedSequence(args.seed)
    children = seed_sequence.spawn(args.replicates * len(SCENARIOS))
    tasks: list[tuple[int, str, int, int]] = []
    index = 0
    for scenario in SCENARIOS:
        for _ in range(args.replicates):
            seed = int(children[index].generate_state(1)[0])
            index += 1
            tasks.append((seed, scenario, args.clusters, args.cluster_size))

    all_rows: list[dict] = []
    if args.jobs == 1:
        task_results = map(_run_task, tasks)
        for rows in task_results:
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for rows in executor.map(_run_task, tasks, chunksize=1):
                all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    summary = summarize(results)
    results.to_csv(args.output / "robustness_replicates.csv", index=False)
    summary.to_csv(args.output / "robustness_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
