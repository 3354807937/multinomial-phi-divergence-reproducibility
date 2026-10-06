#!/usr/bin/env python3
"""Reproduce the equal-weight aggregate BMI sensitivity analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "bmi_aggregated.csv"
OUTPUT = ROOT / "results" / "bmi_aggregate_estimates.csv"
LAMBDAS = (0.0, 2.0 / 3.0, 1.0, 1.5, 2.0)


def main() -> None:
    frame = pd.read_csv(DATA)
    design = np.column_stack(
        [frame["gender"].eq("Male").astype(float), frame["age_group"].eq("65+").astype(float)]
    )
    counts = frame[["acceptable", "overweight", "obese"]].to_numpy(dtype=float)
    sizes = counts.sum(axis=1)
    observed = counts / sizes[:, None]
    nodes, gh_weights = hermgauss(25)
    gh_weights = gh_weights / np.sqrt(np.pi)

    def marginal_probabilities(theta: np.ndarray) -> np.ndarray:
        beta = theta[:4]
        sigma = np.exp(theta[4])
        random_intercept = np.sqrt(2.0) * sigma * nodes
        eta1 = design[:, 0, None] * beta[0] + design[:, 1, None] * beta[1] + random_intercept
        eta2 = design[:, 0, None] * beta[2] + design[:, 1, None] * beta[3] + random_intercept
        linear = np.stack((eta1, eta2, np.zeros_like(eta1)), axis=-1)
        linear -= linear.max(axis=-1, keepdims=True)
        probabilities = np.exp(linear)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return np.sum(probabilities * gh_weights[None, :, None], axis=1)

    def objective(theta: np.ndarray, lam: float) -> float:
        fitted = np.clip(marginal_probabilities(theta), 1e-14, 1.0)
        ratio = np.clip(observed / fitted, 1e-14, 1e14)
        if abs(lam) < 1e-12:
            phi = ratio * np.log(ratio) - ratio + 1.0
        else:
            phi = (ratio ** (lam + 1.0) - ratio - lam * (ratio - 1.0)) / (lam * (lam + 1.0))
        return float(np.sum(sizes[:, None] * fitted * phi) / np.sum(sizes))

    starts = (
        np.r_[np.zeros(4), np.log(0.50)],
        np.r_[[0.8, 0.5, 0.2, 0.1], np.log(1.00)],
        np.r_[[1.5, 1.0, 0.4, 0.25], np.log(0.70)],
    )
    bounds = [(-15.0, 15.0)] * 4 + [(np.log(0.02), np.log(4.0))]
    rows = []
    for lam in LAMBDAS:
        candidates = [
            minimize(
                lambda value, tuning=lam: objective(value, tuning),
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-10},
            )
            for start in starts
        ]
        result = min(candidates, key=lambda item: item.fun)
        rows.append(
            {
                "lambda": lam,
                "beta11_gender": result.x[0],
                "beta12_age": result.x[1],
                "beta21_gender": result.x[2],
                "beta22_age": result.x[3],
                "sigma2": np.exp(2.0 * result.x[4]),
                "objective": result.fun,
                "converged": int(result.success),
                "sigma_at_lower_bound": int(np.isclose(result.x[4], bounds[-1][0], atol=1e-6)),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT, index=False)
    print(output.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
