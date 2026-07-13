from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coefs


def _clean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).dropna()


def mediation_analysis(df: pd.DataFrame, outcome: str, treatment: str = "epsilon",
                        mediator: str = "representation_distortion",
                        n_boot: int = 1000, seed: int = 0) -> dict:
    data = _clean(df, [treatment, mediator, outcome])
    if len(data) < 5:
        return {"outcome": outcome, "error": "insufficient data"}

    t = data[treatment].to_numpy(dtype=float)
    m = data[mediator].to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)

    a_coef = _ols(t.reshape(-1, 1), m)[1]
    bc_coef = _ols(np.column_stack([t, m]), y)
    c_prime, b_coef = bc_coef[1], bc_coef[2]
    c_coef = _ols(t.reshape(-1, 1), y)[1]

    indirect = a_coef * b_coef
    prop_mediated = float(indirect / c_coef) if abs(c_coef) > 1e-9 else float("nan")

    rng = np.random.default_rng(seed)
    n = len(data)
    boot_indirect = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tb, mb, yb = t[idx], m[idx], y[idx]
        a_b = _ols(tb.reshape(-1, 1), mb)[1]
        bc_b = _ols(np.column_stack([tb, mb]), yb)
        boot_indirect.append(a_b * bc_b[2])
    lo, hi = np.percentile(boot_indirect, [2.5, 97.5])

    return {
        "outcome": outcome,
        "path_a_treatment_to_mediator": float(a_coef),
        "path_b_mediator_to_outcome": float(b_coef),
        "path_c_total_effect": float(c_coef),
        "path_c_prime_direct_effect": float(c_prime),
        "indirect_effect": float(indirect),
        "proportion_mediated": prop_mediated,
        "indirect_effect_ci95": [float(lo), float(hi)],
        "n": n,
    }


def decompose_pathways(df: pd.DataFrame, outcome: str,
                        predictors: Optional[List[str]] = None) -> dict:
    if predictors is None:
        predictors = ["epsilon", "representation_distortion", "loss_variance", "calibration_ece"]
    cols = predictors + [outcome]
    data = _clean(df, cols)
    if len(data) < len(predictors) + 2:
        return {"outcome": outcome, "error": "insufficient data"}

    X = data[predictors].to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xz = (X - X_mean) / X_std
    y_std = y.std() if y.std() > 0 else 1.0
    yz = (y - y.mean()) / y_std

    coefs = _ols(Xz, yz)
    intercept, betas = coefs[0], coefs[1:]

    y_hat = intercept + Xz @ betas
    ss_res = float(np.sum((yz - y_hat) ** 2))
    ss_tot = float(np.sum((yz - yz.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "outcome": outcome,
        "standardized_coefficients": dict(zip(predictors, [float(b) for b in betas])),
        "r_squared": r2,
        "n": len(data),
    }


def plot_privacy_alignment_curves(df: pd.DataFrame, metric: str, out_path: str,
                                   mechanism_col: str = "mechanism", epsilon_col: str = "epsilon"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for mech, sub in df.groupby(mechanism_col):
        agg = sub.groupby(epsilon_col)[metric].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", label=mech)
    ax.set_xlabel("Privacy budget (epsilon)")
    ax.set_ylabel(metric)
    ax.set_title(f"Privacy–alignment–utility curve: {metric}")
    ax.legend(fontsize=8)
    ax.invert_xaxis()  # smaller epsilon (stronger privacy) shown left-to-right as increasing privacy
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
