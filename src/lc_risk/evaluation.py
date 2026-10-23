"""Metrics for a probability-of-default model: ranking, calibration, and lending outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def discrimination(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": float(np.max(tpr - fpr)),
        "average_precision": average_precision_score(y, p),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate per predicted-PD decile."""
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    return frame.groupby("bin").agg(
        loans=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    table = calibration_table(y, p, bins)
    weights = table["loans"] / table["loans"].sum()
    return {
        "brier": brier_score_loss(y, p),
        "ece": float((weights * (table["predicted"] - table["observed"]).abs()).sum()),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


APPROVAL_RATES = tuple(np.round(np.linspace(0.1, 1.0, 19), 2))


def approval_curve(y: np.ndarray, score: np.ndarray, approval_rates=APPROVAL_RATES) -> pd.DataFrame:
    """Bad rate among approved loans when approving the lowest-risk share of applicants."""
    order = np.argsort(score, kind="stable")
    y_sorted = np.asarray(y)[order]
    rows = []
    for rate in approval_rates:
        n = max(int(round(rate * len(y_sorted))), 1)
        rows.append({"approval_rate": rate, "bad_rate": y_sorted[:n].mean()})
    return pd.DataFrame(rows)


def realized_return(
    score: np.ndarray, funded: np.ndarray, repaid: np.ndarray, approval_rate: float
) -> float:
    """Net realized return on the approved book: (total repaid - funded) / funded.

    Uses each loan's actual cash flows (``total_pymnt`` includes recoveries), so it
    reflects pricing, prepayment, and loss severity, not just default counts.
    """
    funded, repaid = np.asarray(funded, float), np.asarray(repaid, float)
    if not (np.isfinite(funded).all() and np.isfinite(repaid).all()):
        raise ValueError("funded and repaid amounts must be finite for every loan")
    order = np.argsort(score, kind="stable")
    n = max(int(round(approval_rate * len(order))), 1)
    chosen = order[:n]
    return float(np.sum(repaid[chosen]) / np.sum(funded[chosen]) - 1)


def bootstrap_auc_difference(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 200, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap of AUC(b) - AUC(a) with a 95% interval."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "difference": roc_auc_score(y, p_b) - roc_auc_score(y, p_a),
        "ci_low": float(low),
        "ci_high": float(high),
    }
