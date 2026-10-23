"""Shared plots and markdown helpers for the validation and test reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lc_risk.evaluation import approval_curve, calibration_table


def md_table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(value):
        return floatfmt.format(value) if isinstance(value, float | np.floating) else str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def plot_calibration(y: np.ndarray, curves: dict[str, np.ndarray], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    top = 0.0
    for label, p in curves.items():
        table = calibration_table(y, p)
        ax.plot(100 * table["predicted"], 100 * table["observed"], "o-", label=label)
        top = max(top, table["predicted"].max(), table["observed"].max())
    ax.plot([0, 100 * top], [0, 100 * top], "k--", lw=0.8, label="perfect calibration")
    ax.set_xlabel("predicted default rate (%)")
    ax.set_ylabel("observed default rate (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_approval_curves(
    y: np.ndarray, scores: dict[str, np.ndarray], path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, score in scores.items():
        curve = approval_curve(y, score)
        ax.plot(100 * curve["approval_rate"], 100 * curve["bad_rate"], label=label)
    ax.set_xlabel("approval rate (%): approve the lowest-risk share")
    ax.set_ylabel("default rate among approved (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
