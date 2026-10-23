"""Population Stability Index (PSI) for feature and score drift.

PSI compares the distribution of a variable in a current sample against a reference
sample, binned on the reference. Common reading in credit risk:
< 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
Missing values get their own bin, because a change in missingness is itself drift.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from lc_risk.schema import Field

MISSING = "__missing__"
_EPS = 1e-4  # floor on bin shares so empty bins do not produce infinities


def _shares(binned: pd.Series, categories: list) -> pd.Series:
    counts = binned.value_counts().reindex(categories, fill_value=0)
    return (counts / max(len(binned), 1)).clip(lower=_EPS)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    if pd.api.types.is_numeric_dtype(reference):
        observed = reference.dropna()
        if observed.empty:
            edges = np.array([-np.inf, np.inf])
        else:
            quantiles = np.quantile(observed, np.linspace(0, 1, bins + 1))
            edges = np.unique(np.concatenate(([-np.inf], quantiles[1:-1], [np.inf])))
        labels = list(range(len(edges) - 1))

        def binned(s: pd.Series) -> pd.Series:
            cut = pd.cut(s, edges, labels=labels, include_lowest=True).astype("object")
            return cut.where(s.notna(), MISSING)

        ref_bins, cur_bins = binned(reference), binned(current)
        categories = [*labels, MISSING]
    else:
        ref_bins = reference.astype("object").where(reference.notna(), MISSING)
        cur_bins = current.astype("object").where(current.notna(), MISSING)
        categories = sorted(set(ref_bins) | set(cur_bins), key=str)

    ref, cur = _shares(ref_bins, categories), _shares(cur_bins, categories)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def psi_table(
    reference: pd.DataFrame, current: pd.DataFrame, fields: Iterable[Field], bins: int = 10
) -> pd.DataFrame:
    """PSI per field, largest first."""
    rows = [
        {"field": f.name, "psi": psi(reference[f.name], current[f.name], bins)}
        for f in fields
        if f.kind != "date"
    ]
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
