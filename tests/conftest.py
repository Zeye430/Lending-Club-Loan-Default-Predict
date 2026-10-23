import numpy as np
import pandas as pd
import pytest

from lc_risk import schema


def make_applications(n: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic raw applications that satisfy the input spec, with a learnable target."""
    rng = np.random.default_rng(seed)
    frame = {}
    for field in schema.ALL_INPUT_FIELDS:
        if field.kind == "numeric":
            low = field.ge if field.ge is not None else 0.0
            high = field.le if field.le is not None else low + 100.0
            values = rng.uniform(low, high, n)
            values[rng.random(n) < 0.05] = np.nan
            frame[field.name] = values
        elif field.kind == "categorical":
            frame[field.name] = rng.choice(field.categories, n)
        else:
            frame[field.name] = pd.to_datetime("2000-01-01") + pd.to_timedelta(
                rng.integers(0, 3650, n), unit="D"
            )
    frame[schema.DATE_COL] = pd.to_datetime("2014-06-01")
    applications = pd.DataFrame(frame)

    fico = applications["fico_range_low"].fillna(575)
    logit = -1.5 - 0.01 * (fico - 575) + 0.03 * applications["dti"].fillna(50) - 1.5
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return applications, target


@pytest.fixture(scope="session")
def applications():
    return make_applications(3_000)
