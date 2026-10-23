"""Check a frame against the field spec in ``lc_risk.schema``."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from lc_risk.schema import Field


@dataclass(frozen=True)
class Violation:
    field: str
    problem: str
    count: int

    def __str__(self) -> str:
        return f"{self.field}: {self.problem} ({self.count:,} rows)"


def validate_inputs(frame: pd.DataFrame, fields: Iterable[Field]) -> list[Violation]:
    """Return every spec violation; an empty list means the frame conforms.

    Nulls are allowed everywhere: missingness is handled by the model, and for
    event-history fields it carries meaning.
    """
    violations: list[Violation] = []
    for field in fields:
        if field.name not in frame.columns:
            violations.append(Violation(field.name, "missing column", len(frame)))
            continue
        values = frame[field.name]

        if field.kind == "numeric":
            if not pd.api.types.is_numeric_dtype(values):
                violations.append(Violation(field.name, f"non-numeric dtype {values.dtype}", 0))
                continue
            if field.ge is not None and (n := int((values < field.ge).sum())):
                violations.append(Violation(field.name, f"below minimum {field.ge}", n))
            if field.le is not None and (n := int((values > field.le).sum())):
                violations.append(Violation(field.name, f"above maximum {field.le}", n))

        elif field.kind == "categorical":
            unknown = values.notna() & ~values.isin(field.categories)
            if n := int(unknown.sum()):
                examples = sorted(values[unknown].astype(str).unique())[:5]
                violations.append(Violation(field.name, f"unknown categories {examples}", n))

        elif field.kind == "date" and not pd.api.types.is_datetime64_any_dtype(values):
            violations.append(Violation(field.name, f"non-datetime dtype {values.dtype}", 0))

    return violations
