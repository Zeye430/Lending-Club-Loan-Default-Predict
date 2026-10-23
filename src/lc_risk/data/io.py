"""File access helpers shared by data discovery and the cohort build."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_raw_chunks(
    path: Path, *, usecols: Iterable[str] | None = None, chunk_size: int = 200_000
) -> Iterator[pd.DataFrame]:
    """Stream the raw export. Despite its ``.gzip`` suffix the Kaggle file is plain CSV."""
    return pd.read_csv(
        path,
        usecols=list(usecols) if usecols is not None else None,
        chunksize=chunk_size,
        compression=None,
        low_memory=False,
    )
