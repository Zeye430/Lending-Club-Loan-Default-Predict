"""Record where an artifact came from: code revision and library versions."""

from __future__ import annotations

import hashlib
import subprocess
from importlib import metadata
from pathlib import Path

from lc_risk.config import ROOT

PACKAGE_DIR = Path(__file__).resolve().parent


def source_digest() -> str:
    """SHA-256 over the package's Python sources.

    Identifies the exact code a model was trained with, even from an uncommitted working
    tree: anyone can recompute it from a checkout and compare.
    """
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        digest.update(path.relative_to(PACKAGE_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision() -> dict[str, str | bool | None]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions(*names: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions
