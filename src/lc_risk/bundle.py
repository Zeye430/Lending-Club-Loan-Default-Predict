"""Save and load a released model bundle: ``model.joblib`` + ``metadata.json``.

Loading verifies the file hash recorded at release, and that the installed scikit-learn and
xgboost match the versions the model was pickled with (pickles are not guaranteed to load
correctly across versions of those libraries).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import sklearn
import xgboost

from lc_risk.data.io import sha256_file
from lc_risk.model import RiskModel

log = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
PINNED_LIBRARIES = ("scikit-learn", "xgboost")


def installed_versions() -> dict[str, str]:
    """Versions of the libraries a pickled model depends on.

    Read from the modules rather than package metadata: on Linux the `xgboost` module is
    installed by the `xgboost-cpu` distribution.
    """
    return {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}


class BundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bundle:
    model: RiskModel
    metadata: dict
    path: Path

    @property
    def name(self) -> str:
        return self.metadata["name"]

    @property
    def version(self) -> str:
        return self.metadata["version"]


def save_bundle(model: RiskModel, metadata: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / MODEL_FILE, compress=3)
    metadata = {
        **metadata,
        "libraries": {**metadata.get("libraries", {}), **installed_versions()},
        "model_sha256": sha256_file(out_dir / MODEL_FILE),
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    return metadata


def load_bundle(path: Path, strict_versions: bool = True) -> Bundle:
    metadata = json.loads((path / METADATA_FILE).read_text())
    if sha256_file(path / MODEL_FILE) != metadata["model_sha256"]:
        raise BundleError(f"{path / MODEL_FILE} does not match the hash in {METADATA_FILE}")

    installed = installed_versions()
    expected = {lib: metadata["libraries"].get(lib) for lib in PINNED_LIBRARIES}
    if mismatched := {lib: (expected[lib], installed[lib]) for lib in PINNED_LIBRARIES
                      if expected[lib] != installed[lib]}:  # fmt: skip
        message = f"library versions differ from training (expected, installed): {mismatched}"
        if strict_versions:
            raise BundleError(message)
        log.warning(message)

    return Bundle(model=joblib.load(path / MODEL_FILE), metadata=metadata, path=path)
