import json

import numpy as np
import pytest

from lc_risk.bundle import METADATA_FILE, MODEL_FILE, BundleError, load_bundle, save_bundle
from lc_risk.model import RiskModel
from lc_risk.training.estimators import xgboost_pipeline
from tests.conftest import make_applications


@pytest.fixture
def bundle_dir(tmp_path):
    X, y = make_applications(1_000, seed=5)
    pipeline = xgboost_pipeline(False, {"n_estimators": 10, "max_depth": 2}, n_jobs=1).fit(X, y)
    model = RiskModel(pipeline).with_calibration(X, y)
    save_bundle(model, {"name": "test-model", "version": "0.0.1"}, tmp_path)
    return tmp_path, model, X


def test_round_trip_gives_identical_predictions(bundle_dir):
    path, model, X = bundle_dir
    bundle = load_bundle(path)
    assert (bundle.name, bundle.version) == ("test-model", "0.0.1")
    np.testing.assert_array_equal(bundle.model.predict_pd(X), model.predict_pd(X))


def test_tampered_model_file_is_rejected(bundle_dir):
    path, _, _ = bundle_dir
    with (path / MODEL_FILE).open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(BundleError, match="hash"):
        load_bundle(path)


def test_library_version_mismatch(bundle_dir):
    path, _, _ = bundle_dir
    metadata = json.loads((path / METADATA_FILE).read_text())
    metadata["libraries"]["xgboost"] = "0.0.0"
    (path / METADATA_FILE).write_text(json.dumps(metadata))

    with pytest.raises(BundleError, match="xgboost"):
        load_bundle(path)
    assert load_bundle(path, strict_versions=False).name == "test-model"
