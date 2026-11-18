"""Smoke test of the Streamlit demo, wired to the in-process API instead of the network."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lc_risk.serving.app import DEFAULT_MODEL_DIR, create_app

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["api_client"] = TestClient(create_app(DEFAULT_MODEL_DIR))
    at.run()
    assert not at.exception, at.exception
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


def score(at, preset):
    at.radio[0].set_value(preset).run()
    button(at, "Score").click().run()
    assert not at.exception and not at.error, (at.exception, at.error)
    return float(at.metric[0].value.rstrip("%"))


def test_presets_score_in_a_sensible_order(app):
    strong, typical, stretched = (score(app, p) for p in
                                  ("Strong profile", "Typical applicant", "Stretched borrower"))  # fmt: skip
    assert strong < typical < stretched


def test_what_if_runs_a_batch(app):
    button(app, "Run what-if").click().run()
    assert not app.exception and not app.error


def test_reasons_are_formatted_for_people(app):
    score(app, "Stretched borrower")
    reasons = next(d.value for d in app.dataframe if "contribution" in d.value.columns)
    values = list(reasons["value"])
    assert "not reported" in values  # emp_length is blank in this preset
    assert all(len(v) <= 12 for v in values), values  # no long float tails
