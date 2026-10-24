"""API tests against the committed model bundle in models/lc-default-risk."""

import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lc_risk import schema
from lc_risk.bundle import load_bundle
from lc_risk.serving.app import DEFAULT_MODEL_DIR, create_app
from lc_risk.serving.schemas import EXAMPLE_APPLICATION, MAX_BATCH, REQUIRED_FIELDS
from tests.conftest import make_applications


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(DEFAULT_MODEL_DIR))


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(DEFAULT_MODEL_DIR)


def as_request(row: pd.Series) -> dict:
    """A cohort-style row -> the JSON an API client would send."""
    body = {}
    for field in schema.APPLICATION_FIELDS:
        value = row[field.name]
        if pd.isna(value):
            body[field.name] = None
        elif field.kind == "date":
            body[field.name] = value.date().isoformat()
        elif field.kind == "numeric":
            body[field.name] = float(value)
        else:
            body[field.name] = str(value)
    body["application_date"] = row[schema.DATE_COL].date().isoformat()
    return body


@pytest.fixture(scope="module")
def applications():
    frame, _ = make_applications(40, seed=11)
    for name in REQUIRED_FIELDS:  # required fields cannot be null in a request
        frame[name] = frame[name].fillna(frame[name].dropna().iloc[0])
    return frame


def test_committed_bundle_loads_and_scores(bundle, applications):
    """Guards against schema or library changes that break the released model."""
    pd_ = bundle.model.predict_pd(applications)
    assert np.isfinite(pd_).all() and ((pd_ > 0) & (pd_ < 1)).all()


def test_health_and_model_info(client, bundle):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model").json()
    assert info["version"] == bundle.version
    assert "addr_state" in info["inputs_accepted_but_unused"]
    assert set(info["required_fields"]) == set(REQUIRED_FIELDS)


def test_example_application(client):
    response = client.post("/predict", json=EXAMPLE_APPLICATION)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 < body["probability_of_default"] < 1
    assert 1 <= len(body["reasons"]) <= 3
    assert all(r["contribution"] > 0 and r["description"] for r in body["reasons"])
    assert not {r["feature"] for r in body["reasons"]} & set(schema.EXCLUDED_FROM_MODEL)
    assert body["warnings"] == []
    assert response.headers["x-request-id"]


def test_api_matches_offline_scoring(client, bundle, applications):
    """Train/serve parity: JSON through the API == the cohort-style frame scored directly."""
    offline = bundle.model.predict_pd(applications)
    payload = {"applications": [as_request(row) for _, row in applications.iterrows()]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200, response.text
    online = [r["probability_of_default"] for r in response.json()["results"]]
    np.testing.assert_allclose(online, offline, atol=1e-6)


def test_batch_matches_single_requests(client, applications):
    rows = [as_request(row) for _, row in applications.head(3).iterrows()]
    batch = client.post("/predict/batch", json={"applications": rows}).json()["results"]
    single = [client.post("/predict", json=row).json() for row in rows]
    for b, s in zip(batch, single, strict=True):
        assert b["probability_of_default"] == s["probability_of_default"]
        assert b["reasons"] == s["reasons"]


def test_example_covers_every_field():
    assert set(EXAMPLE_APPLICATION) == {f.name for f in schema.APPLICATION_FIELDS}


def test_nullable_fields_can_be_omitted(client):
    minimal = {k: EXAMPLE_APPLICATION[k] for k in REQUIRED_FIELDS}
    response = client.post("/predict", json=minimal)
    assert response.status_code == 200, response.text
    assert math.isfinite(response.json()["probability_of_default"])
    assert all(r["value"] is not None for r in response.json()["reasons"]
               if r["feature"] in REQUIRED_FIELDS)  # fmt: skip


@pytest.mark.parametrize(
    "change",
    [
        {"fico_range_low": 900},  # above the schema maximum
        {"purpose": "crypto"},  # unknown category
        {"loan_amnt": None},  # required field
        {"num_tl_op_past_12m": None},  # always present in training, so required
        {"fico_score": 700},  # unknown field name (typo protection)
        {"earliest_cr_line": "not-a-date"},
    ],
)
def test_invalid_requests_are_rejected(client, change):
    response = client.post("/predict", json={**EXAMPLE_APPLICATION, **change})
    assert response.status_code == 422


def test_batch_size_limits(client):
    assert client.post("/predict/batch", json={"applications": []}).status_code == 422
    too_many = {"applications": [EXAMPLE_APPLICATION] * (MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_outside_policy_range_is_flagged(client):
    risky = {**EXAMPLE_APPLICATION, "fico_range_low": 620, "dti": 45}
    warnings = client.post("/predict", json=risky).json()["warnings"]
    assert any("FICO" in w for w in warnings) and any("DTI" in w for w in warnings)


def test_higher_fico_never_raises_risk(client):
    scores = [
        client.post("/predict", json={**EXAMPLE_APPLICATION, "fico_range_low": f}).json()[
            "probability_of_default"
        ]
        for f in (660, 700, 740, 780, 820)
    ]
    assert scores == sorted(scores, reverse=True)


def test_state_does_not_change_the_score(client):
    scores = {
        client.post("/predict", json={**EXAMPLE_APPLICATION, "addr_state": s}).json()[
            "probability_of_default"
        ]
        for s in ("CA", "NY", "NV", "MS")
    }
    assert len(scores) == 1
