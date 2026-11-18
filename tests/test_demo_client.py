"""The demo's API client, run against the real FastAPI app and committed model bundle.

This is the contract test between the Streamlit demo and the service: if the API's schema
changes in a way the demo cannot read, CI fails here.
"""

import pytest
from fastapi.testclient import TestClient

from app.client import (
    ApiError,
    application_fields,
    example_application,
    get_json,
    predict,
    predict_batch,
)
from lc_risk import schema
from lc_risk.serving.app import DEFAULT_MODEL_DIR, create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(DEFAULT_MODEL_DIR))


@pytest.fixture(scope="module")
def openapi(client):
    return get_json(client, "/openapi.json")


def test_fields_mirror_the_shared_schema(openapi):
    fields = application_fields(openapi)
    for field in schema.APPLICATION_FIELDS:
        spec = fields[field.name]
        assert spec.required == (not field.nullable), field.name
        if field.kind == "categorical":
            assert spec.kind == "category" and spec.options == field.categories
        elif field.kind == "numeric":
            assert spec.kind == "number"
            assert (spec.minimum, spec.maximum) == (field.ge, field.le), field.name
        else:
            assert spec.kind == "date"
    assert not fields["application_date"].required


def test_example_round_trips_through_the_api(client, openapi):
    example = example_application(openapi)
    assert set(example) == {f.name for f in schema.APPLICATION_FIELDS}
    result = predict(client, example)
    assert 0 < result["probability_of_default"] < 1


def test_validation_errors_name_the_field(client, openapi):
    bad = {**example_application(openapi), "fico_range_low": 900, "revol_util": None}
    with pytest.raises(ApiError) as excinfo:
        predict(client, bad)
    assert excinfo.value.status == 422
    assert any(m.startswith("fico_range_low") for m in excinfo.value.messages)
    assert any(m.startswith("revol_util") for m in excinfo.value.messages)


def test_batch_what_if(client, openapi):
    base = example_application(openapi)
    grid = [{**base, "fico_range_low": f} for f in (660, 720, 780)]
    scores = [r["probability_of_default"] for r in predict_batch(client, grid)]
    assert scores == sorted(scores, reverse=True)
