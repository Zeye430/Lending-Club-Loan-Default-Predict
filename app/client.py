"""Thin client for the scoring API: the demo's only connection to the model.

Everything the UI knows about fields (types, ranges, categories, descriptions, example
values) is read from the API's OpenAPI document, so the form cannot drift from the service
it calls. The demo never loads the model or imports the training package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx

DEFAULT_API_URL = "https://lending-club-loan-default-predict.onrender.com"
# Render's free tier sleeps after ~15 idle minutes; waking takes 30-60 s.
TIMEOUT_SECONDS = 90.0


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: Literal["number", "category", "date"]
    description: str
    required: bool
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[str, ...] = ()


class ApiError(RuntimeError):
    def __init__(self, status: int, messages: list[str]):
        super().__init__("; ".join(messages))
        self.status = status
        self.messages = messages


def make_client(base_url: str = DEFAULT_API_URL) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS)


def _application_schema(openapi: dict) -> dict:
    return openapi["components"]["schemas"]["Application"]


def application_fields(openapi: dict) -> dict[str, FieldSpec]:
    """Field specs for the request body, parsed from the OpenAPI document."""
    schema = _application_schema(openapi)
    required = set(schema.get("required", []))
    fields = {}
    for name, prop in schema["properties"].items():
        # optional fields are `anyOf: [<type>, {"type": "null"}]`
        variant = next(v for v in prop.get("anyOf", [prop]) if v.get("type") != "null")
        if "enum" in variant:
            spec = FieldSpec(name, "category", prop.get("description", ""), name in required,
                             options=tuple(variant["enum"]))  # fmt: skip
        elif variant.get("format") == "date":
            spec = FieldSpec(name, "date", prop.get("description", ""), name in required)
        else:
            spec = FieldSpec(name, "number", prop.get("description", ""), name in required,
                             variant.get("minimum"), variant.get("maximum"))  # fmt: skip
        fields[name] = spec
    return fields


def example_application(openapi: dict) -> dict:
    """The API's own example request; nulls are dropped from OpenAPI examples, so restore them."""
    example = dict(_application_schema(openapi)["example"])
    for name in application_fields(openapi):
        example.setdefault(name, None)
    example.pop("application_date", None)
    return example


def _raise_for_error(response: httpx.Response) -> None:
    if response.status_code == 422:
        detail = response.json().get("detail", [])
        messages = [
            f"{'.'.join(str(p) for p in d.get('loc', [])[1:]) or 'request'}: {d.get('msg')}"
            for d in detail
        ]
        raise ApiError(422, messages or [response.text])
    if response.is_error:
        raise ApiError(response.status_code, [response.text[:300]])


def get_json(client: httpx.Client, path: str) -> dict:
    response = client.get(path)
    _raise_for_error(response)
    return response.json()


def predict(client: httpx.Client, application: dict) -> dict:
    response = client.post("/predict", json=application)
    _raise_for_error(response)
    return response.json()


def predict_batch(client: httpx.Client, applications: list[dict]) -> list[dict]:
    response = client.post("/predict/batch", json={"applications": applications})
    _raise_for_error(response)
    return response.json()["results"]
