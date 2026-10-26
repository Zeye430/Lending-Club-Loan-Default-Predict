"""FastAPI scoring service.

    MODEL_DIR=models/lc-default-risk uv run uvicorn --factory lc_risk.serving.app:create_app

The bundle is loaded once at startup; a bundle that fails its hash or library-version
check stops the service from starting rather than serving a broken model. Each request
writes one JSON log line (latency, model version, batch size, mean PD) and no applicant
data.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from lc_risk import config
from lc_risk.bundle import load_bundle
from lc_risk.serving.schemas import (
    REQUIRED_FIELDS,
    Application,
    BatchRequest,
    BatchResponse,
    ModelInfo,
    ScoreResponse,
)
from lc_risk.serving.scoring import score

log = logging.getLogger("lc_risk.serving")

DEFAULT_MODEL_DIR = config.ROOT / "models" / "lc-default-risk"


def create_app(model_dir: Path | None = None) -> FastAPI:
    if not log.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    bundle = load_bundle(model_dir)
    # One thread per request: the service scales with workers, and single-row scoring
    # gains nothing from a thread pool. Set directly: set_params() on a fitted XGBClassifier
    # re-sends every parameter to the booster and rejects the dict-form monotone constraints.
    threads = int(os.environ.get("MODEL_THREADS", "1"))
    bundle.model.estimator.n_jobs = threads
    bundle.model.estimator.get_booster().set_param("nthread", threads)
    info = ModelInfo(name=bundle.name, version=bundle.version)
    meta = bundle.metadata

    app = FastAPI(
        title="Lending Club default-risk API",
        version=bundle.version,
        description=(
            "Calibrated probability of default for 36-month individual Lending Club loans, "
            "with the top risk factors for each applicant. See the model card in "
            "`models/lc-default-risk/model_card.md`."
        ),
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        request.state.log = {}
        response = await call_next(request)
        record = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(1000 * (time.perf_counter() - started), 2),
            "model_version": bundle.version,
            **request.state.log,
        }
        log.info(json.dumps(record))
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    # Render sets RENDER_GIT_COMMIT for Git-backed services; CI polls /health for it to
    # confirm the commit it tested is the one serving traffic.
    commit = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "commit": commit, "model": info.model_dump()}

    @app.get("/model")
    def model_info() -> dict:
        return {
            **info.model_dump(),
            "run_id": meta["run_id"],
            "trained_on_commit": (meta.get("code") or {}).get("commit"),
            "target": meta["target"],
            "inputs_used": meta["input_fields"],
            "inputs_accepted_but_unused": list(meta.get("excluded_fields") or {}),
            "calibration": meta["calibration"],
            "test": {k: meta["test"][k] for k in ("rows", "default_rate", "auc", "gini", "ks")},
            "required_fields": list(REQUIRED_FIELDS),
        }

    @app.post("/predict", response_model=ScoreResponse)
    def predict(application: Application, request: Request) -> ScoreResponse:
        [result] = score(bundle, [application])
        request.state.log.update(n=1, mean_pd=result.probability_of_default)
        return ScoreResponse(model=info, **result.model_dump())

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        results = score(bundle, batch.applications)
        mean_pd = sum(r.probability_of_default for r in results) / len(results)
        request.state.log.update(n=len(results), mean_pd=round(mean_pd, 6))
        return BatchResponse(model=info, results=results)

    return app
