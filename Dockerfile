# Scoring service image: locked serving dependencies + the released model bundle.
# No training code paths, data, or notebooks are needed at runtime.

# --- build: install the locked serving dependencies into a virtualenv ---
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so code changes do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra serve --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra serve --no-editable

# --- runtime: virtualenv + model bundle, non-root ---
FROM python:3.13-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY models/lc-default-risk /app/models/lc-default-risk
ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_DIR=/app/models/lc-default-risk \
    PORT=8000 \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
# PORT is set by the hosting platform (Render injects it); WEB_CONCURRENCY sets workers.
CMD ["sh", "-c", "exec uvicorn --factory lc_risk.serving.app:create_app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
