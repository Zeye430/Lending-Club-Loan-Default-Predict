"""Streamlit demo for the Lending Club default-risk API.

    uv run --group demo streamlit run app/streamlit_app.py

The app only talks to the deployed API over HTTP (set LC_API_URL to use another one). Field
types, ranges, categories, and descriptions come from the API's OpenAPI document.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import (  # noqa: E402
    DEFAULT_API_URL,
    ApiError,
    FieldSpec,
    application_fields,
    example_application,
    get_json,
    make_client,
    predict,
    predict_batch,
)

REPO = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Zeye430/Lending-Club-Loan-Default-Predict"
API_URL = os.environ.get("LC_API_URL", DEFAULT_API_URL).rstrip("/")

KEY_FIELDS = {
    "loan_amnt": "Loan amount ($)",
    "purpose": "Purpose",
    "annual_inc": "Annual income ($)",
    "emp_length": "Years employed (blank = not given)",
    "home_ownership": "Home ownership",
    "verification_status": "Income verification",
    "dti": "Debt-to-income (%)",
    "fico_range_low": "FICO (lower bound)",
    "earliest_cr_line": "Earliest credit line",
    "revol_util": "Revolving utilization (%)",
    "inq_last_6mths": "Inquiries, last 6 months",
    "acc_open_past_24mths": "Accounts opened, last 24 months",
    "delinq_2yrs": "Delinquencies, last 2 years",
    "mort_acc": "Mortgage accounts",
}
UNUSED_FIELDS = {"addr_state"}  # accepted and recorded by the API, ignored by the model

# Overrides on top of the API's own example (a typical applicant: training medians).
PRESETS = {
    "Typical applicant": {},
    "Stretched borrower": {
        "loan_amnt": 20000, "annual_inc": 42000, "emp_length": None, "home_ownership": "RENT",
        "dti": 31.5, "fico_range_low": 665, "revol_bal": 18650, "revol_util": 88.0,
        "bc_util": 93.0, "percent_bc_gt_75": 100.0, "bc_open_to_buy": 900,
        "inq_last_6mths": 3, "mths_since_recent_inq": 1, "acc_open_past_24mths": 10,
        "num_tl_op_past_12m": 5, "mo_sin_rcnt_tl": 1, "mo_sin_rcnt_rev_tl_op": 2,
        "delinq_2yrs": 1, "mths_since_last_delinq": 14, "pct_tl_nvr_dlq": 90.0, "mort_acc": 0,
        "earliest_cr_line": "2009-05-01", "mo_sin_old_rev_tl_op": 90,
    },
    "Strong profile": {
        "loan_amnt": 15000, "purpose": "credit_card", "annual_inc": 125000, "emp_length": 10,
        "dti": 8.5, "fico_range_low": 780, "revol_bal": 5900, "revol_util": 14.0,
        "total_rev_hi_lim": 42000, "total_bc_limit": 35000, "bc_util": 11.0,
        "percent_bc_gt_75": 0.0, "bc_open_to_buy": 31000, "inq_last_6mths": 0,
        "mths_since_recent_inq": 14, "acc_open_past_24mths": 2, "num_tl_op_past_12m": 0,
        "mort_acc": 3, "earliest_cr_line": "1994-03-01", "mo_sin_old_rev_tl_op": 270,
    },
}  # fmt: skip

# What-if ranges; the first five carry monotone constraints in the model.
WHAT_IF = {
    "fico_range_low": (640, 820),
    "dti": (0, 45),
    "annual_inc": (20_000, 200_000),
    "revol_util": (0, 120),
    "inq_last_6mths": (0, 8),
    "loan_amnt": (1_000, 35_000),
}
CONSTRAINED = {"fico_range_low", "dti", "annual_inc", "revol_util", "inq_last_6mths"}


@st.cache_resource
def _client(url: str):
    return make_client(url)


def api():
    """The HTTP client; tests inject a client bound to the in-process app."""
    return st.session_state.get("api_client") or _client(API_URL)


@st.cache_data(ttl=600, show_spinner=False)
def load_meta(url: str, _client) -> tuple[dict, dict]:
    return get_json(_client, "/openapi.json"), get_json(_client, "/model")


def field_input(spec: FieldSpec, label: str, value, key: str):
    if spec.kind == "category":
        options = list(spec.options) if spec.required else [None, *spec.options]
        return st.selectbox(label, options, index=options.index(value), key=key,
                            help=spec.description, format_func=lambda v: v or "—")  # fmt: skip
    if spec.kind == "date":
        picked = st.date_input(label, date.fromisoformat(value), key=key, help=spec.description,
                               min_value=date(1950, 1, 1), max_value=date.today())  # fmt: skip
        return picked.isoformat()
    return st.number_input(
        label,
        value=None if value is None else float(value),
        min_value=None if spec.minimum is None else float(spec.minimum),
        max_value=None if spec.maximum is None else float(spec.maximum),
        step=1.0,
        key=key,
        help=spec.description,
    )


def build_application(fields: dict[str, FieldSpec], base: dict, preset: str) -> dict:
    values = {**base, **PRESETS[preset]}
    application = dict(values)

    columns = st.columns(3)
    for i, (name, label) in enumerate(KEY_FIELDS.items()):
        with columns[i % 3]:
            application[name] = field_input(fields[name], label, values[name], f"{preset}:{name}")

    rest = [n for n, s in fields.items()
            if n not in KEY_FIELDS and n not in UNUSED_FIELDS and s.kind == "number"]  # fmt: skip
    with st.expander(f"All other credit-bureau fields ({len(rest)})"):
        st.caption("Blank means *none on file*. Only fields marked optional may be blank; the "
                   "API rejects a request missing any other field.")  # fmt: skip
        table = pd.DataFrame(
            {
                "field": rest,
                "value": [values[n] for n in rest],
                "optional": [not fields[n].required for n in rest],
                "description": [fields[n].description for n in rest],
            }
        ).astype({"value": "float64"})
        edited = st.data_editor(
            table,
            key=f"{preset}:rest",
            hide_index=True,
            disabled=["field", "optional", "description"],
            width="stretch",
        )
        for name, value in zip(edited["field"], edited["value"], strict=True):
            application[name] = None if pd.isna(value) else float(value)
        state = fields["addr_state"]
        application["addr_state"] = field_input(
            state, "State (recorded by the API, not used by the model)",
            values["addr_state"], f"{preset}:addr_state",
        )  # fmt: skip
    return application


def _display(value) -> str:
    if value is None:
        return "not reported"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    return str(value)


def show_result(result: dict, average: float) -> None:
    probability = result["probability_of_default"]
    left, right = st.columns([1, 2])
    left.metric(
        "Probability of default (36 months)",
        f"{probability:.1%}",
        delta=f"{probability - average:+.1%} vs 2016 average ({average:.1%})",
        delta_color="inverse",
    )
    for warning in result["warnings"]:
        left.warning(warning)
    with right:
        st.markdown("**What raises this applicant's risk most**")
        if not result["reasons"]:
            st.info("No factor raises this applicant's risk above the baseline.")
        else:
            # format before building the frame: pandas would turn None into NaN
            reasons = pd.DataFrame(
                [{**r, "value": _display(r["value"])} for r in result["reasons"]]
            )
            st.dataframe(
                reasons[["feature", "value", "contribution", "description"]],
                hide_index=True,
                width="stretch",
            )
            st.caption("Contribution: exact TreeSHAP increase in the log-odds of default.")


def score_tab(fields, base, average) -> dict:
    preset = st.radio("Start from", list(PRESETS), horizontal=True)
    application = build_application(fields, base, preset)
    if st.button("Score", type="primary"):
        try:
            st.session_state["result"] = predict(api(), application)
            st.session_state.pop("error", None)
        except ApiError as exc:
            st.session_state["error"] = exc.messages
            st.session_state.pop("result", None)
    if errors := st.session_state.get("error"):
        st.error("The API rejected the request:\n\n" + "\n".join(f"- {m}" for m in errors))
    if result := st.session_state.get("result"):
        show_result(result, average)
    return application


def what_if_tab(application: dict) -> None:
    name = st.selectbox("Vary", list(WHAT_IF), format_func=lambda n: KEY_FIELDS[n])
    low, high = WHAT_IF[name]
    st.caption(
        "Scores the applicant from the first tab 25 times in one batch request, changing only "
        "this field. "
        + ("The model constrains this field to move risk one way only."
           if name in CONSTRAINED else "")
    )  # fmt: skip
    if st.button("Run what-if"):
        grid = [low + (high - low) * i / 24 for i in range(25)]
        try:
            results = predict_batch(api(), [{**application, name: v} for v in grid])
        except ApiError as exc:
            st.error(
                "The API rejected the request:\n\n" + "\n".join(f"- {m}" for m in exc.messages)
            )
            return
        curve = pd.DataFrame(
            {
                KEY_FIELDS[name]: grid,
                "probability of default": [r["probability_of_default"] for r in results],
            }
        ).set_index(KEY_FIELDS[name])
        st.line_chart(curve)
        flagged = [g for g, r in zip(grid, results, strict=True) if r["warnings"]]
        if flagged:
            st.caption(f"{len(flagged)} of 25 points fall outside LC's historical approval range; "
                       "the API flags them because the model never saw such applicants.")  # fmt: skip


def about_tab(model: dict) -> None:
    test = model["test"]
    commit = model.get("trained_on_commit") or ""
    st.markdown(
        f"""
**{model["name"]} v{model["version"]}**: XGBoost with monotone constraints and Platt calibration,
trained on 36-month Lending Club loans issued 2013–14, calibrated on 2015, tested once on
{test["rows"]:,} loans from 2016. Trained from commit [`{commit[:7]}`]({GITHUB}/commit/{commit}).

| out-of-time test (2016) | |
|---|---|
| AUC | {test["auc"]:.3f} |
| Gini | {test["gini"]:.3f} |
| KS | {test["ks"]:.3f} |
| default rate | {test["default_rate"]:.1%} |

- Uses **{len(model["inputs_used"])}** application and credit-bureau fields; it does **not** use
  LC's own grade, the interest rate, or the applicant's state
  ({", ".join(model["inputs_accepted_but_unused"])} is recorded but ignored).
- Ranks risk better than LC's own sub-grade: approving the best 80% of 2016 applicants by this
  model gives a 12.3% default rate against 12.7% by LC grade.
- Monitoring replay: early-default AUC flagged the 2016 degradation **28 months** before
  lifetime outcomes could, while input drift never alerted.

[Model card]({GITHUB}/blob/main/models/lc-default-risk/model_card.md) ·
[Monitoring playbook]({GITHUB}/blob/main/docs/monitoring/README.md) ·
[API docs]({API_URL}/docs) · [Source]({GITHUB})
"""
    )
    for path, caption in (
        (REPO / "models/lc-default-risk/figures/approval_test.png",
         "Default rate among approved loans, by approval rate (2016)"),
        (REPO / "docs/monitoring/figures/alert_timeline.png",
         "When each monitoring alert would have become visible"),
    ):  # fmt: skip
        if path.exists():
            st.image(str(path), caption=caption)


def main() -> None:
    st.set_page_config(page_title="Default-risk demo", page_icon="📉", layout="wide")
    st.title("Lending Club default-risk model")
    try:
        with st.spinner("Connecting to the API. On the free tier the first call after a quiet "
                        "spell can take up to a minute..."):  # fmt: skip
            openapi, model = load_meta(API_URL, api())
    except (httpx.HTTPError, ApiError) as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        st.stop()

    st.caption(
        f"Live API: [{API_URL}]({API_URL}/docs) · model {model['name']} v{model['version']} · "
        f"[source on GitHub]({GITHUB})"
    )
    fields = application_fields(openapi)
    base = example_application(openapi)
    score, what_if, about = st.tabs(["Score an applicant", "What-if", "About the model"])
    with score:
        application = score_tab(fields, base, model["test"]["default_rate"])
    with what_if:
        what_if_tab(application)
    with about:
        about_tab(model)


main()
