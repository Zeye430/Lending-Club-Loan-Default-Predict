import json

import numpy as np
import pandas as pd
import pytest

from lc_risk.data.cohort import RAW_COLUMNS, build_cohort, write_cohort

BASE_ROW = {
    "term": " 36 months",
    "application_type": "Individual",
    "loan_status": "Fully Paid",
    "loan_amnt": 10000,
    "purpose": "debt_consolidation",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "annual_inc": 60000,
    "verification_status": "Verified",
    "addr_state": "CA",
    "dti": 15.0,
    "fico_range_low": 700,
    "earliest_cr_line": "Jan-2005",
    "revol_util": "30.5%",
    "sub_grade": "B3",
    "int_rate": "10.5%",
    "installment": 325.0,
    "funded_amnt": 10000,
    "total_pymnt": 11700.0,
    "recoveries": 0.0,
    "last_pymnt_d": "Jan-2016",
}

# (overrides, expected outcome) for each synthetic raw row
ROWS = [
    ({"issue_d": "Mar-2013"}, "kept"),
    (
        {"issue_d": "Jun-2015", "loan_status": "Charged Off", "dti": -1.0,
         "home_ownership": "ANY", "purpose": "educational"},
        "kept",
    ),
    ({"issue_d": "Feb-2016", "loan_status": "Default", "revol_util": "250%"}, "kept"),
    ({"issue_d": "Mar-2013", "term": " 60 months"}, "term_not_36_months"),
    ({"issue_d": "Mar-2012"}, "issue_year_out_of_scope"),
    ({"issue_d": "Mar-2014", "application_type": "Joint App"}, "joint_application"),
    ({"issue_d": "Mar-2016", "loan_status": "Current"}, "unresolved_status"),
]  # fmt: skip


def write_raw(path, rows):
    records = []
    for i, overrides in enumerate(rows):
        record = dict.fromkeys(RAW_COLUMNS, 1)
        record.update(BASE_ROW, id=1000 + i, **overrides)
        records.append(record)
    pd.DataFrame(records, columns=RAW_COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def raw_csv(tmp_path):
    return write_raw(tmp_path / "raw.csv", [overrides for overrides, _ in ROWS])


def test_filters_and_counts_exclusions(raw_csv):
    cohort, summary = build_cohort(raw_csv, chunk_size=3)

    assert len(cohort) == 3
    assert summary["exclusions"] == {reason: 1 for _, reason in ROWS if reason != "kept"}
    assert cohort["split"].tolist() == ["train", "validation", "test"]
    assert cohort["default"].tolist() == [0, 1, 1]


def test_parses_and_cleans_values(raw_csv):
    cohort, summary = build_cohort(raw_csv)
    first, second, third = (cohort.iloc[i] for i in range(3))

    assert first["emp_length"] == 5.0
    assert first["revol_util"] == 30.5
    assert first["earliest_cr_line"] == pd.Timestamp("2005-01-01")
    assert np.isnan(second["dti"])
    assert second["home_ownership"] == "OTHER"
    assert second["purpose"] == "other"
    assert np.isnan(third["revol_util"])
    assert summary["cleaning_rules_applied"]["dti_negative"] == 1
    assert summary["cleaning_rules_applied"]["revol_util_above_200"] == 1


def test_rejects_values_outside_spec(tmp_path):
    raw = write_raw(tmp_path / "raw.csv", [{"issue_d": "Mar-2013", "fico_range_low": 900}])
    with pytest.raises(ValueError, match="fico_range_low"):
        build_cohort(raw)


def test_write_cohort_round_trips_with_manifest(raw_csv, tmp_path):
    cohort, summary = build_cohort(raw_csv)
    out = tmp_path / "cohort.parquet"
    manifest = write_cohort(cohort, summary, raw_csv, out)

    reloaded = pd.read_parquet(out)
    pd.testing.assert_frame_equal(reloaded, cohort)
    on_disk = json.loads((tmp_path / "cohort_manifest.json").read_text())
    assert on_disk["cohort_file"]["sha256"] == manifest["cohort_file"]["sha256"]
    assert on_disk["splits"]["validation"]["defaults"] == 1
