# Data discovery

This folder holds the analysis that decided **which loans, labels, and columns the model may
use**. The modeling code in `src/lc_risk/` does not import anything from here. The decisions
were made once, from this evidence, and written into `src/lc_risk/config.py` (cohort and
splits) and `src/lc_risk/schema.py` (inputs).

## Reports

| Report | What it answers |
|---|---|
| [`reports/data_audit.md`](reports/data_audit.md) | Is the raw export usable? Which loans have mature labels? Which columns are known at application time? When did LC start populating each field? |
| [`eda.ipynb`](eda.ipynb) | How do default rates, LC's grade, and the inputs behave in the 2013–2015 development years, and what does that mean for modeling? |

Key findings:

1. **Recent vintages are right-censored.** In the 2020Q3 snapshot, 36-month loans are fully
   resolved through 2016, but 60-month loans are only 72% resolved for 2016 and 2017 loans
   are 77% resolved. → Use 36-month loans issued 2013–2016.
2. **LC's export schema changed over time.** About 30 bureau fields were only half-populated
   in 2012, and 14 more first appear in late 2015. → Start the cohort in 2013 and exclude the
   late fields.
3. **LC's grade is the strongest single signal (AUC ≈ 0.67), but its meaning drifts.**
   Default rates rose within every grade from 2013 to 2015. → Compare Model A (no LC rating)
   with Model B (+ `sub_grade`), and calibrate on the most recent labeled year.
4. **Missing values carry signal.** No delinquency on file means lower risk; no stated
   employment length means higher risk. → Keep nulls rather than imputing medians.
5. **Inputs are stable even though outcomes shift.** Train-to-validation PSI is below 0.06
   for every input. → Monitoring must track realized default rates, not only input drift.

## Running it

```bash
# raw-data audit: download, validate, classify columns, label maturity, write the report
uv run python discovery/run_data_discovery.py            # add --skip-download if data/raw exists

# EDA: needs the cohort built from the raw data
uv run python -m lc_risk.data.cohort
uv run jupyter nbconvert --to notebook --execute --inplace discovery/eda.ipynb
```

| Stage | Output |
|---|---|
| `01_download_data.py` | `data/raw/` Kaggle snapshot, plus a source manifest with SHA-256 hashes |
| `02_validate_raw_data.py` | Column profile, data-quality checks, non-null rate by year |
| `03_build_column_catalog.py` | Every column classified by source and availability time |
| `04_analyze_target_cohorts.py` | Label maturity by issue year and term |
| `05_write_audit_report.py` | `reports/data_audit.md` and its figures (committed) |

Stages 01–04 write machine-readable CSV/JSON under `reports/data_discovery/` at the repo root
(git-ignored). Only the human-readable reports in this folder are committed.
