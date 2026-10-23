# Data-discovery scripts

These scripts replace the exploratory parts of the original `src/01-05` workflow. They
stop before cleaning, preprocessing, train/test materialization, or model training.

## Run everything

```bash
python scripts/run_data_discovery.py
```

When the Kaggle files already exist in `data/raw/`:

```bash
python scripts/run_data_discovery.py --skip-download
```

## Stages

1. `01_download_data.py`: downloads the immutable Kaggle snapshot with `kagglehub`, or
   verifies the existing files and writes their hashes.
2. `02_validate_raw_data.py`: performs chunked schema, completeness, dictionary, time
   coverage, ID, and range checks over all 2.9 million rows.
3. `03_build_column_catalog.py`: classifies every field by source and availability, then
   defines provisional Model A and Model B feature policies.
4. `04_analyze_target_cohorts.py`: measures status maturity by issue vintage and term and
   proposes an observation-safe development cohort.
5. `05_run_eda.py`: calculates full-cohort rates and missingness, uses a bounded sample for
   distributions and correlations, and writes plots and findings.
6. `06_compare_original_approach.py`: summarizes data-backed differences from the original
   `src/01-05` decisions.

Outputs are written under `reports/data_discovery/`. Raw files are never modified.

The proposed experiment uses 36-month loans: 2012-2014 for training, 2015 for
validation, and 2016 as an out-of-time test. Feature-target EDA uses only 2012-2015;
2016 outcomes appear only in the pre-freeze maturity and temporal-drift audit.
