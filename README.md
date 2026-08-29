# Lending Club Loan Default Prediction

 We try to predict if a borrower will default on their Lending Club loan, using loan data from 2014-2017.

Dataset: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1

## Some choices we made

- Only using 2014-2017 data (older data is too sparse, newer loans havent matured)
- Target is `charged_off` (1 = defaulted, 0 = fully paid)
- Time-based split so we dont accidentally use future info during training
- Theres a ~4:1 class imbalance (way more paid loans than defaults)
- Final model is XGBoost picked by Optuna in `13_model_selection.py`

## Project Structure

```
Eas587_project/
├── data/
│   ├── raw/                          # archive.zip from Kaggle goes here
│   └── processed/                    # pipeline writes intermediate csvs here
├── models/
│   ├── best_model.pkl                # final XGBoost
│   └── model_results.csv             # Optuna run summary
├── notebooks/databricks/             # Phase 3 - Spark + Delta + macro notebooks
├── presentation/
│   ├── presentation_slides.pdf
│   ├── data/                         # precomputed insight parquets used by the app
│   └── figures/                      # charts the app embeds
├── report/                           # Phase 4 IEEE LaTeX report
│   ├── Phase4Report.tex
│   ├── references.bib
│   └── figures/
├── src/
│   ├── 01_data_loading.py
│   ├── 02_data_inspection.py
│   ├── 03_data_cleaning.py
│   ├── 04_data_splitting.py
│   ├── 05_data_eda.py
│   ├── 06_data_processing_pipeline.py
│   ├── run_pipeline.py
│   ├── app/streamlit_app.py          # Phase 4 Streamlit app
│   ├── mcp/server.py                 # MCP server for Claude Desktop
│   └── models/
│       ├── 07_model_knn.py
│       ├── 08_model_svm.py
│       ├── 09_model_lr.py
│       ├── 10_model_dt.py
│       ├── 11_model_xgb.py
│       ├── 12_model_hgb.py
│       ├── 13_model_selection.py
│       └── 14_final_evaluation.py
├── mcp_config_template.json
├── requirements.txt
└── README.md
```

## Setup

Need Python 3.10+ and these packages:

```bash
pip install -r requirements.txt
```

Then download `archive.zip` from the Kaggle link above and put it in `data/raw/`.

## Running the Pipeline

Run scripts in order:
### we have a `run_pipeline.py` that runs everything in order, you can use:
```bash
python src/run_pipeline.py
```
It skips re-training if `models/best_model.pkl` already exists. Delete that file if you want to force re-tuning.

## if you want to run the scripts manually, run:
```bash
python src/01_data_loading.py      # loads raw data, filters to 2014-2017
python src/02_data_inspection.py   # prints a report about the raw data (optional)
python src/03_data_cleaning.py     # cleans data, drops bad columns, fixes types
python src/04_data_splitting.py    # splits into train/val/test by date
python src/05_data_eda.py          # generates HTML EDA report
python src/06_data_processing_pipeline.py  # builds ML pipeline, scales/encodes features

# (Optional) Individual model training algorithms
python src/models/07_model_knn.py
python src/models/08_model_svm.py
python src/models/09_model_lr.py
python src/models/10_model_dt.py
python src/models/11_model_xgb.py
python src/models/12_model_hgb.py

# Model Selection & Final Evaluation
python src/models/13_model_selection.py  # hyperparameter tuning, compares models, saves best
python src/models/14_final_evaluation.py # evaluates the best model on the hidden test set
```

Each script reads the output of the previous one so they have to be run in order.

## What Each Script Does

**01_data_loading.py** - Extracts the gzip from archive.zip, filters to 2014-2017, saves as csv

**02_data_inspection.py** - Looks at missing values, duplicates, class balance, outliers etc. Prints everything to console. Good to run before cleaning to understand whats going on.

**03_data_cleaning.py** - Drops columns with too many missing values (>30%), removes columns that would leak the outcome (like payment totals), drops useless columns (IDs, free text), fixes data types

**04_data_splitting.py** - Train = 2014-2016 first 80%, Val = 2014-2016 last 20%, Test = all of 2017. Uses chronological order to avoid data leakage.

**05_data_eda.py** - Runs on training set only. Makes an HTML report with histograms, correlation heatmaps, outlier analysis, etc.

**06_data_processing_pipeline.py** - Builds the Scikit-Learn preprocessing pipeline (imputation, scaling, encoding) and saves transformed features for model training.

**models/07_model_knn.py** through **models/12_model_hgb.py** - Individual scripts exploring baseline models and various algorithms.

**models/13_model_selection.py** - Runs multiple models with Optuna hyperparameter tuning, validates them against the validation set, logs results to MLflow, and saves the single best model.

**models/14_final_evaluation.py** - Loads the best saved model and performs a final unbiased evaluation on the holdout 2017 test set, plotting the ROC curve and confusion matrix.

---

## Phase 3 - Databricks, Spark, Delta Lake

Phase 3 rebuilds the pipeline on Databricks with bronze / silver / gold tables. The five notebooks are in `notebooks/databricks/`:

| Notebook | Stage | What it does |
|---|---|---|
| `01_bronze_layer.ipynb` | Bronze | Loads raw csv into Delta table `bronze_loans` |
| `02_silver_layer.ipynb` | Silver | Cleans and types into `silver_loans` |
| `03_gold_layer.ipynb` | Gold | Time-split + stratified sample into `gold_loans_train/val/test` |
| `04_mllib_models.ipynb` | Models | LogisticRegression (baseline + 3-fold CV), XGBoost, HistGradientBoosting on the Gold tables, compared back to Phase 2 |
| `05_macro_integration.ipynb` | Macro | Adds FRED unemployment, 3 insights, combined-features model |

To run them: make a free account at https://community.cloud.databricks.com, run `python src/01_data_loading.py` locally first so you have `data/processed/optimized_data_14_17.csv`, then in Databricks go **Catalog -> Create -> Create Table -> Upload file** and drop the csv in (it registers as a Delta table at `workspace.default.optimized_data_14_17`). Then **Workspace -> Users -> Import** the five `.ipynb` files and run them in order 01 -> 05.

A few notes:
- Notebooks 01-03 are pure Spark + Delta. 04 and 05 fall back to pandas + sklearn for the model fit because Databricks Free Edition blocks `Imputer`, `StringIndexer`, `OneHotEncoder` through its Py4J whitelist, and Free Edition doesn't expose classic compute as an alternative. The scalable data work still happens in Spark.
- Test set is kept at the full 2017 size so the final metric is comparable to Phase 2. Train and val are sampled to 200k / 50k.
- FRED macro data in notebook 05 is fetched directly from https://fred.stlouisfed.org, no API key.

---

## Phase 4 - Streamlit app and MCP server

Phase 4 wraps the trained model two ways.

The Streamlit app drove the live demo. To run it:

```bash
streamlit run src/app/streamlit_app.py
```

9 tabs walk through the project. The Predict tab loads `models/best_model.pkl` and scores loans in real time.

The MCP server exposes the same model to Claude Desktop so you can ask in plain English ("score a $15k 36-month loan, B3 grade, FICO 710, $65k income"). To run it:

```bash
python src/mcp/server.py
```

You point Claude Desktop at it through `claude_desktop_config.json`. There's a template entry at `mcp_config_template.json` you can copy in.
