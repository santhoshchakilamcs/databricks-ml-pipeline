# Databricks ML Pipeline: Customer LTV Prediction

A Databricks-ready ML workflow for customer lifetime value (LTV), built to
cover the Databricks production ML gap for the NerdWallet Senior Data
Scientist (Applied ML) role. Personal project, not job experience, not a
production deployment at NerdWallet or any other employer.

## What it does

- **Time-based target, no lookahead.** Features use only transactions before
  2011-06-01; the target is money spent after that date. Customers with no
  post-split spend are kept with target 0.0, so churn is part of the signal.
- **RFM-style features.** Orders, items, total spend, average order value,
  tenure days, recency days, countries.
- **Model.** RandomForestRegressor (scikit-learn, 200 trees, seed 42).
- **MLflow tracking.** Params, metrics, and model artifact are logged to an
  MLflow experiment (`ltv-databricks-pipeline` locally; the workspace
  experiment on Databricks).
- **Drift check.** `src/drift.py` compares train vs test feature means.
  A large relative change is a flag to investigate, not an automatic failure.
- **Databricks notebook.** `notebooks/ltv_pipeline.py` imports into a
  Databricks workspace and mirrors the local pipeline, so local and
  Databricks results are comparable.

## Verified numbers (Nov 4, 2026, local run)

- Data: UCI Online Retail II, real public data, 406,829 rows after dropping
  missing Customer IDs. 2,718 customers had pre-split history and were used
  (customers first appearing after the split have no pre-split features).
- Train / test: 2,174 / 544 customers (random 80/20 split, seed 42, after the
  time-based target was built).
- **Test RMSE 7,571.59, MAE 1,118.88, R2 0.364**
- **Mean baseline RMSE 9,494.84** — the model beats predicting the average
  by about 20%. RMSE is much larger than MAE because LTV is heavy-tailed:
  a few very large spenders dominate squared error.
- Test repeat-buyer rate: 69.5%.
- Unit test: `tests/test_features.py` passes — pre-split spend only, and
  zero-target customers are kept, not dropped.

## Honest limits

- **Not yet run on a Databricks workspace.** The notebook is written for
  Databricks (Spark load, workspace MLflow), but no Databricks workspace
  was available in this environment, so the verified run is local with
  pandas + MLflow. Do not claim a production Databricks deployment.
- **Not a real-time inference system.** This is batch LTV prediction with
  an MLflow model artifact. Real-time serving, latency, and reliability
  trade-offs are design notes in the notebook, not measured here.
- **Not NerdWallet data.** UCI retail data is a stand-in for consumer
  purchase behaviour, not financial product LTV.
- **XGBoost / LightGBM not used.** The JD lists them as examples; this run
  uses scikit-learn RandomForest. Swapping the estimator is a one-line
  change in `src/train.py`.

## Run it locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/download_data.py   # downloads UCI data, not committed
.venv/bin/python src/train.py --data data/online_retail.csv
.venv/bin/python -m pytest tests/ -q
```

Data files (`data/`, `mlruns/`, `mlflow.db`) are gitignored; the app
re-downloads public data on first run.
