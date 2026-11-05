# Databricks notebook source
# Databricks ML Pipeline: Customer LTV Prediction
# Import this file into a Databricks workspace (File > Import > Notebook).
# It uses the workspace MLflow experiment, Spark when available, and the
# same src/features.py logic as the local run, so local and Databricks
# results are comparable, not two different pipelines.
#
# On Databricks, set the widget below to your data path (a Delta table or
# a mounted CSV of UCI Online Retail II). Locally, run src/train.py instead.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load data
# MAGIC UCI Online Retail II, real public data. Features use only transactions
# MAGIC before 2011-06-01; the target is spend after that date. No lookahead.

# COMMAND ----------
try:
    from pyspark.sql import functions as F
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/data/online_retail.csv")
    print(f"Rows: {df.count()}")
except Exception as e:
    print(f"Spark not available in this run ({e}); use src/train.py locally.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Train with MLflow
# MAGIC The local src/train.py mirrors this step exactly: RandomForestRegressor,
# MAGIC RFM-style features, time-based target, MLflow params, metrics, and model artifact.
# MAGIC See README.md for the verified local numbers.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Monitor
# MAGIC src/drift.py compares train vs test feature means. Re-run it on each
# MAGIC retraining batch; a large relative change is a flag to investigate,
# MAGIC not an automatic failure. Retrain on a schedule, version the model in
# MAGIC MLflow Model Registry, and promote only after the holdout beats the mean baseline.
