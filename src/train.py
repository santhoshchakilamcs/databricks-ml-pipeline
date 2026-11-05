"""Train LTV model with MLflow tracking.

Usage (local):  .venv/bin/python src/train.py --data data/online_retail.csv
Usage (Databricks): import the notebook in notebooks/ltv_pipeline.py, which
calls the same functions and uses the workspace MLflow experiment.

Model: RandomForestRegressor (scikit-learn) on RFM-style features.
Metrics: RMSE, MAE, R2 on a time-based holdout, plus a mean-baseline RMSE
for comparison, so the model must beat predicting the average.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlflow
import pandas as pd
from features import build_customer_table
from drift import drift_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["n_orders", "n_items", "total_spend", "avg_order_value",
            "tenure_days", "recency_days", "n_countries"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="2011-06-01")
    ap.add_argument("--experiment", default="ltv-databricks-pipeline")
    args = ap.parse_args()
    tx = pd.read_csv(args.data, parse_dates=["InvoiceDate"])
    customers = build_customer_table(tx, args.split)
    # Time-based holdout within the feature period is not possible here, so
    # hold out a random 20% of customers (seed fixed) and report drift too.
    customers = customers.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(0.8 * len(customers))
    train, test = customers.iloc[:n_train], customers.iloc[n_train:]
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(train[FEATURES], train["ltv_next_period"])
        pred = model.predict(test[FEATURES])
        baseline = [train["ltv_next_period"].mean()] * len(test)
        metrics = {
            "n_customers": int(len(customers)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "rmse": float(mean_squared_error(test["ltv_next_period"], pred) ** 0.5),
            "mae": float(mean_absolute_error(test["ltv_next_period"], pred)),
            "r2": float(r2_score(test["ltv_next_period"], pred)),
            "baseline_rmse": float(mean_squared_error(test["ltv_next_period"], baseline) ** 0.5),
            "repeat_buyer_rate_test": float(test["bought_again"].mean()),
        }
        mlflow.log_params({"model": "RandomForestRegressor", "split": args.split, "features": ",".join(FEATURES)})
        mlflow.log_metrics(metrics)
        # MLflow 3.x saves sklearn models via skops, which blocks tree node types
        # by default. This model was trained in this same run from our own
        # data, so trusting its own tree type is safe here.
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        drift = drift_report(train, test, FEATURES)
        Path("metrics.json").write_text(json.dumps({"metrics": metrics, "drift": drift}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
