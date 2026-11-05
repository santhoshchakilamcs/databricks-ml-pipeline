"""Live serving app for the Databricks ML Pipeline LTV model.

Production-style, not production-hosted yet: this app runs the trained
model behind a FastAPI service with health, metrics, and drift endpoints,
and a small HTML dashboard. On first start it downloads UCI Online Retail
II, builds the customer table, and trains the model (same code as
src/train.py), so no binary model file is committed to git.

Endpoints:
  GET  /          service status + verified metrics
  GET  /health     liveness, model loaded or not
  GET  /metrics    test metrics + drift report from metrics.json
  POST /predict    {n_orders, n_items, total_spend, avg_order_value,
                    tenure_days, recency_days, n_countries} -> predicted LTV
  GET  /dashboard  simple HTML form to try a prediction
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Databricks ML Pipeline - LTV Serving")
MODEL = None
FEATURES = ["n_orders", "n_items", "total_spend", "avg_order_value",
            "tenure_days", "recency_days", "n_countries"]


class PredictRequest(BaseModel):
    n_orders: float
    n_items: float
    total_spend: float
    avg_order_value: float
    tenure_days: float
    recency_days: float
    n_countries: float = 1.0


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    from download_data import main as download_main
    from features import build_customer_table
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    download_main()
    tx = pd.read_csv("data/online_retail.csv", parse_dates=["InvoiceDate"])
    customers = build_customer_table(tx, "2011-06-01")
    MODEL = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    MODEL.fit(customers[FEATURES], customers["ltv_next_period"])
    return MODEL


@app.on_event("startup")
def startup():
    load_model()


@app.get("/")
def root():
    metrics = {}
    p = Path("metrics.json")
    if p.exists():
        metrics = json.loads(p.read_text())["metrics"]
    return {"service": "ltv-serving", "model": "RandomForestRegressor (200 trees)",
            "model_loaded": MODEL is not None, "verified_metrics": metrics}


@app.get("/health")
def health():
    return {"status": "ok" if MODEL is not None else "loading", "model_loaded": MODEL is not None}


@app.get("/metrics")
def metrics():
    p = Path("metrics.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"error": "metrics.json not found; run src/train.py first"}


@app.post("/predict")
def predict(req: PredictRequest):
    model = load_model()
    row = [[getattr(req, f) for f in FEATURES]]
    return {"predicted_ltv_next_period": float(model.predict(row)[0])}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """<html><head><title>LTV Predictor</title></head><body>
<h1>Customer LTV Predictor</h1>
<p>Databricks ML Pipeline model, served live. Enter pre-split customer features to predict spend in the next period.</p>
<form id="f">
<input name="n_orders" placeholder="n_orders" value="3">
<input name="n_items" placeholder="n_items" value="100">
<input name="total_spend" placeholder="total_spend" value="1000">
<input name="avg_order_value" placeholder="avg_order_value" value="50">
<input name="tenure_days" placeholder="tenure_days" value="60">
<input name="recency_days" placeholder="recency_days" value="30">
<input name="n_countries" placeholder="n_countries" value="1">
<button type="submit">Predict LTV</button></form>
<pre id="out"></pre>
<script>document.getElementById('f').onsubmit=async(e)=>{e.preventDefault();
const d={}; new FormData(e.target).forEach((v,k)=>d[k]=parseFloat(v));
const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);}</script>
</body></html>"""
