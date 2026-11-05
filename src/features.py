"""Feature engineering for customer LTV prediction.

Time-based split, no lookahead: features are computed only from transactions
BEFORE the split date; the target is money spent AFTER the split date.
Customers with no post-split spend get target 0.0 (they are kept, not dropped,
so churn is part of the LTV signal, not hidden by filtering).
"""
import pandas as pd


def build_customer_table(transactions: pd.DataFrame, split_date) -> pd.DataFrame:
    tx = transactions.copy()
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"])
    tx = tx[tx["Customer ID"].notna()]
    tx["amount"] = tx["Quantity"] * tx["Price"]
    tx = tx[tx["amount"] > 0]  # returns/cancellations (negative) excluded from spend
    before = tx[tx["InvoiceDate"] < split_date]
    after = tx[tx["InvoiceDate"] >= split_date]
    feats = before.groupby("Customer ID").agg(
        n_orders=("Invoice", "nunique"),
        n_items=("Quantity", "sum"),
        total_spend=("amount", "sum"),
        avg_order_value=("amount", "mean"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        n_countries=("Country", "nunique"),
    ).reset_index()
    feats["tenure_days"] = (feats["last_purchase"] - feats["first_purchase"]).dt.days
    feats["recency_days"] = (pd.Timestamp(split_date) - feats["last_purchase"]).dt.days
    target = after.groupby("Customer ID")["amount"].sum().rename("ltv_next_period")
    out = feats.merge(target, left_on="Customer ID", right_index=True, how="left")
    out["ltv_next_period"] = out["ltv_next_period"].fillna(0.0)
    out["bought_again"] = (out["ltv_next_period"] > 0).astype(int)
    return out
