import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import build_customer_table


def test_no_lookahead_and_zero_target_kept():
    tx = pd.DataFrame({
        "Invoice": ["A1", "A1", "A2", "A3"],
        "Customer ID": [1.0, 1.0, 1.0, 2.0],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-07-01", "2011-02-01"]),
        "Quantity": [2, 1, 3, 1],
        "Price": [10.0, 5.0, 10.0, 20.0],
        "Country": ["UK", "UK", "UK", "UK"],
    })
    out = build_customer_table(tx, "2011-06-01")
    c1 = out[out["Customer ID"] == 1.0].iloc[0]
    assert c1["total_spend"] == 25.0  # only pre-split spend, not the 30.0 in July
    assert c1["ltv_next_period"] == 30.0
    c2 = out[out["Customer ID"] == 2.0].iloc[0]
    assert c2["ltv_next_period"] == 0.0  # kept, not dropped
    assert c2["bought_again"] == 0
