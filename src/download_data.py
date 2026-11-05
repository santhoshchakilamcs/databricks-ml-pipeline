"""Download UCI Online Retail II and flatten it to data/online_retail.csv.

Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
The raw file is NOT committed to git (large); this script re-downloads it on
first run, the same pattern used by the E-commerce Recommender project.
"""
import sys
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"


def main():
    out = Path("data/online_retail.csv")
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        print(f"Already present: {out}")
        return
    xlsx = Path("/tmp/online_retail.xlsx")
    if not xlsx.exists():
        print(f"Downloading {URL} ...")
        import urllib.request
        urllib.request.urlretrieve(URL, xlsx)
    df = pd.read_excel(xlsx)
    df = df[df["CustomerID"].notna()]
    df = df.rename(columns={"CustomerID": "Customer ID", "InvoiceNo": "Invoice", "UnitPrice": "Price"})
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} rows, {df['Customer ID'].nunique()} customers")


if __name__ == "__main__":
    main()
