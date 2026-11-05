#!/bin/bash
# Local CI mirror of .github/workflows/ci.yml: tests + train + metric gate.
set -e
cd "$(dirname "$0")/.."
.venv/bin/python -m pytest tests/ -q
.venv/bin/python src/train.py --data data/online_retail.csv
.venv/bin/python -c "import json; m=json.load(open('metrics.json'))['metrics']; assert m['rmse'] < m['baseline_rmse']; print('PASS: model beats baseline', m)"
