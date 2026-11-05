"""Simple, honest drift check: compare feature means between train and test.

Returns a dict of {feature: (train_mean, test_mean, relative_change)}.
A large relative change is a flag to investigate, not an automatic failure.
"""
import pandas as pd


def drift_report(train: pd.DataFrame, test: pd.DataFrame, features) -> dict:
    report = {}
    for f in features:
        tm, em = float(train[f].mean()), float(test[f].mean())
        rel = (em - tm) / abs(tm) if tm != 0 else 0.0
        report[f] = {"train_mean": tm, "test_mean": em, "relative_change": rel}
    return report
