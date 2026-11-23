"""Loading, feature building and the time-ordered split."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "creditcard.parquet"
PCA_COLS = [f"V{i}" for i in range(1, 29)]


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not df["Time"].is_monotonic_increasing:
        df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """V1..V28 as given, log amount, and hour of day as a circle (the data covers two days)."""
    out = df[PCA_COLS].copy()
    out["log_amount"] = np.log1p(df["Amount"])
    hour = (df["Time"] % 86400) / 3600
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def time_split(df: pd.DataFrame, train: float = 0.6, val: float = 0.2):
    """First 60% of transactions by time to train, next 20% to validate, last 20% to test.

    A model in production only ever scores transactions that happen after the ones it learned from,
    so a random split (the usual Kaggle setup) measures something easier than the real job.
    """
    n = len(df)
    a, b = int(train * n), int((train + val) * n)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
