"""Constants, data helpers, and cached model training."""
from __future__ import annotations

import os
from typing import Optional, List

import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "energydata_complete.csv")

TARGET_COL   = "Appliances"
DROP_COLS    = ["date", "rv1", "rv2"]

TEST_SIZE    = 0.2
RANDOM_STATE = 42

ALL_MODELS    = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Random Forest",
    "Gradient Boosting",
]
LINEAR_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}
TREE_MODELS   = {"Random Forest", "Gradient Boosting"}


def _load_raw() -> Optional[pd.DataFrame]:
    """Read CSV; returns None if file is missing."""
    try:
        return pd.read_csv(DATA_PATH, parse_dates=["date"])
    except FileNotFoundError:
        return None


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]        = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["part_of_day"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 20, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"],
    ).astype(str)
    return df


def get_numeric_features(df: pd.DataFrame) -> List[str]:
    """Return model-ready numeric feature columns (excludes target + drop cols)."""
    exclude = set(DROP_COLS) | {TARGET_COL}
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def get_data() -> pd.DataFrame:
    df = _load_raw()
    if df is None:
        raise FileNotFoundError("Dataset not found. Place `energydata_complete.csv` next to `app.py`.")
    return _add_time_features(df)
