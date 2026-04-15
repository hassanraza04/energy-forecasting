"""
src/data_loader.py
Constants, data helpers, and cached model-training for the Smart Energy app.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
# src/ is one level below the project root
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "energydata_complete.csv")

# ── Column config ─────────────────────────────────────────────────────────────
TARGET_COL   = "Appliances"
DROP_COLS    = ["date", "rv1", "rv2"]

# ── Model config ──────────────────────────────────────────────────────────────
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


# ── Raw helpers (not cached — used by cached wrappers) ────────────────────────

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


# ── Cached Streamlit resources ────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def get_data() -> pd.DataFrame:
    df = _load_raw()
    if df is None:
        st.error("Dataset not found. Place `energydata_complete.csv` next to `app.py`.")
        st.stop()
    return _add_time_features(df)


@st.cache_resource(show_spinner="Training all models — first load only…")
def train_all_models(_cache_key: str) -> Dict[str, Any]:
    """
    Train all 5 models and cache the full training bundle.
    _cache_key (DATA_PATH) prevents re-hashing large DataFrames.
    """
    df        = get_data()
    feat_cols = get_numeric_features(df)
    X = df[feat_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    definitions = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=1.0, max_iter=5000),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=100,
                                 random_state=RANDOM_STATE,
                                 n_jobs=-1,
                             ),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=100,
                                 random_state=RANDOM_STATE,
                             ),
    }

    results: Dict[str, Any] = {}
    trained: Dict[str, Any] = {}

    for name, model in definitions.items():
        is_linear = name in LINEAR_MODELS
        Xtr = X_train_s if is_linear else X_train.values
        Xte = X_test_s  if is_linear else X_test.values
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        results[name] = {
            "MAE":    float(mean_absolute_error(y_test, preds)),
            "RMSE":   float(mean_squared_error(y_test, preds) ** 0.5),
            "R2":     float(r2_score(y_test, preds)),
            "preds":  preds,
            "y_test": y_test.values,
        }
        trained[name] = model

    return {
        "results":    results,
        "trained":    trained,
        "X_train":    X_train,
        "X_test":     X_test,
        "X_train_s":  X_train_s,
        "X_test_s":   X_test_s,
        "y_train":    y_train,
        "y_test":     y_test,
        "scaler":     scaler,
        "feat_cols":  feat_cols,
    }
