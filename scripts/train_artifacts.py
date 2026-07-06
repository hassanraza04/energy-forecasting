"""Train model artifacts for the public app."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.artifacts import MODEL_BUNDLE_PATH
from src.data_loader import (
    DATA_PATH,
    LINEAR_MODELS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    _add_time_features,
    get_numeric_features,
)


def _load_training_data() -> pd.DataFrame:
    return _add_time_features(pd.read_csv(DATA_PATH, parse_dates=["date"]))


def _candidate_models() -> dict[str, list[tuple[Any, dict[str, Any]]]]:
    return {
        "Linear Regression": [
            (LinearRegression(), {"fit_intercept": True}),
        ],
        "Ridge Regression": [
            (Ridge(alpha=alpha), {"alpha": alpha})
            for alpha in [0.1, 1.0, 10.0, 25.0]
        ],
        "Lasso Regression": [
            (Lasso(alpha=alpha, max_iter=5000), {"alpha": alpha})
            for alpha in [0.001, 0.01, 0.05]
        ],
        "Random Forest": [
            (
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
            ),
            (
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=24,
                    min_samples_leaf=1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                {"n_estimators": 300, "max_depth": 24, "min_samples_leaf": 1},
            ),
            (
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
            ),
        ],
        "Gradient Boosting": [
            (
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
            ),
            (
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
                {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
            ),
            (
                GradientBoostingRegressor(
                    n_estimators=250,
                    learning_rate=0.08,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
                {"n_estimators": 250, "learning_rate": 0.08, "max_depth": 3},
            ),
        ],
    }


def _score_model(model: Any, x_test: Any, y_test: pd.Series) -> dict[str, Any]:
    preds = model.predict(x_test)
    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(mean_squared_error(y_test, preds) ** 0.5),
        "R2": float(r2_score(y_test, preds)),
        "preds": preds,
        "y_test": y_test.values,
    }


def train_bundle() -> dict[str, Any]:
    df = _load_training_data()
    feature_columns = get_numeric_features(df)
    x = df[feature_columns]
    y = df[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    results: dict[str, dict[str, Any]] = {}
    trained_candidates: dict[str, Any] = {}
    selected_params: dict[str, dict[str, Any]] = {}
    candidate_scores: list[dict[str, Any]] = []

    for model_name, candidates in _candidate_models().items():
        best_score: dict[str, Any] | None = None
        best_model = None
        best_params: dict[str, Any] = {}

        for model, params in candidates:
            is_linear = model_name in LINEAR_MODELS
            x_train_input = x_train_scaled if is_linear else x_train.values
            x_test_input = x_test_scaled if is_linear else x_test.values
            model.fit(x_train_input, y_train)
            score = _score_model(model, x_test_input, y_test)
            candidate_scores.append({
                "Model": model_name,
                "Params": params,
                "MAE": score["MAE"],
                "RMSE": score["RMSE"],
                "R2": score["R2"],
            })
            if best_score is None or score["R2"] > best_score["R2"]:
                best_score = score
                best_model = model
                best_params = params

        if best_score is None or best_model is None:
            raise RuntimeError(f"No model trained for {model_name}")
        results[model_name] = best_score
        trained_candidates[model_name] = best_model
        selected_params[model_name] = best_params

    best_model_name = max(results, key=lambda name: results[name]["R2"])

    return {
        "results": results,
        "trained": {best_model_name: trained_candidates[best_model_name]},
        "scaler": scaler,
        "feat_cols": feature_columns,
        "feature_means": {
            column: float(x[column].mean())
            for column in feature_columns
        },
        "target_average": float(y.mean()),
        "target_q99": float(y.quantile(0.99)),
        "X_train": x_train.head(500),
        "X_test": x_test.head(500),
        "X_train_s": x_train_scaled[:500],
        "X_test_s": x_test_scaled[:500],
        "y_train": y_train.head(500),
        "y_test": y_test.head(500),
        "best_model": best_model_name,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset_rows": int(df.shape[0]),
            "dataset_columns": int(df.shape[1]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "selected_params": selected_params,
            "candidate_scores": candidate_scores,
            "best_model": best_model_name,
        },
    }


def main() -> None:
    bundle = train_bundle()
    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_BUNDLE_PATH, compress=3)
    best = bundle["best_model"]
    best_score = bundle["results"][best]
    print(f"Saved {MODEL_BUNDLE_PATH}")
    print(f"Best model: {best}")
    print(f"R2: {best_score['R2']:.4f}")
    print(f"MAE: {best_score['MAE']:.2f} Wh")


if __name__ == "__main__":
    main()
