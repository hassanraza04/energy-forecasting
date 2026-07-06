from __future__ import annotations

from typing import Any

import pandas as pd


def build_leaderboard(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Model": name,
            "MAE": float(values["MAE"]),
            "RMSE": float(values["RMSE"]),
            "R2": float(values["R2"]),
        }
        for name, values in results.items()
    ]
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


def get_best_model(leaderboard: pd.DataFrame) -> pd.Series:
    if leaderboard.empty:
        raise ValueError("Leaderboard is empty.")
    return leaderboard.iloc[0]
