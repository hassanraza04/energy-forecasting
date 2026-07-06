import pandas as pd

from src.modeling import build_leaderboard, get_best_model


def test_build_leaderboard_sorts_by_r2_descending():
    results = {
        "Small Model": {"MAE": 12.0, "RMSE": 20.0, "R2": 0.35},
        "Better Model": {"MAE": 8.0, "RMSE": 12.0, "R2": 0.72},
    }

    leaderboard = build_leaderboard(results)

    assert leaderboard["Model"].tolist() == ["Better Model", "Small Model"]
    assert leaderboard.loc[0, "MAE"] == 8.0
    assert leaderboard.loc[0, "RMSE"] == 12.0
    assert leaderboard.loc[0, "R2"] == 0.72


def test_get_best_model_returns_first_ranked_row():
    leaderboard = pd.DataFrame(
        [
            {"Model": "Better Model", "MAE": 8.0, "RMSE": 12.0, "R2": 0.72},
            {"Model": "Small Model", "MAE": 12.0, "RMSE": 20.0, "R2": 0.35},
        ]
    )

    best = get_best_model(leaderboard)

    assert best["Model"] == "Better Model"
