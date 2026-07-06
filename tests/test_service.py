from src.service import build_app_config, predict_energy


class FakeModel:
    def predict(self, rows):
        return [rows[0][0] * 2 + rows[0][1]]


def _bundle():
    return {
        "best_model": "Random Forest",
        "trained": {"Random Forest": FakeModel()},
        "feat_cols": ["lights", "T2"],
        "feature_means": {"lights": 3.0, "T2": 20.0},
        "target_average": 100.0,
        "target_q99": 420.0,
        "results": {"Random Forest": {"R2": 0.56, "MAE": 30.8, "RMSE": 55.1}},
        "metadata": {
            "selected_params": {
                "Random Forest": {
                    "n_estimators": 300,
                    "max_depth": 24,
                    "min_samples_leaf": 1,
                }
            },
            "dataset_rows": 19735,
        },
    }


def test_predict_energy_uses_payload_values_and_saved_means():
    result = predict_energy(_bundle(), {"lights": 10.0})

    assert result["model"] == "Random Forest"
    assert result["prediction"] == 40.0
    assert result["unit"] == "Wh"
    assert result["description"]["level"] == "Below average"


def test_build_app_config_exposes_product_ready_metadata():
    config = build_app_config(_bundle())

    assert config["appName"] == "Home Energy Estimator"
    assert config["model"]["name"] == "Random Forest"
    assert config["model"]["params"]["n_estimators"] == 300
    assert config["inputs"][0]["key"] == "lights"
    assert config["presets"][0]["name"] == "Typical afternoon"
