import numpy as np

from src.prediction import (
    build_prediction_vector,
    clip_energy_prediction,
    describe_prediction,
)


def test_clip_energy_prediction_never_returns_negative_values():
    assert clip_energy_prediction(-4.5) == 0.0
    assert clip_energy_prediction(12.25) == 12.25


def test_build_prediction_vector_uses_form_values_then_feature_means():
    vector = build_prediction_vector(
        ["lights", "T2", "RH_2"],
        {"lights": 4.0},
        {"T2": 21.5, "RH_2": 42.0},
    )

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (1, 3)
    assert vector.tolist() == [[4.0, 21.5, 42.0]]


def test_describe_prediction_compares_result_to_dataset_average():
    low = describe_prediction(60.0, 100.0)
    normal = describe_prediction(110.0, 100.0)
    high = describe_prediction(140.0, 100.0)

    assert low["level"] == "Below average"
    assert normal["level"] == "Near average"
    assert high["level"] == "Above average"
    assert "lower than the dataset average" in low["message"]
