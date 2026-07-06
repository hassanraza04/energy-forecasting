from __future__ import annotations

from typing import TypedDict

import numpy as np


class PredictionDescription(TypedDict):
    level: str
    message: str


def clip_energy_prediction(value: float) -> float:
    return max(0.0, float(value))


def build_prediction_vector(
    feature_columns: list[str],
    form_values: dict[str, float],
    feature_means: dict[str, float],
) -> np.ndarray:
    values = [
        float(form_values.get(feature, feature_means.get(feature, 0.0)))
        for feature in feature_columns
    ]
    return np.array(values, dtype=float).reshape(1, -1)


def describe_prediction(prediction: float, average_energy: float) -> PredictionDescription:
    if prediction < average_energy * 0.75:
        return {
            "level": "Below average",
            "message": (
                "This estimate is lower than the dataset average for appliance energy. "
                "In the source data, similar conditions sit in the lower consumption range."
            ),
        }
    if prediction < average_energy * 1.25:
        return {
            "level": "Near average",
            "message": (
                "This estimate is close to the dataset average. The selected conditions "
                "look typical compared with the source data."
            ),
        }
    return {
        "level": "Above average",
        "message": (
            "This estimate is higher than the dataset average. Similar conditions appear "
            "in the higher consumption range of the source data."
        ),
    }
