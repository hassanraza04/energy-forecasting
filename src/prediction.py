from __future__ import annotations

import numpy as np


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
