"""Prediction service helpers for the custom web app."""
from __future__ import annotations

from typing import Any

from src.prediction import (
    build_prediction_vector,
    clip_energy_prediction,
    describe_prediction,
)


APP_NAME = "Home Energy Estimator"
APP_SUBTITLE = "Estimate appliance energy from room conditions and weather."


INPUTS = [
    {
        "key": "lights",
        "label": "Lighting use",
        "unit": "Wh",
        "min": 0,
        "max": 70,
        "step": 1,
        "default": 0,
    },
    {
        "key": "T2",
        "label": "Kitchen temperature",
        "unit": "C",
        "min": 14,
        "max": 26,
        "step": 0.5,
        "default": 20,
    },
    {
        "key": "RH_2",
        "label": "Kitchen humidity",
        "unit": "%",
        "min": 20,
        "max": 60,
        "step": 1,
        "default": 40,
    },
    {
        "key": "T_out",
        "label": "Outdoor temperature",
        "unit": "C",
        "min": -5,
        "max": 28,
        "step": 0.5,
        "default": 6,
    },
    {
        "key": "RH_out",
        "label": "Outdoor humidity",
        "unit": "%",
        "min": 20,
        "max": 100,
        "step": 1,
        "default": 75,
    },
    {
        "key": "Windspeed",
        "label": "Wind speed",
        "unit": "m/s",
        "min": 0,
        "max": 14,
        "step": 0.5,
        "default": 4,
    },
    {
        "key": "Visibility",
        "label": "Visibility",
        "unit": "km",
        "min": 1,
        "max": 66,
        "step": 1,
        "default": 40,
    },
    {
        "key": "hour",
        "label": "Hour of day",
        "unit": "",
        "min": 0,
        "max": 23,
        "step": 1,
        "default": 12,
    },
    {
        "key": "is_weekend",
        "label": "Weekend",
        "unit": "0 or 1",
        "min": 0,
        "max": 1,
        "step": 1,
        "default": 0,
    },
]


PRESETS = [
    {
        "name": "Typical afternoon",
        "values": {
            "lights": 0,
            "T2": 20,
            "RH_2": 40,
            "T_out": 6,
            "RH_out": 75,
            "Windspeed": 4,
            "Visibility": 40,
            "hour": 14,
            "is_weekend": 0,
        },
    },
    {
        "name": "Evening load",
        "values": {
            "lights": 20,
            "T2": 22,
            "RH_2": 44,
            "T_out": 8,
            "RH_out": 78,
            "Windspeed": 3,
            "Visibility": 32,
            "hour": 19,
            "is_weekend": 0,
        },
    },
    {
        "name": "Weekend morning",
        "values": {
            "lights": 5,
            "T2": 19,
            "RH_2": 42,
            "T_out": 4,
            "RH_out": 82,
            "Windspeed": 5,
            "Visibility": 28,
            "hour": 9,
            "is_weekend": 1,
        },
    },
]


def _params_for_best_model(bundle: dict[str, Any]) -> dict[str, Any]:
    model_name = bundle["best_model"]
    return bundle.get("metadata", {}).get("selected_params", {}).get(model_name, {})


def build_app_config(bundle: dict[str, Any]) -> dict[str, Any]:
    model_name = bundle["best_model"]
    score = bundle["results"][model_name]
    return {
        "appName": APP_NAME,
        "subtitle": APP_SUBTITLE,
        "inputs": INPUTS,
        "presets": PRESETS,
        "model": {
            "name": model_name,
            "params": _params_for_best_model(bundle),
            "r2": round(float(score["R2"]), 4),
            "mae": round(float(score["MAE"]), 2),
            "rmse": round(float(score["RMSE"]), 2),
            "datasetRows": int(bundle.get("metadata", {}).get("dataset_rows", 0)),
        },
        "averageEnergy": round(float(bundle["target_average"]), 2),
    }


def predict_energy(bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    model_name = bundle["best_model"]
    model = bundle["trained"][model_name]
    vector = build_prediction_vector(
        bundle["feat_cols"],
        {key: float(value) for key, value in payload.items()},
        bundle["feature_means"],
    )
    prediction = clip_energy_prediction(float(model.predict(vector)[0]))
    average = float(bundle["target_average"])
    description = describe_prediction(prediction, average)
    return {
        "model": model_name,
        "prediction": round(prediction, 2),
        "unit": "Wh",
        "average": round(average, 2),
        "deltaPercent": round(((prediction - average) / average) * 100, 1),
        "description": description,
    }
