"""Saved model artifact helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "model_bundle.joblib"


def load_artifact_bundle(path: str | Path = MODEL_BUNDLE_PATH) -> dict[str, Any]:
    return joblib.load(Path(path))


_MODEL_BUNDLE: dict[str, Any] | None = None


def get_model_bundle(path: str = str(MODEL_BUNDLE_PATH)) -> dict[str, Any]:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Saved model artifact is missing. Run `python scripts/train_artifacts.py` "
            "before starting the app."
        )
    _MODEL_BUNDLE = load_artifact_bundle(artifact_path)
    return _MODEL_BUNDLE
