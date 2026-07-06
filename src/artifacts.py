"""Saved model artifact helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "model_bundle.joblib"


def load_artifact_bundle(path: str | Path = MODEL_BUNDLE_PATH) -> dict[str, Any]:
    return joblib.load(Path(path))


@st.cache_resource(show_spinner="Loading saved model bundle...")
def get_model_bundle(path: str = str(MODEL_BUNDLE_PATH)) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        st.error(
            "Saved model artifact is missing. Run `python scripts/train_artifacts.py` "
            "before starting the app."
        )
        st.stop()
    return load_artifact_bundle(artifact_path)
