from pathlib import Path

import joblib

from src.artifacts import load_artifact_bundle


def test_load_artifact_bundle_reads_joblib_bundle(tmp_path: Path):
    artifact_path = tmp_path / "model_bundle.joblib"
    expected = {
        "best_model": "Random Forest",
        "metadata": {"trained_at": "2026-07-06"},
        "results": {},
        "trained": {},
    }
    joblib.dump(expected, artifact_path)

    bundle = load_artifact_bundle(artifact_path)

    assert bundle["best_model"] == "Random Forest"
    assert bundle["metadata"]["trained_at"] == "2026-07-06"
