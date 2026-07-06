# Energy Forecasting Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current Streamlit class project into a cleaner public analytics app named Energy Forecasting Lab.

**Architecture:** Keep Streamlit and the existing six-page flow. Move shared labels, chart theme values, prediction helpers, and model summary helpers into focused modules so page files can focus on rendering. Add tests for non-UI logic before changing production code.

**Tech Stack:** Python, Streamlit, Pandas, NumPy, scikit-learn, Plotly, SHAP, W&B, pytest.

## Global Constraints

- Keep Streamlit.
- Work on the same repo.
- Use plain wording.
- Avoid emoji in page titles, labels, buttons, captions, and success messages.
- Avoid classroom framing such as final project, graded requirement, project complete, developed for.
- Do not imply live building control, utility integration, or real production automation.
- Keep Hugging Face Spaces compatibility.
- Use short natural commit messages.

---

### Task 1: Shared App Helpers

**Files:**
- Create: `src/content.py`
- Create: `src/modeling.py`
- Create: `src/prediction.py`
- Create: `tests/test_modeling.py`
- Create: `tests/test_prediction.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `src.content.APP_NAME: str`
- Produces: `src.content.PAGE_OPTIONS: list[str]`
- Produces: `src.content.PAGE_TITLES: dict[str, str]`
- Produces: `src.modeling.build_leaderboard(results: dict[str, dict[str, object]]) -> pandas.DataFrame`
- Produces: `src.modeling.get_best_model(leaderboard: pandas.DataFrame) -> pandas.Series`
- Produces: `src.prediction.clip_energy_prediction(value: float) -> float`
- Produces: `src.prediction.build_prediction_vector(feature_columns: list[str], form_values: dict[str, float], feature_means: dict[str, float]) -> numpy.ndarray`
- Consumes: existing model result dictionaries from `src.data_loader.train_all_models`

- [ ] **Step 1: Write failing tests for model summaries**

```python
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
```

- [ ] **Step 2: Write failing tests for prediction helpers**

```python
import numpy as np

from src.prediction import build_prediction_vector, clip_energy_prediction


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
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_modeling.py tests/test_prediction.py -v`

Expected: FAIL with import errors because `src.modeling` and `src.prediction` do not exist.

- [ ] **Step 4: Implement shared helpers**

Create `src/content.py`:

```python
APP_NAME = "Energy Forecasting Lab"
APP_TAGLINE = "Estimate appliance energy use from indoor, outdoor, and time based features."

PAGE_OVERVIEW = "Overview"
PAGE_EXPLORE = "Explore Data"
PAGE_FORECAST = "Forecast"
PAGE_EXPLAIN = "Explain Model"
PAGE_TUNE = "Tune Models"
PAGE_FINDINGS = "Findings"

PAGE_OPTIONS = [
    PAGE_OVERVIEW,
    PAGE_EXPLORE,
    PAGE_FORECAST,
    PAGE_EXPLAIN,
    PAGE_TUNE,
    PAGE_FINDINGS,
]

PAGE_TITLES = {
    PAGE_OVERVIEW: "Overview",
    PAGE_EXPLORE: "Explore Data",
    PAGE_FORECAST: "Forecast",
    PAGE_EXPLAIN: "Explain Model",
    PAGE_TUNE: "Tune Models",
    PAGE_FINDINGS: "Findings",
}
```

Create `src/modeling.py`:

```python
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
```

Create `src/prediction.py`:

```python
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
```

Add `pytest` to `requirements.txt`.

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_modeling.py tests/test_prediction.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add requirements.txt src/content.py src/modeling.py src/prediction.py tests/test_modeling.py tests/test_prediction.py
git commit -m "added simple tests"
```

---

### Task 2: App Shell Cleanup

**Files:**
- Modify: `app.py`
- Modify: `src/content.py`

**Interfaces:**
- Consumes: `APP_NAME`, `APP_TAGLINE`, `PAGE_OPTIONS`, and page constants from `src.content`
- Produces: clean page routing with plain page labels

- [ ] **Step 1: Update app shell**

Use `src.content` for page config, sidebar copy, and route comparisons. Remove emoji labels from navigation and sidebar text. Keep cached data loading as is.

- [ ] **Step 2: Run import check**

Run: `python -m compileall app.py src`

Expected: compile completes without syntax errors.

- [ ] **Step 3: Commit**

Run:

```bash
git add app.py src/content.py
git commit -m "cleaned up the app"
```

---

### Task 3: Page Copy And Logic Cleanup

**Files:**
- Modify: `src/page1_business.py`
- Modify: `src/page2_eda.py`
- Modify: `src/page3_predictions.py`
- Modify: `src/page4_shap.py`
- Modify: `src/page5_tuning.py`
- Modify: `src/page6_conclusions.py`
- Modify: `src/modeling.py`
- Modify: `src/prediction.py`

**Interfaces:**
- Consumes: `build_leaderboard`, `get_best_model`, `build_prediction_vector`, `clip_energy_prediction`
- Produces: clean Streamlit pages with consistent labels and copy

- [ ] **Step 1: Update Overview**

Rewrite `page1_business.render` around dataset overview, modeling goal, feature groups, and preview tabs. Remove emoji and broad business claims.

- [ ] **Step 2: Update Explore Data**

Keep all five chart views. Rename page title, sidebar controls, chart labels, and section copy.

- [ ] **Step 3: Update Forecast**

Use `build_leaderboard`, `build_prediction_vector`, and `clip_energy_prediction`. Rename controls and result copy. Keep model comparison for the same input.

- [ ] **Step 4: Update Explain Model**

Keep SHAP behavior. Replace explainer copy with one plain sentence. Remove emoji and overly technical decorative labels.

- [ ] **Step 5: Update Tune Models**

Keep grid search and W&B logging. Remove grading language. Make W&B optional and quiet when credentials are missing.

- [ ] **Step 6: Update Findings**

Use `build_leaderboard` and `get_best_model`. Replace claims with honest findings and limits.

- [ ] **Step 7: Run tests and compile**

Run: `pytest -v`

Expected: PASS.

Run: `python -m compileall app.py src tests`

Expected: compile completes without syntax errors.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/page1_business.py src/page2_eda.py src/page3_predictions.py src/page4_shap.py src/page5_tuning.py src/page6_conclusions.py src/modeling.py src/prediction.py
git commit -m "better project copy"
```

---

### Task 4: README And Space Metadata

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: final app name and live Space URL
- Produces: public README with setup, features, limits, and project structure

- [ ] **Step 1: Rewrite README**

Include the live URL, project purpose, dataset, features, setup, environment variables, structure, and limits. Remove placeholder links and class project language.

- [ ] **Step 2: Scan README**

Run: `rg -n "final project|graded|project complete|developed for|next generation|seamless|unlock|revolutionize|🚀|⚡|🛠|📦|👥|—|–" README.md`

Expected: no output.

- [ ] **Step 3: Commit**

Run:

```bash
git add README.md
git commit -m "readme cleanup"
```

---

### Task 5: Final Verification

**Files:**
- Read: all changed files

**Interfaces:**
- Consumes: complete app changes
- Produces: verified repo ready for push

- [ ] **Step 1: Run tests**

Run: `pytest -v`

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run: `python -m compileall app.py src tests`

Expected: compile completes without syntax errors.

- [ ] **Step 3: Scan visible copy**

Run:

```bash
rg -n "🚀|⚡|🛠|📦|👥|🤖|📊|🔍|⚙|🏁|🎯|💡|✨|final project|graded requirement|project complete|developed for|next generation|seamless|unlock|revolutionize|—|–" app.py src README.md
```

Expected: no output, except if the dataset or third party name requires a matched word.

- [ ] **Step 4: Check git state**

Run: `git status --short --branch`

Expected: branch is ahead of origin with no unstaged changes.
