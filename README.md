---
title: Home Energy Estimator
sdk: docker
app_port: 7860
pinned: false
---

# Home Energy Estimator

Home Energy Estimator is a custom web app that estimates appliance energy from room conditions, weather, and time of day.

The app does not use Streamlit. It serves a static frontend and a small Python prediction API from `app.py`.

## How It Works

- The model is trained offline with `scripts/train_artifacts.py`.
- The best saved model is stored in `artifacts/model_bundle.joblib`.
- The website loads that artifact once.
- Visitors adjust sliders and run fast inference.
- No visitor can retrain the model from the public website.

Current saved model:

- Model: Random Forest
- Parameters: `n_estimators=300`, `max_depth=24`, `min_samples_leaf=1`
- R2: `0.5663`
- MAE: `30.88 Wh`

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open:

```text
http://localhost:8501
```

Run tests:

```bash
pytest -v
```

## Rebuild The Model Artifact

Run this after changing the dataset or candidate model settings:

```bash
python scripts/train_artifacts.py
```

## Project Structure

```text
.
├── app.py
├── Dockerfile
├── artifacts
│   └── model_bundle.joblib
├── public
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── scripts
│   └── train_artifacts.py
├── src
│   ├── artifacts.py
│   ├── data_loader.py
│   ├── modeling.py
│   ├── prediction.py
│   └── service.py
└── tests
    ├── test_artifacts.py
    ├── test_modeling.py
    ├── test_prediction.py
    └── test_service.py
```

## Model Limits

This is an estimator built from a public dataset. It is not connected to a real building, utility account, or control system. Real use would need fresh local data, monitoring, and review before operational decisions.
