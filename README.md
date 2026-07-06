---
title: Energy Forecasting Lab
colorFrom: blue
colorTo: cyan
sdk: streamlit
app_file: app.py
pinned: false
---

# Energy Forecasting Lab

Energy Forecasting Lab is a Streamlit app for estimating appliance energy use from indoor sensor readings, outdoor weather, and time based features.

Live app: https://huggingface.co/spaces/hassanraza04/ds_final_proj

## What It Does

The app uses the UCI Appliances Energy Prediction dataset to train and compare five regression models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting

The app includes data exploration, live forecasting, model comparison, SHAP based explanations, small grid search experiments, and a short findings page.

## Main Features

- Dataset overview with row counts, missing values, feature types, and summary statistics
- Distribution, time series, period, scatter, and correlation views
- Forecast form for testing one set of conditions across trained models
- Model ranking by R2, MAE, and RMSE
- SHAP charts for feature impact and single prediction explanations
- Optional Weights & Biases logging for tuning runs
- Clear notes on model limits before real operational use

## Local Setup

Clone the repo:

```bash
git clone https://github.com/hassanraza04/energy-forecasting.git
cd energy-forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python -m streamlit run app.py
```

Run tests:

```bash
pytest -v
```

## Environment Variables

Weights & Biases logging is optional. Set these values only if you want tuning runs logged outside the app:

```env
WANDB_API_KEY=your_key_here
WANDB_ENTITY=your_entity_here
WANDB_PROJECT=energy-forecasting
```

The app still runs without these values.

## Project Structure

```text
.
├── app.py
├── energydata_complete.csv
├── requirements.txt
├── src
│   ├── content.py
│   ├── data_loader.py
│   ├── modeling.py
│   ├── prediction.py
│   ├── page1_business.py
│   ├── page2_eda.py
│   ├── page3_predictions.py
│   ├── page4_shap.py
│   ├── page5_tuning.py
│   ├── page6_conclusions.py
│   └── secrets.py
└── tests
    ├── conftest.py
    ├── test_modeling.py
    └── test_prediction.py
```

## Model Limits

This app is a forecasting demo built from a public dataset. It is not connected to a live building, utility account, or appliance control system.

Before using a model like this for real decisions, it would need fresh local data, sensor quality checks, retraining, drift monitoring, versioned model artifacts, and human review around any operational actions.
