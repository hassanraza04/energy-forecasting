# Energy Forecasting Redesign

Date: 2026-07-06

## Goal

Turn the current data science project into a cleaner public Streamlit app.

The app should feel like a serious portfolio project. It should still be honest about the dataset, the model limits, and the workflow.

## Current State

The project is a Streamlit app that runs on Hugging Face Spaces. GitHub is the source repo. The Space repo is a sync target.

Main files:

- `app.py` handles page setup and navigation.
- `src/data_loader.py` loads the dataset, adds time features, trains models, and returns cached model bundles.
- `src/page1_business.py` introduces the project and dataset.
- `src/page2_eda.py` contains the EDA views.
- `src/page3_predictions.py` contains live prediction, model ranking, model inspection, and model comparison.
- `src/page4_shap.py` explains model outputs with SHAP.
- `src/page5_tuning.py` runs grid search and logs to Weights & Biases when credentials exist.
- `src/page6_conclusions.py` summarizes results and recommendations.

The app works as a class project, but the copy and labels feel too generic. It uses many emoji labels, classroom framing, and phrases like "smart" and "strategic recommendations" too often. Some pages also mix product logic, copy, and chart setup in ways that make future changes harder.

## Product Direction

Keep Streamlit. Rework the project into a polished analytics app named **Energy Forecasting Lab**.

The app should feel:

- clear
- calm
- practical
- data led
- portfolio ready

The app should not feel:

- like a generated demo
- like a class slide deck
- like a marketing landing page
- like a fake production platform

## App Structure

Use the existing six-page flow, but rename and sharpen it:

1. Overview
2. Explore Data
3. Forecast
4. Explain Model
5. Tune Models
6. Findings

The sidebar should use plain labels. No emoji labels. The project title should be short and stable.

## Copy Rules

Rewrite visible copy across the app.

Use plain wording. Avoid vague phrases like:

- smart
- next generation
- strategic
- robust
- seamless
- unlock
- leverage
- revolutionize

Avoid classroom framing such as:

- final project
- graded requirement
- project complete
- developed for

Avoid emoji in page titles, labels, buttons, captions, and success messages.

Keep claims tied to what the app actually does. Do not imply it is connected to a live building, a utility account, or a deployed control system.

## Visual Design

Use a restrained dark analytics style with one accent color. The current blue heavy theme can stay, but it should be cleaner and less loud.

Design rules:

- simpler sidebar
- tighter page headers
- consistent chart colors
- clearer metric labels
- no decorative labels
- no fake precision beyond model metrics
- no celebratory success copy

Keep Streamlit custom CSS small. Do not fight Streamlit too much.

## Data And Model Flow

Keep the UCI Appliances Energy Prediction dataset.

Keep the five baseline models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting

Keep cached loading and cached training.

Improve boundaries where useful:

- move shared labels and constants into small modules
- keep chart theme helpers in one place
- keep prediction input config in one place
- keep model summary helpers out of page files

Do not introduce a database, auth, background jobs, or live API calls.

## Page Changes

### Overview

Show the dataset, target, row count, feature count, missing values, and the real modeling goal.

Replace broad business claims with a simple statement:

"Estimate appliance energy use from indoor, outdoor, and time based features."

### Explore Data

Keep the current chart views. Clean the labels and explanations.

Make the controls easier to scan. Keep distribution, time series, period comparison, scatter, and correlation views.

### Forecast

Keep the live prediction form. Rename inputs so they read like real controls.

Show:

- selected model
- predicted appliance energy
- dataset average
- model score
- same input across all models

Keep clipping negative predictions to zero.

### Explain Model

Keep SHAP support. Explain SHAP in one plain sentence.

Keep bar, beeswarm, waterfall, and dependency plots.

Make errors clear if SHAP is not installed.

### Tune Models

Keep grid search. Keep W&B logging only when credentials exist.

Remove language that says W&B is a grading requirement. Treat it as optional experiment tracking.

### Findings

Summarize the best model, model error, feature patterns, and limits.

Make the conclusion honest. Do not claim the app is ready for real building automation. Say what would be needed before real use.

## Reliability

Add focused tests for non-UI logic where practical.

Good targets:

- time feature creation
- numeric feature selection
- model leaderboard creation
- prediction vector building
- negative prediction clipping

Do not try to snapshot Streamlit pages.

## Documentation

Rewrite the README as a real public project page.

Include:

- what the app does
- live demo link
- dataset source
- main features
- local setup
- environment variables
- project structure
- model limits

Remove placeholder deployment links and class project wording.

## Commit Style

Use short, natural commit messages.

Examples:

- planned cleanup
- cleaned up the app
- better project copy
- polished predictions
- clearer findings
- added simple tests
- readme cleanup

## Out Of Scope

Do not rewrite the app in React or Next.js.

Do not add user accounts.

Do not add a database.

Do not invent live energy data.

Do not claim production readiness beyond the app quality itself.

## Success Criteria

The work is done when:

- the app has no AI slop style copy
- the visible labels are clean and consistent
- the README reads like a real project
- the app still runs on Streamlit and Hugging Face Spaces
- basic logic tests pass
- commits are short and natural
