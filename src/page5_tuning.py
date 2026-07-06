"""Hyperparameter tuning and optional experiment tracking."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV

from src.data_loader import ALL_MODELS, LINEAR_MODELS, RANDOM_STATE
from src.secrets import get_secret


def render(bundle: Dict[str, Any]) -> None:
    st.title("Tune Models")
    st.caption("Run small grid searches and compare each result with the baseline models.")

    results = bundle["results"]
    x_train = bundle["X_train"]
    x_train_scaled = bundle["X_train_s"]
    y_train = bundle["y_train"]

    env_api_key = get_secret("WANDB_API_KEY")
    env_entity = get_secret("WANDB_ENTITY")
    env_project = get_secret("WANDB_PROJECT", "energy-forecasting")

    if env_api_key and "_wb_auto_logged_in" not in st.session_state:
        try:
            import wandb
            wandb.login(key=env_api_key, relogin=False)
            st.session_state["_wb_auto_logged_in"] = True
        except Exception:
            pass

    use_wb = bool(env_api_key)
    wb_project = env_project or "energy-forecasting"
    wb_entity = env_entity or None

    st.subheader("Experiment setup")
    model_col, cv_col = st.columns([2, 1])
    with model_col:
        tune_model = st.selectbox("Model to tune", ALL_MODELS, key="tune_model")
    with cv_col:
        cv_folds = st.slider("Cross-validation folds", 2, 10, 3, key="tune_cv")

    is_linear = tune_model in LINEAR_MODELS
    param_grid: Dict[str, Any] = {}

    st.markdown(f"#### Search grid for {tune_model}")
    with st.expander("Configure search grid", expanded=True):
        if tune_model == "Linear Regression":
            st.info("Linear Regression has no regularization settings in this app.")
            param_grid = {"fit_intercept": [True]}

        elif tune_model == "Ridge Regression":
            alphas = st.multiselect(
                "Alpha",
                [0.01, 0.1, 1.0, 10.0, 100.0],
                default=[0.1, 1.0, 10.0],
                key="tune_ridge_alpha",
                help="Lower alpha means weaker regularization.",
            )
            param_grid = {"alpha": alphas or [1.0]}

        elif tune_model == "Lasso Regression":
            alphas = st.multiselect(
                "Alpha",
                [0.001, 0.01, 0.1, 1.0, 10.0],
                default=[0.01, 0.1, 1.0],
                key="tune_lasso_alpha",
                help="Higher alpha can shrink more coefficients toward zero.",
            )
            param_grid = {"alpha": alphas or [1.0]}

        elif tune_model == "Random Forest":
            n_estimators = st.multiselect(
                "Number of trees",
                [50, 100, 200, 300],
                default=[50, 100],
                key="tune_rf_nest",
            )
            max_depth = st.multiselect(
                "Max depth",
                [5, 10, 20, None],
                default=[5, 10],
                key="tune_rf_depth",
            )
            param_grid = {
                "n_estimators": n_estimators or [100],
                "max_depth": max_depth or [10],
            }

        elif tune_model == "Gradient Boosting":
            n_estimators = st.multiselect(
                "Number of estimators",
                [50, 100, 200],
                default=[50, 100],
                key="tune_gb_nest",
            )
            learning_rate = st.multiselect(
                "Learning rate",
                [0.01, 0.05, 0.1, 0.2],
                default=[0.05, 0.1],
                key="tune_gb_lr",
            )
            max_depth = st.multiselect(
                "Max depth",
                [3, 5, 7],
                default=[3, 5],
                key="tune_gb_depth",
            )
            param_grid = {
                "n_estimators": n_estimators or [100],
                "learning_rate": learning_rate or [0.1],
                "max_depth": max_depth or [3],
            }

    total_configs = 1
    for values in param_grid.values():
        total_configs *= len(values)
    st.caption(
        f"The grid will test {total_configs} configuration(s) across {cv_folds} folds."
    )

    if st.button("Run grid search", type="primary", key="tune_run"):
        estimator_map = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(max_iter=5000),
            "Lasso Regression": Lasso(max_iter=5000),
            "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        }
        estimator = estimator_map[tune_model]
        x_train_input = x_train_scaled if is_linear else x_train

        with st.spinner(
            f"Running {cv_folds}-fold grid search for {tune_model}"
        ):
            grid_search = GridSearchCV(
                estimator,
                param_grid,
                cv=cv_folds,
                scoring="r2",
                n_jobs=-1,
                verbose=0,
                return_train_score=True,
            )
            grid_search.fit(x_train_input, y_train)

        st.success(
            f"Best parameters: {grid_search.best_params_}. "
            f"CV R2: {grid_search.best_score_:.4f}"
        )

        result_columns = [
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "rank_test_score",
        ]
        param_columns = [
            column for column in pd.DataFrame(grid_search.cv_results_).columns
            if column.startswith("param_")
        ]
        cv_df = (
            pd.DataFrame(grid_search.cv_results_)[param_columns + result_columns]
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )
        cv_df.columns = [column.replace("param_", "") for column in cv_df.columns]
        clean_param_columns = [column.replace("param_", "") for column in param_columns]

        st.dataframe(
            cv_df.style
            .highlight_max(subset=["mean_test_score"], color="#064e3b")
            .highlight_min(subset=["std_test_score"], color="#064e3b")
            .format({
                "mean_test_score": "{:.4f}",
                "std_test_score": "{:.4f}",
                "mean_train_score": "{:.4f}",
            }),
            width="stretch",
        )

        if use_wb and wb_project:
            try:
                import wandb
                if env_api_key:
                    wandb.login(key=env_api_key, relogin=False)

                logged = 0
                progress = st.progress(0, text="Logging runs to W&B")
                for index, row in cv_df.iterrows():
                    config = {param: row[param] for param in clean_param_columns}
                    config["model"] = tune_model
                    config["cv_folds"] = cv_folds

                    final_entity = wb_entity.strip() if wb_entity else ""
                    kwargs = {
                        "project": wb_project.strip() if wb_project else "energy-forecasting",
                        "name": f"{tune_model.replace(' ', '_')}_run_{index + 1}",
                        "config": config,
                        "reinit": True,
                    }
                    if final_entity:
                        kwargs["entity"] = final_entity

                    run = wandb.init(**kwargs)
                    wandb.log({
                        "cv_r2_mean": float(row["mean_test_score"]),
                        "cv_r2_std": float(row["std_test_score"]),
                        "train_r2_mean": float(row["mean_train_score"]),
                        "rank": int(row["rank_test_score"]),
                    })
                    if int(row["rank_test_score"]) == 1:
                        wandb.run.tags = ["best"]
                    run.finish()
                    logged += 1
                    progress.progress(logged / len(cv_df), text=f"Logged {logged}/{len(cv_df)}")

                progress.empty()
                entity_str = wb_entity if wb_entity else "<your-entity>"
                run_url = f"https://wandb.ai/{entity_str}/{wb_project}"
                st.success(f"Logged {logged} run(s) to W&B: {run_url}")
            except Exception as exc:
                st.error(f"W&B logging failed: {exc}")
        else:
            st.info("Set W&B credentials to log tuning runs outside this app.")

        st.subheader("Tuning results")
        sorted_cv = cv_df.sort_values("mean_test_score")
        fig = px.bar(
            sorted_cv,
            y=sorted_cv.index.astype(str),
            x="mean_test_score",
            orientation="h",
            error_x="std_test_score",
            color="mean_test_score",
            color_continuous_scale="teal",
            labels={"mean_test_score": "CV R2", "y": "Config"},
            template="plotly_dark",
            title=f"{tune_model}: configurations ranked by CV R2",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            cv_df,
            x="mean_train_score",
            y="mean_test_score",
            error_y="std_test_score",
            color="rank_test_score",
            color_continuous_scale="RdYlGn_r",
            labels={
                "mean_train_score": "Train R2",
                "mean_test_score": "CV test R2",
                "rank_test_score": "Rank",
            },
            title="Train score vs CV score",
            template="plotly_dark",
        )
        min_value = min(cv_df["mean_train_score"].min(), cv_df["mean_test_score"].min()) - 0.02
        max_value = max(cv_df["mean_train_score"].max(), cv_df["mean_test_score"].max()) + 0.02
        fig2.add_shape(
            type="line",
            x0=min_value,
            y0=min_value,
            x1=max_value,
            y1=max_value,
            line=dict(color="white", dash="dash"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        if len(clean_param_columns) == 2:
            try:
                p1, p2 = clean_param_columns[0], clean_param_columns[1]
                pivot = cv_df.pivot_table(index=p1, columns=p2, values="mean_test_score")
                fig3 = px.imshow(
                    pivot,
                    text_auto=".3f",
                    color_continuous_scale="Blues",
                    labels={"color": "CV R2"},
                    title=f"{tune_model}: parameter grid",
                    template="plotly_dark",
                )
                st.plotly_chart(fig3, use_container_width=True)
            except Exception:
                pass

    st.divider()
    st.subheader("Baseline performance")

    metric_view = st.radio(
        "Metric",
        ["R2", "MAE", "RMSE"],
        horizontal=True,
        key="tune_baseline_metric",
    )
    base_df = pd.DataFrame([
        {"Model": name, "MAE": values["MAE"], "RMSE": values["RMSE"], "R2": values["R2"]}
        for name, values in results.items()
    ]).sort_values(metric_view, ascending=(metric_view != "R2"))

    best_model = base_df.iloc[0]["Model"]
    st.info(
        f"Best baseline model: {best_model}. "
        f"R2: {base_df.iloc[0]['R2']:.4f}. "
        f"MAE: {base_df.iloc[0]['MAE']:.2f} Wh."
    )

    fig_b = px.bar(
        base_df,
        x="Model",
        y=metric_view,
        color=metric_view,
        color_continuous_scale="teal" if metric_view == "R2" else "Reds",
        template="plotly_dark",
        text_auto=".3f",
    )
    fig_b.update_traces(textposition="outside")
    st.plotly_chart(fig_b, use_container_width=True)

    st.caption(
        "Run a grid search to compare tuned settings with the baseline model settings."
    )
