"""
src/page5_tuning.py
Page 5 — Hyperparameter Tuning & W&B Tracking — all 5 models
W&B is a core graded requirement; this page makes it the centrepiece.
"""
from __future__ import annotations

from src.secrets import get_secret
from typing import Dict, Any, List

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from src.data_loader import ALL_MODELS, LINEAR_MODELS, RANDOM_STATE


def render(bundle: Dict[str, Any]) -> None:
    st.title("⚙️ Hyperparameter Tuning & W&B Tracking")
    st.caption(
        "Run grid-search experiments on any model, track every run in "
        "**Weights & Biases**, and select the best-performing configuration."
    )

    results   = bundle["results"]
    X_train   = bundle["X_train"]
    X_train_s = bundle["X_train_s"]
    y_train   = bundle["y_train"]

    # ── Load credentials from .env (fallback to empty string) ─────────────────
    _env_api_key = get_secret("WANDB_API_KEY")
    _env_entity  = get_secret("WANDB_ENTITY")
    _env_project = get_secret("WANDB_PROJECT", "energy-forecasting")

    # Auto-login with env key if available (silent, once per session)
    if _env_api_key and "_wb_auto_logged_in" not in st.session_state:
        try:
            import wandb
            wandb.login(key=_env_api_key, relogin=False)
            st.session_state["_wb_auto_logged_in"] = True
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # W&B SETUP SECTION
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🔐 Weights & Biases Setup")
    with st.expander("W&B Authentication & Project Config", expanded=True):
        st.markdown("""
        **Weights & Biases** (wandb) tracks every experiment run — hyperparameters,
        metrics, and model performance — so you can compare and reproduce results.

        **Setup Steps:**
        1. Create a free account at [wandb.ai](https://wandb.ai)
        2. Copy your API key from **Settings → API Keys**
        3. Paste it below and click **Login**
        """)

        col_key, col_proj = st.columns([2, 1])
        with col_key:
            wb_api_key = st.text_input(
                "W&B API Key",
                value=_env_api_key,
                type="password",
                placeholder="Auto-loaded from .env",
                key="wb_api_key",
            )
        with col_proj:
            wb_project = st.text_input(
                "W&B Project Name",
                value=_env_project,
                key="wb_project",
            )

        wb_entity = st.text_input(
            "W&B Entity (username or team)",
            value=_env_entity,
            key="wb_entity",
            placeholder="Auto-loaded from .env",
        )

        use_wb = st.toggle(
            "Enable W&B Logging",
            value=bool(_env_api_key),   # auto-enable when key is in .env
            key="wb_enable",
        )

        if _env_api_key and st.session_state.get("_wb_auto_logged_in"):
            st.success("✅ Auto-logged in via .env credentials.")
        elif use_wb and st.button("🔑 Login to W&B", key="wb_login_btn"):
            if wb_api_key:
                try:
                    import wandb
                    wandb.login(key=wb_api_key, relogin=True)
                    st.session_state["_wb_auto_logged_in"] = True
                    st.success("✅ Logged in to Weights & Biases successfully!")
                except Exception as e:
                    st.error(f"W&B login failed: {e}")
            else:
                st.warning("Please enter your W&B API key first.")

        if use_wb:
            st.info(
                f"After running the grid search, results will be logged to "
                f"**{wb_project}** on wandb.ai. "
                f"View your runs at: https://wandb.ai/{'<entity>/' if not wb_entity else wb_entity+'/'}{wb_project}"
            )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL & GRID CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🎛️ Experiment Configuration")

    col_m, col_cv = st.columns([2, 1])
    with col_m:
        tune_model = st.selectbox("Select model to tune", ALL_MODELS, key="tune_model")
    with col_cv:
        cv_folds = st.slider("Cross-validation folds", 2, 10, 3, key="tune_cv")

    is_linear  = tune_model in LINEAR_MODELS
    param_grid: Dict[str, Any] = {}

    st.markdown(f"**Hyperparameter Grid — {tune_model}**")

    with st.expander("Configure search grid", expanded=True):

        if tune_model == "Linear Regression":
            st.info(
                "Linear Regression has no free hyperparameters. "
                "A single CV-scored run will be logged to W&B."
            )
            param_grid = {"fit_intercept": [True]}

        elif tune_model == "Ridge Regression":
            alphas = st.multiselect(
                "alpha (regularisation strength)",
                [0.01, 0.1, 1.0, 10.0, 100.0],
                default=[0.1, 1.0, 10.0],
                key="tune_ridge_alpha",
                help="Lower α → less regularisation; higher α → stronger regularisation.",
            )
            param_grid = {"alpha": alphas or [1.0]}

        elif tune_model == "Lasso Regression":
            alphas = st.multiselect(
                "alpha (regularisation strength)",
                [0.001, 0.01, 0.1, 1.0, 10.0],
                default=[0.01, 0.1, 1.0],
                key="tune_lasso_alpha",
            )
            param_grid = {"alpha": alphas or [1.0]}

        elif tune_model == "Random Forest":
            n_est = st.multiselect(
                "n_estimators (number of trees)",
                [50, 100, 200, 300],
                default=[50, 100],
                key="tune_rf_nest",
            )
            m_dep = st.multiselect(
                "max_depth (tree depth)",
                [5, 10, 20, None],
                default=[5, 10],
                key="tune_rf_depth",
            )
            param_grid = {
                "n_estimators": n_est or [100],
                "max_depth":    m_dep or [10],
            }

        elif tune_model == "Gradient Boosting":
            n_est = st.multiselect(
                "n_estimators",
                [50, 100, 200],
                default=[50, 100],
                key="tune_gb_nest",
            )
            lr = st.multiselect(
                "learning_rate",
                [0.01, 0.05, 0.1, 0.2],
                default=[0.05, 0.1],
                key="tune_gb_lr",
            )
            m_dep = st.multiselect(
                "max_depth",
                [3, 5, 7],
                default=[3, 5],
                key="tune_gb_depth",
            )
            param_grid = {
                "n_estimators":  n_est or [100],
                "learning_rate": lr    or [0.1],
                "max_depth":     m_dep or [3],
            }

    total_configs = 1
    for v in param_grid.values():
        total_configs *= len(v)
    st.caption(
        f"Grid will test **{total_configs} configuration(s)** × {cv_folds} folds "
        f"= **{total_configs * cv_folds} model fits**."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RUN GRID SEARCH
    # ══════════════════════════════════════════════════════════════════════════
    if st.button("▶  Run Grid Search & Log to W&B", type="primary", key="tune_run"):

        estimator_map = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression":  Ridge(max_iter=5000),
            "Lasso Regression":  Lasso(max_iter=5000),
            "Random Forest":     RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        }
        estimator = estimator_map[tune_model]
        Xtr       = X_train_s if is_linear else X_train

        with st.spinner(
            f"Running {cv_folds}-fold GridSearchCV for {tune_model} "
            f"({total_configs} configs)…"
        ):
            gs = GridSearchCV(
                estimator, param_grid,
                cv=cv_folds, scoring="r2",
                n_jobs=-1, verbose=0,
                return_train_score=True,
            )
            gs.fit(Xtr, y_train)

        # ── Results ────────────────────────────────────────────────────────────
        st.success(
            f"✅ Best params: **{gs.best_params_}**  |  "
            f"CV R²: **{gs.best_score_:.4f}**"
        )

        result_cols = [
            "mean_test_score", "std_test_score",
            "mean_train_score", "rank_test_score",
        ]
        param_cols = [
            c for c in pd.DataFrame(gs.cv_results_).columns
            if c.startswith("param_")
        ]
        cv_df = (
            pd.DataFrame(gs.cv_results_)[param_cols + result_cols]
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )
        cv_df.columns = [c.replace("param_", "") for c in cv_df.columns]
        clean_param_cols = [c.replace("param_", "") for c in param_cols]

        st.dataframe(
            cv_df.style
                 .highlight_max(subset=["mean_test_score"], color="#064e3b")
                 .highlight_min(subset=["std_test_score"],  color="#064e3b")
                 .format({
                     "mean_test_score":  "{:.4f}",
                     "std_test_score":   "{:.4f}",
                     "mean_train_score": "{:.4f}",
                 }),
            width="stretch",
        )

        # ── W&B Logging ────────────────────────────────────────────────────────
        if use_wb and wb_project:
            try:
                import wandb
                if wb_api_key:
                    wandb.login(key=wb_api_key, relogin=False)

                logged = 0
                prog   = st.progress(0, text="Logging to W&B…")
                for i, row in cv_df.iterrows():
                    cfg = {p: row[p] for p in clean_param_cols}
                    cfg["model"]    = tune_model
                    cfg["cv_folds"] = cv_folds

                    # Use .strip() and check for empty strings
                    final_entity = wb_entity.strip() if wb_entity else ""

                    kwargs = dict(
                        project=wb_project.strip() if wb_project else "energy-forecasting",
                        name=f"{tune_model.replace(' ', '_')}_run_{i+1}",
                        config=cfg,
                        reinit=True,
                    )
                    if final_entity:
                        kwargs["entity"] = final_entity

                    run = wandb.init(**kwargs)
                    wandb.log({
                        "cv_r2_mean":    float(row["mean_test_score"]),
                        "cv_r2_std":     float(row["std_test_score"]),
                        "train_r2_mean": float(row["mean_train_score"]),
                        "rank":          int(row["rank_test_score"]),
                    })
                    # Tag best run
                    if int(row["rank_test_score"]) == 1:
                        wandb.run.tags = ["best"]
                    run.finish()
                    logged += 1
                    prog.progress(logged / len(cv_df), text=f"Logged run {logged}/{len(cv_df)}")

                prog.empty()

                entity_str = wb_entity if wb_entity else "<your-entity>"
                run_url    = f"https://wandb.ai/{entity_str}/{wb_project}"
                st.success(
                    f"✅ Logged **{logged} runs** to W&B project "
                    f"**{wb_project}**.  \n"
                    f"[🔗 View in W&B dashboard]({run_url})"
                )
            except Exception as e:
                st.error(f"W&B logging failed: {e}")
                st.info(
                    "Tip: make sure you've clicked **Login to W&B** above "
                    "and your API key is correct."
                )
        elif not use_wb:
            st.info("Enable W&B Logging above to track these experiments in wandb.ai.")

        # ── Visualisations ─────────────────────────────────────────────────────
        st.subheader("📊 Results Visualisation")

        sorted_cv = cv_df.sort_values("mean_test_score")
        fig = px.bar(
            sorted_cv,
            y=sorted_cv.index.astype(str),
            x="mean_test_score",
            orientation="h",
            error_x="std_test_score",
            color="mean_test_score",
            color_continuous_scale="teal",
            labels={"mean_test_score": "CV R²", "y": "Config"},
            template="plotly_dark",
            title=f"{tune_model} — All configs ranked by CV R²",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Train vs test R² (overfitting check)
        fig2 = px.scatter(
            cv_df,
            x="mean_train_score",
            y="mean_test_score",
            error_y="std_test_score",
            color="rank_test_score",
            color_continuous_scale="RdYlGn_r",
            labels={
                "mean_train_score": "Train R²",
                "mean_test_score":  "CV Test R²",
                "rank_test_score":  "Rank",
            },
            title="Train vs CV R² (overfitting check) — closer to diagonal = better",
            template="plotly_dark",
        )
        mn_val = min(cv_df["mean_train_score"].min(), cv_df["mean_test_score"].min()) - 0.02
        mx_val = max(cv_df["mean_train_score"].max(), cv_df["mean_test_score"].max()) + 0.02
        fig2.add_shape(type="line", x0=mn_val, y0=mn_val, x1=mx_val, y1=mx_val,
                       line=dict(color="white", dash="dash"))
        st.plotly_chart(fig2, use_container_width=True)

        # Heatmap for exactly 2 param dimensions
        if len(clean_param_cols) == 2:
            try:
                p1, p2 = clean_param_cols[0], clean_param_cols[1]
                pivot  = cv_df.pivot_table(index=p1, columns=p2,
                                            values="mean_test_score")
                fig3 = px.imshow(
                    pivot, text_auto=".3f",
                    color_continuous_scale="Blues",
                    labels={"color": "CV R²"},
                    title=f"{tune_model} — Parameter Grid Heatmap",
                    template="plotly_dark",
                )
                st.plotly_chart(fig3, use_container_width=True)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE COMPARISON (always visible)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📝 Baseline Performance — All Models (default hyperparameters)")

    metric_view = st.radio(
        "Metric", ["R²", "MAE", "RMSE"], horizontal=True, key="tune_baseline_metric"
    )
    base_df = pd.DataFrame([
        {"Model": name, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]}
        for name, v in results.items()
    ]).sort_values(metric_view, ascending=(metric_view != "R²"))

    # Highlight best
    best_model = base_df.iloc[0]["Model"]
    st.info(f"🏆 Best baseline model: **{best_model}** "
            f"(R²: {base_df.iloc[0]['R²']:.4f}, "
            f"MAE: {base_df.iloc[0]['MAE']:.2f} Wh)")

    fig_b = px.bar(
        base_df, x="Model", y=metric_view,
        color=metric_view,
        color_continuous_scale="teal" if metric_view == "R²" else "Reds",
        template="plotly_dark", text_auto=".3f",
    )
    fig_b.update_traces(textposition="outside")
    st.plotly_chart(fig_b, use_container_width=True)

    st.caption(
        "Run the grid search above to compare tuned vs baseline performance. "
        "All tuning runs are automatically logged to W&B when enabled."
    )
