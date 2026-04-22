"""
src/page4_shap.py
Page 4 — Explainability (SHAP) — all 5 models
"""
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from src.data_loader import ALL_MODELS, LINEAR_MODELS


def render(bundle: Dict[str, Any]) -> None:
    st.title("🔍 Model Explainability — SHAP")
    st.info(
        "SHAP (SHapley Additive exPlanations) values quantify each feature's contribution "
        "to a model prediction. Supported for all 5 models."
    )

    try:
        import shap
    except ImportError:
        st.error("Install shap:  `pip install shap`")
        st.stop()

    trained   = bundle["trained"]
    X_train   = bundle["X_train"]
    X_test    = bundle["X_test"]
    X_train_s = bundle["X_train_s"]
    X_test_s  = bundle["X_test_s"]
    feat_cols = bundle["feat_cols"]

    # ── Controls ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        model_choice = st.selectbox("Model", ALL_MODELS, key="shap_model")
    with c2:
        n_explain = st.slider("Samples to explain", 50, 300, 100, 25, key="shap_n")
    with c3:
        plot_type = st.selectbox(
            "Plot type",
            ["Bar (mean |SHAP|)", "Beeswarm", "Waterfall"],
            key="shap_plot",
        )

    is_linear = model_choice in LINEAR_MODELS
    model     = trained[model_choice]

    with st.spinner(f"Computing SHAP values for {model_choice}…"):
        if is_linear:
            # LinearExplainer uses training data as the background distribution
            X_bg  = X_train_s[:200]
            X_exp = X_test_s[:n_explain]
            explainer = shap.LinearExplainer(model, X_bg)
            shap_vals = explainer.shap_values(X_exp)
            X_df      = pd.DataFrame(X_exp, columns=feat_cols)
            
            # Safe conversion of expected_value (can be array or list in some SHAP versions)
            ev = explainer.expected_value
            base_val = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        else:
            # TreeExplainer — no background needed
            X_exp = X_test.values[:n_explain]
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_exp)
            X_df      = pd.DataFrame(X_exp, columns=feat_cols)
            
            # Safe conversion of expected_value
            ev = explainer.expected_value
            base_val = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)

    # ── Bar (mean |SHAP|) ──────────────────────────────────────────────────────
    if plot_type == "Bar (mean |SHAP|)":
        top_n     = st.slider("Top N features", 5, len(feat_cols), 20, key="shap_topn")
        shap_df   = pd.DataFrame(np.abs(shap_vals), columns=feat_cols)
        mean_shap = shap_df.mean().sort_values(ascending=False).head(top_n)

        fig = px.bar(
            x=mean_shap.values, y=mean_shap.index,
            orientation="h", color=mean_shap.values,
            color_continuous_scale="Blues",
            labels={"x": "Mean |SHAP value|", "y": "Feature"},
            title=f"{model_choice} — Top {top_n} Feature Importances",
            template="plotly_dark",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Beeswarm ───────────────────────────────────────────────────────────────
    elif plot_type == "Beeswarm":
        max_disp = st.slider("Max features displayed", 5, 25, 15, key="shap_beeswarm_n")
        fig2, _ = plt.subplots(figsize=(10, 7))
        fig2.patch.set_facecolor("#0f172a")
        shap.summary_plot(
            shap_vals, X_df,
            plot_type="dot", show=False, max_display=max_disp,
        )
        plt.tight_layout()
        st.pyplot(fig2)

    # ── Waterfall ──────────────────────────────────────────────────────────────
    elif plot_type == "Waterfall":
        idx = st.slider("Test-set sample index", 0, n_explain - 1, 0, key="shap_idx")
        exp = shap.Explanation(
            values=shap_vals[int(idx)],
            base_values=base_val,
            data=X_df.iloc[int(idx)].values,
            feature_names=feat_cols,
        )
        fig3, _ = plt.subplots(figsize=(10, 6))
        fig3.patch.set_facecolor("#0f172a")
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        st.pyplot(fig3)

    # ── Bonus: top feature dependency plot ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 SHAP Dependency — Top Feature")
    shap_df   = pd.DataFrame(np.abs(shap_vals), columns=feat_cols)
    top_feat  = shap_df.mean().idxmax()
    feat_idx  = feat_cols.index(top_feat)
    shap_sign = shap_vals[:, feat_idx]
    feat_vals = X_df[top_feat].values

    interaction_col = st.selectbox(
        "Colour interaction by",
        [f for f in feat_cols if f != top_feat],
        key="shap_dep_color",
    )
    color_vals = X_df[interaction_col].values

    fig4 = px.scatter(
        x=feat_vals, y=shap_sign,
        color=color_vals,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=float(np.median(color_vals)),
        opacity=0.7, template="plotly_dark",
        labels={"x": top_feat, "y": "SHAP value", "color": interaction_col},
        title=f"SHAP dependency for top feature: {top_feat}",
    )
    fig4.add_hline(y=0, line_dash="dot", line_color="white")
    st.plotly_chart(fig4, use_container_width=True)
