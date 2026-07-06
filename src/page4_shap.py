"""Model explanation views using SHAP."""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import LINEAR_MODELS


def render(bundle: Dict[str, Any]) -> None:
    st.title("Explain Model")
    st.info("SHAP estimates how much each feature contributes to a model prediction.")

    try:
        import shap
    except ImportError:
        st.error("Install SHAP to use this page: `pip install shap`.")
        st.stop()

    trained = bundle["trained"]
    x_train = bundle["X_train"]
    x_test = bundle["X_test"]
    x_train_scaled = bundle["X_train_s"]
    x_test_scaled = bundle["X_test_s"]
    feature_columns = bundle["feat_cols"]

    model_names = list(trained.keys())

    c1, c2, c3 = st.columns(3)
    with c1:
        model_choice = st.selectbox("Saved model", model_names, key="shap_model")
    with c2:
        sample_count = st.slider("Samples to explain", 50, 300, 100, 25, key="shap_n")
    with c3:
        plot_type = st.selectbox(
            "Plot type",
            ["Mean impact", "Beeswarm", "Waterfall"],
            key="shap_plot",
        )

    is_linear = model_choice in LINEAR_MODELS
    model = trained[model_choice]

    with st.spinner(f"Computing SHAP values for {model_choice}"):
        if is_linear:
            x_background = x_train_scaled[:200]
            x_explain = x_test_scaled[:sample_count]
            explainer = shap.LinearExplainer(model, x_background)
            shap_values = explainer.shap_values(x_explain)
            x_df = pd.DataFrame(x_explain, columns=feature_columns)
            expected_value = explainer.expected_value
        else:
            x_explain = x_test.values[:sample_count]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_explain)
            x_df = pd.DataFrame(x_explain, columns=feature_columns)
            expected_value = explainer.expected_value

        base_value = (
            float(expected_value[0])
            if isinstance(expected_value, (list, np.ndarray))
            else float(expected_value)
        )

    if plot_type == "Mean impact":
        top_n = st.slider("Top features", 5, len(feature_columns), 20, key="shap_topn")
        shap_df = pd.DataFrame(np.abs(shap_values), columns=feature_columns)
        mean_shap = shap_df.mean().sort_values(ascending=False).head(top_n)

        fig = px.bar(
            x=mean_shap.values,
            y=mean_shap.index,
            orientation="h",
            color=mean_shap.values,
            color_continuous_scale="Blues",
            labels={"x": "Mean absolute SHAP value", "y": "Feature"},
            title=f"{model_choice}: top feature impact",
            template="plotly_dark",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed", dtick=1),
            height=300 + (top_n * 25),
            margin=dict(l=150),
        )
        st.plotly_chart(fig, width="stretch")

    elif plot_type == "Beeswarm":
        max_display = st.slider("Max features displayed", 5, 25, 15, key="shap_beeswarm_n")
        fig2, _ = plt.subplots(figsize=(10, 7))
        fig2.patch.set_facecolor("#0f172a")
        shap.summary_plot(
            shap_values,
            x_df,
            plot_type="dot",
            show=False,
            max_display=max_display,
        )
        plt.tight_layout()
        st.pyplot(fig2)

    elif plot_type == "Waterfall":
        sample_index = st.slider(
            "Test sample index",
            0,
            sample_count - 1,
            0,
            key="shap_idx",
        )
        explanation = shap.Explanation(
            values=shap_values[int(sample_index)],
            base_values=base_value,
            data=x_df.iloc[int(sample_index)].values,
            feature_names=feature_columns,
        )
        fig3, _ = plt.subplots(figsize=(10, 6))
        fig3.patch.set_facecolor("#0f172a")
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        st.pyplot(fig3)

    st.divider()
    st.subheader("Top feature dependency")
    shap_df = pd.DataFrame(np.abs(shap_values), columns=feature_columns)
    top_feature = shap_df.mean().idxmax()
    feature_index = feature_columns.index(top_feature)
    shap_sign = shap_values[:, feature_index]
    feature_values = x_df[top_feature].values

    interaction_column = st.selectbox(
        "Color by",
        [feature for feature in feature_columns if feature != top_feature],
        key="shap_dep_color",
    )
    color_values = x_df[interaction_column].values

    fig4 = px.scatter(
        x=feature_values,
        y=shap_sign,
        color=color_values,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=float(np.median(color_values)),
        opacity=0.7,
        template="plotly_dark",
        labels={"x": top_feature, "y": "SHAP value", "color": interaction_column},
        title=f"Dependency for top feature: {top_feature}",
    )
    fig4.add_hline(y=0, line_dash="dot", line_color="white")
    st.plotly_chart(fig4, width="stretch")
