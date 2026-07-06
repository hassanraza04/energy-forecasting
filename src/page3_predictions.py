"""Forecast and model comparison views."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import LINEAR_MODELS
from src.modeling import build_leaderboard
from src.prediction import (
    build_prediction_vector,
    clip_energy_prediction,
    describe_prediction,
)


FORM_FEATURES = {
    "lights": ("Lights energy (Wh)", 0.0, 70.0, 1.0, 0.0),
    "T2": ("Kitchen temperature (C)", 14.0, 26.0, 0.5, 20.0),
    "T6": ("North side outdoor temperature (C)", -5.0, 28.0, 0.5, 7.0),
    "T_out": ("Outdoor temperature (C)", -5.0, 28.0, 0.5, 6.0),
    "RH_2": ("Kitchen humidity (%)", 20.0, 60.0, 1.0, 40.0),
    "RH_out": ("Outdoor humidity (%)", 20.0, 100.0, 1.0, 75.0),
    "Windspeed": ("Wind speed (m/s)", 0.0, 14.0, 0.5, 4.0),
    "Visibility": ("Visibility (km)", 1.0, 66.0, 1.0, 40.0),
    "hour": ("Hour of day", 0, 23, 1, 12),
    "month": ("Month", 1, 12, 1, 1),
    "is_weekend": ("Weekend flag", 0, 1, 1, 0),
}


def _predict_for_model(
    model_name: str,
    input_vector: np.ndarray,
    trained: dict[str, Any],
    scaler: Any,
) -> float:
    model = trained[model_name]
    if model_name in LINEAR_MODELS:
        prediction = float(model.predict(scaler.transform(input_vector))[0])
    else:
        prediction = float(model.predict(input_vector)[0])
    return clip_energy_prediction(prediction)


def render(bundle: Dict[str, Any]) -> None:
    st.title("Forecast")
    st.caption("Run fast predictions with models trained offline and loaded from saved artifacts.")

    results = bundle["results"]
    feature_columns = bundle["feat_cols"]
    trained = bundle["trained"]
    scaler = bundle["scaler"]
    best_model = bundle.get("best_model", "Random Forest")
    leaderboard = build_leaderboard(results)

    view = st.radio(
        "View",
        ["Live forecast", "Model ranking", "Inspect saved model"],
        horizontal=True,
        key="pred_view",
    )
    st.divider()

    if view == "Live forecast":
        st.subheader("Live forecast")
        st.write(
            "Adjust the conditions below. The app builds one input row and sends it "
            "through the saved model. It does not retrain on this page."
        )

        feature_means = bundle["feature_means"]

        with st.form("prediction_form"):
            st.markdown("#### Conditions")
            form_values: Dict[str, float] = {}

            form_keys = list(FORM_FEATURES.keys())
            rows = [form_keys[index:index + 3] for index in range(0, len(form_keys), 3)]
            for row in rows:
                cols = st.columns(len(row))
                for col_widget, feature in zip(cols, row):
                    label, minimum, maximum, step, default = FORM_FEATURES[feature]
                    with col_widget:
                        if isinstance(step, int):
                            form_values[feature] = float(st.slider(
                                label,
                                int(minimum),
                                int(maximum),
                                int(default),
                                step,
                                key=f"live_{feature}",
                            ))
                        else:
                            form_values[feature] = st.slider(
                                label,
                                minimum,
                                maximum,
                                default,
                                step,
                                key=f"live_{feature}",
                            )

            live_model = best_model
            st.caption(f"Using saved production model: {live_model}")

            submitted = st.form_submit_button(
                "Run forecast",
                type="primary",
                width="stretch",
            )

        if submitted:
            input_vector = build_prediction_vector(feature_columns, form_values, feature_means)
            prediction = _predict_for_model(live_model, input_vector, trained, scaler)

            st.divider()
            average_energy = float(bundle["target_average"])
            delta_pct = (prediction - average_energy) / average_energy * 100
            delta_label = f"{delta_pct:+.1f}% vs dataset average"
            description = describe_prediction(prediction, average_energy)

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(
                label=f"Predicted energy with {live_model}",
                value=f"{prediction:.1f} Wh",
                delta=delta_label,
                delta_color="inverse",
            )
            res_col2.metric("Dataset average", f"{average_energy:.1f} Wh")
            res_col3.metric("Model R2", f"{results[live_model]['R2']:.4f}")

            st.markdown(f"#### {description['level']}")
            st.write(description["message"])

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={"reference": average_energy, "valueformat": ".1f"},
                title={"text": "Predicted appliance energy", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {
                        "range": [0, float(bundle["target_q99"])],
                        "tickcolor": "#e2e8f0",
                    },
                    "bar": {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0, average_energy * 0.75], "color": "#064e3b"},
                        {
                            "range": [average_energy * 0.75, average_energy * 1.25],
                            "color": "#854d0e",
                        },
                        {
                            "range": [
                                average_energy * 1.25,
                                float(bundle["target_q99"]),
                            ],
                            "color": "#7f1d1d",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "#38bdf8", "width": 3},
                        "thickness": 0.75,
                        "value": average_energy,
                    },
                },
                number={"suffix": " Wh", "font": {"color": "#e2e8f0"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0f172a",
                font={"color": "#e2e8f0"},
                height=320,
            )
            st.plotly_chart(fig_gauge, width="stretch")

            st.divider()
            st.subheader("Why the result changed")
            st.write(
                "Each slider value becomes one feature in the input row. The saved model "
                "uses the relationships learned during offline training to estimate the "
                "target value for that row."
            )

    elif view == "Model ranking":
        st.write(
            "These scores were calculated during offline training and saved with the model bundle."
        )
        sort_by = st.selectbox("Sort by", ["R2", "MAE", "RMSE"], key="lb_sort")
        ascending = sort_by in ["MAE", "RMSE"]
        ranked = leaderboard.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        st.dataframe(
            ranked.style
            .highlight_max(subset=["R2"], color="#064e3b")
            .highlight_min(subset=["MAE", "RMSE"], color="#064e3b")
            .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}),
            width="stretch",
        )

        metric = st.selectbox("Metric to chart", ["R2", "MAE", "RMSE"], key="lb_vis")
        chart_sorted = ranked.sort_values(metric, ascending=(metric != "R2"))
        fig = px.bar(
            chart_sorted,
            x="Model",
            y=metric,
            color=metric,
            color_continuous_scale="teal" if metric == "R2" else "Reds",
            template="plotly_dark",
            text_auto=".3f",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

    elif view == "Inspect saved model":
        chosen = st.selectbox(
            "Model to inspect",
            list(trained.keys()),
            index=0,
            key="ins_model",
        )
        result = results[chosen]
        y_true = result["y_test"]
        y_pred = result["preds"]

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{result['MAE']:.2f} Wh")
        c2.metric("RMSE", f"{result['RMSE']:.2f} Wh")
        c3.metric("R2", f"{result['R2']:.4f}")

        actual_tab, residual_tab, error_tab = st.tabs(
            ["Actual vs predicted", "Residuals", "Error analysis"]
        )

        with actual_tab:
            point_count = st.slider(
                "Points to show",
                100,
                len(y_true),
                min(500, len(y_true)),
                100,
                key="avp_pts",
            )
            fig = px.scatter(
                x=y_true[:point_count],
                y=y_pred[:point_count],
                labels={"x": "Actual (Wh)", "y": "Predicted (Wh)"},
                title=f"{chosen}: actual vs predicted",
                opacity=0.6,
                color_discrete_sequence=["#38bdf8"],
                template="plotly_dark",
            )
            minimum, maximum = float(y_true.min()), float(y_true.max())
            fig.add_shape(
                type="line",
                x0=minimum,
                y0=minimum,
                x1=maximum,
                y1=maximum,
                line=dict(color="#e2e8f0", dash="dash"),
            )
            st.plotly_chart(fig, width="stretch")

        with residual_tab:
            residuals = y_true - y_pred
            col1, col2 = st.columns(2)
            with col1:
                bins = st.slider("Histogram bins", 20, 120, 60, key="res_bins")
                fig2 = px.histogram(
                    x=residuals,
                    nbins=bins,
                    labels={"x": "Residual (Wh)", "y": "Count"},
                    color_discrete_sequence=["#38bdf8"],
                    template="plotly_dark",
                )
                st.plotly_chart(fig2, width="stretch")
            with col2:
                fig3 = px.scatter(
                    x=y_pred,
                    y=residuals,
                    labels={"x": "Predicted (Wh)", "y": "Residual (Wh)"},
                    opacity=0.5,
                    color_discrete_sequence=["#38bdf8"],
                    template="plotly_dark",
                )
                fig3.add_hline(y=0, line_dash="dash", line_color="white")
                st.plotly_chart(fig3, width="stretch")

        with error_tab:
            abs_error = np.abs(y_true - y_pred)
            pct_error = abs_error / (y_true + 1e-9) * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean abs error", f"{abs_error.mean():.2f} Wh")
            c2.metric("Median abs error", f"{np.median(abs_error):.2f} Wh")
            c3.metric("Mean pct error", f"{pct_error.mean():.1f}%")

            fig4 = px.histogram(
                x=pct_error,
                nbins=60,
                labels={"x": "Percentage error (%)", "y": "Count"},
                color_discrete_sequence=["#38bdf8"],
                template="plotly_dark",
                title="Percentage error distribution",
            )
            st.plotly_chart(fig4, width="stretch")
