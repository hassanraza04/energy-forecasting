"""Forecast and model comparison views."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import ALL_MODELS, LINEAR_MODELS, TARGET_COL, get_data
from src.modeling import build_leaderboard
from src.prediction import build_prediction_vector, clip_energy_prediction


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
    st.caption("Test one set of conditions across the trained regression models.")

    results = bundle["results"]
    feature_columns = bundle["feat_cols"]
    trained = bundle["trained"]
    scaler = bundle["scaler"]
    leaderboard = build_leaderboard(results)

    view = st.radio(
        "View",
        ["Live forecast", "Model ranking", "Inspect model", "Compare models"],
        horizontal=True,
        key="pred_view",
    )
    st.divider()

    if view == "Live forecast":
        st.subheader("Live forecast")
        st.write(
            "Adjust the conditions below. Features not shown in the form are filled "
            "with their dataset averages."
        )

        df_raw = get_data()
        feature_means = {
            column: float(df_raw[column].mean())
            for column in feature_columns
            if column in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[column])
        }

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

            model_col, _ = st.columns([1, 2])
            with model_col:
                live_model = st.selectbox("Model", ALL_MODELS, key="live_model_select")

            submitted = st.form_submit_button(
                "Run forecast",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            input_vector = build_prediction_vector(feature_columns, form_values, feature_means)
            prediction = _predict_for_model(live_model, input_vector, trained, scaler)

            st.divider()
            average_energy = float(df_raw[TARGET_COL].mean())
            delta_pct = (prediction - average_energy) / average_energy * 100
            delta_label = f"{delta_pct:+.1f}% vs dataset average"

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(
                label=f"Predicted energy with {live_model}",
                value=f"{prediction:.1f} Wh",
                delta=delta_label,
                delta_color="inverse",
            )
            res_col2.metric("Dataset average", f"{average_energy:.1f} Wh")
            res_col3.metric("Model R2", f"{results[live_model]['R2']:.4f}")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={"reference": average_energy, "valueformat": ".1f"},
                title={"text": "Predicted appliance energy", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {
                        "range": [0, float(df_raw[TARGET_COL].quantile(0.99))],
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
                                float(df_raw[TARGET_COL].quantile(0.99)),
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
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prediction < average_energy * 0.75:
                st.success("This estimate is below the dataset average.")
            elif prediction < average_energy * 1.25:
                st.info("This estimate is close to the dataset average.")
            else:
                st.warning("This estimate is above the dataset average.")

            st.divider()
            st.subheader("Same input across all models")
            all_predictions = [
                {
                    "Model": model_name,
                    "Predicted energy (Wh)": _predict_for_model(
                        model_name,
                        input_vector,
                        trained,
                        scaler,
                    ),
                }
                for model_name in ALL_MODELS
            ]
            prediction_df = pd.DataFrame(all_predictions)
            fig_all = px.bar(
                prediction_df,
                x="Model",
                y="Predicted energy (Wh)",
                color="Predicted energy (Wh)",
                color_continuous_scale="Blues",
                template="plotly_dark",
                text_auto=".1f",
                title="Forecast by model",
            )
            fig_all.add_hline(
                y=average_energy,
                line_dash="dash",
                line_color="#38bdf8",
                annotation_text=f"Dataset average: {average_energy:.0f} Wh",
                annotation_font_color="#38bdf8",
            )
            fig_all.update_traces(textposition="outside")
            st.plotly_chart(fig_all, use_container_width=True)

    elif view == "Model ranking":
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
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Inspect model":
        chosen = st.selectbox("Model to inspect", ALL_MODELS, key="ins_model")
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
            st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig2, use_container_width=True)
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
                st.plotly_chart(fig3, use_container_width=True)

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
            st.plotly_chart(fig4, use_container_width=True)

    elif view == "Compare models":
        selected = st.multiselect(
            "Models to compare",
            ALL_MODELS,
            default=ALL_MODELS,
            key="comp_models",
        )
        if not selected:
            st.warning("Select at least one model.")
            return

        metric = st.radio("Metric", ["R2", "MAE", "RMSE"], horizontal=True, key="comp_metric")
        selected_leaderboard = leaderboard[leaderboard["Model"].isin(selected)]
        sorted_leaderboard = selected_leaderboard.sort_values(
            metric,
            ascending=(metric != "R2"),
        )

        fig = px.bar(
            sorted_leaderboard,
            x="Model",
            y=metric,
            color="Model",
            template="plotly_dark",
            text_auto=".3f",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction overlay")
        palette = ["#38bdf8", "#f472b6", "#a78bfa", "#34d399", "#fbbf24"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(300)),
            y=results[selected[0]]["y_test"][:300].tolist(),
            mode="lines",
            name="Actual",
            line=dict(color="white", width=1.5),
        ))
        for index, model_name in enumerate(selected):
            fig2.add_trace(go.Scatter(
                x=list(range(300)),
                y=results[model_name]["preds"][:300].tolist(),
                mode="lines",
                name=model_name,
                line=dict(color=palette[index % len(palette)], width=1),
            ))
        fig2.update_layout(
            template="plotly_dark",
            xaxis_title="Test sample index",
            yaxis_title="Energy (Wh)",
        )
        st.plotly_chart(fig2, use_container_width=True)
