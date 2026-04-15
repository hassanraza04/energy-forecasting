"""
src/page3_predictions.py
Page 3 — Model Predictions (dynamic)
Views:
  • 🎯 Live Prediction  — user inputs conditions → instant forecast
  • 📋 Leaderboard      — ranked model comparison
  • 🔬 Inspect Model    — detailed per-model diagnostics
  • 📊 Compare Models   — side-by-side overlay
"""
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import ALL_MODELS, LINEAR_MODELS, TARGET_COL, get_data


# ── Key features exposed in the live prediction form ─────────────────────────
# (others are filled with dataset means)
FORM_FEATURES = {
    # name          label                          min    max   step  default
    "lights":    ("💡 Lights energy (Wh)",          0.0,   70.0,  1.0,  0.0),
    "T2":        ("🍳 Kitchen temp (°C)",          14.0,   26.0,  0.5, 20.0),
    "T6":        ("🏠 Outside-N bldg temp (°C)",   -5.0,   28.0,  0.5,  7.0),
    "T_out":     ("🌡️ Outdoor temp (°C)",          -5.0,   28.0,  0.5,  6.0),
    "RH_2":      ("💧 Kitchen humidity (%)",       20.0,   60.0,  1.0, 40.0),
    "RH_out":    ("🌧️ Outdoor humidity (%)",       20.0,  100.0,  1.0, 75.0),
    "Windspeed": ("💨 Wind speed (m/s)",             0.0,   14.0,  0.5,  4.0),
    "Visibility":("👁️ Visibility (km)",             1.0,   66.0,  1.0, 40.0),
    "hour":      ("🕐 Hour of day",                  0,     23,    1,   12),
    "month":     ("📆 Month",                         1,     12,    1,    1),
    "is_weekend":("🏖️ Weekend? (0 = No, 1 = Yes)",  0,      1,    1,    0),
}


def render(bundle: Dict[str, Any]) -> None:
    st.title("🤖 Model Predictions")

    results   = bundle["results"]
    feat_cols = bundle["feat_cols"]
    trained   = bundle["trained"]
    scaler    = bundle["scaler"]

    # Leaderboard DataFrame (reused across views)
    leaderboard = pd.DataFrame([
        {"Model": name, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]}
        for name, v in results.items()
    ]).sort_values("R²", ascending=False).reset_index(drop=True)

    # ── View toggle ────────────────────────────────────────────────────────────
    view = st.radio(
        "View",
        ["🎯 Live Prediction", "📋 Leaderboard", "🔬 Inspect Model", "📊 Compare Models"],
        horizontal=True,
        key="pred_view",
    )
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # 🎯 LIVE PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    if view == "🎯 Live Prediction":
        st.subheader("🎯 Predict Appliance Energy Consumption")
        st.write(
            "Adjust the environmental and time conditions below, select a model, "
            "and get an instant **energy consumption forecast** for your building."
        )

        # Load dataset means to fill non-exposed features
        df_raw = get_data()
        feat_means = {col: float(df_raw[col].mean()) for col in feat_cols
                      if col in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[col])}

        # ── Input form ────────────────────────────────────────────────────────
        with st.form("prediction_form"):
            st.markdown("#### 🏠 Building & Environmental Conditions")
            form_vals: Dict[str, float] = {}

            # Render inputs in a 3-column grid
            form_keys = list(FORM_FEATURES.keys())
            rows = [form_keys[i:i+3] for i in range(0, len(form_keys), 3)]
            for row in rows:
                cols = st.columns(len(row))
                for col_widget, feat in zip(cols, row):
                    label, mn, mx, step, default = FORM_FEATURES[feat]
                    with col_widget:
                        if isinstance(step, int):
                            form_vals[feat] = float(st.slider(
                                label, int(mn), int(mx), int(default), step,
                                key=f"live_{feat}",
                            ))
                        else:
                            form_vals[feat] = st.slider(
                                label, mn, mx, default, step,
                                key=f"live_{feat}",
                            )

            st.markdown("#### 🤖 Select Model")
            model_col, _ = st.columns([1, 2])
            with model_col:
                live_model = st.selectbox(
                    "Model", ALL_MODELS, key="live_model_select"
                )

            submitted = st.form_submit_button("⚡ Predict Energy Consumption",
                                              type="primary",
                                              use_container_width=True)

        # ── Prediction ────────────────────────────────────────────────────────
        if submitted:
            # Build feature vector: exposed features + dataset means for rest
            input_vec = np.array([
                form_vals.get(f, feat_means.get(f, 0.0))
                for f in feat_cols
            ]).reshape(1, -1)

            model     = trained[live_model]
            is_linear = live_model in LINEAR_MODELS

            if is_linear:
                input_scaled = scaler.transform(input_vec)
                prediction   = float(model.predict(input_scaled)[0])
            else:
                prediction = float(model.predict(input_vec)[0])

            prediction = max(0.0, prediction)   # energy can't be negative

            # ── Result display ────────────────────────────────────────────────
            st.markdown("---")
            avg_energy  = float(df_raw[TARGET_COL].mean())
            delta_pct   = (prediction - avg_energy) / avg_energy * 100
            delta_label = f"{delta_pct:+.1f}% vs dataset avg"

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(
                label=f"⚡ Predicted Energy ({live_model})",
                value=f"{prediction:.1f} Wh",
                delta=delta_label,
                delta_color="inverse",
            )
            res_col2.metric("📊 Dataset Average", f"{avg_energy:.1f} Wh")
            res_col3.metric(
                "📈 Model R²",
                f"{results[live_model]['R2']:.4f}",
                help="Higher is better (max 1.0)",
            )

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={"reference": avg_energy, "valueformat": ".1f"},
                title={"text": "Predicted Appliance Energy (Wh)", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, float(df_raw[TARGET_COL].quantile(0.99))],
                             "tickcolor": "#e2e8f0"},
                    "bar":  {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0, avg_energy * 0.75],             "color": "#064e3b"},
                        {"range": [avg_energy * 0.75, avg_energy * 1.25], "color": "#854d0e"},
                        {"range": [avg_energy * 1.25, float(df_raw[TARGET_COL].quantile(0.99))],
                         "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line": {"color": "#f472b6", "width": 3},
                        "thickness": 0.75,
                        "value": avg_energy,
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

            # Business interpretation
            if prediction < avg_energy * 0.75:
                st.success("🟢 **Low consumption** — Good time for energy-intensive tasks outside peak hours.")
            elif prediction < avg_energy * 1.25:
                st.info("🟡 **Moderate consumption** — Typical usage; monitor closely for cost management.")
            else:
                st.warning("🔴 **High consumption** — Consider deferring non-essential appliance loads to reduce costs.")

            # All-model comparison for same input
            st.markdown("---")
            st.subheader("All Models — Predictions for Same Conditions")
            all_preds = []
            for mname in ALL_MODELS:
                mod = trained[mname]
                if mname in LINEAR_MODELS:
                    p = float(mod.predict(scaler.transform(input_vec))[0])
                else:
                    p = float(mod.predict(input_vec)[0])
                all_preds.append({"Model": mname, "Predicted (Wh)": max(0.0, p)})

            pred_df = pd.DataFrame(all_preds)
            fig_all = px.bar(
                pred_df, x="Model", y="Predicted (Wh)",
                color="Predicted (Wh)", color_continuous_scale="Blues",
                template="plotly_dark", text_auto=".1f",
                title="Model Predictions for Current Input",
            )
            fig_all.add_hline(y=avg_energy, line_dash="dash", line_color="#f472b6",
                              annotation_text=f"Dataset avg: {avg_energy:.0f} Wh",
                              annotation_font_color="#f472b6")
            fig_all.update_traces(textposition="outside")
            st.plotly_chart(fig_all, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 📋 LEADERBOARD
    # ══════════════════════════════════════════════════════════════════════════
    elif view == "📋 Leaderboard":
        sort_by   = st.selectbox("Sort by", ["R²", "MAE", "RMSE"], key="lb_sort")
        ascending = sort_by in ["MAE", "RMSE"]
        lb = leaderboard.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        st.dataframe(
            lb.style
              .highlight_max(subset=["R²"],           color="#064e3b")
              .highlight_min(subset=["MAE", "RMSE"],  color="#064e3b")
              .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"}),
            width="stretch",
        )

        metric    = st.selectbox("Metric to visualise", ["R²", "MAE", "RMSE"], key="lb_vis")
        asc_bar   = metric in ["MAE", "RMSE"]
        lb_sorted = lb.sort_values(metric, ascending=asc_bar)

        fig = px.bar(
            lb_sorted, x="Model", y=metric,
            color=metric,
            color_continuous_scale="teal" if metric == "R²" else "Reds",
            template="plotly_dark", text_auto=".3f",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 🔬 INSPECT MODEL
    # ══════════════════════════════════════════════════════════════════════════
    elif view == "🔬 Inspect Model":
        chosen = st.selectbox("Model to inspect", ALL_MODELS, key="ins_model")
        r      = results[chosen]
        y_true = r["y_test"]
        y_pred = r["preds"]

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  f"{r['MAE']:.2f} Wh")
        c2.metric("RMSE", f"{r['RMSE']:.2f} Wh")
        c3.metric("R²",   f"{r['R2']:.4f}")

        tab_avp, tab_res, tab_err = st.tabs(
            ["Actual vs Predicted", "Residuals", "Error Analysis"]
        )

        with tab_avp:
            n_pts = st.slider("Points to show", 100, len(y_true),
                              min(500, len(y_true)), 100, key="avp_pts")
            fig = px.scatter(
                x=y_true[:n_pts], y=y_pred[:n_pts],
                labels={"x": "Actual (Wh)", "y": "Predicted (Wh)"},
                title=f"{chosen} — Actual vs Predicted",
                opacity=0.6, color_discrete_sequence=["#38bdf8"],
                template="plotly_dark",
            )
            mn, mx = float(y_true.min()), float(y_true.max())
            fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                         line=dict(color="#f472b6", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

        with tab_res:
            residuals = y_true - y_pred
            col1, col2 = st.columns(2)
            with col1:
                nbins = st.slider("Histogram bins", 20, 120, 60, key="res_bins")
                fig2  = px.histogram(
                    x=residuals, nbins=nbins,
                    labels={"x": "Residual (Wh)", "y": "Count"},
                    color_discrete_sequence=["#a78bfa"],
                    template="plotly_dark",
                )
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = px.scatter(
                    x=y_pred, y=residuals,
                    labels={"x": "Predicted (Wh)", "y": "Residual (Wh)"},
                    opacity=0.5, color_discrete_sequence=["#f472b6"],
                    template="plotly_dark",
                )
                fig3.add_hline(y=0, line_dash="dash", line_color="white")
                st.plotly_chart(fig3, use_container_width=True)

        with tab_err:
            abs_err = np.abs(y_true - y_pred)
            pct_err = abs_err / (y_true + 1e-9) * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Abs Error",   f"{abs_err.mean():.2f} Wh")
            c2.metric("Median Abs Error", f"{np.median(abs_err):.2f} Wh")
            c3.metric("Mean Pct Error",   f"{pct_err.mean():.1f}%")

            fig4 = px.histogram(
                x=pct_err, nbins=60,
                labels={"x": "Percentage Error (%)", "y": "Count"},
                color_discrete_sequence=["#34d399"],
                template="plotly_dark",
                title="Percentage Error Distribution",
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 📊 COMPARE MODELS
    # ══════════════════════════════════════════════════════════════════════════
    elif view == "📊 Compare Models":
        selected = st.multiselect(
            "Models to compare", ALL_MODELS, default=ALL_MODELS, key="comp_models"
        )
        if not selected:
            st.warning("Select at least one model.")
            return

        metric    = st.radio("Metric", ["R²", "MAE", "RMSE"], horizontal=True, key="comp_metric")
        sub_lb    = leaderboard[leaderboard["Model"].isin(selected)]
        asc_bar   = metric in ["MAE", "RMSE"]
        sub_sorted = sub_lb.sort_values(metric, ascending=asc_bar)

        fig = px.bar(sub_sorted, x="Model", y=metric,
                     color="Model", template="plotly_dark", text_auto=".3f")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predictions Overlay (first 300 test points)")
        palette = ["#38bdf8", "#f472b6", "#a78bfa", "#34d399", "#fbbf24"]
        fig2    = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(300)),
            y=results[selected[0]]["y_test"][:300].tolist(),
            mode="lines", name="Actual",
            line=dict(color="white", width=1.5),
        ))
        for i, m in enumerate(selected):
            fig2.add_trace(go.Scatter(
                x=list(range(300)),
                y=results[m]["preds"][:300].tolist(),
                mode="lines", name=m,
                line=dict(color=palette[i % len(palette)], width=1),
            ))
        fig2.update_layout(
            template="plotly_dark",
            xaxis_title="Test sample index",
            yaxis_title="Energy (Wh)",
        )
        st.plotly_chart(fig2, use_container_width=True)
