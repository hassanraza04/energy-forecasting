"""
src/page2_eda.py
Page 2 — Exploratory Data Analysis (fully dynamic)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import TARGET_COL, RANDOM_STATE


def render(df: pd.DataFrame, num_cols: List[str]) -> None:
    st.title("📊 Exploratory Data Analysis")

    # ── Sidebar section selector ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📊 EDA Controls")
        chart_type = st.selectbox(
            "Chart section",
            ["Distribution", "Time Series", "By Time Period", "Scatter", "Correlation Heatmap"],
            key="eda_chart_type",
        )

    # ── 1. Distribution ────────────────────────────────────────────────────────
    if chart_type == "Distribution":
        st.subheader("Feature Distribution")
        all_feats = [TARGET_COL] + num_cols
        col_pick  = st.selectbox("Feature", all_feats, key="dist_feat")
        col1, col2 = st.columns([3, 1])
        with col2:
            nbins    = st.slider("Bins", 10, 150, 80, key="dist_bins")
            log_y    = st.checkbox("Log Y-axis", False, key="dist_logy")
            show_box = st.checkbox("Marginal box plot", True, key="dist_box")
        with col1:
            fig = px.histogram(
                df, x=col_pick, nbins=nbins,
                marginal="box" if show_box else None,
                color_discrete_sequence=["#38bdf8"],
                labels={col_pick: col_pick},
                template="plotly_dark", log_y=log_y,
            )
            fig.update_layout(bargap=0.03)
            st.plotly_chart(fig, use_container_width=True)

        if pd.api.types.is_numeric_dtype(df[col_pick]):
            s = df[col_pick].describe()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean",   f"{s['mean']:.2f}")
            m2.metric("Std",    f"{s['std']:.2f}")
            m3.metric("Min",    f"{s['min']:.0f}")
            m4.metric("Max",    f"{s['max']:.0f}")

    # ── 2. Time Series ─────────────────────────────────────────────────────────
    elif chart_type == "Time Series":
        st.subheader("Energy Consumption Over Time")
        c1, c2, c3 = st.columns(3)
        with c1:
            feature  = st.selectbox("Feature", [TARGET_COL] + num_cols, key="ts_feat")
        with c2:
            freq     = st.selectbox("Resample frequency", ["10min", "H", "D", "W"],
                                    index=1, key="ts_freq")
        with c3:
            agg_func = st.selectbox("Aggregation", ["mean", "median", "max", "min", "sum"],
                                    key="ts_agg")

        ts = df.set_index("date")[feature].resample(freq).agg(agg_func).reset_index()
        fig = px.line(
            ts, x="date", y=feature,
            color_discrete_sequence=["#f472b6"],
            labels={"date": "Time", feature: f"{agg_func.title()} {feature}"},
            template="plotly_dark",
        )
        fig.update_traces(line=dict(width=1.2))
        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show rolling average overlay", True, key="ts_rolling_toggle"):
            window = st.slider("Rolling window (periods)", 2, 48, 12, key="ts_window")
            ts["rolling"] = ts[feature].rolling(window).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=ts["date"], y=ts[feature],
                                      mode="lines", name=feature,
                                      line=dict(color="#f472b6", width=1)))
            fig2.add_trace(go.Scatter(x=ts["date"], y=ts["rolling"],
                                      mode="lines", name=f"Rolling {window}",
                                      line=dict(color="#fbbf24", width=2)))
            fig2.update_layout(template="plotly_dark",
                               xaxis_title="Time", yaxis_title=feature)
            st.plotly_chart(fig2, use_container_width=True)

    # ── 3. By Time Period ──────────────────────────────────────────────────────
    elif chart_type == "By Time Period":
        st.subheader("Average Consumption by Time Period")
        c1, c2, c3 = st.columns(3)
        with c1:
            feature  = st.selectbox("Feature", [TARGET_COL] + num_cols, key="bp_feat")
        with c2:
            group_by = st.selectbox(
                "Group by",
                ["hour", "day_of_week", "month", "is_weekend", "part_of_day"],
                key="bp_group",
            )
        with c3:
            agg_func = st.selectbox("Aggregation", ["mean", "median", "max", "sum"],
                                    key="bp_agg")

        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
                   4: "Fri", 5: "Sat", 6: "Sun"}
        grp = df.groupby(group_by)[feature].agg(agg_func).reset_index()
        if group_by == "day_of_week":
            grp["label"] = grp[group_by].map(day_map)
            x_col = "label"
        else:
            x_col = group_by

        fig = px.bar(
            grp, x=x_col, y=feature,
            color=feature, color_continuous_scale="Blues",
            labels={feature: f"{agg_func.title()} {feature}"},
            template="plotly_dark", text_auto=".0f",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── 4. Scatter ─────────────────────────────────────────────────────────────
    elif chart_type == "Scatter":
        st.subheader("Feature vs Feature Scatter")
        all_num = [TARGET_COL] + num_cols
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x_feat   = st.selectbox("X axis", all_num,
                                    index=all_num.index("T2") if "T2" in all_num else 1,
                                    key="sc_x")
        with c2:
            y_feat   = st.selectbox("Y axis", all_num, index=0, key="sc_y")
        with c3:
            color_by = st.selectbox(
                "Colour by",
                ["hour", "day_of_week", "month", "is_weekend", "part_of_day"],
                key="sc_color",
            )
        with c4:
            n_pts = st.slider("Sample points", 500, len(df), 2000, 500, key="sc_pts")

        sample = df.sample(n_pts, random_state=RANDOM_STATE)
        fig = px.scatter(
            sample, x=x_feat, y=y_feat,
            color=color_by, opacity=0.6,
            color_continuous_scale="Viridis",
            labels={x_feat: x_feat, y_feat: y_feat},
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Add OLS trend line", key="sc_ols"):
            fig2 = px.scatter(
                sample, x=x_feat, y=y_feat,
                color=color_by, opacity=0.5,
                trendline="ols", trendline_color_override="#f472b6",
                color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── 5. Correlation Heatmap ─────────────────────────────────────────────────
    elif chart_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        c1, c2 = st.columns([3, 1])
        with c2:
            method    = st.radio("Method", ["pearson", "spearman", "kendall"],
                                 key="corr_method")
            show_vals = st.checkbox("Show values", False, key="corr_annot")
            n_top     = st.slider("Top N features", 5, 30, 15, key="corr_ntop")

        num_df   = df[[TARGET_COL] + num_cols]
        corr     = num_df.corr(method=method)
        top_feats = (
            corr[TARGET_COL].abs()
            .sort_values(ascending=False)
            .head(n_top)
            .index.tolist()
        )
        corr_sub = num_df[top_feats].corr(method=method)

        with c1:
            figc, ax = plt.subplots(figsize=(14, 9))
            figc.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            sns.heatmap(
                corr_sub, ax=ax, cmap="coolwarm",
                annot=show_vals, fmt=".2f" if show_vals else "",
                linewidths=0.3, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7},
            )
            ax.tick_params(colors="#e2e8f0", labelsize=7)
            plt.tight_layout()
            st.pyplot(figc)

        st.subheader(f"Correlation with {TARGET_COL}")
        corr_target = corr[TARGET_COL].drop(TARGET_COL).sort_values()
        fig = px.bar(
            x=corr_target.values, y=corr_target.index,
            orientation="h",
            color=corr_target.values,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"x": f"{method.title()} r", "y": "Feature"},
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
