"""Exploratory data views."""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from src.data_loader import RANDOM_STATE, TARGET_COL


def render(df: pd.DataFrame, num_cols: List[str]) -> None:
    st.title("Explore Data")
    st.caption("Inspect distributions, time patterns, feature relationships, and correlations.")

    with st.sidebar:
        st.markdown("### Data controls")
        chart_type = st.selectbox(
            "View",
            ["Distribution", "Time series", "Time period", "Scatter", "Correlation"],
            key="eda_chart_type",
        )

    if chart_type == "Distribution":
        st.subheader("Feature distribution")
        all_features = [TARGET_COL] + num_cols
        selected_feature = st.selectbox("Feature", all_features, key="dist_feat")
        chart_col, control_col = st.columns([3, 1])
        with control_col:
            bins = st.slider("Bins", 10, 150, 80, key="dist_bins")
            log_y = st.checkbox("Log y-axis", False, key="dist_logy")
            show_box = st.checkbox("Show box plot", True, key="dist_box")
        with chart_col:
            fig = px.histogram(
                df,
                x=selected_feature,
                nbins=bins,
                marginal="box" if show_box else None,
                color_discrete_sequence=["#38bdf8"],
                labels={selected_feature: selected_feature},
                template="plotly_dark",
                log_y=log_y,
            )
            fig.update_layout(bargap=0.03)
            st.plotly_chart(fig, width="stretch")

        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            summary = df[selected_feature].describe()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean", f"{summary['mean']:.2f}")
            m2.metric("Std", f"{summary['std']:.2f}")
            m3.metric("Min", f"{summary['min']:.0f}")
            m4.metric("Max", f"{summary['max']:.0f}")

    elif chart_type == "Time series":
        st.subheader("Energy over time")
        c1, c2, c3 = st.columns(3)
        with c1:
            feature = st.selectbox("Feature", [TARGET_COL] + num_cols, key="ts_feat")
        with c2:
            frequency = st.selectbox(
                "Resample frequency",
                ["10min", "h", "D", "W"],
                index=1,
                key="ts_freq",
            )
        with c3:
            aggregation = st.selectbox(
                "Aggregation",
                ["mean", "median", "max", "min", "sum"],
                key="ts_agg",
            )

        ts = df.set_index("date")[feature].resample(frequency).agg(aggregation).reset_index()
        fig = px.line(
            ts,
            x="date",
            y=feature,
            color_discrete_sequence=["#38bdf8"],
            labels={"date": "Time", feature: f"{aggregation.title()} {feature}"},
            template="plotly_dark",
        )
        fig.update_traces(line=dict(width=1.2))
        st.plotly_chart(fig, width="stretch")

        if st.checkbox("Show rolling average", True, key="ts_rolling_toggle"):
            window = st.slider("Rolling window", 2, 48, 12, key="ts_window")
            ts["rolling"] = ts[feature].rolling(window).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=ts["date"],
                y=ts[feature],
                mode="lines",
                name=feature,
                line=dict(color="#38bdf8", width=1),
            ))
            fig2.add_trace(go.Scatter(
                x=ts["date"],
                y=ts["rolling"],
                mode="lines",
                name=f"Rolling {window}",
                line=dict(color="#fbbf24", width=2),
            ))
            fig2.update_layout(
                template="plotly_dark",
                xaxis_title="Time",
                yaxis_title=feature,
            )
            st.plotly_chart(fig2, width="stretch")

    elif chart_type == "Time period":
        st.subheader("Average by time period")
        c1, c2, c3 = st.columns(3)
        with c1:
            feature = st.selectbox("Feature", [TARGET_COL] + num_cols, key="bp_feat")
        with c2:
            group_by = st.selectbox(
                "Group by",
                ["hour", "day_of_week", "month", "is_weekend", "part_of_day"],
                key="bp_group",
            )
        with c3:
            aggregation = st.selectbox(
                "Aggregation",
                ["mean", "median", "max", "sum"],
                key="bp_agg",
            )

        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        grouped = df.groupby(group_by)[feature].agg(aggregation).reset_index()
        if group_by == "day_of_week":
            grouped["label"] = grouped[group_by].map(day_map)
            x_column = "label"
        else:
            x_column = group_by

        fig = px.bar(
            grouped,
            x=x_column,
            y=feature,
            color=feature,
            color_continuous_scale="Blues",
            labels={feature: f"{aggregation.title()} {feature}"},
            template="plotly_dark",
            text_auto=".0f",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

    elif chart_type == "Scatter":
        st.subheader("Feature relationship")
        all_numeric = [TARGET_COL] + num_cols
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x_feature = st.selectbox(
                "X axis",
                all_numeric,
                index=all_numeric.index("T2") if "T2" in all_numeric else 1,
                key="sc_x",
            )
        with c2:
            y_feature = st.selectbox("Y axis", all_numeric, index=0, key="sc_y")
        with c3:
            color_by = st.selectbox(
                "Color by",
                ["hour", "day_of_week", "month", "is_weekend", "part_of_day"],
                key="sc_color",
            )
        with c4:
            point_count = st.slider("Sample points", 500, len(df), 2000, 500, key="sc_pts")

        sample = df.sample(point_count, random_state=RANDOM_STATE)
        fig = px.scatter(
            sample,
            x=x_feature,
            y=y_feature,
            color=color_by,
            opacity=0.6,
            color_continuous_scale="Viridis",
            labels={x_feature: x_feature, y_feature: y_feature},
            template="plotly_dark",
        )
        st.plotly_chart(fig, width="stretch")

        if st.checkbox("Add OLS trend line", key="sc_ols"):
            fig2 = px.scatter(
                sample,
                x=x_feature,
                y=y_feature,
                color=color_by,
                opacity=0.5,
                trendline="ols",
                trendline_color_override="#38bdf8",
                color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            st.plotly_chart(fig2, width="stretch")

    elif chart_type == "Correlation":
        st.subheader("Correlation matrix")
        chart_col, control_col = st.columns([3, 1])
        with control_col:
            method = st.radio("Method", ["pearson", "spearman", "kendall"], key="corr_method")
            show_values = st.checkbox("Show values", False, key="corr_annot")
            top_n = st.slider("Top features", 5, 30, 15, key="corr_ntop")

        numeric_df = df[[TARGET_COL] + num_cols]
        corr = numeric_df.corr(method=method)
        top_features = (
            corr[TARGET_COL]
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        corr_subset = numeric_df[top_features].corr(method=method)

        with chart_col:
            figc, ax = plt.subplots(figsize=(14, 9))
            figc.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            sns.heatmap(
                corr_subset,
                ax=ax,
                cmap="coolwarm",
                annot=show_values,
                fmt=".2f" if show_values else "",
                linewidths=0.3,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7},
            )
            ax.tick_params(colors="#e2e8f0", labelsize=7)
            plt.tight_layout()
            st.pyplot(figc)

        st.subheader(f"Correlation with {TARGET_COL}")
        corr_target = corr[TARGET_COL].drop(TARGET_COL).sort_values()
        fig = px.bar(
            x=corr_target.values,
            y=corr_target.index,
            orientation="h",
            color=corr_target.values,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"x": f"{method.title()} r", "y": "Feature"},
            template="plotly_dark",
        )
        st.plotly_chart(fig, width="stretch")
