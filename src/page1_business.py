"""
src/page1_business.py
Page 1 — Business Case & Data
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import TARGET_COL, DROP_COLS


def render(df: pd.DataFrame) -> None:
    st.title("⚡ Smart Energy Consumption Forecasting")
    st.caption(
        "Predicting appliance energy use to drive smarter, cost-efficient building management."
    )

    # ── KPI strip ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",     f"{df.shape[0]:,}")
    c2.metric("Total Columns",  df.shape[1])
    c3.metric("Target Column",  TARGET_COL)
    c4.metric("Missing Values", int(df.isna().sum().sum()))

    st.markdown("---")

    # ── Business problem ──────────────────────────────────────────────────────
    st.subheader("🎯 Business Problem")
    st.write("""
    Buildings account for roughly **40%** of global energy consumption, with appliances being a
    major contributor. This project builds a **regression model** that predicts appliance energy
    consumption (Wh) using:
    - 🌡️ Indoor temperature & humidity sensors (9 rooms)
    - 🌤️ Outdoor weather indicators
    - 🕐 Time-of-day and calendar features

    **Business value:** facility managers can anticipate high-consumption periods, schedule
    loads intelligently, reduce electricity bills, and lower CO₂ emissions.
    """)

    st.markdown("---")

    # ── Column overview ───────────────────────────────────────────────────────
    st.subheader("📋 Dataset Column Overview")

    # Cast Mean uniformly to str to avoid Arrow mixed-type errors
    mean_vals = [
        str(round(float(df[c].mean()), 2))
        if pd.api.types.is_numeric_dtype(df[c]) else "—"
        for c in df.columns
    ]
    col_info = pd.DataFrame({
        "Column":   list(df.columns),
        "Dtype":    [str(df[c].dtype) for c in df.columns],
        "Non-Null": [int(df[c].count()) for c in df.columns],
        "Mean":     mean_vals,
    })
    st.dataframe(col_info, width="stretch", height=350)

    st.markdown("---")

    # ── Tabs: preview + stats ─────────────────────────────────────────────────
    tab_prev, tab_stats = st.tabs(["🔎 Data Preview", "📊 Descriptive Statistics"])

    with tab_prev:
        n_rows = st.slider("Rows to preview", 5, 100, 10, 5, key="p1_preview_rows")
        st.dataframe(df.head(n_rows), width="stretch")

    with tab_stats:
        st.dataframe(
            df.describe().T.style.background_gradient(cmap="Blues"),
            width="stretch",
        )
