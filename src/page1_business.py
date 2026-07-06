"""Overview page for the energy forecasting app."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.content import APP_NAME, APP_TAGLINE
from src.data_loader import DROP_COLS, TARGET_COL


def render(df: pd.DataFrame) -> None:
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Target", TARGET_COL)
    c4.metric("Missing cells", int(df.isna().sum().sum()))

    st.divider()

    st.subheader("What this app estimates")
    st.write(
        "This app estimates appliance energy use from indoor sensor readings, "
        "outdoor weather, and time based features. It is built around the UCI "
        "Appliances Energy Prediction dataset."
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Modeling target")
        st.write(
            f"The target column is `{TARGET_COL}`, measured in watt-hours. "
            "Each prediction is an estimate for appliance energy use under a "
            "specific set of environmental conditions."
        )
        st.markdown("#### Excluded columns")
        st.write(
            "The model excludes the raw timestamp and random variables used in "
            f"the source dataset: `{', '.join(DROP_COLS)}`."
        )

    with right:
        st.markdown("#### Feature groups")
        st.write(
            "- Indoor temperature and humidity readings\n"
            "- Outdoor temperature, humidity, wind, and visibility\n"
            "- Calendar features derived from the timestamp\n"
            "- Lighting energy as a contextual input"
        )

    st.divider()

    st.subheader("Dataset columns")
    mean_values = [
        str(round(float(df[column].mean()), 2))
        if pd.api.types.is_numeric_dtype(df[column]) else "Not numeric"
        for column in df.columns
    ]
    column_info = pd.DataFrame({
        "Column": list(df.columns),
        "Type": [str(df[column].dtype) for column in df.columns],
        "Non-null rows": [int(df[column].count()) for column in df.columns],
        "Mean": mean_values,
    })
    st.dataframe(column_info, width="stretch", height=350)

    st.divider()

    preview_tab, stats_tab = st.tabs(["Preview", "Summary statistics"])

    with preview_tab:
        row_count = st.slider("Rows to show", 5, 100, 10, 5, key="p1_preview_rows")
        st.dataframe(df.head(row_count), width="stretch")

    with stats_tab:
        st.dataframe(
            df.describe().T.style.background_gradient(cmap="Blues"),
            width="stretch",
        )
