"""Findings page for model results and limits."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.modeling import build_leaderboard, get_best_model


def render(bundle: Dict[str, Any]) -> None:
    st.title("Findings")
    st.caption("A short summary of model performance, patterns, and limits.")

    results = bundle["results"]
    leaderboard = build_leaderboard(results)
    best = get_best_model(leaderboard)

    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", best["Model"])
    c2.metric("Best R2", f"{best['R2']:.4f}")
    c3.metric("Mean abs error", f"{best['MAE']:.2f} Wh")

    st.write(
        f"The strongest baseline model in this run is {best['Model']}. "
        f"It reaches an R2 score of {best['R2']:.4f} with a mean absolute "
        f"error of {best['MAE']:.2f} Wh on the held out test set."
    )

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("What the model is picking up")
        st.write(
            "- Indoor temperature and humidity readings carry useful signal.\n"
            "- Time based features help capture daily usage patterns.\n"
            "- Outdoor weather features add context, but they are not the whole story."
        )

    with right:
        st.subheader("What this app should not claim")
        st.write(
            "- It is not connected to a live building.\n"
            "- It does not control appliances or energy schedules.\n"
            "- It should be validated on a new building before operational use."
        )

    st.divider()

    st.subheader("Next steps before real use")
    with st.expander("Data checks", expanded=True):
        st.write(
            "Validate sensor quality, missing data handling, timestamp consistency, "
            "and whether the training data matches the building where the model would run."
        )

    with st.expander("Model checks"):
        st.write(
            "Compare the baseline models with more recent data, track drift over time, "
            "and retrain when usage patterns or seasons change."
        )

    with st.expander("Deployment checks"):
        st.write(
            "Add monitoring, versioned model artifacts, controlled releases, and a human "
            "review process before using predictions in operational decisions."
        )

    st.divider()
    st.caption("Model comparison by R2 score")
    chart_df = pd.DataFrame(leaderboard)
    fig = px.bar(
        chart_df,
        x="R2",
        y="Model",
        orientation="h",
        color="R2",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        range_x=[0, 1],
    )
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
