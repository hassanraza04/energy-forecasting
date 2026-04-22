"""
src/page6_conclusions.py
Page 6 — Conclusions & Recommendations
Final summary of model performance, feature impact, and business value.
"""
from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import TARGET_COL


def render(bundle: Dict[str, Any]) -> None:
    st.title("🏁 Conclusions & Strategic Recommendations")
    
    results = bundle["results"]
    feat_cols = bundle["feat_cols"]
    
    # ── Leaderboard logic for summary ──────────────────────────────────────────
    lb = pd.DataFrame([
        {"Model": name, "R²": v["R2"], "MAE": v["MAE"]}
        for name, v in results.items()
    ]).sort_values("R²", ascending=False)
    
    best_model = lb.iloc[0]["Model"]
    best_r2 = lb.iloc[0]["R²"]
    best_mae = lb.iloc[0]["MAE"]

    # ── Executive Summary ─────────────────────────────────────────────────────
    st.subheader("📌 Executive Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Model", best_model)
    c2.metric("Best R² Score", f"{best_r2:.4f}")
    c3.metric("Avg Prediction Error", f"{best_mae:.2f} Wh")

    st.markdown(f"""
    Based on our analysis and multi-model experimentation, the **{best_model}** emerged as the 
    most reliable predictor for appliance energy consumption. With an R² of **{best_r2:.4f}**, 
    the model explains a significant portion of the variance in building energy usage.
    """)

    st.markdown("---")

    # ── Key Findings 2-Column Layout ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💡 Key Insights")
        st.write("""
        - **Temperature is King:** Indoor temperatures (especially in common areas like the Kitchen and Living Room) are the strongest predictors of appliance energy spikes.
        - **Temporal Patterns:** Energy consumption follows a strict circadian rhythm, peaking in the late afternoon/evening and dropping significantly during late-night hours.
        - **External Factors:** While indoor conditions dominate, outdoor humidity and windspeed play a secondary but measurable role in heat dissipation and energy load.
        """)

    with col2:
        st.subheader("📈 Business Impact")
        st.write(f"""
        - **Cost Savings:** By anticipating high-load periods, facilities can implement load-shifting strategies to avoid peak utility pricing.
        - **Efficiency:** The **{best_mae:.2f} Wh** average error provides enough precision to set automated alerts for "unusual" consumption patterns.
        - **Sustainability:** Optimising the energy footprint of appliances contributes directly to corporate ESG (Environmental, Social, and Governance) goals.
        """)

    st.markdown("---")

    # ── Strategic Recommendations ──────────────────────────────────────────────
    st.subheader("🚀 Strategic Recommendations")
    
    with st.expander("Short-Term (Quick Wins)", expanded=True):
        st.write("""
        1. **Smart Scheduling:** Defer high-energy appliance cycles to 'Morning' or 'Late Night' periods where model predictions are consistently lower.
        2. **Sensor Calibration:** Focus maintenance on sensors in rooms with high feature importance (determined via SHAP) to ensure data quality remains high.
        """)

    with st.expander("Long-Term (Infrastructure)"):
        st.write("""
        1. **Model Retraining:** Implement a seasonal retraining pipeline (e.g., via W&B) as building thermal dynamics change between winter and summer.
        2. **HVAC Integration:** Link these appliance forecasts with HVAC control systems to optimise the building's total thermal load.
        """)

    # ── Final Visual Celebration ──────────────────────────────────────────────
    st.markdown("---")
    st.caption("Final Model Comparison (R² Score)")
    fig = px.bar(
        lb, x="R²", y="Model", orientation="h",
        color="R²", color_continuous_scale="Viridis",
        template="plotly_dark",
        range_x=[0, 1]
    )
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.success("✨ Project complete. The system is ready for deployment and decision-support integration.")
