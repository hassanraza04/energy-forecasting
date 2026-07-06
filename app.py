"""Streamlit entry point for Energy Forecasting Lab."""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.content import (
    APP_NAME,
    APP_TAGLINE,
    PAGE_EXPLAIN,
    PAGE_EXPLORE,
    PAGE_FINDINGS,
    PAGE_FORECAST,
    PAGE_OPTIONS,
    PAGE_OVERVIEW,
    PAGE_TUNE,
)

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_loader import (
    get_data,
    train_all_models,
    get_numeric_features,
    DATA_PATH,
)
from src import (
    page1_business,
    page2_eda,
    page3_predictions,
    page4_shap,
    page5_tuning,
    page6_conclusions,
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    h1 {
        color: #e0f2fe;
        letter-spacing: 0;
    }
    h2, h3 {
        color: #bae6fd;
        letter-spacing: 0;
    }
    .stMetric {
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    .stMetric label {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"## {APP_NAME}")
    st.caption(APP_TAGLINE)
    st.divider()
    page = st.radio("Navigate", PAGE_OPTIONS, label_visibility="collapsed")
    st.divider()
    st.caption("UCI Appliances Energy Prediction dataset")

df       = get_data()
num_cols = get_numeric_features(df)

if page == PAGE_OVERVIEW:
    page1_business.render(df)

elif page == PAGE_EXPLORE:
    page2_eda.render(df, num_cols)

elif page == PAGE_FORECAST:
    bundle = train_all_models(DATA_PATH)
    page3_predictions.render(bundle)

elif page == PAGE_EXPLAIN:
    bundle = train_all_models(DATA_PATH)
    page4_shap.render(bundle)

elif page == PAGE_TUNE:
    bundle = train_all_models(DATA_PATH)
    page5_tuning.render(bundle)

elif page == PAGE_FINDINGS:
    bundle = train_all_models(DATA_PATH)
    page6_conclusions.render(bundle)
