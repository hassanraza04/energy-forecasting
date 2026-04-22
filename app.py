"""
app.py — Smart Energy Consumption Forecasting
Entry point: page config, CSS, sidebar navigation, and page routing.
All page logic lives in src/.
"""
from __future__ import annotations

# Load .env first so all os.getenv() calls work everywhere
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── src imports ───────────────────────────────────────────────────────────────
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

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    h1 { color: #38bdf8; }
    h2, h3 { color: #7dd3fc; }
    .stMetric label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = [
    "🏠 Business Case & Data",
    "📊 Data Visualisation",
    "🤖 Model Predictions",
    "🔍 Explainability (SHAP)",
    "⚙️  Hyperparameter Tuning",
    "🏁 Conclusions",
]

with st.sidebar:
    st.markdown("## ⚡ Smart Energy\nForecasting Dashboard")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("DS Final Project · Energy Dataset")

# ── Load data (cached) ────────────────────────────────────────────────────────
df       = get_data()
num_cols = get_numeric_features(df)

# ── Route to page ─────────────────────────────────────────────────────────────
if page == PAGES[0]:
    page1_business.render(df)

elif page == PAGES[1]:
    page2_eda.render(df, num_cols)

elif page == PAGES[2]:
    bundle = train_all_models(DATA_PATH)
    page3_predictions.render(bundle)

elif page == PAGES[3]:
    bundle = train_all_models(DATA_PATH)
    page4_shap.render(bundle)

elif page == PAGES[4]:
    bundle = train_all_models(DATA_PATH)
    page5_tuning.render(bundle)

elif page == PAGES[5]:
    bundle = train_all_models(DATA_PATH)
    page6_conclusions.render(bundle)
