"""
src/secrets.py
Unified credential loader — works across all environments:
  - Local dev  → reads from .env  (via python-dotenv loaded in app.py)
  - Streamlit Cloud → reads from st.secrets (set in dashboard)
  - HuggingFace Spaces → reads from environment variables (set in Space settings)
"""
from __future__ import annotations

import os
import streamlit as st


def get_secret(key: str, default: str = "") -> str:
    """
    Fetch a secret with graceful fallback across all deployment targets.

    Priority order:
      1. st.secrets  (Streamlit Cloud — set via app dashboard)
      2. os.environ  (HuggingFace Spaces or local .env loaded by dotenv)
      3. default     (empty string — user fills in via UI)
    """
    # 1. Streamlit secrets (Streamlit Cloud)
    try:
        return str(st.secrets[key])
    except (KeyError, FileNotFoundError):
        pass

    # 2. OS environment variable (HuggingFace / local dotenv)
    return os.getenv(key, default)
