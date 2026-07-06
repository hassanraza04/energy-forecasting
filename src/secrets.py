"""Credential loader for local, Streamlit, and Hugging Face deployments."""
from __future__ import annotations

import os
import streamlit as st


def get_secret(key: str, default: str = "") -> str:
    """Fetch a secret with fallback across supported deployment targets."""
    try:
        return str(st.secrets[key])
    except (KeyError, FileNotFoundError):
        pass

    return os.getenv(key, default)
