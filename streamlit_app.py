"""AssetFund Streamlit entrypoint for the deployed investor product."""

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
from src import app_data, app_pages  # noqa: E402

st.set_page_config(
    page_title="AssetFund",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


@st.cache_data(ttl=86_400, show_spinner="Loading AssetFund research...")
def _load_artifacts() -> app_data.AppArtifacts:
    """Load validated precomputed outputs without rerunning any model."""
    return app_data.load_app_artifacts(PROJECT_ROOT)


try:
    app_pages.render_app(_load_artifacts())
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error("AssetFund research files are unavailable or invalid.")
    st.code(str(exc), language=None)
    st.caption("Run `python scripts/run_part_b.py` from the project root.")
    st.stop()
