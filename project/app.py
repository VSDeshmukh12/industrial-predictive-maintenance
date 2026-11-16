"""Phase 6 Streamlit dashboard entry point.   streamlit run app.py

Read-only consumer of the frozen Phase 4 bundle and the Phase 5 ExplanationEngine. Never trains, tunes, calibrates or
changes the 0.365 threshold; never writes into artifacts/.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🏭", layout="wide", initial_sidebar_state="auto")

import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

if importlib.util.find_spec("pm") is None:     # works even if `pip install -e .` was skipped: use the project's own src/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:
    from pm.dashboard import errors  # noqa: E402
    from pm.dashboard import theme  # noqa: E402
    from pm.dashboard.context import get_context  # noqa: E402
    from pm.dashboard.pages import analytics, engineering, explainability, monitoring, overview, prediction, reports, twf  # noqa: E402
except ImportError as _e:                      # missing package: say so instead of showing a traceback
    st.error(f"**A required Python package is not installed: `{getattr(_e, 'name', None) or _e}`.**  \nInstall the pinned dependencies with Python 3.11: "
             "`pip install -r requirements-dashboard.txt`, then restart `streamlit run app.py`.", icon="🚫")
    st.stop()

PAGES = {m.TITLE: m for m in (overview, monitoring, prediction, engineering, explainability, twf, analytics, reports)}


def main() -> None:
    theme.inject()
    try:
        ctx = get_context()
    except errors.DATA_ERRORS as e:            # missing / modified / corrupt dataset, split or model bundle
        errors.startup_failed(e)
        st.stop()
    with st.sidebar:
        st.markdown("## 🏭 Predictive Maintenance")
        st.caption("Phase 6 dashboard · AI4I 2020 (synthetic)")
        choice = st.radio("Section", list(PAGES), format_func=lambda t: f"{PAGES[t].ICON}  {t}", key="nav", label_visibility="collapsed")
        st.divider()
        st.caption(f"Frozen Random Forest · alert level **{ctx.threshold:.3f}** (provisional)")
        st.caption("Actual data ≠ hypothetical input ≠ model prediction. Attribution is not causation. The TWF advisory is not a prediction.")
    PAGES[choice].render(ctx)


main()
