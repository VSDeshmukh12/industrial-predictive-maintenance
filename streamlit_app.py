"""Streamlit Community Cloud entry point (repository root).

Thin launcher: runs project/app.py unchanged. All application, model and data code stays under project/.
Local equivalent:  streamlit run streamlit_app.py   (or, from project/:  streamlit run app.py)
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "project" / "app.py"), run_name="__main__")
