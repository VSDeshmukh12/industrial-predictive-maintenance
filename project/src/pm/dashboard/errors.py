"""User-facing error handling. Missing, modified or corrupt files and missing dependencies produce a clear message with a
remedy instead of a Python traceback; programming errors (TypeError, AttributeError, ...) are NOT swallowed."""
from __future__ import annotations

import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

try:                                            # these pull in pandas/sklearn; if a package is missing app.py reports it instead
    from pm import config
    from pm.data.loader import DatasetIntegrityError
    from pm.final.bundle import BundleIntegrityError
except ImportError:                             # pragma: no cover - exercised by test_missing_dependency_is_reported
    config = None

    class DatasetIntegrityError(Exception): ...
    class BundleIntegrityError(Exception): ...

# Data/IO failures that a damaged or incomplete project folder can cause.
DATA_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, DatasetIntegrityError, BundleIntegrityError)
RESTORE = "Restore the file from the delivered ZIP. Do not regenerate Phase 2-5 artifacts: they are frozen."


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def describe(exc: BaseException) -> tuple[str, str]:
    """(what happened, what to do) in plain words."""
    if isinstance(exc, FileNotFoundError):
        return f"A required file is missing: {_rel(exc.filename) if exc.filename else exc}", RESTORE
    if isinstance(exc, DatasetIntegrityError):
        return "The dataset file does not match its pinned SHA-256: it has been modified or is not the original AI4I 2020 file.", RESTORE
    if isinstance(exc, BundleIntegrityError):
        return "The frozen model bundle failed its integrity check (model file or frozen specification differs from the manifest).", RESTORE
    if isinstance(exc, KeyError):
        return f"A saved table or file is missing an expected field: {exc}.", RESTORE
    if isinstance(exc, (zipfile.BadZipFile, EOFError)):
        return "A saved data file is corrupt or truncated.", RESTORE
    if isinstance(exc, PermissionError):
        return f"A required file cannot be read (permission denied): {_rel(exc.filename) if exc.filename else exc}", "Check the file permissions."
    if isinstance(exc, OSError):
        return f"A required file could not be read: {exc}", RESTORE
    return f"A saved file could not be interpreted: {exc}", RESTORE


def _render(area: str, exc: BaseException, fatal: bool = False) -> None:
    what, todo = describe(exc)
    head = f"**{area}**" if fatal else f"**{area} unavailable.**"
    st.error(f"{head} {what}  \n{todo}" + ("" if not fatal else "  \nThe dashboard cannot start until this is fixed; reload the page afterwards."), icon="🚫")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception_only(type(exc), exc)).strip())


@contextmanager
def guarded(area: str):
    """Run a page section; a damaged/missing data file shows an error for THIS section only and the rest of the page keeps working."""
    try:
        yield
    except DATA_ERRORS as e:
        _render(area, e)


def safe(area: str, fn, *args, default=None):
    """Call fn(*args); on a data error show the message and return `default`."""
    try:
        return fn(*args)
    except DATA_ERRORS as e:
        _render(area, e)
        return default


def startup_failed(exc: BaseException) -> None:
    _render("The dashboard cannot start;", exc, fatal=True)


def dependency_failed(exc: BaseException) -> None:
    name = getattr(exc, "name", None) or str(exc)
    st.error(f"**A required Python package is not installed: `{name}`.**  \nInstall the pinned dependencies (Python 3.11): "
             "`pip install -r requirements-dashboard.txt && pip install -e .`, then restart `streamlit run app.py`.", icon="🚫")
    with st.expander("Technical details"):
        st.code(repr(exc))
