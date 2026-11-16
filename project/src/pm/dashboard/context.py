"""Application context: cached, read-only loading of the data, the frozen bundle and the Phase 5 engine."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from pm.dashboard import data_access as da
from pm.explain.engine import ExplanationEngine
from pm.final.bundle import Bundle


@dataclass
class AppContext:
    bundle: Bundle
    engine: ExplanationEngine
    records: pd.DataFrame            # all 10,000 ACTUAL records + partition + frozen-model scoring (inference only)
    ranges: dict
    defaults: dict
    info: dict

    @property
    def threshold(self) -> float:
        return self.bundle.threshold


@st.cache_resource(show_spinner="Loading the frozen model, dataset and explanation engine…")
def get_context() -> AppContext:
    bundle, engine = da.load_system()
    records = da.score_records(bundle, da.load_records())
    return AppContext(bundle, engine, records, da.training_ranges(records), da.input_defaults(records), da.system_info(bundle, records))


@st.cache_data(show_spinner=False)
def explain_cached(record_items: tuple, _engine: ExplanationEngine) -> dict:
    """One full Phase 5 explanation. Cache key = the six input values only (the engine argument is not hashed)."""
    return _engine.explain(pd.DataFrame([dict(record_items)]))[0]


def explain_record(ctx: AppContext, record: dict) -> dict:
    return explain_cached(tuple(sorted((k, record[k]) for k in da.RAW)), ctx.engine)


@st.cache_data(show_spinner=False)
def global_tables() -> dict:
    return da.global_shap_tables()


@st.cache_data(show_spinner=False)
def global_training_values() -> dict:
    z = da.global_shap_training_values()
    disp = pd.read_csv(da.P5 / "global_shap_train_display_values.csv")
    return {"values": z["values"], "features": z["features"], "display": disp}


@st.cache_data(show_spinner=False)
def advisory_tables() -> dict:
    return da.advisory_tables()


@st.cache_data(show_spinner=False)
def phase4_report() -> dict:
    return da.phase4_report()
