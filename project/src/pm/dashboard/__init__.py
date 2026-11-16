"""Phase 6: Streamlit dashboard around the FROZEN Phase 4 model and the Phase 5 ExplanationEngine.

The dashboard is a read-only consumer. It never fits, tunes, calibrates or re-thresholds anything, never writes into
`artifacts/`, and never materialises the consumed hold-out through the protected hold-out loader. Layers:

  data_access  pure data/model loading and batch scoring (no Streamlit)
  provenance   the three-way vocabulary: actual data / hypothetical input / model prediction (+ rule, advisory, attribution)
  charts       Plotly figure builders (pure functions)
  export       CSV and PDF report builders (pure functions)
  theme, components   look-and-feel and reusable widgets (Streamlit)
  pages/*      one module per dashboard section, each exposing render(ctx)
"""
