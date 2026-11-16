# Industrial Predictive Maintenance Dashboard (AI4I 2020)

Repository: https://github.com/VSDeshmukh12/industrial-predictive-maintenance

A Streamlit dashboard around a frozen Random Forest failure classifier, SHAP explanations, engineering calculations and a separate tool-wear (TWF) advisory.

> **The AI4I 2020 dataset is synthetic.** Scores measure how well a model recovers the data generator's rules. This project makes **no** claim of real-time prediction, remaining-useful-life estimation or real-world industrial validation. SHAP values are model attribution, not causation. The TWF advisory is not a model prediction.

## What is frozen
| Item | Value |
|---|---|
| Model | Random Forest (`random_forest\|tuned`), trained once on the 8,000-row training partition |
| Inputs | Nine features (`type` + five process variables, engineered physics features) |
| Alert threshold | **0.365** (provisional) |
| Hold-out | Evaluated exactly once in Phase 4 and now **consumed** (`project/artifacts/phase4/HOLDOUT_EVALUATED.lock.json`) |

The app is read-only. It never trains, tunes, recalibrates or changes the threshold, and never writes into `project/artifacts/`. Do not regenerate or edit anything under `project/artifacts/`, `project/data/` or the model bundle.

## Repository layout
```
streamlit_app.py        Streamlit Community Cloud entry point (runs project/app.py unchanged)
requirements.txt        exact pins for deployment (flattened from project/requirements*.txt)
.streamlit/config.toml  theme + server settings (no secrets)
project/                application, model bundle, data, artifacts, tests, scripts, docs
  app.py  src/pm/  data/raw/  artifacts/  tests/  scripts/  docs/
deliverables/           phase reports, figures, screenshots, sample reports
GITHUB_UPLOAD_GUIDE.md  step-by-step upload + deployment instructions
```

## Run locally (Python 3.11, CPU only)
```bash
git clone https://github.com/VSDeshmukh12/industrial-predictive-maintenance.git && cd industrial-predictive-maintenance
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py                            # http://localhost:8501
```
Python **3.11** is required: the model bundle was built and verified on it, and other versions may fail to unpickle it.
First load takes about 10-20 s (integrity checks, model load, SHAP explainer, inference on 10,000 records).

## Tests
```bash
pip install pytest==9.2.0
cd project && pytest          # 425 tests
```
**Fresh clones:** Git does not store the read-only permission bit, so after cloning one test (`test_original_file_is_read_only`) fails until you run `chmod a-w project/data/raw/ai4i2020.csv` (Windows: `attrib +R project\data\raw\ai4i2020.csv`). The file's content and SHA-256 are identical; only the permission differs, and the dashboard does not depend on it.

Do not run `scripts/14_*`, `17_*`, `25_*`, `30_*`, `32_*` on a delivered copy: they rewrite earlier verification files.

## Deploy to Streamlit Community Cloud
See `GITHUB_UPLOAD_GUIDE.md`. Main file path `streamlit_app.py`, Python 3.11, no secrets required.

## Documentation
`project/docs/MODEL_CARD.md` · `project/docs/PHASE4_TECHNICAL_REPORT.md` (final hold-out) · `project/docs/PHASE5_TECHNICAL_REPORT.md` (explainability) · `project/docs/DASHBOARD_SETUP.md` · `project/DATA_CARD.md` · `project/docs/PHASE8_DEPLOYMENT_REPORT.md`

## Data and license
Dataset: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," AI4I 2020, doi:10.1109/AI4I49448.2020.00023; UCI doi:10.24432/C5HS5C; CC BY 4.0 (attribution required, see `project/DATA_CARD.md`).
Code license: **not yet chosen**; add a `LICENSE` file before publishing if you want others to reuse the code.
