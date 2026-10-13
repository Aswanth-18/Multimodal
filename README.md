# CardioScope3D: coronary risk prediction mapped onto a 3D heart

Hackathon 2026, Track A. Predicts overall CAD and LAD / LCX / RCA stenosis from clinical data
(Extension of the Z-Alizadeh Sani dataset, 303 patients) and colours the coronary arteries of an
interactive 3D heart by predicted probability, with a SHAP explanation for every prediction.

> **Decision-support / educational use only. Not a substitute for formal diagnostic imaging or clinical judgement.**
> The same disclaimer is shown permanently at the top of the UI.

## Quick start
```bash
pip install -r requirements.txt
python -m backend.ml.train                 # optional: retrain (models + results/ are already included)
uvicorn backend.app:app --port 8000        # serves the API and the built frontend
# open http://localhost:8000
```
Frontend development (hot reload; API proxy in `vite.config.js`):
```bash
cd frontend && npm install && npm run dev      # production build: npm run build
```

## What is where
| Path | Purpose |
|---|---|
| `backend/ml/data.py, features.py` | Cleaning, encoding, feature schema (single source of truth for the form) |
| `backend/ml/train.py` | Model comparison (CV), calibration, saving weights, plots |
| `backend/models/` | Trained `cad/lad/lcx/rca.joblib`, `schema.json`, `presets.json` |
| `backend/app.py` | FastAPI: `/api/schema`, `/api/predict`, `/api/metrics`, `/api/presets` |
| `frontend/src/scene.js` | Three.js scene: heart shell, vessel meshes, picking, colour animation |
| `frontend/src/main.js` | Dashboard: generated form, risk cards, SHAP breakdown, metrics tab |
| `tools/convert_obj.py` | Converts BodyParts3D `.obj` files to the viewer's coordinate frame |
| `results/` | `metrics.md/csv/json`, ROC curves, confusion matrices |

## Method
- **Targets:** CAD (`Cath`), LAD, LCX, RCA, one binary classifier each.
- **No leakage:** `LAD`, `LCX`, `RCA`, `Cath` are removed from the inputs (asserted in `data.py`).
- **Preprocessing:** text categories mapped to numbers (Y/N, Sex, VHD severity, BBB split into LBBB/RBBB); constant columns dropped.
- **Models:** logistic regression, random forest, XGBoost. Validation is 5-fold stratified CV repeated 3 times. The best model per target by mean ROC-AUC is kept and sigmoid-calibrated.
- **Explanations:** SHAP (TreeExplainer) per prediction, shown as the top drivers with the patient's value and share of the explanation.

## Results (cross-validated; see `results/metrics.md`)
| Target | Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| CAD | RandomForest | 0.859 | 0.906 | 0.926 |
| LAD | RandomForest | 0.787 | 0.830 | 0.857 |
| LCX | XGBoost | 0.690 | 0.561 | 0.742 |
| RCA | XGBoost | 0.686 | 0.531 | 0.720 |

CAD and LAD are predicted well. LCX and RCA are genuinely harder from clinical data alone. Their recall is low (about 0.3 to 0.5), so a low probability for those vessels should not be read as reassurance.

## 3D anatomy and its limits
- The heart shell, aorta, pulmonary trunk, **LAD, LCX and the RCA trunk are procedurally generated** tubes, placed on the shell surface.
- The seven RCA side branches are **real BodyParts3D meshes** (CC BY-SA 2.1 JP, Database Center for Life Science), converted with `tools/convert_obj.py`.
- To use real left-coronary meshes: convert them, put them in `frontend/public/models/`, and list them under `LAD` / `LCX` in `manifest.json`. The procedural tube for that vessel is then skipped automatically.
- Dataset features are clinical measurements, not coordinates. Colours show a vessel-level probability, not a lesion location.

## Extending
- **New feature:** add it to `backend/ml/features.py`; the form, API and explanations pick it up.
- **New model/vessel:** drop `<target>.joblib` into `backend/models/` (the registry loads it) and add it to `VESSELS` in `scene.js`.
- **New anatomy:** add `.obj` files to `manifest.json`.

## Limitations
303 patients from one centre, no external validation; probabilities are calibrated on the same data. Not a medical device.
