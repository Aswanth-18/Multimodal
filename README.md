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

## Doctor access and patient records
The prediction screen and patient-record APIs require a Firebase-authenticated doctor whose
Firestore profile has been explicitly approved by an administrator. New email accounts must
verify their email; Google sign-in is supported. Registration alone never grants access.

1. Create a Firebase project, enable **Authentication → Email/Password** and **Google**, create a
   Web App, and enable Cloud Firestore. Add `localhost` to Authentication's authorized domains.
   Use `http://localhost:5173` in the browser for Google sign-in; Firebase may reject
   `127.0.0.1` as an unauthorized OAuth domain.
2. Copy `frontend/.env.example` to `frontend/.env.local` and fill in the Firebase Web App values.
   These are public client configuration values; never put a service-account private key there.
3. Apply `firestore.rules` using Firebase Console or the Firebase CLI (`firebase deploy --only firestore:rules`).
   The rules deny all direct client access; the FastAPI server accesses Firestore using its
   service identity after verifying Firebase ID tokens.
4. The backend reads only `FIREBASE_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS`, and
   `FRONTEND_ORIGIN` from `backend/.env.local` (without replacing values already set in the process).
   The frontend's `VITE_FIREBASE_PROJECT_ID` is separate and does not configure Firebase Admin in
   FastAPI. Set `GOOGLE_APPLICATION_CREDENTIALS` there to a secure path to a service-account JSON
   file, or configure Application Default Credentials. For local development with Google Cloud CLI
   installed, run `gcloud auth application-default login`. Never put service-account JSON in
   frontend files or commit it. Set `FRONTEND_ORIGIN` only when using a non-default frontend origin.
5. Start the backend and frontend using the commands above. Vite proxies `/api` to port 8000.
6. A doctor registers or signs in and submits their full name and registration number. Until
   reviewed, their profile is `pending`. A trusted administrator verifies the doctor and changes
   `doctors/{firebase_uid}.status` to `approved` in the Firebase Console. Do not expose an
   approval endpoint to ordinary users.

Patient data is scoped under `doctors/{uid}/patients/{record_id}` and each consultation is a
separate document in its `consultations` subcollection. The patient ID/clinic record number is
claimed uniquely per doctor; return visits add consultation documents instead of replacing history.
AI probabilities, model names, inputs, and generation time are stored with the consultation when a
prediction exists, and are labelled as estimates rather than confirmed diagnoses. Audit entries
avoid copying patient names or clinical notes.

Before storing real patient information, configure the Firebase project’s data location, backups,
retention, access controls, and operational privacy/security practices for the applicable
jurisdiction. This demo does not by itself establish regulatory compliance or clinical validation.

Protected data endpoints include `GET/POST /api/auth/profile`, `GET/POST /api/patients`,
`POST /api/patients/search`, `GET/PUT /api/patients/{record_id}`, and
`GET/POST /api/patients/{record_id}/consultations`. A single-consultation GET is also available.
Clinical values are sent in request bodies rather than search URLs. The demo's saved example
presets are also doctor-protected because they contain rows sampled from the training inputs.

## What is where
| Path | Purpose |
|---|---|
| `backend/ml/data.py, features.py` | Cleaning, encoding, feature schema (single source of truth for the form) |
| `backend/ml/train.py` | Model comparison (CV), calibration, saving weights, plots |
| `backend/models/` | Trained `cad/lad/lcx/rca.joblib`, `schema.json`, `presets.json` |
| `backend/app.py` | FastAPI model endpoints; prediction and patient presets require an approved doctor |
| `backend/firebase_security.py` | Firebase ID-token verification, approval guard, and audit support |
| `backend/patient_records.py` | Doctor profiles, owner-scoped patient records, and append-only consultation history |
| `firestore.rules`, `firebase.json` | Deny direct client access to Firestore; Firebase CLI rules target |
| `frontend/src/scene.js` | Three.js scene: heart shell, vessel meshes, picking, colour animation |
| `frontend/src/auth.js`, `frontend/src/main.js` | Firebase sign-in and approved-doctor dashboard, existing prediction UI, patient management |
| `frontend/.env.example` | Public Firebase Web App configuration variable names |
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

## 3D anatomy and local model asset
- The viewer loads `frontend/public/models/heart.glb` with Three.js `GLTFLoader`. The asset was built from BodyParts3D 4.0 by the Heart Atlas contributors and is distributed under **CC BY-SA 4.0**. Sources: [Heart Atlas model and attribution](https://github.com/w3lld1/heart-atlas/tree/8e46df9a3528c6bcda1bc1e60c4a06d00d178191/assets/models/heart), [Heart Atlas content license](https://github.com/w3lld1/heart-atlas/blob/8e46df9a3528c6bcda1bc1e60c4a06d00d178191/LICENSE-CONTENT), and [BodyParts3D](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html). The viewer includes a source/license credit above the heart. Preserve the CC BY-SA 4.0 terms and attribution when redistributing the model.
- The source model has 27 named meshes, including right/left atria and ventricles, aorta, pulmonary trunk, superior/inferior vena cava, `right-coronary-artery`, `left-anterior-descending-artery`, and `circumflex-artery`. Its source workflow documents conversion from millimetres/Z-up to metres/Y-up and centering. The viewer checks the mesh bounds, scales its height to 160 scene units, and centers it. The three coronary artery meshes map to RCA/LAD/LCX predictions; individual branch OBJ files are not overlaid when the corresponding full-artery mesh is present.
- The seven separately converted RCA branch OBJs in `manifest.json` are skipped when the full RCA mesh loads: their converted millimetre coordinates do not align with the centered, normalized GLB transform. The existing OBJ loading remains available for branch-only models when a full artery mesh is absent.
- Vessel meshes start in a neutral tissue color and change only after actual backend probabilities arrive, using the shared `colors.js` probability mapping. The interface labels these as AI estimates, not confirmed diagnoses. If the GLB fails to load, the existing schematic fallback is displayed and explicitly identified as such.
- Dataset features are clinical measurements, not coordinates. Colours show a vessel-level AI probability, not a lesion location or confirmed diagnosis.

## Extending
- **New feature:** add it to `backend/ml/features.py`; the form, API and explanations pick it up.
- **New model/vessel:** drop `<target>.joblib` into `backend/models/` (the registry loads it) and add it to `VESSELS` in `scene.js`.
- **New anatomy:** add `.obj` files to `manifest.json`.

## Limitations
303 patients from one centre, no external validation; probabilities are calibrated on the same data. Not a medical device.
