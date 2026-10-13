"""FastAPI service: /schema, /predict, /metrics, /presets (+ serves the built frontend).
Run:  uvicorn backend.app:app --port 8000
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, joblib, shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODELS, RES, DIST = ROOT / "backend" / "models", ROOT / "results", ROOT / "frontend" / "dist"

class Registry:
    """Model registry: target -> trained model bundle. Add a vessel/model by dropping <target>.joblib in models/."""
    def __init__(self):
        with open(MODELS / "schema.json", encoding="utf-8") as fh:
            self.schema = json.load(fh)
        with open(MODELS / "presets.json", encoding="utf-8") as fh:
            self.presets = json.load(fh)
        self.meta = {f["name"]: f for f in self.schema}
        self.models, self.explainers = {}, {}
        for p in sorted(MODELS.glob("*.joblib")):
            b = joblib.load(p); t = p.stem
            self.models[t] = b
            base = b["base"]
            if b["kind"] in ("RandomForest", "XGBoost"):
                self.explainers[t] = shap.TreeExplainer(base)
            else:
                self.explainers[t] = shap.Explainer(lambda d, m=base: m.predict_proba(pd.DataFrame(d, columns=b["features"]))[:, 1], b["background"])
        self.features = next(iter(self.models.values()))["features"]

    def frame(self, values: dict) -> pd.DataFrame:
        row = {}
        for f in self.features:
            m = self.meta[f]; v = values.get(f)
            row[f] = float(m["default"]) if v is None else float(v)   # missing -> training median
        return pd.DataFrame([row])[self.features]

    def shap_values(self, t, x):
        sv = self.explainers[t](x) if not hasattr(self.explainers[t], "shap_values") else self.explainers[t].shap_values(x)
        sv = getattr(sv, "values", sv)
        sv = np.asarray(sv)
        if sv.ndim == 3: sv = sv[..., 1]          # (n, features, classes) -> positive class
        return sv[0]

R = Registry()
app = FastAPI(title="Cardio-3D risk API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Patient(BaseModel):
    values: dict[str, float] = {}

@app.get("/api/health")
def health(): return {"ok": True, "targets": list(R.models)}

@app.get("/api/schema")
def schema(): return R.schema

@app.get("/api/presets")
def presets(): return R.presets

@app.get("/api/metrics")
def metrics():
    with open(RES / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)

@app.post("/api/predict")
def predict(p: Patient, top_k: int = 8):
    unknown = set(p.values) - set(R.features)
    if unknown: raise HTTPException(422, f"Unknown features: {sorted(unknown)}")
    x = R.frame(p.values)
    out = {"probabilities": {}, "explanations": {}, "model": {}}
    for t, b in R.models.items():
        prob = float(b["calibrated"].predict_proba(x)[0, 1])
        sv = R.shap_values(t, x)
        total = float(np.abs(sv).sum()) or 1.0
        order = np.argsort(-np.abs(sv))[:top_k]
        out["probabilities"][t] = prob
        out["model"][t] = b["kind"]
        out["explanations"][t] = [dict(
            feature=R.features[i], label=R.meta[R.features[i]]["label"], unit=R.meta[R.features[i]]["unit"],
            category=R.meta[R.features[i]]["category"], value=float(x.iloc[0, i]),
            display=(R.meta[R.features[i]]["options"] or {}).get(str(int(x.iloc[0, i]))) if R.meta[R.features[i]]["options"] else None,
            contribution=float(sv[i]), share=float(abs(sv[i]) / total),
            direction="increases" if sv[i] > 0 else "decreases") for i in order]
    return out

app.mount("/results", StaticFiles(directory=RES), name="results")
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")
    app.mount("/models", StaticFiles(directory=DIST / "models"), name="models")
    @app.get("/")
    def index(): return FileResponse(DIST / "index.html")
