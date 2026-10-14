import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend import firebase_security
from backend.app import R, app
from backend.firebase_security import approved_doctor
from backend.ml.data import load
from google.auth.exceptions import DefaultCredentialsError


X, Y = load()

def test_backend_environment_loader_only_imports_server_settings(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "FIREBASE_PROJECT_ID=test-project\n"
        "GOOGLE_APPLICATION_CREDENTIALS=C:/secure/service-account.json\n"
        "VITE_FIREBASE_API_KEY=frontend-only\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("VITE_FIREBASE_API_KEY", raising=False)

    firebase_security._load_backend_environment(env_file)

    assert firebase_security.os.environ["FIREBASE_PROJECT_ID"] == "test-project"
    assert firebase_security.os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "C:/secure/service-account.json"
    assert "VITE_FIREBASE_API_KEY" not in firebase_security.os.environ


def test_dataset_and_targets_load():
    assert X.shape == (303, 55)
    assert Y.shape == (303, 4)
    assert set(Y.columns) == {"cad", "lad", "lcx", "rca"}
    assert X.isna().sum().sum() == 0


def test_every_model_loads_and_predicts():
    for target in ["cad", "lad", "lcx", "rca"]:
        bundle = R.models[target]
        assert bundle["kind"] in {"LogisticRegression", "RandomForest", "XGBoost"}
        assert len(bundle["features"]) == 55
        assert set(bundle["features"]) == set(X.columns)
        probs = bundle["calibrated"].predict_proba(
            pd.DataFrame([np.zeros(len(bundle["features"]))], columns=bundle["features"])
        )[:, 1]
        assert probs.shape == (1,)
        assert np.isfinite(probs).all()
        assert ((0.0 <= probs) & (probs <= 1.0)).all()


def test_api_health_and_prediction_contract():
    client = TestClient(app)
    assert client.get("/api/patients").status_code == 401
    assert client.get("/api/auth/profile").status_code == 401
    assert client.get("/api/presets").status_code == 401
    health = client.get("/api/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["ok"] is True
    assert set(payload["targets"]) == {"cad", "lad", "lcx", "rca"}

    schema = client.get("/api/schema")
    assert schema.status_code == 200
    assert len(schema.json()) == 55

    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    assert set(metrics.json()) == {"cv_all", "selected"}

    example = {
        "Age": 60, "Sex": 1, "Weight": 80, "Length": 170, "BMI": 27.7, "DM": 0, "HTN": 1,
        "Current Smoker": 0, "EX-Smoker": 0, "FH": 0, "Obesity": 0, "CRF": 0, "CVA": 0,
        "Airway disease": 0, "Thyroid Disease": 0, "CHF": 0, "DLP": 1, "BP": 130, "PR": 75,
        "Edema": 0, "Weak Peripheral Pulse": 0, "Lung rales": 0, "Systolic Murmur": 0,
        "Diastolic Murmur": 0, "Typical Chest Pain": 1, "Dyspnea": 0, "Atypical": 0,
        "Nonanginal": 0, "LowTH Ang": 0, "Function Class": 2, "Q Wave": 0, "St Elevation": 0,
        "St Depression": 0, "Tinversion": 0, "LVH": 0, "Poor R Progression": 0, "LBBB": 0,
        "RBBB": 0, "FBS": 92, "CR": 1.1, "TG": 120, "LDL": 120, "HDL": 40, "BUN": 20,
        "ESR": 12, "HB": 14.5, "K": 4.2, "Na": 140, "WBC": 7.1, "Lymph": 30, "Neut": 60,
        "PLT": 220, "EF-TTE": 60, "Region RWMA": 0, "VHD": 0,
    }
    assert client.post("/api/predict", json={"values": example}).status_code == 401
    app.dependency_overrides[approved_doctor] = lambda: {"uid": "test-doctor"}
    try:
        assert client.get("/api/presets").status_code == 200
        response = client.post("/api/predict", json={"values": example})
        assert response.status_code == 200
        body = response.json()
        for target, prob in body["probabilities"].items():
            assert math.isfinite(prob)
            assert 0.0 <= prob <= 1.0
            assert target in body["explanations"]
        invalid = client.post("/api/predict", json={"values": {"Bogus": 42}})
        assert invalid.status_code == 422
    finally:
        app.dependency_overrides.pop(approved_doctor, None)


def test_missing_firebase_admin_credentials_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(firebase_security, "_firebase_app", lambda: object())

    def missing_credentials(*_args, **_kwargs):
        raise DefaultCredentialsError("test credentials missing")

    monkeypatch.setattr(firebase_security.auth, "verify_id_token", missing_credentials)
    response = TestClient(app).get(
        "/api/auth/profile",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Firebase server credentials are not configured."
