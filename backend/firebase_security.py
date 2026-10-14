"""Firebase identity verification and role-based authorization."""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import firebase_admin
from dotenv import dotenv_values
from firebase_admin import auth, firestore
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import GoogleAPICallError

bearer = HTTPBearer(auto_error=False)
_BACKEND_ENV_KEYS = {"FIREBASE_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS", "FRONTEND_ORIGIN"}


def _load_backend_environment(env_file: Path) -> None:
    for key, value in dotenv_values(env_file).items():
        if key in _BACKEND_ENV_KEYS and value is not None:
            os.environ.setdefault(key, value)


_load_backend_environment(Path(__file__).with_name(".env.local"))


@lru_cache(maxsize=1)
def _firebase_app():
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    if not project_id:
        raise HTTPException(503, "Firebase is not configured: set FIREBASE_PROJECT_ID.")
    try:
        return firebase_admin.get_app()
    except ValueError:
        try:
            return firebase_admin.initialize_app(options={"projectId": project_id})
        except DefaultCredentialsError as exc:
            raise HTTPException(503, "Firebase server credentials are not configured.") from exc


def firestore_client():
    try:
        return firestore.client(_firebase_app())
    except (DefaultCredentialsError, GoogleAPICallError) as exc:
        raise HTTPException(503, "Firebase services are temporarily unavailable.") from exc


def _doctor_profile(user: dict) -> dict:
    snapshot = firestore_client().collection("doctors").document(user["uid"]).get()
    if not snapshot.exists:
        return {}
    return snapshot.to_dict() or {}


def verified_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Sign in is required.")
    try:
        decoded = auth.verify_id_token(credentials.credentials, app=_firebase_app(), check_revoked=True)
    except DefaultCredentialsError as exc:
        raise HTTPException(503, "Firebase server credentials are not configured.") from exc
    except (auth.InvalidIdTokenError, auth.ExpiredIdTokenError, auth.RevokedIdTokenError) as exc:
        raise HTTPException(401, "Your sign-in has expired. Please sign in again.") from exc
    except auth.CertificateFetchError as exc:
        raise HTTPException(503, "Unable to verify sign-in right now.") from exc
    if not decoded.get("email_verified", False):
        raise HTTPException(403, "Verify your email address before continuing.")
    return decoded


def approved_doctor(user: dict = Depends(verified_user)) -> dict:
    profile = _doctor_profile(user)
    if not profile:
        raise HTTPException(403, "Doctor registration is required.")
    role = profile.get("role", "doctor")
    if role == "master":
        return {**user, "doctor_profile": profile, "role": role}
    if role != "doctor":
        raise HTTPException(403, "This account does not have an authorized role.")
    if profile.get("status") == "rejected":
        raise HTTPException(403, "Doctor account was rejected. Contact the master administrator.")
    if profile.get("status") != "approved":
        raise HTTPException(403, "Doctor account approval is pending.")
    return {**user, "doctor_profile": profile, "role": role}


def master_user(user: dict = Depends(verified_user)) -> dict:
    profile = _doctor_profile(user)
    if profile.get("role") != "master":
        raise HTTPException(403, "Master access is required.")
    return {**user, "doctor_profile": profile, "role": "master"}


def patient_code_key(uid: str, patient_code: str) -> str:
    normalized = patient_code.strip().casefold()
    return hashlib.sha256(f"{uid}:{normalized}".encode("utf-8")).hexdigest()


def audit_record(db, uid: str, action: str, patient_id: str) -> None:
    """Record an access or change without copying patient details into the audit log."""
    audit_id = hashlib.sha256(f"{uid}:{patient_id}".encode("utf-8")).hexdigest()
    db.collection("audit_logs").add({
        "doctor_uid": uid,
        "action": action,
        "patient_ref": audit_id,
        "created_at": firestore.SERVER_TIMESTAMP,
    })
