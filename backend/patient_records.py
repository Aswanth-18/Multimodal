"""Firestore-backed doctor-owned patient records and consultation history."""
from datetime import date, datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from firebase_admin import firestore
from google.api_core.exceptions import GoogleAPICallError
from pydantic import BaseModel, Field, field_validator, model_validator

from .firebase_security import (
    approved_doctor,
    audit_record,
    firestore_client,
    master_user,
    patient_code_key,
    verified_user,
)

router = APIRouter(prefix="/api")


class DoctorProfileUpdate(BaseModel):
    full_name: str = Field(min_length=1, max_length=160)
    doctor_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9._/-]+$")

    @field_validator("full_name", "doctor_id")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field is required.")
        return value


class DoctorDecision(BaseModel):
    status: Literal["approved", "rejected"]


class PatientDetails(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    date_of_birth: date | None = None
    age: int | None = Field(default=None, ge=0, le=125)
    sex: str = Field(default="", max_length=40)
    contact: str = Field(default="", max_length=200)
    medical_history: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def require_age_or_birth_date(self):
        if self.age is None and self.date_of_birth is None:
            raise ValueError("Provide either age or date of birth.")
        if self.date_of_birth and self.date_of_birth > date.today():
            raise ValueError("Date of birth cannot be in the future.")
        return self

    @field_validator("name")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field is required.")
        return value

    @field_validator("sex")
    @classmethod
    def strip_sex(cls, value: str) -> str:
        return value.strip()


class AIPredictionSnapshot(BaseModel):
    probabilities: dict[str, float] = Field(max_length=4)
    model: dict[str, str] = Field(max_length=4)
    inputs: dict[str, float] = Field(max_length=55)
    generated_at: datetime
    disclaimer: str = Field(max_length=250)

    @model_validator(mode="after")
    def validate_output_targets(self):
        targets = {"cad", "lad", "lcx", "rca"}
        if set(self.probabilities) != targets or set(self.model) != targets:
            raise ValueError("AI snapshot must include the four supported prediction targets.")
        if any(not 0 <= probability <= 1 for probability in self.probabilities.values()):
            raise ValueError("AI prediction probabilities must be between 0 and 1.")
        return self


class ConsultationInput(BaseModel):
    consultation_date: date
    symptoms: str = Field(default="", max_length=5000)
    diagnosis: str = Field(default="", max_length=5000)
    test_results: str = Field(default="", max_length=10000)
    treatment: str = Field(default="", max_length=5000)
    notes: str = Field(default="", max_length=10000)
    ai_prediction: AIPredictionSnapshot | None = None


class PatientCreate(PatientDetails):
    patient_code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._/-]+$")
    consultation: ConsultationInput


class PatientUpdate(PatientDetails):
    patient_code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._/-]+$")

class PatientSearch(BaseModel):
    q: str = Field(default="", max_length=160)


def _patient_ref(db, uid: str, patient_id: str):
    return db.collection("doctors").document(uid).collection("patients").document(patient_id)


def _consultation_dict(value: ConsultationInput) -> dict:
    consultation = value.model_dump(mode="json")
    consultation["created_at"] = datetime.now(timezone.utc).isoformat()
    return consultation


def _serialize_patient(snapshot) -> dict:
    data = snapshot.to_dict() or {}
    data["id"] = snapshot.id
    return data


def _require_patient(ref):
    snapshot = ref.get()
    if not snapshot.exists:
        raise HTTPException(404, "Patient record not found.")
    return snapshot


@router.get("/auth/profile")
def get_doctor_profile(user: dict = Depends(verified_user)):
    snapshot = firestore_client().collection("doctors").document(user["uid"]).get()
    if not snapshot.exists:
        return None
    profile = snapshot.to_dict() or {}
    profile.setdefault("role", "doctor")
    return profile


@router.post("/auth/profile")
def save_doctor_profile(payload: DoctorProfileUpdate, user: dict = Depends(verified_user)):
    db = firestore_client()
    ref = db.collection("doctors").document(user["uid"])
    snapshot = ref.get()
    current = snapshot.to_dict() if snapshot.exists else {}
    if current.get("role", "doctor") != "doctor":
        raise HTTPException(403, "Only doctor accounts can submit a doctor profile.")
    if current.get("status") == "approved" and current.get("doctor_id") != payload.doctor_id:
        raise HTTPException(403, "An approved doctor's registration ID cannot be changed.")
    if current.get("doctor_id") and current["doctor_id"] != payload.doctor_id:
        raise HTTPException(409, "A doctor registration ID cannot be changed after submission.")
    profile = {
        "uid": user["uid"],
        "role": "doctor",
        "email": user.get("email", ""),
        "full_name": payload.full_name.strip(),
        "doctor_id": payload.doctor_id.strip(),
        "status": current.get("status", "pending"),
        "updated_at": firestore.SERVER_TIMESTAMP,
    }
    if not snapshot.exists:
        profile["created_at"] = firestore.SERVER_TIMESTAMP
    try:
        ref.set(profile, merge=True)
    except GoogleAPICallError as exc:
        raise HTTPException(503, "Unable to save doctor registration right now.") from exc
    return {**profile, "created_at": current.get("created_at"), "updated_at": None}


@router.get("/master/doctors")
def list_doctor_registrations(user: dict = Depends(master_user)):
    doctors = firestore_client().collection("doctors").stream()
    registrations = []
    for snapshot in doctors:
        profile = snapshot.to_dict() or {}
        if profile.get("role", "doctor") != "doctor":
            continue
        registrations.append({
            "uid": snapshot.id,
            "full_name": profile.get("full_name", ""),
            "email": profile.get("email", ""),
            "doctor_id": profile.get("doctor_id", ""),
            "status": profile.get("status", "pending"),
        })
    registrations.sort(key=lambda item: (item["status"] != "pending", item["full_name"].casefold()))
    return registrations


@router.post("/master/doctors/{doctor_uid}/decision")
def decide_doctor_registration(
    doctor_uid: str,
    payload: DoctorDecision,
    user: dict = Depends(master_user),
):
    if doctor_uid == user["uid"]:
        raise HTTPException(400, "A master account cannot review itself as a doctor.")
    db = firestore_client()
    ref = db.collection("doctors").document(doctor_uid)
    snapshot = ref.get()
    if not snapshot.exists:
        raise HTTPException(404, "Doctor registration not found.")
    profile = snapshot.to_dict() or {}
    if profile.get("role", "doctor") != "doctor" or not profile.get("doctor_id"):
        raise HTTPException(404, "Doctor registration not found.")
    if profile.get("status") == "approved":
        raise HTTPException(409, "This doctor account is already approved.")
    try:
        ref.update({
            "role": "doctor",
            "status": payload.status,
            "reviewed_by": user["uid"],
            "reviewed_at": firestore.SERVER_TIMESTAMP,
        })
    except GoogleAPICallError as exc:
        raise HTTPException(503, "Unable to update the doctor registration right now.") from exc
    return {"uid": doctor_uid, "status": payload.status}


def _list_patients(db, uid: str, q: str):
    snapshots = db.collection("doctors").document(uid).collection("patients").stream()
    search = q.strip().casefold()
    patients = [_serialize_patient(item) for item in snapshots]
    if search:
        patients = [
            item for item in patients
            if search in str(item.get("name", "")).casefold()
            or search in str(item.get("patient_code", "")).casefold()
        ]
    patients.sort(key=lambda item: item.get("latest_consultation_date", ""), reverse=True)
    audit_record(db, uid, "list_patients", "collection")
    return patients


@router.get("/patients")
def list_patients(user: dict = Depends(approved_doctor)):
    return _list_patients(firestore_client(), user["uid"], "")


@router.post("/patients/search")
def search_patients(payload: PatientSearch, user: dict = Depends(approved_doctor)):
    return _list_patients(firestore_client(), user["uid"], payload.q)


@router.post("/patients", status_code=201)
def create_patient(payload: PatientCreate, user: dict = Depends(master_user)):
    db = firestore_client()
    uid = user["uid"]
    patient_id = uuid4().hex
    patient_ref = _patient_ref(db, uid, patient_id)
    consultation_id = uuid4().hex
    consultation_ref = patient_ref.collection("consultations").document(consultation_id)
    claim_ref = (
        db.collection("doctors").document(uid).collection("patient_code_claims")
        .document(patient_code_key(uid, payload.patient_code))
    )
    consultation = _consultation_dict(payload.consultation)
    patient = payload.model_dump(exclude={"consultation"}, mode="json")
    patient.update({
        "doctor_uid": uid,
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "latest_consultation_date": consultation["consultation_date"],
        "latest_diagnosis": consultation["diagnosis"],
        "consultation_count": 1,
    })

    transaction = db.transaction()

    @firestore.transactional
    def commit_create(tx):
        if claim_ref.get(transaction=tx).exists:
            raise HTTPException(409, "This patient ID is already registered. Search for the existing patient.")
        tx.create(claim_ref, {"patient_id": patient_id})
        tx.create(patient_ref, patient)
        tx.create(consultation_ref, consultation)
        tx.set(db.collection("audit_logs").document(), {
            "doctor_uid": uid, "action": "create_patient",
            "patient_ref": patient_id, "created_at": firestore.SERVER_TIMESTAMP,
        })

    try:
        commit_create(transaction)
    except GoogleAPICallError as exc:
        raise HTTPException(503, "Unable to save the patient record right now.") from exc
    return {"id": patient_id, **{k: v for k, v in patient.items() if k not in {"created_at", "updated_at"}}}


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str, user: dict = Depends(approved_doctor)):
    db = firestore_client()
    ref = _patient_ref(db, user["uid"], patient_id)
    patient = _serialize_patient(_require_patient(ref))
    audit_record(db, user["uid"], "view_patient", patient_id)
    return patient


@router.put("/patients/{patient_id}")
def update_patient(patient_id: str, payload: PatientUpdate, user: dict = Depends(approved_doctor)):
    db = firestore_client()
    ref = _patient_ref(db, user["uid"], patient_id)
    existing = _require_patient(ref).to_dict() or {}
    if existing.get("patient_code") != payload.patient_code:
        raise HTTPException(409, "Patient ID is immutable. Create a new record only for a different patient.")
    data = payload.model_dump(mode="json")
    data.update({"updated_at": firestore.SERVER_TIMESTAMP})
    try:
        ref.update(data)
    except GoogleAPICallError as exc:
        raise HTTPException(503, "Unable to update the patient record right now.") from exc
    audit_record(db, user["uid"], "update_patient", patient_id)
    return {"id": patient_id, **payload.model_dump(mode="json")}


@router.get("/patients/{patient_id}/consultations")
def list_consultations(patient_id: str, user: dict = Depends(approved_doctor)):
    db = firestore_client()
    ref = _patient_ref(db, user["uid"], patient_id)
    _require_patient(ref)
    items = []
    for snapshot in ref.collection("consultations").stream():
        item = snapshot.to_dict() or {}
        item["id"] = snapshot.id
        items.append(item)
    items.sort(key=lambda item: item.get("consultation_date", ""), reverse=True)
    audit_record(db, user["uid"], "view_consultations", patient_id)
    return items


@router.get("/patients/{patient_id}/consultations/{consultation_id}")
def get_consultation(
    patient_id: str,
    consultation_id: str,
    user: dict = Depends(approved_doctor),
):
    db = firestore_client()
    ref = _patient_ref(db, user["uid"], patient_id)
    _require_patient(ref)
    snapshot = ref.collection("consultations").document(consultation_id).get()
    if not snapshot.exists:
        raise HTTPException(404, "Consultation not found.")
    audit_record(db, user["uid"], "view_consultation", patient_id)
    return {"id": snapshot.id, **(snapshot.to_dict() or {})}


@router.post("/patients/{patient_id}/consultations", status_code=201)
def add_consultation(
    patient_id: str,
    payload: ConsultationInput,
    user: dict = Depends(approved_doctor),
):
    db = firestore_client()
    uid = user["uid"]
    patient_ref = _patient_ref(db, uid, patient_id)
    _require_patient(patient_ref)
    consultation_id = uuid4().hex
    consultation_ref = patient_ref.collection("consultations").document(consultation_id)
    consultation = _consultation_dict(payload)
    update = {
        "updated_at": firestore.SERVER_TIMESTAMP,
        "latest_consultation_date": consultation["consultation_date"],
        "latest_diagnosis": consultation["diagnosis"],
        "consultation_count": firestore.Increment(1),
    }
    transaction = db.transaction()

    @firestore.transactional
    def commit_consultation(tx):
        current = tx.get(patient_ref)
        if not current.exists:
            raise HTTPException(404, "Patient record not found.")
        current_data = current.to_dict() or {}
        tx.create(consultation_ref, consultation)
        if consultation["consultation_date"] >= current_data.get("latest_consultation_date", ""):
            tx.update(patient_ref, update)
        else:
            tx.update(patient_ref, {
                "updated_at": firestore.SERVER_TIMESTAMP,
                "consultation_count": firestore.Increment(1),
            })
        tx.set(db.collection("audit_logs").document(), {
            "doctor_uid": uid, "action": "add_consultation",
            "patient_ref": patient_id, "created_at": firestore.SERVER_TIMESTAMP,
        })

    try:
        commit_consultation(transaction)
    except GoogleAPICallError as exc:
        raise HTTPException(503, "Unable to save the consultation right now.") from exc
    return {"id": consultation_id, **consultation}
