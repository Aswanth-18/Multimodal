const BASE = import.meta.env.VITE_API_URL ?? "";
const j = async (r) => {
  if (!r.ok) {
    const text = await r.text();
    let message = text;
    try { message = JSON.parse(text).detail ?? text; } catch {}
    throw new Error(`${r.status} ${message}`);
  }
  return r.json();
};
const authFetch = (path, token, options = {}) => fetch(`${BASE}${path}`, {
  ...options,
  headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), Authorization: `Bearer ${token}`, ...options.headers },
}).then(j);
export const getSchema = () => fetch(`${BASE}/api/schema`).then(j);
export const getHealth = () => fetch(`${BASE}/api/health`).then(j);
export const getPresets = (token) => authFetch("/api/presets", token);
export const getMetrics = () => fetch(`${BASE}/api/metrics`).then(j);
export const getDoctorProfile = (token) => authFetch("/api/auth/profile", token);
export const saveDoctorProfile = (token, profile) => authFetch("/api/auth/profile", token, {
  method: "POST", body: JSON.stringify(profile),
});
export const getDoctorRegistrations = (token) => authFetch("/api/master/doctors", token);
export const decideDoctorRegistration = (token, uid, status) =>
  authFetch(`/api/master/doctors/${encodeURIComponent(uid)}/decision`, token, {
    method: "POST", body: JSON.stringify({ status }),
  });
export const predict = (values, signal, token) => authFetch("/api/predict", token, {
  method: "POST", body: JSON.stringify({ values }), signal,
});
export const getPatients = (token, q = "") =>
  authFetch("/api/patients/search", token, {
    method: "POST", body: JSON.stringify({ q }),
  });
export const getPatient = (token, id) => authFetch(`/api/patients/${encodeURIComponent(id)}`, token);
export const createPatient = (token, patient) => authFetch("/api/patients", token, {
  method: "POST", body: JSON.stringify(patient),
});
export const updatePatient = (token, id, patient) => authFetch(`/api/patients/${encodeURIComponent(id)}`, token, {
  method: "PUT", body: JSON.stringify(patient),
});
export const getConsultations = (token, id) =>
  authFetch(`/api/patients/${encodeURIComponent(id)}/consultations`, token);
export const addConsultation = (token, id, consultation) =>
  authFetch(`/api/patients/${encodeURIComponent(id)}/consultations`, token, {
    method: "POST", body: JSON.stringify(consultation),
  });
