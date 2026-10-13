const BASE = import.meta.env.VITE_API_URL ?? "";
const j = async (r) => { if (!r.ok) throw new Error(`${r.status} ${await r.text()}`); return r.json(); };
export const getSchema = () => fetch(`${BASE}/api/schema`).then(j);
export const getPresets = () => fetch(`${BASE}/api/presets`).then(j);
export const getMetrics = () => fetch(`${BASE}/api/metrics`).then(j);
export const predict = (values, signal) =>
  fetch(`${BASE}/api/predict`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }), signal }).then(j);
