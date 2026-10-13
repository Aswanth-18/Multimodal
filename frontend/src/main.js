import { createScene } from "./scene.js";
import { getSchema, getPresets, getMetrics, predict } from "./api.js";
import { probToCSS, riskLabel, pct } from "./colors.js";

const $ = (s, r = document) => r.querySelector(s);
const el = (t, props = {}, html = "") => Object.assign(Object.assign(document.createElement(t), props), html ? { innerHTML: html } : {});
const TARGET_LABEL = { cad: "Overall CAD", lad: "LAD", lcx: "LCX", rca: "RCA" };
const CHIP = (p) => `<span class="chip" style="background:${probToCSS(p)}">${riskLabel(p)}</span>`;

const state = { schema: [], values: {}, result: null, target: "cad", selected: null, scene: null, ctrl: null, metrics: null };

// ---------- form (generated from /api/schema, so new features need no UI code) ----------
function buildForm() {
  const host = $("#form"); host.innerHTML = "";
  const cats = [...new Set(state.schema.map(f => f.category))];
  cats.forEach((cat, i) => {
    const d = el("details", { open: i < 2 }); d.append(el("summary", {}, cat));
    const wrap = el("div", { className: "fields" });
    for (const f of state.schema.filter(x => x.category === cat)) wrap.append(field(f));
    d.append(wrap); host.append(d);
  });
}
function field(f) {
  const w = el("div", { className: "field" }), id = "f_" + f.name.replace(/\W/g, "_");
  const unit = f.unit ? ` <span class="muted">${f.unit}</span>` : "";
  const lab = el("label", { htmlFor: id }, `<span>${f.label}${unit}</span><b data-val="${f.name}"></b>`);
  w.append(lab);
  const set = (v) => { state.values[f.name] = Number(v); $(`[data-val="${f.name}"]`, w).textContent = display(f, v); schedule(); };
  if (f.options) {
    const s = el("select", { id }); for (const [k, v] of Object.entries(f.options)) s.append(el("option", { value: k }, v));
    s.value = String(Math.round(state.values[f.name])); s.oninput = () => set(s.value); w.append(s); w._sync = (v) => { s.value = String(Math.round(v)); $(`[data-val="${f.name}"]`, w).textContent = display(f, v); };
  } else {
    const step = (f.max - f.min) > 100 ? 1 : (f.max - f.min) > 10 ? 0.5 : 0.1;
    const lo = Math.floor(f.min - (f.max - f.min) * 0.05), hi = Math.ceil(f.max + (f.max - f.min) * 0.05);
    const row = el("div", { className: "num" });
    const r = el("input", { type: "range", min: lo, max: hi, step, value: state.values[f.name] });
    const n = el("input", { type: "number", id, step, value: state.values[f.name], "aria-label": f.label });
    r.oninput = () => { n.value = r.value; set(r.value); };
    n.oninput = () => { if (n.value !== "") { r.value = n.value; set(n.value); } };
    row.append(r, n); w.append(row);
    w._sync = (v) => { r.value = n.value = Number(v).toFixed(step < 1 ? 1 : 0); $(`[data-val="${f.name}"]`, w).textContent = ""; };
  }
  queueMicrotask(() => { $(`[data-val="${f.name}"]`, w).textContent = f.options ? display(f, state.values[f.name]) : ""; });
  w.dataset.name = f.name; return w;
}
const display = (f, v) => f.options ? (f.options[String(Math.round(v))] ?? v) : v;
function setAll(values) {
  for (const f of state.schema) {
    state.values[f.name] = values[f.name] ?? f.default;
    const w = $(`.field[data-name="${CSS.escape(f.name)}"]`); w?._sync?.(state.values[f.name]);
  }
  schedule(0);
}

// ---------- prediction (debounced, cancels stale requests => live updates) ----------
let timer, ctl;
function schedule(delay = 300) { clearTimeout(timer); timer = setTimeout(run, delay); }
async function run() {
  ctl?.abort(); ctl = new AbortController();
  setStatus("Predicting…");
  try {
    state.result = await predict(state.values, ctl.signal);
    setStatus("Live · updated"); render();
    state.scene?.setProbabilities(state.result.probabilities);
  } catch (e) { if (e.name !== "AbortError") setStatus("API unreachable — is the backend running?", true); }
}
const setStatus = (t, err) => { const s = $("#status"); s.textContent = t; s.classList.toggle("err", !!err); };

// ---------- rendering ----------
function render() {
  const r = state.result, P = r.probabilities;
  const cad = P.cad, pos = cad >= 0.5;
  $("#cad-card").innerHTML = `<div class="lbl">Overall coronary artery disease</div>
    <div class="row"><span class="verdict" style="color:${probToCSS(cad)}">${pos ? "CAD likely" : "CAD unlikely"}</span><span class="p">${pct(cad)}</span></div>
    <div class="meter"><i style="width:${cad * 100}%;background:${probToCSS(cad)}"></i></div>
    <div class="muted small" style="margin-top:6px">Predicted probability of ≥50% narrowing in a major vessel · ${CHIP(cad)}</div>`;
  const rows = $("#vessel-rows"); rows.innerHTML = "";
  for (const v of ["lad", "lcx", "rca"]) {
    const b = el("button", { className: "vrow" + (state.selected === v.toUpperCase() ? " sel" : ""), title: "Select in 3D view" },
      `<span class="nm">${v.toUpperCase()}</span><span class="meter"><i style="width:${P[v] * 100}%;background:${probToCSS(P[v])}"></i></span><span class="v">${pct(P[v])} ${CHIP(P[v])}</span>`);
    b.onclick = () => state.scene.select(v.toUpperCase(), true);
    rows.append(b);
  }
  const seg = $("#target-seg"); seg.innerHTML = "";
  for (const t of Object.keys(TARGET_LABEL)) {
    const b = el("button", { className: t === state.target ? "on" : "" }, TARGET_LABEL[t]);
    b.onclick = () => { state.target = t; render(); }; seg.append(b);
  }
  drivers(); vesselCard();
}
function drivers() {
  const r = state.result, t = state.target, ex = r.explanations[t];
  $("#explain-note").textContent = `${TARGET_LABEL[t]} · model: ${r.model[t]}. Red bars push this patient's risk up, blue bars push it down (SHAP values), shown with each measurement's relative share of the explanation.`;
  const mx = Math.max(...ex.map(e => Math.abs(e.contribution))) || 1;
  $("#drivers").innerHTML = ex.map(e => {
    const w = Math.abs(e.contribution) / mx * 50, up = e.contribution > 0;
    const val = e.display ?? `${Number(e.value.toFixed(2))}${e.unit ? " " + e.unit : ""}`;
    return `<div class="drv"><div class="n" title="${e.label}">${e.label} <small>= ${val}</small></div><div class="s">${up ? "▲" : "▼"} ${(e.share * 100).toFixed(0)}%</div>
      <div class="diverge"><i style="${up ? `left:50%` : `right:50%`};width:${w}%;background:var(${up ? "--up" : "--down"})"></i></div></div>`;
  }).join("") + `<div class="axis"><span>◀ lowers risk</span><span>raises risk ▶</span></div>`;
}
function vesselCard() {
  const c = $("#vcard"), v = state.selected;
  if (!v || !state.result) { c.hidden = true; return; }
  const k = v.toLowerCase(), p = state.result.probabilities[k], info = state.scene.VESSELS[v];
  const top = state.result.explanations[k].slice(0, 3);
  c.hidden = false;
  c.innerHTML = `<button class="x" aria-label="Close">×</button><h4>${info.name}</h4><div class="sup">Supplies: ${info.supplies}</div>
    <div class="big"><b style="color:${probToCSS(p)}">${pct(p)}</b>${CHIP(p)}</div><div class="muted small">stenosis probability</div>
    <ul>${top.map(e => `<li><span>${e.label}</span><span style="color:var(--${e.contribution > 0 ? "up" : "down"})">${e.contribution > 0 ? "▲ raises" : "▼ lowers"}</span></li>`).join("")}</ul>`;
  $(".x", c).onclick = () => state.scene.select(null);
}

// ---------- model-performance tab ----------
function perfTab() {
  const m = state.metrics, host = $("#tab-perf"); if (!m) return;
  const f = (x) => x.toFixed(3);
  const sel = Object.entries(m.selected).map(([t, s]) => `<tr class="best"><td>${t.toUpperCase()}</td><td>${s.model}</td><td>${f(s.accuracy)}</td><td>${f(s.precision)}</td><td>${f(s.recall)}</td><td>${f(s.f1)}</td><td>${f(s.roc_auc)}</td></tr>`).join("");
  const bestBy = Object.fromEntries(Object.entries(m.selected).map(([t, s]) => [t.toUpperCase(), s.model]));
  const all = m.cv_all.map(r => `<tr class="${bestBy[r.target] === r.model ? "best" : ""}"><td>${r.target}</td><td>${r.model}</td><td>${f(r.accuracy)}</td><td>${f(r.precision)}</td><td>${f(r.recall)}</td><td>${f(r.f1)}</td><td>${f(r.roc_auc)}±${r.roc_auc_std.toFixed(2)}</td></tr>`).join("");
  const head = "<tr><th>Target</th><th>Model</th><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th><th>AUC</th></tr>";
  host.innerHTML = `<h3>Deployed models (calibrated, out-of-fold)</h3><p class="muted small">5-fold stratified cross-validated predictions on 303 patients; threshold 0.5.</p><table>${head}${sel}</table>
    <h3 style="margin-top:16px">Model comparison (5-fold × 3 repeats)</h3><p class="muted small">Mean over 15 folds. Highlighted = selected by ROC-AUC.</p><table>${head}${all}</table>
    <img src="/results/roc_curves.png" alt="ROC curves" onerror="this.remove()">`;
}

// ---------- boot ----------
async function boot() {
  try {
    [state.schema, state.presets, state.metrics] = await Promise.all([getSchema(), getPresets(), getMetrics().catch(() => null)]);
  } catch (e) { setStatus("API unreachable — start the backend (see README)", true); return; }
  for (const f of state.schema) state.values[f.name] = f.default;
  buildForm(); perfTab();
  const pb = $("#presets");
  const reset = el("button", {}, "Median patient"); reset.onclick = () => setAll({}); pb.append(reset);
  for (const [name, vals] of Object.entries(state.presets)) { const b = el("button", {}, name.replace(" example", "")); b.onclick = () => setAll(vals); pb.append(b); }
  document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === t));
    $("#tab-risk").hidden = t.dataset.tab !== "risk"; $("#tab-perf").hidden = t.dataset.tab !== "perf";
  });
  $("#viewbar").onclick = (e) => e.target.dataset.v && state.scene.setView(e.target.dataset.v);
  const tip = $("#tip");
  state.scene = await createScene($("#c"), {
    onSelect(v) { state.selected = v; if (v) state.target = v.toLowerCase(); else if (state.target !== "cad") state.target = "cad"; if (state.result) render(); },
    onHover(v, e) {
      if (!v || !e) { tip.hidden = true; return; }
      const r = $(".stage").getBoundingClientRect(), p = state.result?.probabilities[v.toLowerCase()];
      tip.hidden = false; tip.style.left = e.clientX - r.left + 14 + "px"; tip.style.top = e.clientY - r.top + 14 + "px";
      tip.innerHTML = `<b>${v}</b>${p != null ? ` · ${pct(p)} (${riskLabel(p)})` : ""}`;
    },
  });
  setAll(state.presets["Low-risk example"] ?? {});
}
boot();
