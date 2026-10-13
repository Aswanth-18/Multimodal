import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { probToHex } from "./colors.js";

/* Coordinate frame (mm): origin near the heart centre, +x = patient's left, +y = up, +z = anterior.
   RCA branch meshes come from BodyParts3D (converted with tools/convert_obj.py). Anything without a mesh in
   models/manifest.json (LAD, LCX, RCA trunk, heart shell) is generated procedurally. */
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const HEART_C = V(20, 10, 0);
const AXIS = V(0.8, -0.5, 0.33).normalize();       // base -> apex direction

export const VESSELS = {
  LAD: { name: "Left Anterior Descending (LAD)", supplies: "Front wall of the left ventricle and most of the septum" },
  LCX: { name: "Left Circumflex (LCX)", supplies: "Side and back wall of the left ventricle" },
  RCA: { name: "Right Coronary Artery (RCA)", supplies: "Right ventricle, bottom (inferior) wall, SA and AV nodes" },
};
const PROC_RADIUS = { LAD: 2.4, LCX: 2.1, RCA: 2.4 };

function heartShellGeometry() {
  const g = new THREE.SphereGeometry(1, 72, 56);
  const p = g.attributes.position;
  const ez = AXIS.clone(), ex = V(0, 1, 0).cross(ez).normalize(), ey = ez.clone().cross(ex);
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    let a = v.dot(ez), b = v.dot(ex), c = v.dot(ey);
    const t = Math.max(0, a);
    const taper = 1 - 0.55 * t * t;                       // pointed apex
    const flat = c < 0 ? 0.85 : 1;                        // flatter diaphragmatic surface
    v.copy(ez).multiplyScalar(a * 66).addScaledVector(ex, b * 42 * taper).addScaledVector(ey, c * 36 * taper * flat).add(HEART_C);
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}

export async function createScene(canvas, { onSelect, onHover }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 1, 2000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.09; controls.minDistance = 90; controls.maxDistance = 520;
  controls.target.copy(HEART_C).add(V(0, 8, 0));

  scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x2a1e24, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(120, 180, 260); scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.7); rim.position.set(-200, 60, -160); scene.add(rim);

  // ---- anatomy (context) ---------------------------------------------------------------
  const anatomy = new THREE.Group(); scene.add(anatomy);
  const shellMat = new THREE.MeshStandardMaterial({ color: 0xc98a8f, roughness: 0.55, metalness: 0, transparent: true,
    opacity: 0.38, side: THREE.DoubleSide, depthWrite: false });
  const shell = new THREE.Mesh(heartShellGeometry(), shellMat); shell.renderOrder = 1; anatomy.add(shell);
  const ctxMat = shellMat.clone(); ctxMat.opacity = 0.42; ctxMat.color.set(0xb77a80);
  const sph = (c, r, s = [1, 1, 1]) => { const m = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 24), ctxMat); m.position.copy(c); m.scale.set(...s); m.renderOrder = 1; anatomy.add(m); };
  sph(V(-26, 24, 2), 20, [1, 1.1, 1]);              // right atrium
  sph(V(30, 44, -22), 21, [1.2, .8, 1]);            // left atrium (posterior)
  const tube = (pts, r, mat = ctxMat) => { const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 40, r, 16), mat); m.renderOrder = 1; anatomy.add(m); return m; };
  tube([V(-7, 38, 8), V(-6, 60, 4), V(-12, 84, -6), V(-30, 98, -18)], 11);   // aorta
  tube([V(8, 36, 32), V(14, 56, 26), V(22, 72, 8)], 9.5);                        // pulmonary trunk

  // ---- vessels ---------------------------------------------------------------------------
  const rayc = new THREE.Raycaster();
  const snap = (p, off = 1.6) => {
    const d = p.clone().sub(HEART_C).normalize();
    rayc.set(HEART_C.clone().addScaledVector(d, 200), d.clone().negate()); // from outside, inward
    const hit = rayc.intersectObject(shell)[0];
    return hit ? hit.point.clone().addScaledVector(d, off) : p;
  };
  shell.updateMatrixWorld(true);

  const mats = {}, groups = {};
  for (const k of Object.keys(VESSELS)) {
    mats[k] = new THREE.MeshStandardMaterial({ color: probToHex(0), roughness: 0.4, metalness: 0.05, emissive: 0x000000, side: THREE.DoubleSide });
    groups[k] = new THREE.Group(); groups[k].name = k; scene.add(groups[k]);
  }
  const addProc = (k, pts, snapFrom = 0) => {
    const P = pts.map((p, i) => i < snapFrom ? p : snap(p));
    const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(P, false, "centripetal"), 70, PROC_RADIUS[k], 12), mats[k]);
    m.userData.vessel = k; groups[k].add(m);
    for (const e of [P[0], P[P.length - 1]]) { const c = new THREE.Mesh(new THREE.SphereGeometry(PROC_RADIUS[k], 12, 10), mats[k]); c.position.copy(e); c.userData.vessel = k; groups[k].add(c); }
  };
  const manifest = await fetch(`${import.meta.env.BASE_URL}models/manifest.json`).then(r => r.json()).catch(() => ({}));
  if (!(manifest.LAD || []).length)
    addProc("LAD", [V(-6, 44, 10), V(6, 40, 34), V(22, 26, 41), V(38, 6, 38), V(52, -14, 28), V(62, -34, 18)], 1);
  if (!(manifest.LCX || []).length)
    addProc("LCX", [V(-4, 44, 6), V(14, 46, -2), V(36, 38, -10), V(54, 20, -18), V(58, -2, -24), V(46, -8, -34)], 1);
  // RCA trunk passes through the proximal ends of the real BodyParts3D branch meshes
  addProc("RCA", [V(-13, 49, 5), V(-24, 42, 14), V(-31, 35, 17), V(-34, 23, 15), V(-36, 15, 10), V(-32, 10, 0), V(-25, 6, -7), V(-21, 5, -10)], 99);

  const loader = new OBJLoader();
  await Promise.all(Object.entries(manifest).filter(([k]) => groups[k]).flatMap(([k, files]) => files.map(f =>
    loader.loadAsync(`${import.meta.env.BASE_URL}models/${f}`).then(o => {
      o.traverse(c => { if (c.isMesh) { c.material = mats[k]; c.userData.vessel = k; c.geometry.computeVertexNormals(); } });
      groups[k].add(o);
    }).catch(e => console.warn("mesh failed", f, e)))));

  // ---- interaction ---------------------------------------------------------------------
  let needsRender = true, selected = null, hover = null, anim = null, hoverEvt = null;
  const request = () => { needsRender = true; };
  controls.addEventListener("change", request);
  const pick = (e) => {
    const r = canvas.getBoundingClientRect();
    rayc.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const hit = rayc.intersectObjects(Object.values(groups), true)[0];
    return hit ? hit.object.userData.vessel : null;
  };
  let down = null;
  canvas.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener("pointerup", e => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) api.select(pick(e), true);
    down = null;
  });
  canvas.addEventListener("pointermove", e => { hoverEvt = e; });
  canvas.addEventListener("pointerleave", () => { hoverEvt = null; if (hover) { hover = null; onHover?.(null); } });

  const applyHighlight = () => {
    for (const k of Object.keys(mats)) {
      const on = k === selected, h = k === hover && !on;
      mats[k].emissive.setHex(on ? 0x666666 : h ? 0x333333 : 0x000000);
    }
    request();
  };

  // ---- public API -----------------------------------------------------------------------
  const cur = {}, api = {
    VESSELS,
    setProbabilities(p) {   // animate colours towards new predictions
      const from = Object.fromEntries(Object.keys(mats).map(k => [k, cur[k] ?? 0]));
      const to = Object.fromEntries(Object.keys(mats).map(k => [k, p[k.toLowerCase()] ?? cur[k] ?? 0]));
      anim = { t0: performance.now(), from, to };
    },
    select(v, fromUser = false) { selected = (v === selected && fromUser) ? null : v; applyHighlight(); onSelect?.(selected, fromUser); },
    get selected() { return selected; },
    setView(name) {
      const d = 270, c = controls.target.clone();
      const dirs = { front: V(0, 0.12, 1), back: V(0, 0.12, -1), left: V(1, 0.1, 0.1), right: V(-1, 0.1, 0.1), top: V(0, 1, 0.2) };
      camera.position.copy(c).addScaledVector((dirs[name] || dirs.front).clone().normalize(), d); camera.lookAt(c); controls.update(); request();
    },
    resize() {
      const r = canvas.parentElement.getBoundingClientRect();
      renderer.setSize(r.width, r.height, false); camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix(); request();
    },
  };
  api.setView("front"); api.resize();
  new ResizeObserver(() => api.resize()).observe(canvas.parentElement);

  const col = new THREE.Color();
  (function loop(now) {
    requestAnimationFrame(loop);
    if (hoverEvt) { const h = pick(hoverEvt); if (h !== hover) { hover = h; canvas.style.cursor = h ? "pointer" : "grab"; applyHighlight(); onHover?.(h, hoverEvt); } else if (h) onHover?.(h, hoverEvt); }
    if (anim) {
      const t = Math.min(1, (now - anim.t0) / 650), e = t * t * (3 - 2 * t);
      for (const k of Object.keys(mats)) { cur[k] = anim.from[k] + (anim.to[k] - anim.from[k]) * e; mats[k].color.setHex(probToHex(cur[k])); }
      if (t >= 1) anim = null; request();
    }
    const moved = controls.update();
    if (needsRender || moved) { needsRender = false; renderer.render(scene, camera); }
  })(performance.now());
  return api;
}
