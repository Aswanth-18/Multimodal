import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { probToHex } from "./colors.js";

/* Expected local heart asset: models/heart.glb, Y-up, anterior toward +Z, apex toward -Y.
   RCA branch meshes come from BodyParts3D (converted with tools/convert_obj.py). */
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const HEART_C = V(20, 10, 0);
const AXIS = V(0.8, -0.5, 0.33).normalize();       // base -> apex direction
const NEUTRAL_VESSEL_COLOR = 0xc4a39b;

export const VESSELS = {
  LAD: { name: "Left Anterior Descending (LAD)", supplies: "Front wall of the left ventricle and most of the septum" },
  LCX: { name: "Left Circumflex (LCX)", supplies: "Side and back wall of the left ventricle" },
  RCA: { name: "Right Coronary Artery (RCA)", supplies: "Right ventricle, bottom (inferior) wall, SA and AV nodes" },
};
const PROC_RADIUS = { LAD: 2.4, LCX: 2.1, RCA: 2.4 };

function vesselFromName(name = "") {
  const normalized = name.toLowerCase().replace(/[^a-z0-9]/g, "");
  if (normalized.startsWith("lad") || normalized.includes("leftanteriordescending")) return "LAD";
  if (normalized.startsWith("lcx") || normalized.includes("leftcircumflex") || normalized.includes("circumflexartery")) return "LCX";
  if (normalized.startsWith("rca") || normalized.includes("rightcoronary")) return "RCA";
  return null;
}

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

export async function createScene(canvas, { onSelect, onHover, onModelStatus }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 1, 2000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.09; controls.minDistance = 90; controls.maxDistance = 520;
  controls.target.copy(HEART_C).add(V(0, 8, 0));
  camera.position.copy(controls.target).add(V(0, 32, 268));

  scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x2a1e24, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(120, 180, 260); scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.7); rim.position.set(-200, 60, -160); scene.add(rim);

  // ---- local anatomical model or clearly identified schematic fallback ----------------
  const anatomy = new THREE.Group(); scene.add(anatomy);
  const rayc = new THREE.Raycaster();
  const surfaceMeshes = [], assetVessels = { LAD: [], LCX: [], RCA: [] };
  let loadedModel = null;
  const reportModelStatus = (state, message) => onModelStatus?.(state, message);
  reportModelStatus("loading", "Loading local anatomical heart model…");
  try {
    const gltf = await new GLTFLoader().loadAsync(`${import.meta.env.BASE_URL}models/heart.glb`);
    loadedModel = new THREE.Group();
    loadedModel.add(gltf.scene);
    const bounds = new THREE.Box3().setFromObject(loadedModel);
    const size = bounds.getSize(new THREE.Vector3());
    const longestSide = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(longestSide) || longestSide <= 0) throw new Error("The GLB contains no measurable 3D geometry.");
    const scale = 160 / (size.y > 0 ? size.y : longestSide);
    const center = bounds.getCenter(new THREE.Vector3());
    loadedModel.scale.setScalar(scale);
    loadedModel.position.copy(HEART_C).addScaledVector(center, -scale);
    anatomy.add(loadedModel);
    loadedModel.updateMatrixWorld(true);

    const markVessels = (node, inheritedVessel = null) => {
      const vessel = vesselFromName(node.name) || inheritedVessel;
      if (node.isMesh) {
        node.castShadow = true;
        node.receiveShadow = true;
        if (vessel) assetVessels[vessel].push(node);
        else surfaceMeshes.push(node);
      }
      for (const child of node.children) markVessels(child, vessel);
    };
    markVessels(loadedModel);
    if (!surfaceMeshes.length) throw new Error("The GLB contains no heart-surface meshes.");
    reportModelStatus("loaded", "Anatomical Heart Atlas model loaded.");
  } catch (error) {
    if (loadedModel) anatomy.remove(loadedModel);
    surfaceMeshes.length = 0;
    Object.values(assetVessels).forEach(meshes => { meshes.length = 0; });
    const shellMat = new THREE.MeshStandardMaterial({ color: 0xc98a8f, roughness: 0.55, metalness: 0,
      transparent: true, opacity: 0.38, side: THREE.DoubleSide, depthWrite: false });
    const fallbackShell = new THREE.Mesh(heartShellGeometry(), shellMat);
    fallbackShell.renderOrder = 1;
    anatomy.add(fallbackShell);
    surfaceMeshes.push(fallbackShell);
    const ctxMat = shellMat.clone();
    ctxMat.opacity = 0.42;
    ctxMat.color.set(0xb77a80);
    const sph = (c, r, s = [1, 1, 1]) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 24), ctxMat);
      m.position.copy(c);
      m.scale.set(...s);
      m.renderOrder = 1;
      anatomy.add(m);
    };
    sph(V(-26, 24, 2), 20, [1, 1.1, 1]);
    sph(V(30, 44, -22), 21, [1.2, .8, 1]);
    const tube = (pts, r) => {
      const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 40, r, 16), ctxMat);
      m.renderOrder = 1;
      anatomy.add(m);
    };
    tube([V(-7, 38, 8), V(-6, 60, 4), V(-12, 84, -6), V(-30, 98, -18)], 11);
    tube([V(8, 36, 32), V(14, 56, 26), V(22, 72, 8)], 9.5);
    const reason = error instanceof Error ? error.message : "unknown model loading error";
    reportModelStatus("fallback", `Anatomical GLB unavailable (${reason}). Showing schematic fallback; add models/heart.glb.`);
  }

  // ---- prediction-coloured coronary arteries --------------------------------------------
  const snap = (p, off = 1.6) => {
    const d = p.clone().sub(HEART_C).normalize();
    rayc.set(HEART_C.clone().addScaledVector(d, 200), d.clone().negate()); // from outside, inward
    const hit = rayc.intersectObjects(surfaceMeshes, false)[0];
    return hit ? hit.point.clone().addScaledVector(d, off) : p;
  };
  anatomy.updateMatrixWorld(true);

  const mats = {}, riskMaterials = {}, groups = {};
  for (const k of Object.keys(VESSELS)) {
    mats[k] = new THREE.MeshStandardMaterial({ color: NEUTRAL_VESSEL_COLOR, roughness: 0.42, metalness: 0, emissive: 0x000000, side: THREE.DoubleSide });
    riskMaterials[k] = [mats[k]];
    groups[k] = new THREE.Group(); groups[k].name = k; scene.add(groups[k]);
    for (const mesh of assetVessels[k]) {
      const clones = Array.isArray(mesh.material)
        ? mesh.material.map(material => material.clone())
        : mesh.material.clone();
      mesh.material = clones;
      for (const material of Array.isArray(clones) ? clones : [clones]) {
        material.color?.setHex(NEUTRAL_VESSEL_COLOR);
        material.emissive?.setHex(0x000000);
        riskMaterials[k].push(material);
      }
      mesh.userData.vessel = k;
    }
  }
  const addProc = (k, pts, snapFrom = 0) => {
    const P = pts.map((p, i) => i < snapFrom ? p : snap(p));
    const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(P, false, "centripetal"), 70, PROC_RADIUS[k], 12), mats[k]);
    m.userData.vessel = k; groups[k].add(m);
    for (const e of [P[0], P[P.length - 1]]) { const c = new THREE.Mesh(new THREE.SphereGeometry(PROC_RADIUS[k], 12, 10), mats[k]); c.position.copy(e); c.userData.vessel = k; groups[k].add(c); }
  };
  const manifest = await fetch(`${import.meta.env.BASE_URL}models/manifest.json`).then(r => r.json()).catch(() => ({}));
  if (!assetVessels.LAD.length && !(manifest.LAD || []).length)
    addProc("LAD", [V(-6, 44, 10), V(6, 40, 34), V(22, 26, 41), V(38, 6, 38), V(52, -14, 28), V(62, -34, 18)], 1);
  if (!assetVessels.LCX.length && !(manifest.LCX || []).length)
    addProc("LCX", [V(-4, 44, 6), V(14, 46, -2), V(36, 38, -10), V(54, 20, -18), V(58, -2, -24), V(46, -8, -34)], 1);
  if (!assetVessels.RCA.length)
    addProc("RCA", [V(-13, 49, 5), V(-24, 42, 14), V(-31, 35, 17), V(-34, 23, 15), V(-36, 15, 10), V(-32, 10, 0), V(-25, 6, -7), V(-21, 5, -10)], 99);

  const loader = new OBJLoader();
  // The bundled heart mesh already contains the complete RCA. Its normalized, centered
  // coordinates differ from the separately converted millimetre OBJ branch coordinates.
  await Promise.all(Object.entries(manifest).filter(([k]) => groups[k] && !assetVessels[k].length).flatMap(([k, files]) => files.map(f =>
    loader.loadAsync(`${import.meta.env.BASE_URL}models/${f}`).then(o => {
      o.traverse(c => {
        if (c.isMesh) {
          c.material = mats[k];
          c.userData.vessel = k;
          c.geometry.computeVertexNormals();
        }
      });
      groups[k].add(o);
    }).catch(e => console.warn("mesh failed", f, e)))));

  // ---- interaction ---------------------------------------------------------------------
  let needsRender = true, selected = null, hover = null, anim = null, hoverEvt = null;
  let disposed = false, frameHandle = 0;
  const request = () => { needsRender = true; };
  controls.addEventListener("change", request);
  const namedVesselMeshes = Object.values(assetVessels).flat();
  const pick = (e) => {
    const r = canvas.getBoundingClientRect();
    rayc.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const hit = rayc.intersectObjects([...Object.values(groups), ...namedVesselMeshes], true)[0];
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
      for (const material of riskMaterials[k]) material.emissive?.setHex(on ? 0x222222 : h ? 0x111111 : 0x000000);
    }
    request();
  };

  // ---- public API -----------------------------------------------------------------------
  const api = {
    VESSELS,
    setProbabilities(p) {
      const keys = Object.keys(mats);
      const values = Object.fromEntries(keys.map(k => [k, p?.[k.toLowerCase()]]));
      if (keys.some(k => !Number.isFinite(values[k]) || values[k] < 0 || values[k] > 1)) return false;
      anim = {
        t0: performance.now(),
        fromColors: Object.fromEntries(keys.map(k => [k, riskMaterials[k].map(material => material.color.clone())])),
        to: values,
      };
      return true;
    },
    select(v, fromUser = false) { selected = (v === selected && fromUser) ? null : v; applyHighlight(); onSelect?.(selected, fromUser); },
    get selected() { return selected; },
    setView(name) {
      const d = 270, c = HEART_C.clone().add(V(0, 8, 0));
      const dirs = { front: V(0, 0.12, 1), back: V(0, 0.12, -1), left: V(1, 0.1, 0.1), right: V(-1, 0.1, 0.1), top: V(0, 1, 0.2) };
      const position = c.clone().addScaledVector((dirs[name] || dirs.front).clone().normalize(), d);
      viewTransition = { t0: performance.now(), fromPosition: camera.position.clone(), toPosition: position,
        fromTarget: controls.target.clone(), toTarget: c };
      request();
    },
    resize() {
      const r = canvas.parentElement.getBoundingClientRect();
      renderer.setSize(r.width, r.height, false); camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix(); request();
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frameHandle);
      resizeObserver.disconnect();
      controls.dispose();
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
        else object.material?.dispose();
      });
      renderer.dispose();
    },
  };
  let viewTransition = null;
  api.setView("front"); api.resize();
  const resizeObserver = new ResizeObserver(() => api.resize());
  resizeObserver.observe(canvas.parentElement);

  (function loop(now) {
    if (disposed) return;
    frameHandle = requestAnimationFrame(loop);
    if (hoverEvt) { const h = pick(hoverEvt); if (h !== hover) { hover = h; canvas.style.cursor = h ? "pointer" : "grab"; applyHighlight(); onHover?.(h, hoverEvt); } else if (h) onHover?.(h, hoverEvt); }
    if (anim) {
      const t = Math.min(1, (now - anim.t0) / 650), e = t * t * (3 - 2 * t);
      for (const k of Object.keys(mats)) {
        const targetColor = new THREE.Color(probToHex(anim.to[k]));
        for (const [i, material] of riskMaterials[k].entries()) material.color.lerpColors(anim.fromColors[k][i], targetColor, e);
      }
      if (t >= 1) anim = null;
      request();
    }
    if (viewTransition) {
      const t = Math.min(1, (now - viewTransition.t0) / 500), e = t * t * (3 - 2 * t);
      camera.position.lerpVectors(viewTransition.fromPosition, viewTransition.toPosition, e);
      controls.target.lerpVectors(viewTransition.fromTarget, viewTransition.toTarget, e);
      if (t >= 1) viewTransition = null;
      request();
    }
    const moved = controls.update();
    if (needsRender || moved) { needsRender = false; renderer.render(scene, camera); }
  })(performance.now());
  return api;
}
