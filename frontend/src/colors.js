// Single source of truth for the probability -> colour/label mapping.
// The 3D vessels, legend, probability bars and detail cards ALL call these functions,
// so the visualisation can never disagree with the numbers shown in the dashboard.
const STOPS = [
  [0.0, [47, 163, 107]],   // green  - low
  [0.5, [242, 183, 5]],    // amber  - moderate
  [1.0, [215, 38, 61]],    // red    - high
];
export const THRESHOLDS = { low: 0.3, high: 0.6 };

export function probToRGB(p) {
  p = Math.min(1, Math.max(0, p));
  for (let i = 1; i < STOPS.length; i++) {
    if (p <= STOPS[i][0]) {
      const [p0, c0] = STOPS[i - 1], [p1, c1] = STOPS[i];
      const t = (p - p0) / (p1 - p0);
      return c0.map((v, k) => Math.round(v + (c1[k] - v) * t));
    }
  }
  return STOPS[STOPS.length - 1][1];
}
export const probToCSS = (p) => `rgb(${probToRGB(p).join(",")})`;
export const probToHex = (p) => { const [r, g, b] = probToRGB(p); return (r << 16) | (g << 8) | b; };
export function riskLabel(p) {
  return p < THRESHOLDS.low ? "Low" : p < THRESHOLDS.high ? "Moderate" : "High";
}
export const pct = (p) => `${Math.round(p * 100)}%`;
