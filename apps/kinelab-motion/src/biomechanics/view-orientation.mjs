/**
 * Estimates how the body is turned relative to the camera, so each joint angle can
 * be read from the representation that is most accurate for that view.
 *
 * yaw 0° = pure lateral (side-on) view, 90° = facing the camera. The estimate uses
 * the shoulder and hip lines of the 3D world landmarks and is smoothed over time:
 * per-frame depth is noisy, but body orientation changes slowly.
 */

import { JOINTS } from './landmark-indices.mjs';

function lateralYaw(a, b) {
  if (!a || !b) return null;
  const dx = b.x - a.x;
  const dz = (b.z ?? 0) - (a.z ?? 0);
  const len = Math.hypot(dx, dz);
  if (len < 0.05) return null;
  // Lateral axis along depth (dz) => side view (yaw 0); along x => frontal (yaw 90).
  return Math.atan2(Math.abs(dx), Math.abs(dz)) * 180 / Math.PI;
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export class ViewOrientationEstimator {
  /**
   * @param {Object} [o]
   * @param {number} [o.timeConstantMs=600] EMA time constant for the yaw estimate
   * @param {number} [o.minVisibility=0.5]
   */
  constructor({ timeConstantMs = 600, minVisibility = 0.5 } = {}) {
    this.timeConstantMs = timeConstantMs;
    this.minVisibility = minVisibility;
    this.reset();
  }

  reset() {
    this.yawDeg = null;
    this.lastTs = null;
  }

  _visible(p) {
    return p && Number.isFinite(p.x) && (p.visibility ?? 1) >= this.minVisibility;
  }

  /**
   * @param {Array<Object>|null} world 33 world landmarks
   * @param {number} timestampMs
   * @returns {{ yawDeg: number|null, sagittal2DWeight: number, frontal2DWeight: number }}
   */
  update(world, timestampMs) {
    if (Array.isArray(world)) {
      const ls = world[JOINTS.leftShoulder];
      const rs = world[JOINTS.rightShoulder];
      // Shoulders are rarely occluded in either view; hips are used only as a fallback
      // because the far hip is pulled toward the near one in lateral views.
      let sample = this._visible(ls) && this._visible(rs) ? lateralYaw(ls, rs) : null;
      if (sample === null) {
        const lh = world[JOINTS.leftHip];
        const rh = world[JOINTS.rightHip];
        if (this._visible(lh) && this._visible(rh)) sample = lateralYaw(lh, rh);
      }
      if (sample !== null) {
        if (this.yawDeg === null || this.lastTs === null) {
          this.yawDeg = sample;
        } else {
          const dt = Math.max(0, timestampMs - this.lastTs);
          const alpha = 1 - Math.exp(-dt / this.timeConstantMs);
          this.yawDeg += alpha * (sample - this.yawDeg);
        }
        this.lastTs = timestampMs;
      }
    }
    return this.getWeights();
  }

  /**
   * 2D weights: sagittal-plane angles (knee/hip flexion) project faithfully in a side
   * view; frontal-plane movement (e.g. shoulder abduction) in a front view.
   */
  getWeights() {
    if (this.yawDeg === null) return { yawDeg: null, sagittal2DWeight: 0, frontal2DWeight: 0 };
    return {
      yawDeg: Math.round(this.yawDeg * 10) / 10,
      sagittal2DWeight: 1 - smoothstep(35, 55, this.yawDeg),
      frontal2DWeight: smoothstep(35, 55, this.yawDeg),
    };
  }
}

/** Angle keys measured in the sagittal plane for the exercises this app tracks. */
export const SAGITTAL_ANGLE_KEYS = Object.freeze(['leftKnee', 'rightKnee', 'leftHip', 'rightHip']);

/**
 * Blends 2D and 3D angles by view: lower-limb sagittal angles use the 2D image
 * angle in lateral views (precise) and the 3D world angle in oblique/frontal views
 * (no foreshortening). Other joints keep the 3D angle, falling back to 2D.
 */
export function blendAnglesByView(angles2D, angles3D, weights) {
  const out = {};
  const keys = new Set([...Object.keys(angles2D || {}), ...Object.keys(angles3D || {})]);
  for (const key of keys) {
    const a2 = angles2D?.[key];
    const a3 = angles3D?.[key];
    const w = SAGITTAL_ANGLE_KEYS.includes(key) ? (weights?.sagittal2DWeight ?? 0) : 0;
    if (Number.isFinite(a2) && Number.isFinite(a3)) out[key] = Math.round((w * a2 + (1 - w) * a3) * 100) / 100;
    else out[key] = Number.isFinite(a3) ? a3 : (Number.isFinite(a2) ? a2 : null);
  }
  return out;
}
