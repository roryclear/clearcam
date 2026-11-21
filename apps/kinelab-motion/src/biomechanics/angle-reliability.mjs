/**
 * Per-angle reliability flag for display.
 *
 * Side-on, the far (occluded) limb is much less accurate: error spread +69% at the
 * knee and +108% at the hip vs the camera-side limb (Wade 2023, PMC10635560). Low
 * landmark confidence is the other main cause of large errors.
 */
import { ANGLE_DEFINITIONS, bilateralSide } from './angle-calculations.mjs';

/** Mean landmark confidence below which an angle is flagged. */
export const LOW_CONFIDENCE = 0.7;

function conf(p) {
  if (!p) return 0;
  return Math.min(p.visibility ?? 1, p.presence ?? 1);
}

/**
 * @param {Object} angles
 * @param {Array<Object>|null} landmarks
 * @param {{ sagittal2DWeight?: number }|null} view
 * @returns {Object<string, { low: boolean, reason: string|null }>} by angle key
 */
export function angleReliability(angles, landmarks, view) {
  const out = {};
  const sideOn = (view?.sagittal2DWeight ?? 0) >= 0.5;
  const nearSide = {};
  for (const def of ANGLE_DEFINITIONS) {
    const joint = def.key.replace(/^(left|right)/, '');
    if (!(joint in nearSide)) nearSide[joint] = bilateralSide(angles, landmarks, joint);
    const side = def.key.startsWith('left') ? 'left' : 'right';
    if (!Number.isFinite(angles?.[def.key])) {
      out[def.key] = { low: false, reason: null };
      continue;
    }
    const mean = Array.isArray(landmarks)
      ? (conf(landmarks[def.from]) + conf(landmarks[def.joint]) + conf(landmarks[def.to])) / 3
      : 1;
    let reason = null;
    if (sideOn && (nearSide[joint] === 'left' || nearSide[joint] === 'right') && nearSide[joint] !== side) {
      reason = 'Far-side limb: partly hidden side-on, larger error';
    } else if (mean < LOW_CONFIDENCE) {
      reason = `Low landmark confidence (${Math.round(mean * 100)}%)`;
    }
    out[def.key] = { low: reason !== null, reason };
  }
  return out;
}
