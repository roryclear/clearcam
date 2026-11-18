/**
 * 2D projected joint angle calculations, geometric metrics, and confidence indicators.
 */

import { JOINTS, KEY_TRACKED_INDICES } from './landmark-indices.mjs';

export const ANGLE_DEFINITIONS = Object.freeze([
  { key: 'leftKnee', label: 'LEFT KNEE', from: JOINTS.leftHip, joint: JOINTS.leftKnee, to: JOINTS.leftAnkle, group: 'Lower body' },
  { key: 'rightKnee', label: 'RIGHT KNEE', from: JOINTS.rightHip, joint: JOINTS.rightKnee, to: JOINTS.rightAnkle, group: 'Lower body' },
  { key: 'leftHip', label: 'LEFT HIP', from: JOINTS.leftShoulder, joint: JOINTS.leftHip, to: JOINTS.leftKnee, group: 'Lower body' },
  { key: 'rightHip', label: 'RIGHT HIP', from: JOINTS.rightShoulder, joint: JOINTS.rightHip, to: JOINTS.rightKnee, group: 'Lower body' },
  { key: 'leftElbow', label: 'LEFT ELBOW', from: JOINTS.leftShoulder, joint: JOINTS.leftElbow, to: JOINTS.leftWrist, group: 'Upper body' },
  { key: 'rightElbow', label: 'RIGHT ELBOW', from: JOINTS.rightShoulder, joint: JOINTS.rightElbow, to: JOINTS.rightWrist, group: 'Upper body' },
  { key: 'leftShoulder', label: 'LEFT SHOULDER', from: JOINTS.leftElbow, joint: JOINTS.leftShoulder, to: JOINTS.leftHip, group: 'Upper body' },
  { key: 'rightShoulder', label: 'RIGHT SHOULDER', from: JOINTS.rightElbow, joint: JOINTS.rightShoulder, to: JOINTS.rightHip, group: 'Upper body' },
]);

/**
 * Checks whether a landmark point has valid coordinates and satisfies visibility threshold.
 */
export function reliable(point, minVisibility = 0.55) {
  return Boolean(
    point &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    (point.visibility === undefined || point.visibility >= minVisibility) &&
    (point.presence === undefined || point.presence >= minVisibility)
  );
}

/**
 * The internal angle at joint B formed by rays BA and BC, in image-plane degrees.
 * Uses canvas pixel width/height to avoid aspect-ratio skew.
 * Returns null if points are occluded, degenerate, or invalid.
 */
export function angleAt(a, b, c, width, height, minVisibility = 0.55) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return null;
  if (![a, b, c].every(point => reliable(point, minVisibility))) return null;

  const ux = (a.x - b.x) * width;
  const uy = (a.y - b.y) * height;
  const vx = (c.x - b.x) * width;
  const vy = (c.y - b.y) * height;

  const denominator = Math.hypot(ux, uy) * Math.hypot(vx, vy);
  if (denominator < 1e-6) return null;

  const cos = Math.max(-1, Math.min(1, (ux * vx + uy * vy) / denominator));
  return Math.round(Math.acos(cos) * 18000 / Math.PI) / 100;
}

/**
 * Computes an estimated uncertainty score [0, 1] for an angle based on landmark visibility.
 * 0 = high confidence, 1 = maximum uncertainty.
 */
export function angleUncertainty(a, b, c) {
  if (!a || !b || !c) return 1.0;
  const va = a.visibility ?? 0.5;
  const vb = b.visibility ?? 0.5;
  const vc = c.visibility ?? 0.5;
  const avg = (va + vb + vc) / 3;
  return Math.max(0, Math.min(1, 1 - avg));
}

/**
 * Calculates joint angles for all 8 standard definitions.
 */
export function getJointAngles(landmarks, width, height, threshold = 0.55) {
  return Object.fromEntries(
    ANGLE_DEFINITIONS.map(def => [
      def.key,
      angleAt(landmarks?.[def.from], landmarks?.[def.joint], landmarks?.[def.to], width, height, threshold),
    ])
  );
}

/**
 * Checks whether a 3D metric world landmark has valid spatial coordinates and satisfies visibility threshold.
 */
export function reliable3D(point, minVisibility = 0.55) {
  return Boolean(
    point &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    Number.isFinite(point.z ?? 0) &&
    (point.visibility === undefined || point.visibility >= minVisibility) &&
    (point.presence === undefined || point.presence >= minVisibility)
  );
}

/**
 * Calculates the perspective-invariant 3D spatial joint angle at joint B
 * formed by rays BA and BC using real-world metric coordinates (in meters).
 * Eliminates 2D camera foreshortening and oblique perspective distortions.
 * Returns null if points are occluded, degenerate, or zero-length.
 */
export function angle3DAt(a, b, c, minVisibility = 0.55) {
  if (![a, b, c].every(point => reliable3D(point, minVisibility))) return null;

  const ux = a.x - b.x;
  const uy = a.y - b.y;
  const uz = (a.z ?? 0) - (b.z ?? 0);

  const vx = c.x - b.x;
  const vy = c.y - b.y;
  const vz = (c.z ?? 0) - (b.z ?? 0);

  const magU = Math.hypot(ux, uy, uz);
  const magV = Math.hypot(vx, vy, vz);

  if (magU < 1e-6 || magV < 1e-6) return null;

  const dot = ux * vx + uy * vy + uz * vz;
  const cos = Math.max(-1, Math.min(1, dot / (magU * magV)));
  return Math.round(Math.acos(cos) * 18000 / Math.PI) / 100;
}

/**
 * Calculates joint angles for all 8 standard definitions using 3D metric worldLandmarks.
 */
export function getJointAngles3D(worldLandmarks, threshold = 0.55) {
  return Object.fromEntries(
    ANGLE_DEFINITIONS.map(def => [
      def.key,
      angle3DAt(worldLandmarks?.[def.from], worldLandmarks?.[def.joint], worldLandmarks?.[def.to], threshold),
    ])
  );
}

/**
 * Calculates trunk inclination angle relative to vertical (in degrees).
 * Useful for posture monitoring and sit-to-stand trunk lean.
 */
export function calculateTrunkAngle(landmarks, width, height, threshold = 0.55) {
  if (!Array.isArray(landmarks)) return null;
  const ls = landmarks[JOINTS.leftShoulder];
  const rs = landmarks[JOINTS.rightShoulder];
  const lh = landmarks[JOINTS.leftHip];
  const rh = landmarks[JOINTS.rightHip];

  if (!reliable(ls, threshold) || !reliable(rs, threshold) ||
      !reliable(lh, threshold) || !reliable(rh, threshold)) {
    return null;
  }

  // Mid-shoulder and mid-hip in pixel space
  const midShoulderX = ((ls.x + rs.x) / 2) * width;
  const midShoulderY = ((ls.y + rs.y) / 2) * height;
  const midHipX = ((lh.x + rh.x) / 2) * width;
  const midHipY = ((lh.y + rh.y) / 2) * height;

  const dx = midShoulderX - midHipX;
  const dy = midShoulderY - midHipY; // pointing up is negative y

  // Angle with vertical (0 deg = upright vertical)
  const angleRad = Math.atan2(Math.abs(dx), -dy);
  return Math.round((angleRad * 180 / Math.PI) * 10) / 10;
}

/**
 * Assesses overall body visibility and tracking stability.
 */
export function trackingStatus(landmarks, threshold = 0.55) {
  if (!Array.isArray(landmarks) || landmarks.length < 29) return 'lost';
  const tracked = KEY_TRACKED_INDICES.filter(index => reliable(landmarks[index], threshold)).length;
  if (tracked >= 10) return 'good';
  return tracked >= 5 ? 'partial' : 'lost';
}

/**
 * Calculates representative knee angle (averaging visible legs or single leg).
 */
export function dominantKneeAngle(angles) {
  const values = [angles?.leftKnee, angles?.rightKnee].filter(Number.isFinite);
  if (!values.length) return null;
  return Math.round((values.reduce((sum, v) => sum + v, 0) / values.length) * 10) / 10;
}
