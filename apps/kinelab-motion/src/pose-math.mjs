/** Pure, testable 2D geometry for a single-camera movement analysis prototype. */
export const JOINTS = Object.freeze({
  leftShoulder: 11, rightShoulder: 12,
  leftElbow: 13, rightElbow: 14,
  leftWrist: 15, rightWrist: 16,
  leftHip: 23, rightHip: 24,
  leftKnee: 25, rightKnee: 26,
  leftAnkle: 27, rightAnkle: 28,
});

export const ANGLE_DEFINITIONS = Object.freeze([
  { key: 'leftKnee', label: 'LEFT KNEE', from: 23, joint: 25, to: 27, group: 'Lower body' },
  { key: 'rightKnee', label: 'RIGHT KNEE', from: 24, joint: 26, to: 28, group: 'Lower body' },
  { key: 'leftHip', label: 'LEFT HIP', from: 11, joint: 23, to: 25, group: 'Lower body' },
  { key: 'rightHip', label: 'RIGHT HIP', from: 12, joint: 24, to: 26, group: 'Lower body' },
  { key: 'leftElbow', label: 'LEFT ELBOW', from: 11, joint: 13, to: 15, group: 'Upper body' },
  { key: 'rightElbow', label: 'RIGHT ELBOW', from: 12, joint: 14, to: 16, group: 'Upper body' },
  { key: 'leftShoulder', label: 'LEFT SHOULDER', from: 13, joint: 11, to: 23, group: 'Upper body' },
  { key: 'rightShoulder', label: 'RIGHT SHOULDER', from: 14, joint: 12, to: 24, group: 'Upper body' },
]);

export const SKELETON_CONNECTIONS = Object.freeze([
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28], [27, 29], [29, 31], [28, 30], [30, 32],
]);

export function reliable(point, minVisibility = 0.55) {
  return Boolean(point && Number.isFinite(point.x) && Number.isFinite(point.y)
    && (point.visibility === undefined || point.visibility >= minVisibility)
    && (point.presence === undefined || point.presence >= minVisibility));
}

/** The internal angle at B, in image-plane degrees, NOT a calibrated 3D joint angle. */
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

export function getJointAngles(landmarks, width, height, threshold = 0.55) {
  return Object.fromEntries(ANGLE_DEFINITIONS.map(def => [
    def.key,
    angleAt(landmarks?.[def.from], landmarks?.[def.joint], landmarks?.[def.to], width, height, threshold),
  ]));
}

const TRACKED_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
export function trackingStatus(landmarks, threshold = 0.55) {
  if (!Array.isArray(landmarks) || landmarks.length < 29) return 'lost';
  const tracked = TRACKED_INDICES.filter(index => reliable(landmarks[index], threshold)).length;
  if (tracked >= 10) return 'good';
  return tracked >= 5 ? 'partial' : 'lost';
}

/** Hysteresis-based movement counting: a down phase followed by a clear return to extension. */
export function createSquatState() {
  return {
    phase: 'ready', reps: 0, downFrames: 0, upFrames: 0,
    enteredDownAt: null, lastRepAt: -Infinity, status: 'waiting',
  };
}

export function advanceSquat(state, kneeAngle, timestampMs) {
  const next = { ...state };
  if (!Number.isFinite(kneeAngle)) {
    next.status = 'tracking-lost';
    next.phase = 'ready';
    next.enteredDownAt = null;
    next.downFrames = 0;
    next.upFrames = 0;
    return next;
  }
  if (state.phase === 'ready') {
    next.upFrames = 0;
    next.downFrames = kneeAngle <= 115 ? next.downFrames + 1 : 0;
    if (next.downFrames >= 2) {
      next.phase = 'down';
      next.enteredDownAt = timestampMs;
      next.downFrames = 0;
      next.status = 'rise';
    } else {
      next.status = kneeAngle > 145 ? 'ready' : 'lowering';
    }
  } else {
    next.downFrames = 0;
    next.upFrames = kneeAngle >= 155 ? next.upFrames + 1 : 0;
    if (next.upFrames >= 2 && timestampMs - next.enteredDownAt >= 350
      && timestampMs - next.lastRepAt >= 500) {
      next.phase = 'ready';
      next.reps += 1;
      next.upFrames = 0;
      next.lastRepAt = timestampMs;
      next.status = 'completed';
    } else {
      next.status = kneeAngle >= 155 ? 'finishing' : 'rise';
    }
  }
  return next;
}

export function dominantKneeAngle(angles) {
  const values = [angles?.leftKnee, angles?.rightKnee].filter(Number.isFinite);
  if (!values.length) return null;
  // Average if both legs are visible; use the tracked leg otherwise.
  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length * 10) / 10;
}
