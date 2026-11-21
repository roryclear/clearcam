/**
 * Squat movement module with hysteresis, minimum phase duration, and tracking loss protection.
 * Preserves the exact algorithm and behavior of the existing KineLab squat prototype.
 */

import { formatAngle } from '../biomechanics/angle-convention.mjs';
import { BaseExercise } from './exercise-base.mjs';
import { JOINTS } from '../biomechanics/landmark-indices.mjs';
import { selectBilateralAngle, reliable } from '../biomechanics/angle-calculations.mjs';

/**
 * Vertical hip-above-ankle span in the image, relative to the standing span.
 * Rotation about the vertical axis does not change vertical image positions, and
 * no depth is involved, so this squat-depth cue is unaffected by camera yaw and by
 * monocular depth noise (which can hold 3D knee angles below the "standing" threshold).
 * @returns {number|null} current span in normalized image units
 */
export function hipAnkleSpan(landmarks, minVisibility = 0.5) {
  if (!Array.isArray(landmarks)) return null;
  const hips = [JOINTS.leftHip, JOINTS.rightHip].map(i => landmarks[i]).filter(p => reliable(p, minVisibility));
  const ankles = [JOINTS.leftAnkle, JOINTS.rightAnkle].map(i => landmarks[i]).filter(p => reliable(p, minVisibility));
  if (!hips.length || !ankles.length) return null;
  const hipY = hips.reduce((s, p) => s + p.y, 0) / hips.length;
  const ankleY = ankles.reduce((s, p) => s + p.y, 0) / ankles.length;
  const span = ankleY - hipY;
  return span > 0.02 ? span : null;
}

export function createSquatState() {
  return {
    phase: 'ready',
    reps: 0,
    downFrames: 0,
    upFrames: 0,
    enteredDownAt: null,
    lastRepAt: -Infinity,
    status: 'waiting',
    lastKneeAngle: null,
    standSpan: null, // decaying max of hip-ankle span (standing reference)
    standSpanAt: null,
    depthRatio: null, // current span / standing span (1 = standing)
  };
}

/** Updates the standing hip-ankle reference and returns the depth ratio. */
export function updateDepthRatio(state, span, timestampMs, decayPerSec = 0.03, { frozen = false } = {}) {
  if (!Number.isFinite(span)) return { standSpan: state.standSpan, standSpanAt: state.standSpanAt, depthRatio: null };
  let ref = state.standSpan;
  // During a repetition the reference must not decay: holding the bottom would
  // otherwise slowly turn the unchanged bottom height into "standing".
  if (!frozen && Number.isFinite(ref) && Number.isFinite(state.standSpanAt)) {
    // Slow decay lets the reference follow a user who steps further from the camera.
    ref *= Math.pow(1 - decayPerSec, Math.max(0, timestampMs - state.standSpanAt) / 1000);
  }
  ref = Number.isFinite(ref) ? Math.max(ref, span) : span;
  return { standSpan: ref, standSpanAt: timestampMs, depthRatio: Math.round((span / ref) * 1000) / 1000 };
}

export function advanceSquat(state, kneeAngle, timestampMs, config = {}) {
  const next = { ...state, lastKneeAngle: kneeAngle, justHitDepth: false, justCompletedRep: false };
  const downThreshold = config.downThreshold ?? 115;
  const upThreshold = config.upThreshold ?? 155;
  const minDurationMs = config.minDurationMs ?? 350;
  const minIntervalMs = config.minIntervalMs ?? 500;
  const holdFrames = config.holdFrames ?? 2;
  // Optional depth-free channel: hip height relative to standing (see hipAnkleSpan).
  const depthRatio = Number.isFinite(config.depthRatio) ? config.depthRatio : null;
  // Geometry: hips sit at ~0.84 of standing height at a 115° knee and ~0.97 at 155°.
  const downRatio = config.downRatio ?? 0.92;
  const upRatio = config.upRatio ?? 0.97;

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
    // Depth needs the knee angle; when the hip cue is available it must agree, which
    // rejects depth-noise dips of a 3D knee angle while the person is still standing.
    const isDown = kneeAngle <= downThreshold && (depthRatio === null || depthRatio <= downRatio);
    next.downFrames = isDown ? next.downFrames + 1 : 0;
    if (next.downFrames >= holdFrames) {
      next.phase = 'down';
      next.enteredDownAt = timestampMs;
      next.downFrames = 0;
      next.status = 'rise';
      next.justHitDepth = true;
    } else {
      next.status = kneeAngle > (upThreshold - 10) ? 'ready' : 'lowering';
    }
  } else {
    next.downFrames = 0;
    // Standing is confirmed by either cue: correlated depth error can keep a 3D knee
    // angle under the threshold while the hips are clearly back at standing height.
    const isUp = kneeAngle >= upThreshold || (depthRatio !== null && depthRatio >= upRatio);
    next.upFrames = isUp ? next.upFrames + 1 : 0;
    if (
      next.upFrames >= holdFrames &&
      timestampMs - next.enteredDownAt >= minDurationMs &&
      timestampMs - next.lastRepAt >= minIntervalMs
    ) {
      next.phase = 'ready';
      next.reps += 1;
      next.upFrames = 0;
      next.lastRepAt = timestampMs;
      next.status = 'completed';
      next.justCompletedRep = true;
    } else {
      next.status = isUp ? 'finishing' : 'rise';
    }
  }

  return next;
}

export class SquatExercise extends BaseExercise {
  constructor(options = {}) {
    super({
      id: 'squat',
      name: 'Squat',
      type: 'repetition',
      recommendedCamera: 'side',
      description: 'Tracks knee flexion and return to standing extension. Side camera angle recommended.',
      requiredLandmarks: [
        JOINTS.leftHip, JOINTS.rightHip,
        JOINTS.leftKnee, JOINTS.rightKnee,
        JOINTS.leftAnkle, JOINTS.rightAnkle,
      ],
      ...options,
    });
  }

  createState() {
    return createSquatState();
  }

  update(state, { angles, landmarks, timestampMs, trackingQuality }) {
    if (trackingQuality === 'lost') {
      return advanceSquat(state, null, timestampMs);
    }
    // Trust the clearly more visible leg; the occluded one is unreliable in side view.
    const kneeAngle = selectBilateralAngle(angles, landmarks, 'Knee');
    const depth = updateDepthRatio(state, hipAnkleSpan(landmarks), timestampMs, undefined, {
      frozen: state.phase !== 'ready',
    });
    const preset = this.getPresetConfig();
    const config = {
      downThreshold: 115 + preset.angleToleranceDeg,
      upThreshold: 155 - Math.max(0, preset.angleToleranceDeg / 2),
      minDurationMs: preset.minPhaseDurationMs,
      holdFrames: preset.requiredHoldFrames,
      depthRatio: depth.depthRatio,
    };
    return { ...advanceSquat(state, kneeAngle, timestampMs, config), ...depth };
  }

  getFeedback(state) {
    if (state.status === 'tracking-lost' || !Number.isFinite(state.lastKneeAngle)) {
      return 'Knee landmarks unavailable. Keep whole body and legs inside camera view.';
    }
    const feedbackMap = {
      waiting: 'Stand sideways to camera, keep hips, knees, and ankles visible.',
      ready: 'Ready. Bend your knees to start a repetition.',
      lowering: 'Knee bend detected. Continue moving downward smoothly.',
      rise: 'Depth threshold reached. Drive upward toward standing.',
      finishing: 'Straighten legs fully to complete the repetition.',
      completed: 'Repetition counted! Reset for the next repetition.',
    };
    return feedbackMap[state.status] ?? 'Maintain full-body visibility.';
  }

  getTelemetry(state) {
    return [
      { label: 'REPS', value: state.reps, highlight: true },
      { label: 'PHASE', value: state.phase.toUpperCase() },
      { label: 'KNEE', value: formatAngle('Knee', state.lastKneeAngle) },
    ];
  }
}
