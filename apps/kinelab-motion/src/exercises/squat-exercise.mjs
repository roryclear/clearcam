/**
 * Squat movement module with hysteresis, minimum phase duration, and tracking loss protection.
 * Preserves the exact algorithm and behavior of the existing KineLab squat prototype.
 */

import { BaseExercise } from './exercise-base.mjs';
import { JOINTS } from '../biomechanics/landmark-indices.mjs';
import { dominantKneeAngle } from '../biomechanics/angle-calculations.mjs';

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
  };
}

export function advanceSquat(state, kneeAngle, timestampMs, config = {}) {
  const next = { ...state, lastKneeAngle: kneeAngle };
  const downThreshold = config.downThreshold ?? 115;
  const upThreshold = config.upThreshold ?? 155;
  const minDurationMs = config.minDurationMs ?? 350;
  const minIntervalMs = config.minIntervalMs ?? 500;
  const holdFrames = config.holdFrames ?? 2;

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
    next.downFrames = kneeAngle <= downThreshold ? next.downFrames + 1 : 0;
    if (next.downFrames >= holdFrames) {
      next.phase = 'down';
      next.enteredDownAt = timestampMs;
      next.downFrames = 0;
      next.status = 'rise';
    } else {
      next.status = kneeAngle > (upThreshold - 10) ? 'ready' : 'lowering';
    }
  } else {
    next.downFrames = 0;
    next.upFrames = kneeAngle >= upThreshold ? next.upFrames + 1 : 0;
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
    } else {
      next.status = kneeAngle >= upThreshold ? 'finishing' : 'rise';
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

  update(state, { angles, timestampMs, trackingQuality }) {
    if (trackingQuality === 'lost') {
      return advanceSquat(state, null, timestampMs);
    }
    const kneeAngle = dominantKneeAngle(angles);
    const preset = this.getPresetConfig();
    const config = {
      downThreshold: 115 + preset.angleToleranceDeg,
      upThreshold: 155 - Math.max(0, preset.angleToleranceDeg / 2),
      minDurationMs: preset.minPhaseDurationMs,
      holdFrames: preset.requiredHoldFrames,
    };
    return advanceSquat(state, kneeAngle, timestampMs, config);
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
      { label: 'KNEE ANGLE', value: Number.isFinite(state.lastKneeAngle) ? `${Math.round(state.lastKneeAngle)}°` : '—' },
    ];
  }
}
