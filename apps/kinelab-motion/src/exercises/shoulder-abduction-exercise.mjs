/**
 * Shoulder Flexion / Abduction exercise module (Upper body mobility).
 * Measures arm elevation from rest at side to overhead reach, comparing left and right symmetry.
 */

import { BaseExercise } from './exercise-base.mjs';
import { JOINTS } from '../biomechanics/landmark-indices.mjs';

export class ShoulderAbductionExercise extends BaseExercise {
  constructor(options = {}) {
    super({
      id: 'shoulder_abduction',
      name: 'Shoulder Abduction',
      type: 'repetition',
      recommendedCamera: 'front',
      description: 'Tracks arm elevation from sides to overhead with left/right range comparison. Front camera recommended.',
      requiredLandmarks: [
        JOINTS.leftShoulder, JOINTS.rightShoulder,
        JOINTS.leftElbow, JOINTS.rightElbow,
        JOINTS.leftHip, JOINTS.rightHip,
      ],
      ...options,
    });
  }

  createState() {
    return {
      phase: 'rest', // 'rest' | 'elevating' | 'peak' | 'lowering'
      reps: 0,
      peakHoldFrames: 0,
      restFrames: 0,
      enteredPeakAt: null,
      lastRepAt: -Infinity,
      status: 'waiting',
      leftAngle: null,
      rightAngle: null,
      maxElevationDeg: 0,
    };
  }

  update(state, { angles, timestampMs, trackingQuality }) {
    const next = { ...state };

    if (trackingQuality === 'lost') {
      next.status = 'tracking-lost';
      next.phase = 'rest';
      next.peakHoldFrames = 0;
      next.restFrames = 0;
      next.leftAngle = null;
      next.rightAngle = null;
      return next;
    }

    const la = angles?.leftShoulder;
    const ra = angles?.rightShoulder;

    next.leftAngle = la;
    next.rightAngle = ra;

    const validAngles = [la, ra].filter(Number.isFinite);
    if (!validAngles.length) {
      next.status = 'partial-tracking';
      return next;
    }

    // Use peak elevation of available arms
    const currentElevation = Math.max(...validAngles);
    next.maxElevationDeg = Math.max(next.maxElevationDeg, currentElevation);

    const preset = this.getPresetConfig();
    const peakThreshold = 140 - preset.angleToleranceDeg;
    const restThreshold = 45 + Math.max(0, preset.angleToleranceDeg / 2);
    const holdFrames = preset.requiredHoldFrames;

    switch (state.phase) {
      case 'rest':
        if (currentElevation > restThreshold + 15) {
          next.phase = 'elevating';
          next.status = 'raising';
          next.peakHoldFrames = 0;
          next.maxElevationDeg = currentElevation;
        } else {
          next.status = 'arms-down';
        }
        break;

      case 'elevating':
        if (currentElevation >= peakThreshold) {
          next.peakHoldFrames++;
          if (next.peakHoldFrames >= holdFrames) {
            next.phase = 'peak';
            next.enteredPeakAt = timestampMs;
            next.status = 'peak-reached';
            next.peakHoldFrames = 0;
          }
        } else if (currentElevation < restThreshold) {
          next.phase = 'rest';
          next.status = 'arms-down';
        }
        break;

      case 'peak':
        if (currentElevation < peakThreshold - 15) {
          next.phase = 'lowering';
          next.status = 'lowering';
          next.restFrames = 0;
        }
        break;

      case 'lowering':
        if (currentElevation <= restThreshold) {
          next.restFrames++;
          if (
            next.restFrames >= holdFrames &&
            timestampMs - next.enteredPeakAt >= preset.minPhaseDurationMs &&
            timestampMs - next.lastRepAt >= 500
          ) {
            next.reps++;
            next.lastRepAt = timestampMs;
            next.phase = 'rest';
            next.status = 'completed';
            next.restFrames = 0;
            next.maxElevationDeg = 0;
          }
        } else if (currentElevation >= peakThreshold) {
          next.phase = 'peak';
          next.status = 'peak-reached';
        }
        break;
    }

    return next;
  }

  getFeedback(state) {
    if (state.status === 'tracking-lost') {
      return 'Upper body lost. Face camera with torso and arms visible.';
    }
    const map = {
      waiting: 'Stand facing camera with arms relaxed at your sides.',
      'arms-down': 'Ready. Raise your arms out to the sides and overhead.',
      raising: 'Raising arms... Reach upward as high as comfortable.',
      'peak-reached': 'Peak elevation reached! Lower arms slowly back to sides.',
      lowering: 'Lowering arms in control...',
      completed: 'Repetition completed! Left/right range recorded.',
    };
    return map[state.status] ?? 'Maintain upper body visibility.';
  }

  getTelemetry(state) {
    return [
      { label: 'REPS', value: state.reps, highlight: true },
      { label: 'LEFT ARM', value: Number.isFinite(state.leftAngle) ? `${Math.round(state.leftAngle)}°` : '—' },
      { label: 'RIGHT ARM', value: Number.isFinite(state.rightAngle) ? `${Math.round(state.rightAngle)}°` : '—' },
    ];
  }
}
