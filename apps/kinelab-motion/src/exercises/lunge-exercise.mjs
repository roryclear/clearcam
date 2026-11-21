/**
 * Lunge movement module.
 * Tracks alternating or single-leg forward/reverse lunges with depth threshold and return to extension.
 */

import { formatAngle } from '../biomechanics/angle-convention.mjs';
import { BaseExercise } from './exercise-base.mjs';
import { JOINTS } from '../biomechanics/landmark-indices.mjs';

export class LungeExercise extends BaseExercise {
  constructor(options = {}) {
    super({
      id: 'lunge',
      name: 'Lunge',
      type: 'repetition',
      recommendedCamera: 'side',
      description: 'Monitors lead-knee flexion and upright recovery during lunges. Side camera angle recommended.',
      requiredLandmarks: [
        JOINTS.leftHip, JOINTS.rightHip,
        JOINTS.leftKnee, JOINTS.rightKnee,
        JOINTS.leftAnkle, JOINTS.rightAnkle,
      ],
      ...options,
    });
  }

  createState() {
    return {
      phase: 'standing', // 'standing' | 'descending' | 'deep_lunge' | 'recovering'
      reps: 0,
      leadLeg: null, // 'left' | 'right'
      deepFrames: 0,
      standingFrames: 0,
      enteredDeepAt: null,
      lastRepAt: -Infinity,
      status: 'waiting',
      leadKneeAngle: null,
    };
  }

  update(state, { landmarks, angles, timestampMs, trackingQuality }) {
    const next = { ...state, justHitDepth: false, justCompletedRep: false };

    if (trackingQuality === 'lost') {
      next.status = 'tracking-lost';
      next.phase = 'standing';
      next.deepFrames = 0;
      next.standingFrames = 0;
      next.leadKneeAngle = null;
      return next;
    }

    const lk = angles?.leftKnee;
    const rk = angles?.rightKnee;

    if (!Number.isFinite(lk) && !Number.isFinite(rk)) {
      next.status = 'partial-tracking';
      return next;
    }

    // Determine lead knee (the more deeply flexed leg)
    let leadKnee = null;
    let leadSide = 'unknown';

    if (Number.isFinite(lk) && Number.isFinite(rk)) {
      if (lk < rk) {
        leadKnee = lk;
        leadSide = 'left';
      } else {
        leadKnee = rk;
        leadSide = 'right';
      }
    } else {
      leadKnee = Number.isFinite(lk) ? lk : rk;
      leadSide = Number.isFinite(lk) ? 'left' : 'right';
    }

    next.leadKneeAngle = leadKnee;
    next.leadLeg = leadSide;

    const preset = this.getPresetConfig();
    const lungeDepthThreshold = 105 + preset.angleToleranceDeg;
    const uprightThreshold = 150 - Math.max(0, preset.angleToleranceDeg / 2);
    const holdFrames = preset.requiredHoldFrames;

    switch (state.phase) {
      case 'standing':
        if (leadKnee <= 130) {
          next.phase = 'descending';
          next.status = 'lowering';
          next.deepFrames = 0;
        } else {
          next.status = 'standing-ready';
        }
        break;

      case 'descending':
        if (leadKnee <= lungeDepthThreshold) {
          next.deepFrames++;
          if (next.deepFrames >= holdFrames) {
            next.phase = 'deep_lunge';
            next.enteredDeepAt = timestampMs;
            next.status = 'depth-reached';
            next.deepFrames = 0;
            next.justHitDepth = true;
          }
        } else if (leadKnee > 145) {
          next.phase = 'standing';
          next.status = 'standing-ready';
        }
        break;

      case 'deep_lunge':
        if (leadKnee > lungeDepthThreshold + 15) {
          next.phase = 'recovering';
          next.status = 'rising';
          next.standingFrames = 0;
        }
        break;

      case 'recovering':
        if (leadKnee >= uprightThreshold) {
          next.standingFrames++;
          if (
            next.standingFrames >= holdFrames &&
            timestampMs - next.enteredDeepAt >= preset.minPhaseDurationMs &&
            timestampMs - next.lastRepAt >= 600
          ) {
            next.reps++;
            next.lastRepAt = timestampMs;
            next.phase = 'standing';
            next.status = 'completed';
            next.standingFrames = 0;
            next.justCompletedRep = true;
          }
        } else if (leadKnee <= lungeDepthThreshold) {
          next.phase = 'deep_lunge';
          next.status = 'depth-reached';
        }
        break;
    }

    return next;
  }

  getFeedback(state) {
    if (state.status === 'tracking-lost' || !Number.isFinite(state.leadKneeAngle)) {
      return 'Leg landmarks lost. Step into full view of the camera.';
    }
    const map = {
      waiting: 'Stand upright with side view to camera.',
      'standing-ready': 'Ready. Step forward or backward into a lunge.',
      lowering: 'Lowering... Drop back knee toward floor while keeping torso upright.',
      'depth-reached': 'Target lunge depth reached. Push through front foot to stand.',
      rising: 'Returning to upright position...',
      completed: 'Lunge repetition completed! Ready for next rep.',
    };
    return map[state.status] ?? 'Maintain side-profile visibility.';
  }

  getTelemetry(state) {
    return [
      { label: 'LUNGES', value: state.reps, highlight: true },
      { label: 'LEAD LEG', value: state.leadLeg ? state.leadLeg.toUpperCase() : '—' },
      { label: 'KNEE', value: formatAngle('Knee', state.leadKneeAngle) },
    ];
  }
}
