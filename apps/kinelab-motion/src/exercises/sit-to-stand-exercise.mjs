/**
 * Sit-to-Stand movement module (Chair stand test / Functional lower body mobility).
 * Detects transition from seated to full upright standing and return.
 */

import { formatAngle } from '../biomechanics/angle-convention.mjs';
import { BaseExercise } from './exercise-base.mjs';
import { JOINTS } from '../biomechanics/landmark-indices.mjs';
import { selectBilateralAngle } from '../biomechanics/angle-calculations.mjs';

export class SitToStandExercise extends BaseExercise {
  constructor(options = {}) {
    super({
      id: 'sit_to_stand',
      name: 'Sit-to-Stand',
      type: 'repetition',
      recommendedCamera: 'side',
      description: 'Measures transitions between seated chair position and upright standing. Side camera angle recommended.',
      requiredLandmarks: [
        JOINTS.leftShoulder, JOINTS.rightShoulder,
        JOINTS.leftHip, JOINTS.rightHip,
        JOINTS.leftKnee, JOINTS.rightKnee,
        JOINTS.leftAnkle, JOINTS.rightAnkle,
      ],
      ...options,
    });
  }

  createState() {
    return {
      phase: 'seated', // 'seated' | 'rising' | 'standing' | 'sitting'
      // A stand only counts once a seated posture has actually been observed, so a
      // user who starts standing and sits down is not credited with a repetition.
      seatedConfirmed: false,
      reps: 0,
      seatedFrames: 0,
      standingFrames: 0,
      enteredStandingAt: null,
      lastRepAt: -Infinity,
      status: 'waiting',
      kneeAngle: null,
      hipAngle: null,
    };
  }

  update(state, { angles, landmarks, timestampMs, trackingQuality }) {
    const next = { ...state };

    if (trackingQuality === 'lost') {
      next.status = 'tracking-lost';
      next.phase = 'seated';
      next.seatedConfirmed = false;
      next.standingFrames = 0;
      next.seatedFrames = 0;
      next.kneeAngle = null;
      next.hipAngle = null;
      return next;
    }

    // Side-on, the far leg is occluded and unreliable: use the clearly more visible side.
    const knee = selectBilateralAngle(angles, landmarks, 'Knee');
    const hipRaw = selectBilateralAngle(angles, landmarks, 'Hip');
    const hip = Number.isFinite(hipRaw) ? Math.round(hipRaw) : null;

    next.kneeAngle = knee;
    next.hipAngle = hip;

    if (!Number.isFinite(knee) || !Number.isFinite(hip)) {
      next.status = 'partial-tracking';
      return next;
    }

    const preset = this.getPresetConfig();
    const standThreshold = 155 - preset.angleToleranceDeg;
    const sitThreshold = 110 + preset.angleToleranceDeg;
    const holdFrames = preset.requiredHoldFrames;

    switch (state.phase) {
      case 'seated':
        if (knee <= sitThreshold) {
          next.seatedConfirmed = true;
        }
        if (!next.seatedConfirmed) {
          next.status = 'sit-to-begin';
        } else if (knee > sitThreshold + 15) {
          next.phase = 'rising';
          next.status = 'rising';
          next.standingFrames = 0;
        } else {
          next.status = 'seated-ready';
        }
        break;

      case 'rising':
        if (knee >= standThreshold && hip >= (standThreshold - 10)) {
          next.standingFrames++;
          if (next.standingFrames >= holdFrames) {
            next.phase = 'standing';
            next.enteredStandingAt = timestampMs;
            next.status = 'standing-hold';
            next.standingFrames = 0;
          }
        } else if (knee < sitThreshold) {
          // Returned back without reaching stand
          next.phase = 'seated';
          next.status = 'seated-ready';
        }
        break;

      case 'standing':
        if (knee < standThreshold - 15) {
          next.phase = 'sitting';
          next.status = 'lowering';
          next.seatedFrames = 0;
        }
        break;

      case 'sitting':
        if (knee <= sitThreshold && hip <= (sitThreshold + 10)) {
          next.seatedFrames++;
          if (
            next.seatedFrames >= holdFrames &&
            timestampMs - next.lastRepAt >= preset.minPhaseDurationMs
          ) {
            next.reps++;
            next.lastRepAt = timestampMs;
            next.phase = 'seated';
            next.status = 'completed';
            next.seatedFrames = 0;
          }
        } else if (knee >= standThreshold) {
          next.phase = 'standing';
          next.status = 'standing-hold';
        }
        break;
    }

    return next;
  }

  getFeedback(state) {
    if (state.status === 'tracking-lost' || !Number.isFinite(state.kneeAngle)) {
      return 'Body tracking lost. Ensure chair and full body are visible from the side.';
    }
    const map = {
      waiting: 'Start seated on a stable chair with side profile to camera.',
      'sit-to-begin': 'Sit down on the chair to begin. Counting starts from a seated position.',
      'seated-ready': 'Seated position detected. Stand up smoothly to begin.',
      rising: 'Rising... Extend hips and knees fully into standing.',
      'standing-hold': 'Standing detected. Now sit back down in a controlled motion.',
      lowering: 'Lowering toward the seat...',
      completed: 'Full repetition counted! Ready for next stand.',
    };
    return map[state.status] ?? 'Maintain side-profile visibility.';
  }

  getTelemetry(state) {
    return [
      { label: 'STANDS', value: state.reps, highlight: true },
      { label: 'PHASE', value: state.phase.toUpperCase() },
      { label: 'KNEE', value: formatAngle('Knee', state.kneeAngle) },
      { label: 'HIP', value: formatAngle('Hip', state.hipAngle) },
    ];
  }
}
