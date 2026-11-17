/**
 * Single-Leg Balance exercise module (Static balance, stability, and postural sway).
 * Tracks timed continuous single-leg hold duration and identifies the stance leg.
 */

import { BaseExercise } from './exercise-base.mjs';
import { JOINTS } from '../biomechanics/landmark-indices.mjs';
import { reliable } from '../biomechanics/angle-calculations.mjs';

export class SingleLegBalanceExercise extends BaseExercise {
  constructor(options = {}) {
    super({
      id: 'single_leg_balance',
      name: 'Single-Leg Balance',
      type: 'hold',
      recommendedCamera: 'front',
      description: 'Measures continuous hold time on a single leg. Front camera view recommended.',
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
      phase: 'two_feet', // 'two_feet' | 'holding' | 'touchdown'
      stanceLeg: null, // 'left' | 'right'
      holdStartedAt: null,
      currentHoldSec: 0,
      bestHoldSec: 0,
      totalHoldSec: 0,
      status: 'waiting',
      ankleElevationDiff: 0,
    };
  }

  update(state, { landmarks, timestampMs, trackingQuality }) {
    const next = { ...state };

    if (trackingQuality === 'lost' || !Array.isArray(landmarks)) {
      if (state.phase === 'holding') {
        // Hold was interrupted by tracking loss
        next.phase = 'two_feet';
        next.holdStartedAt = null;
        next.currentHoldSec = 0;
      }
      next.status = 'tracking-lost';
      next.ankleElevationDiff = 0;
      return next;
    }

    const la = landmarks[JOINTS.leftAnkle];
    const ra = landmarks[JOINTS.rightAnkle];
    const lh = landmarks[JOINTS.leftHip];
    const rh = landmarks[JOINTS.rightHip];

    if (!reliable(la) || !reliable(ra) || !reliable(lh) || !reliable(rh)) {
      if (state.phase === 'holding') {
        next.phase = 'two_feet';
        next.holdStartedAt = null;
        next.currentHoldSec = 0;
      }
      next.status = 'partial-tracking';
      return next;
    }

    // In screen coordinates: y is 0 at top, 1 at bottom.
    // Higher foot on screen has smaller y.
    // Ankle elevation diff = stance_ankle.y - lifted_ankle.y
    const yDiff = ra.y - la.y; // Positive if left foot is lifted higher than right
    const absDiff = Math.abs(yDiff);
    next.ankleElevationDiff = Math.round(absDiff * 1000) / 1000;

    const liftThreshold = 0.055; // ~5.5% of frame height

    if (absDiff >= liftThreshold) {
      const activeStance = yDiff > 0 ? 'right' : 'left'; // if left is lifted, stance is right

      if (state.phase === 'two_feet' || state.phase === 'touchdown') {
        // Began single-leg hold
        next.phase = 'holding';
        next.stanceLeg = activeStance;
        next.holdStartedAt = timestampMs;
        next.currentHoldSec = 0;
        next.status = 'holding';
      } else if (state.phase === 'holding') {
        if (state.stanceLeg === activeStance) {
          // Continue holding
          const elapsed = Math.max(0, (timestampMs - state.holdStartedAt) / 1000);
          next.currentHoldSec = Math.round(elapsed * 10) / 10;
          next.bestHoldSec = Math.max(state.bestHoldSec, next.currentHoldSec);
          next.status = 'holding';
        } else {
          // Switched legs mid-hold
          next.phase = 'two_feet';
          next.holdStartedAt = null;
          next.currentHoldSec = 0;
          next.status = 'leg-switched';
        }
      }
    } else {
      // Both feet on the ground
      if (state.phase === 'holding') {
        const finalHold = state.currentHoldSec;
        next.totalHoldSec = Math.round((state.totalHoldSec + finalHold) * 10) / 10;
        next.phase = 'touchdown';
        next.holdStartedAt = null;
        next.currentHoldSec = 0;
        next.status = 'touchdown';
      } else {
        next.phase = 'two_feet';
        next.status = 'ready-to-lift';
      }
    }

    return next;
  }

  getFeedback(state) {
    if (state.status === 'tracking-lost') {
      return 'Feet or legs not visible. Step back so lower body is in view.';
    }
    const map = {
      waiting: 'Stand facing camera with feet shoulder-width apart.',
      'ready-to-lift': 'Both feet down. Lift one foot off the ground to begin hold.',
      holding: `Balancing on ${state.stanceLeg ?? 'one'} leg! Maintain focus and stability.`,
      touchdown: `Foot touched down! Hold duration recorded: ${state.bestHoldSec}s.`,
      'leg-switched': 'Stance leg switched. Reset feet to start new hold.',
    };
    return map[state.status] ?? 'Keep full body in view.';
  }

  getTelemetry(state) {
    return [
      { label: 'CURRENT HOLD', value: `${state.currentHoldSec}s`, highlight: state.phase === 'holding' },
      { label: 'BEST HOLD', value: `${state.bestHoldSec}s` },
      { label: 'STANCE LEG', value: state.stanceLeg ? state.stanceLeg.toUpperCase() : 'BOTH' },
    ];
  }
}
