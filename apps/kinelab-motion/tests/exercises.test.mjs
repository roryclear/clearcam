import test from 'node:test';
import assert from 'node:assert/strict';
import { ExerciseRegistry, defaultExerciseRegistry } from '../src/exercises/exercise-registry.mjs';
import { SquatExercise } from '../src/exercises/squat-exercise.mjs';
import { SitToStandExercise } from '../src/exercises/sit-to-stand-exercise.mjs';
import { LungeExercise } from '../src/exercises/lunge-exercise.mjs';
import { ShoulderAbductionExercise } from '../src/exercises/shoulder-abduction-exercise.mjs';
import { SingleLegBalanceExercise } from '../src/exercises/balance-exercise.mjs';

const point = (x, y, vis = 0.95) => ({ x, y, visibility: vis, presence: vis });
const blankPose = () => Array.from({ length: 33 }, () => point(0.5, 0.5));

test('ExerciseRegistry registers and discovers the 5 core exercises', () => {
  const reg = defaultExerciseRegistry;
  assert.equal(reg.has('squat'), true);
  assert.equal(reg.has('sit_to_stand'), true);
  assert.equal(reg.has('sit-to-stand'), true);
  assert.equal(reg.has('lunge'), true);
  assert.equal(reg.has('shoulder_abduction'), true);
  assert.equal(reg.has('shoulder-abduction'), true);
  assert.equal(reg.has('single_leg_balance'), true);
  assert.equal(reg.has('single-leg-balance'), true);
  assert.ok(reg.get('sit-to-stand'));
  assert.equal(reg.get('sit-to-stand').id, 'sit_to_stand');
  assert.equal(reg.getAll().length >= 5, true);
});

test('SquatExercise counts rep after controlled down and return to extension', () => {
  const squat = new SquatExercise();
  let state = squat.createState();

  const anglesSeq = [
    { leftKnee: 170, rightKnee: 170, time: 0 },
    { leftKnee: 110, rightKnee: 110, time: 100 },
    { leftKnee: 105, rightKnee: 105, time: 200 },
    { leftKnee: 100, rightKnee: 100, time: 450 },
    { leftKnee: 160, rightKnee: 160, time: 800 },
    { leftKnee: 165, rightKnee: 165, time: 900 },
  ];

  for (const item of anglesSeq) {
    state = squat.update(state, {
      angles: { leftKnee: item.leftKnee, rightKnee: item.rightKnee },
      timestampMs: item.time,
      trackingQuality: 'good',
    });
  }

  assert.equal(state.reps, 1);
  assert.equal(state.phase, 'ready');
  assert.ok(squat.getFeedback(state).includes('Repetition counted'));
});

test('SitToStandExercise tracks transition from seated to standing and back', () => {
  const sts = new SitToStandExercise();
  let state = sts.createState();

  // 1. Initial seated position: knee ~90 deg, hip ~90 deg
  state = sts.update(state, {
    angles: { leftKnee: 90, rightKnee: 90, leftHip: 90, rightHip: 90 },
    timestampMs: 0,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'seated');

  // 2. Rising
  state = sts.update(state, {
    angles: { leftKnee: 135, rightKnee: 135, leftHip: 130, rightHip: 130 },
    timestampMs: 200,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'rising');

  // 3. Standing: knees and hips extended >= 155 deg for 2 frames
  state = sts.update(state, {
    angles: { leftKnee: 165, rightKnee: 165, leftHip: 165, rightHip: 165 },
    timestampMs: 400,
    trackingQuality: 'good',
  });
  state = sts.update(state, {
    angles: { leftKnee: 165, rightKnee: 165, leftHip: 165, rightHip: 165 },
    timestampMs: 500,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'standing');

  // 4. Sitting back down
  state = sts.update(state, {
    angles: { leftKnee: 125, rightKnee: 125, leftHip: 125, rightHip: 125 },
    timestampMs: 700,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'sitting');

  // 5. Returned to seat
  state = sts.update(state, {
    angles: { leftKnee: 92, rightKnee: 92, leftHip: 90, rightHip: 90 },
    timestampMs: 900,
    trackingQuality: 'good',
  });
  state = sts.update(state, {
    angles: { leftKnee: 90, rightKnee: 90, leftHip: 90, rightHip: 90 },
    timestampMs: 1000,
    trackingQuality: 'good',
  });

  assert.equal(state.reps, 1);
  assert.equal(state.phase, 'seated');
});

test('SitToStandExercise drops incomplete repetition when tracking is lost mid-stand', () => {
  const sts = new SitToStandExercise();
  let state = sts.createState();

  // Rising
  state = sts.update(state, {
    angles: { leftKnee: 135, rightKnee: 135, leftHip: 130, rightHip: 130 },
    timestampMs: 100,
    trackingQuality: 'good',
  });

  // Tracking drops mid-movement
  state = sts.update(state, {
    angles: {},
    timestampMs: 200,
    trackingQuality: 'lost',
  });

  assert.equal(state.status, 'tracking-lost');
  assert.equal(state.reps, 0);
  assert.equal(state.phase, 'seated');
});

test('LungeExercise tracks lead-leg depth and standing recovery', () => {
  const lunge = new LungeExercise();
  let state = lunge.createState();

  // Standing upright
  state = lunge.update(state, {
    angles: { leftKnee: 170, rightKnee: 170 },
    timestampMs: 0,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'standing');

  // Stepping forward into lunge: left knee flexes
  state = lunge.update(state, {
    angles: { leftKnee: 125, rightKnee: 155 },
    timestampMs: 200,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'descending');
  assert.equal(state.leadLeg, 'left');

  // Reaching deep lunge position (<= 105 deg) for 2 frames
  state = lunge.update(state, {
    angles: { leftKnee: 95, rightKnee: 110 },
    timestampMs: 400,
    trackingQuality: 'good',
  });
  state = lunge.update(state, {
    angles: { leftKnee: 92, rightKnee: 108 },
    timestampMs: 500,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'deep_lunge');

  // Recovering back to standing
  state = lunge.update(state, {
    angles: { leftKnee: 135, rightKnee: 140 },
    timestampMs: 700,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'recovering');

  // Returned to upright standing >= 150 deg for 2 frames
  state = lunge.update(state, {
    angles: { leftKnee: 165, rightKnee: 165 },
    timestampMs: 950,
    trackingQuality: 'good',
  });
  state = lunge.update(state, {
    angles: { leftKnee: 168, rightKnee: 168 },
    timestampMs: 1100,
    trackingQuality: 'good',
  });

  assert.equal(state.reps, 1);
  assert.equal(state.phase, 'standing');
});

test('ShoulderAbductionExercise tracks overhead arm elevation and lowering', () => {
  const abd = new ShoulderAbductionExercise();
  let state = abd.createState();

  // Arms at side: ~30 deg
  state = abd.update(state, {
    angles: { leftShoulder: 30, rightShoulder: 30 },
    timestampMs: 0,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'rest');

  // Raising arms
  state = abd.update(state, {
    angles: { leftShoulder: 90, rightShoulder: 92 },
    timestampMs: 200,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'elevating');

  // Peak overhead reach (>= 140 deg for 2 frames)
  state = abd.update(state, {
    angles: { leftShoulder: 155, rightShoulder: 150 },
    timestampMs: 450,
    trackingQuality: 'good',
  });
  state = abd.update(state, {
    angles: { leftShoulder: 160, rightShoulder: 155 },
    timestampMs: 550,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'peak');

  // Lowering arms
  state = abd.update(state, {
    angles: { leftShoulder: 95, rightShoulder: 90 },
    timestampMs: 750,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'lowering');

  // Returned to rest (<= 45 deg for 2 frames)
  state = abd.update(state, {
    angles: { leftShoulder: 35, rightShoulder: 32 },
    timestampMs: 950,
    trackingQuality: 'good',
  });
  state = abd.update(state, {
    angles: { leftShoulder: 32, rightShoulder: 30 },
    timestampMs: 1050,
    trackingQuality: 'good',
  });

  assert.equal(state.reps, 1);
  assert.equal(state.phase, 'rest');
});

test('SingleLegBalanceExercise tracks timed hold and touchdown', () => {
  const bal = new SingleLegBalanceExercise();
  let state = bal.createState();

  const pose = blankPose();
  // Two feet down: left and right ankles at y = 0.8
  pose[27] = point(0.4, 0.8); // left ankle
  pose[28] = point(0.6, 0.8); // right ankle

  state = bal.update(state, {
    landmarks: pose,
    timestampMs: 0,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'two_feet');

  // Lift left foot up (ankle moves to y = 0.7, diff = 0.8 - 0.7 = 0.1 > 0.055)
  // Stance leg is RIGHT
  pose[27] = point(0.4, 0.7);
  state = bal.update(state, {
    landmarks: pose,
    timestampMs: 100,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'holding');
  assert.equal(state.stanceLeg, 'right');

  // Hold for 3.0 seconds
  state = bal.update(state, {
    landmarks: pose,
    timestampMs: 3100,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'holding');
  assert.equal(state.currentHoldSec, 3.0);
  assert.equal(state.bestHoldSec, 3.0);

  // Touch left foot down
  pose[27] = point(0.4, 0.8);
  state = bal.update(state, {
    landmarks: pose,
    timestampMs: 3200,
    trackingQuality: 'good',
  });
  assert.equal(state.phase, 'touchdown');
  assert.equal(state.bestHoldSec, 3.0);
});
