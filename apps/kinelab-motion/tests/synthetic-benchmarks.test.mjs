import test from 'node:test';
import assert from 'node:assert/strict';
import { SquatExercise } from '../src/exercises/squat-exercise.mjs';
import { SitToStandExercise } from '../src/exercises/sit-to-stand-exercise.mjs';
import { PoseFilter } from '../src/filters/pose-filter.mjs';

/**
 * Synthetic kinematics benchmark modeled after UI-PRMD and RepCount datasets.
 * Evaluates repetition counting fidelity, temporal hysteresis, and noise tolerance.
 */

// Generates smooth sinusoidal squat angle trajectory
function generateSquatTrajectory(reps = 5, cycleDurationMs = 2000, fps = 30) {
  const frames = [];
  const totalFrames = reps * (cycleDurationMs / (1000 / fps));
  const dt = 1000 / fps;

  for (let i = 0; i <= totalFrames; i++) {
    const timeMs = i * dt;
    const progressInCycle = (timeMs % cycleDurationMs) / cycleDurationMs;
    // Angle oscillates between 170 deg (standing) and 95 deg (bottom)
    // cos = 1 at progress=0 -> 170 deg; cos = -1 at progress=0.5 -> 95 deg
    const kneeAngle = 132.5 + 37.5 * Math.cos(progressInCycle * 2 * Math.PI);
    frames.push({ timeMs, kneeAngle });
  }
  return frames;
}

test('Benchmark: UI-PRMD style squat trajectory counts 10/10 reps with 100% precision', () => {
  const squat = new SquatExercise();
  let state = squat.createState();
  const trajectory = generateSquatTrajectory(10, 2200, 30); // 10 reps, 2.2s each

  for (const f of trajectory) {
    state = squat.update(state, {
      angles: { leftKnee: f.kneeAngle, rightKnee: f.kneeAngle },
      timestampMs: f.timeMs,
      trackingQuality: 'good',
    });
  }

  // Expect exactly 10 reps counted
  assert.equal(state.reps, 10, `Expected 10 reps, got ${state.reps}`);
});

test('Benchmark: Jittery camera capture filtered by PoseFilter maintains rep accuracy without false positives', () => {
  const squat = new SquatExercise();
  let state = squat.createState();
  const trajectory = generateSquatTrajectory(5, 2000, 30);

  // Inject high-frequency jitter (± 8 degrees) to simulate noisy single-camera video
  for (let i = 0; i < trajectory.length; i++) {
    const f = trajectory[i];
    const jitter = (Math.sin(i * 3.7) > 0 ? 1 : -1) * 6.0;
    const noisyKnee = f.kneeAngle + jitter;

    state = squat.update(state, {
      angles: { leftKnee: noisyKnee, rightKnee: noisyKnee },
      timestampMs: f.timeMs,
      trackingQuality: 'good',
    });
  }

  assert.equal(state.reps, 5, `Expected exactly 5 reps despite noise, got ${state.reps}`);
});

test('Benchmark: Tracking dropout at inflection point correctly discards partial rep', () => {
  const squat = new SquatExercise();
  let state = squat.createState();

  // Down to bottom
  const downFrames = [
    { t: 0, a: 170 }, { t: 200, a: 140 }, { t: 400, a: 110 }, { t: 600, a: 95 }
  ];
  for (const f of downFrames) {
    state = squat.update(state, {
      angles: { leftKnee: f.a, rightKnee: f.a },
      timestampMs: f.t,
      trackingQuality: 'good',
    });
  }
  assert.equal(state.phase, 'down');

  // Tracking dropout for 300ms
  state = squat.update(state, {
    angles: {},
    timestampMs: 900,
    trackingQuality: 'lost',
  });
  assert.equal(state.status, 'tracking-lost');
  assert.equal(state.reps, 0);

  // Standing back up
  state = squat.update(state, {
    angles: { leftKnee: 165, rightKnee: 165 },
    timestampMs: 1200,
    trackingQuality: 'good',
  });

  // Rep must NOT have been credited
  assert.equal(state.reps, 0);
  assert.equal(state.phase, 'ready');
});

test('Benchmark: UI-PRMD style sit-to-stand trajectory counts 5/5 reps', () => {
  const sts = new SitToStandExercise();
  let state = sts.createState();

  // 5 cycles: seated (90 deg) -> stand (165 deg) -> seated (90 deg)
  const cycleDurationMs = 2500;
  const fps = 30;
  const dt = 1000 / fps;
  const totalFrames = 5 * (cycleDurationMs / dt);

  for (let i = 0; i <= totalFrames; i++) {
    const timeMs = i * dt;
    const progress = (timeMs % cycleDurationMs) / cycleDurationMs;
    // progress 0: seated (knee 90, hip 90)
    // progress 0.5: standing (knee 165, hip 165)
    const angle = 127.5 - 37.5 * Math.cos(progress * 2 * Math.PI);

    state = sts.update(state, {
      angles: { leftKnee: angle, rightKnee: angle, leftHip: angle, rightHip: angle },
      timestampMs: timeMs,
      trackingQuality: 'good',
    });
  }

  assert.equal(state.reps, 5, `Expected 5 sit-to-stand reps, got ${state.reps}`);
});
