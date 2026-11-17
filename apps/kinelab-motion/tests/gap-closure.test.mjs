import test from 'node:test';
import assert from 'node:assert/strict';
import { ShoulderFlexionExercise } from '../src/exercises/shoulder-flexion-exercise.mjs';
import { defaultExerciseRegistry } from '../src/exercises/exercise-registry.mjs';
import { computeVelocity, directionVisible } from '../src/overlays/direction-indicators.mjs';
import { compareSessions } from '../src/sessions/session-compare.mjs';
import { validateSettings } from '../src/settings/settings-schema.mjs';

test('ShoulderFlexionExercise is registered as a distinct side-camera module', () => {
  assert.equal(defaultExerciseRegistry.has('shoulder_flexion'), true);
  const ex = new ShoulderFlexionExercise();
  assert.equal(ex.id, 'shoulder_flexion');
  assert.equal(ex.recommendedCamera, 'side');
});

test('ShoulderFlexionExercise counts a forward raise and discards on tracking loss', () => {
  const ex = new ShoulderFlexionExercise();
  let s = ex.createState();
  const seq = [
    [{ leftShoulder: 25, rightShoulder: 25 }, 0],
    [{ leftShoulder: 95, rightShoulder: 95 }, 200],
    [{ leftShoulder: 155, rightShoulder: 155 }, 450],
    [{ leftShoulder: 158, rightShoulder: 158 }, 550],
    [{ leftShoulder: 90, rightShoulder: 90 }, 750],
    [{ leftShoulder: 30, rightShoulder: 30 }, 950],
    [{ leftShoulder: 28, rightShoulder: 28 }, 1050],
  ];
  for (const [angles, t] of seq) s = ex.update(s, { angles, timestampMs: t, trackingQuality: 'good' });
  assert.equal(s.reps, 1);

  let s2 = ex.createState();
  s2 = ex.update(s2, { angles: { leftShoulder: 95, rightShoulder: 95 }, timestampMs: 100, trackingQuality: 'good' });
  s2 = ex.update(s2, { angles: {}, timestampMs: 200, trackingQuality: 'lost' });
  assert.equal(s2.reps, 0);
  assert.equal(s2.status, 'tracking-lost');
});

test('direction indicators: velocity math and visibility threshold, no invented motion', () => {
  const v = computeVelocity({ x: 0.5, y: 0.5 }, { x: 0.6, y: 0.4 }, 100);
  assert.ok(Math.abs(v.vx - 1.0) < 1e-9);
  assert.ok(Math.abs(v.vy + 1.0) < 1e-9);
  assert.equal(directionVisible(v), true);
  assert.equal(directionVisible(computeVelocity({ x: 0.5, y: 0.5 }, { x: 0.501, y: 0.5 }, 100)), false);
  assert.equal(computeVelocity({ x: 0.5, y: 0.5 }, null, 100), null);
  assert.equal(computeVelocity({ x: 0.5, y: 0.5 }, { x: 0.6, y: 0.6 }, 0), null);
});

test('compareSessions reports measured deltas and nulls for missing angle data', () => {
  const mk = (vals, reps, dur) => ({ finalReps: reps, durationMs: dur, frames: vals.map((v, i) => ({ timeMs: i * 100, angles: { leftKnee: v }, reps })) });
  const r = compareSessions(mk([170, 100, 170], 1, 200), mk([170, 110, 170], 3, 400), 'leftKnee');
  assert.equal(r.repDelta, 2);
  assert.equal(r.durationDeltaMs, 200);
  assert.ok(Math.abs(r.meanDeltaDeg - (10 / 3)) < 0.01);
  const missing = compareSessions(mk([170], 0, 0), { finalReps: 0, durationMs: 0, frames: [{ timeMs: 0, angles: { leftKnee: null } }] }, 'leftKnee');
  assert.equal(missing.meanDeltaDeg, null);
});

test('settings baseline: valid kept, invalid becomes null, recording stays opt-in', () => {
  const ok = validateSettings({ exercise: { baseline: { standingKneeDeg: 172, standingHipDeg: 175 } } });
  assert.deepEqual(ok.exercise.baseline, { standingKneeDeg: 172, standingHipDeg: 175 });
  assert.equal(validateSettings({ exercise: { baseline: { standingKneeDeg: 40, standingHipDeg: 175 } } }).exercise.baseline, null);
  assert.equal(validateSettings({ exercise: { baseline: { standingKneeDeg: NaN, standingHipDeg: 170 } } }).exercise.baseline, null);
  assert.equal(validateSettings(null).exercise.baseline, null);
  assert.equal(validateSettings({ exercise: { activeExercise: 'shoulder_flexion' } }).exercise.activeExercise, 'shoulder_flexion');
});
