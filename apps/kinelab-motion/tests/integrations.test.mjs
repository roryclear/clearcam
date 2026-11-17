import test from 'node:test';
import assert from 'node:assert/strict';
import { ExtensionManager, FEATURE_FLAGS, createExtensionIfEnabled } from '../src/integrations/extension-contracts.mjs';
import { MovementChallenge } from '../src/integrations/movement-challenge.mjs';
import { SquatExercise } from '../src/exercises/squat-exercise.mjs';

test('ExtensionManager: interactive flags default off, audio on', () => {
  const m = new ExtensionManager();
  assert.equal(m.isFeatureEnabled(FEATURE_FLAGS.MOVEMENT_CHALLENGES), false);
  assert.equal(m.isFeatureEnabled(FEATURE_FLAGS.GESTURE_CONTROLS), false);
  assert.equal(m.isFeatureEnabled(FEATURE_FLAGS.THREE_VRM_AVATAR), false);
  assert.equal(m.isFeatureEnabled(FEATURE_FLAGS.MOTION_PAINTING), false);
  assert.equal(m.isFeatureEnabled(FEATURE_FLAGS.AUDIO_CUES), true);
});

test('createExtensionIfEnabled never constructs when flag is off', () => {
  const m = new ExtensionManager();
  let constructed = 0;
  const off = createExtensionIfEnabled(m, FEATURE_FLAGS.MOVEMENT_CHALLENGES, () => { constructed++; return new MovementChallenge(); });
  assert.equal(off, null);
  assert.equal(constructed, 0);
  m.setFeatureEnabled(FEATURE_FLAGS.MOVEMENT_CHALLENGES, true);
  const on = createExtensionIfEnabled(m, FEATURE_FLAGS.MOVEMENT_CHALLENGES, () => { constructed++; return new MovementChallenge(); });
  assert.ok(on instanceof MovementChallenge);
  assert.equal(on.isActive, false);
  assert.equal(constructed, 1);
});

test('registering a challenge extension does not alter exercise counting', () => {
  const m = new ExtensionManager();
  const challenge = new MovementChallenge();
  challenge.id = 'test-challenge'; // ExtensionManager contract requires an id
  m.registerExtension('challenge', challenge);
  const squat = new SquatExercise();
  let s = squat.createState();
  const seq = [[170, 0], [110, 100], [100, 450], [165, 800], [168, 900]];
  for (const [a, t] of seq) s = squat.update(s, { angles: { leftKnee: a, rightKnee: a }, timestampMs: t, trackingQuality: 'good' });
  assert.equal(s.reps, 1);
});
