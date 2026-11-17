import test from 'node:test';
import assert from 'node:assert/strict';
import { MovementChallenge, CHALLENGE_TYPES } from '../src/integrations/movement-challenge.mjs';

const point = (x, y, vis = 0.95) => ({ x, y, visibility: vis, presence: vis });
const blankPose = () => Array.from({ length: 33 }, () => point(0.5, 0.5));

test('MovementChallenge starts inactive and activates on start()', () => {
  const challenge = new MovementChallenge();
  assert.equal(challenge.isActive, false);

  challenge.start();
  assert.equal(challenge.isActive, true);
  assert.equal(challenge.isCompleted, false);
});

test('MovementChallenge evaluates overhead wrist targets', () => {
  const challenge = new MovementChallenge({
    type: CHALLENGE_TYPES.OVERHEAD_REACH,
    targetDurationSec: 2, // 2 seconds
  });
  challenge.start();

  const pose = blankPose();
  // Wrists below target line (targetHeightNorm = 0.25)
  pose[15] = point(0.4, 0.5); // left wrist
  pose[16] = point(0.6, 0.5); // right wrist

  let res = challenge.update(pose, {}, 100);
  assert.equal(res.status, 'reach_higher');
  assert.equal(res.progress, 0);

  // Wrists above target line (y = 0.15 < 0.25)
  pose[15] = point(0.4, 0.15);
  pose[16] = point(0.6, 0.15);

  // Hold for 1 second (50% progress)
  res = challenge.update(pose, {}, 1000);
  assert.equal(res.status, 'holding');
  assert.equal(res.progress, 0.5);

  // Hold for another 1 second (100% progress -> completed)
  res = challenge.update(pose, {}, 1000);
  assert.equal(res.status, 'completed');
  assert.equal(res.progress, 1.0);
  assert.equal(res.isCompleted, true);
});
