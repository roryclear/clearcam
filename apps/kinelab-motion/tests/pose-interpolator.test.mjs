import test from 'node:test';
import assert from 'node:assert/strict';
import { PoseInterpolator } from '../src/filters/pose-interpolator.mjs';

test('PoseInterpolator smoothly interpolates between frames at 60/120 FPS', () => {
  const interpolator = new PoseInterpolator({ lambda: 25.0 });

  const startPose = Array(33).fill(null).map(() => ({ x: 0.1, y: 0.2, z: 0.0, visibility: 0.9 }));
  interpolator.setTarget(startPose, 0);

  // Initial step sets baseline
  const p0 = interpolator.step(0);
  assert.equal(p0[0].x, 0.1);

  // New target arrived at t = 33.3ms (30 FPS inference)
  const targetPose = Array(33).fill(null).map(() => ({ x: 0.2, y: 0.3, z: 0.0, visibility: 0.9 }));
  interpolator.setTarget(targetPose, 33.3);

  // Intermediate render frame at t = 16.6ms (60 FPS render tick)
  const pIntermediate = interpolator.step(16.6);
  assert.ok(pIntermediate[0].x > 0.1 && pIntermediate[0].x < 0.2, `Expected smooth transition, got ${pIntermediate[0].x}`);
  assert.ok(pIntermediate[0].y > 0.2 && pIntermediate[0].y < 0.3, `Expected smooth transition, got ${pIntermediate[0].y}`);

  // Stepping further in time approaches target
  const pFinal = interpolator.step(100.0);
  assert.ok(Math.abs(pFinal[0].x - 0.2) < 0.01, `Expected convergence, got ${pFinal[0].x}`);
});

test('PoseInterpolator snaps immediately on large teleport jumps to prevent dragging', () => {
  const interpolator = new PoseInterpolator({ maxSnapDistance: 0.3 });
  const startPose = [{ x: 0.1, y: 0.1, visibility: 0.9 }];
  interpolator.setTarget(startPose, 0);
  interpolator.step(0);

  // Teleport jump across screen (dx = 0.5 > 0.3)
  const jumpPose = [{ x: 0.8, y: 0.8, visibility: 0.9 }];
  interpolator.setTarget(jumpPose, 33.3);

  const pStep = interpolator.step(34.0);
  assert.equal(pStep[0].x, 0.8, 'Expected immediate snap on teleport');
});
