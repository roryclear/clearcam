import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraTiltFromGravity, correctImageRoll, correctWorldRoll, DeviceLevel } from '../src/tracking/device-level.mjs';
import { calculateTrunkAngle } from '../src/biomechanics/angle-calculations.mjs';

const G = 9.81;
const rad = d => (d * Math.PI) / 180;
/** Spec convention: upright portrait phone reports y ≈ +g; rotated clockwise by θ (seen on screen). */
const gravityFor = (rollDeg, pitchDeg = 0) => ({
  x: -G * Math.sin(rad(rollDeg)) * Math.cos(rad(pitchDeg)),
  y: G * Math.cos(rad(rollDeg)) * Math.cos(rad(pitchDeg)),
  z: G * Math.sin(rad(pitchDeg)),
});

test('Roll and pitch from gravity; independent of the sensor sign convention', () => {
  for (const roll of [-20, -5, 0, 7, 30]) {
    const t = cameraTiltFromGravity(gravityFor(roll, 4));
    assert.ok(Math.abs(t.rollDeg - roll) < 1e-6, `${roll} → ${t.rollDeg}`);
    assert.ok(Math.abs(t.pitchDeg - 4) < 1e-6);
    // Some browsers report the opposite sign for all axes.
    const g = gravityFor(roll, 4);
    const flipped = cameraTiltFromGravity({ x: -g.x, y: -g.y, z: -g.z });
    assert.ok(Math.abs(flipped.rollDeg - roll) < 1e-6);
  }
  // Landscape: the device x axis is vertical; 90° and 270° agree.
  const g = { x: G * Math.cos(rad(6)), y: G * Math.sin(rad(6)), z: 0 };
  const a = cameraTiltFromGravity(g, 90);
  const b = cameraTiltFromGravity(g, 270);
  assert.ok(Math.abs(Math.abs(a.rollDeg) - 6) < 1e-6);
  assert.ok(Math.abs(a.rollDeg - b.rollDeg) < 1e-9);
  assert.equal(cameraTiltFromGravity({ x: 0, y: 0, z: G }), null, 'flat phone');
  assert.equal(cameraTiltFromGravity(null), null);
});

/** An upright trunk (vertical in the world) seen by a camera rolled by rollDeg. */
function trunkSeenRolled(rollDeg, facing) {
  // Content rotates opposite to the camera for the rear camera, with it for the front camera.
  const a = rad((facing === 'user' ? 1 : -1) * rollDeg);
  const w = 720;
  const h = 1280;
  const pt = (x, y) => {
    const px = x * w - w / 2;
    const py = y * h - h / 2;
    return { x: (w / 2 + px * Math.cos(a) - py * Math.sin(a)) / w, y: (h / 2 + px * Math.sin(a) + py * Math.cos(a)) / h, visibility: 0.95 };
  };
  const lm = Array(33).fill(null).map(() => ({ x: 0.5, y: 0.5, visibility: 0.1 }));
  lm[11] = pt(0.45, 0.3); lm[12] = pt(0.55, 0.3); // shoulders
  lm[23] = pt(0.46, 0.55); lm[24] = pt(0.54, 0.55); // hips
  return { lm, w, h };
}

test('Roll correction restores segment-to-vertical angles for both cameras', () => {
  for (const facing of ['environment', 'user']) {
    const { lm, w, h } = trunkSeenRolled(8, facing);
    const before = calculateTrunkAngle(lm, w, h);
    assert.ok(Math.abs(before - 8) < 0.5, `rolled camera reads ${before}°`);
    const after = calculateTrunkAngle(correctImageRoll(lm, 8, w, h, facing), w, h);
    assert.ok(after < 0.5, `${facing}: corrected ${after}°`);
  }
  assert.equal(correctImageRoll(null, 5, 1, 1), null);
  const world = [{ x: 0, y: -0.5, z: 0 }];
  const r = correctWorldRoll(world, 90, 'environment');
  assert.ok(Math.abs(r[0].x - 0.5) < 1e-9 && Math.abs(r[0].y) < 1e-9);
});

test('DeviceLevel smooths motion events, needs permission on iOS and goes stale', async () => {
  let now = 0;
  const listeners = {};
  const target = {
    DeviceMotionEvent: { requestPermission: async () => 'granted' },
    addEventListener: (n, f) => { listeners[n] = f; },
    removeEventListener: n => { delete listeners[n]; },
  };
  const level = new DeviceLevel({ target, now: () => now, alpha: 1 });
  assert.equal(level.tilt(0), null);
  assert.equal(await level.start(), true);
  listeners.devicemotion({ accelerationIncludingGravity: gravityFor(5) });
  assert.ok(Math.abs(level.tilt(0).rollDeg - 5) < 1e-6);
  now = 5000;
  assert.equal(level.tilt(0), null, 'stale');
  level.stop();
  assert.equal(listeners.devicemotion, undefined);
  const denied = new DeviceLevel({ target: { DeviceMotionEvent: { requestPermission: async () => 'denied' } } });
  assert.equal(await denied.start(), false);
  assert.equal(await new DeviceLevel({ target: {} }).start(), false);
});

test('DeviceLevel: stop() during a pending permission prompt prevents a late listener (PR #16)', async () => {
  let resolve;
  const added = [];
  const target = {
    DeviceMotionEvent: { requestPermission: () => new Promise(r => { resolve = r; }) },
    addEventListener: (type, fn) => added.push(type),
    removeEventListener: () => {},
  };
  const level = new DeviceLevel({ target });
  const pending = level.start();
  level.stop();
  resolve('granted');
  assert.equal(await pending, false);
  assert.deepEqual(added, []);
  // A normal granted start still attaches.
  const again = level.start();
  resolve('granted');
  assert.equal(await again, true);
  assert.deepEqual(added, ['devicemotion']);
});
