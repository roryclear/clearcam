import test from 'node:test';
import assert from 'node:assert/strict';
import { OneEuroFilter, LowPassFilter } from '../src/filters/one-euro-filter.mjs';
import { OutlierFilter } from '../src/filters/outlier-filter.mjs';
import { PoseFilter } from '../src/filters/pose-filter.mjs';

test('LowPassFilter initializes and averages sequential values', () => {
  const lpf = new LowPassFilter(0.5);
  assert.equal(lpf.filter(10), 10);
  assert.equal(lpf.filter(20), 15);
  assert.equal(lpf.filter(20), 17.5);
});

test('OneEuroFilter reduces jitter on static signal', () => {
  const filter = new OneEuroFilter({ minCutoff: 1.0, beta: 0.007 });
  const rawJitter = [];
  const filtered = [];

  // Generate 50 frames with synthetic high-frequency noise at 30 FPS (~33.3ms dt)
  for (let i = 0; i < 50; i++) {
    const t = i * 33.3;
    const noise = (Math.sin(i * 1.7) > 0 ? 1 : -1) * 2.0;
    const rawVal = 100 + noise;
    rawJitter.push(rawVal);
    filtered.push(filter.filter(rawVal, t));
  }

  // Calculate variance of raw vs filtered in the steady state (frames 20-50)
  const calcVar = arr => {
    const slice = arr.slice(20);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    return slice.reduce((sum, val) => sum + (val - mean) ** 2, 0) / slice.length;
  };

  const rawVar = calcVar(rawJitter);
  const filteredVar = calcVar(filtered);

  // Filtered variance should be significantly lower (> 70% reduction)
  assert.ok(filteredVar < rawVar * 0.3, `Filtered variance (${filteredVar}) should be < 30% of raw (${rawVar})`);
});

test('OneEuroFilter responds rapidly to high-speed step input without excessive lag', () => {
  const filter = new OneEuroFilter({ minCutoff: 1.0, beta: 0.05 });
  // Settle at 0
  for (let i = 0; i < 10; i++) {
    filter.filter(0, i * 33.3);
  }

  // Large step from 0 to 100 at frame 10 (333ms)
  let val = filter.filter(100, 333);
  // Next frame (366ms)
  val = filter.filter(100, 366);

  // Due to high derivative, beta boosts cutoff so the filter reaches > 80% of step within 2 frames
  assert.ok(val > 75, `Expected fast response (>75), got ${val}`);
});

test('OutlierFilter rejects impossible instantaneous jumps', () => {
  const outlier = new OutlierFilter({ maxSpeed: 4.0, maxInterpolateFrames: 2 });
  const point = (x, y) => ({ x, y, visibility: 0.95 });

  // Baseline point
  const r1 = outlier.evaluate(0, point(0.5, 0.5), 0);
  assert.equal(r1.accepted, true);
  assert.equal(r1.wasRejected, false);

  // Realistic movement (delta 0.02 in 33ms -> speed ~0.6 < 4.0)
  const r2 = outlier.evaluate(0, point(0.52, 0.5), 33);
  assert.equal(r2.accepted, true);
  assert.equal(r2.wasRejected, false);

  // Unphysical glitch: teleport to 0.95 in 33ms -> speed = 0.43 / 0.033 = 13.0 > 4.0
  const r3 = outlier.evaluate(0, point(0.95, 0.5), 66);
  assert.equal(r3.accepted, true);
  assert.equal(r3.wasRejected, true); // Interpolated from previous position
  assert.equal(r3.point.x, 0.52);
});

test('PoseFilter smooths full pose and handles occluded landmarks', () => {
  const pf = new PoseFilter({ preset: 'medium', minVisibility: 0.55 });
  const makePose = (hipX = 0.5, kneeVis = 0.9) => {
    return Array.from({ length: 33 }, (_, i) => ({
      x: i === 23 ? hipX : 0.5,
      y: 0.5,
      z: 0.0,
      visibility: i === 25 ? kneeVis : 0.9,
      presence: 0.9,
    }));
  };

  // Frame 1
  const f1 = pf.filter(makePose(0.5), 0);
  assert.ok(f1);
  assert.equal(f1.length, 33);
  assert.equal(f1[23].x, 0.5);

  // Frame 2: knee drops visibility to 0.2 (occluded)
  const f2 = pf.filter(makePose(0.51, 0.2), 33);
  assert.equal(f2[25].isReliable, false);

  // Frame 3: pose is null (lost tracking)
  const f3 = pf.filter(null, 66);
  assert.equal(f3, null);
});
