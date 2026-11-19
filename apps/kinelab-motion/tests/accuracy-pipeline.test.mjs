import test from 'node:test';
import assert from 'node:assert/strict';
import { generateSquatSequence } from '../bench/synthetic-pose.mjs';
import { runPipeline, PIPELINES } from '../bench/accuracy-bench.mjs';
import { WorldPoseFilter } from '../src/filters/world-pose-filter.mjs';
import { selectBilateralAngle, bilateralSide } from '../src/biomechanics/angle-calculations.mjs';
import { ViewOrientationEstimator, blendAnglesByView } from '../src/biomechanics/view-orientation.mjs';
import { SquatExercise, hipAnkleSpan, updateDepthRatio } from '../src/exercises/squat-exercise.mjs';
import { AdaptiveController } from '../src/tracking/adaptive-controller.mjs';
import { MediaPipeRunner, MODEL_LADDER } from '../src/tracking/mediapipe-runner.mjs';
import { FramingAssistant } from '../src/tracking/framing-assistant.mjs';
import { SMOOTHING_PRESETS } from '../src/filters/pose-filter.mjs';

const OLD = PIPELINES['current (3D raw + bone filter, avg legs)'];
const NEW = PIPELINES['view-blend (yaw EMA), best leg'];
const pt = (x, y, z = 0, visibility = 0.95) => ({ x, y, z, visibility, presence: 0.99 });

function mean(seqOpts, cfg, seeds = [1, 2, 3]) {
  const rs = seeds.map(seed => runPipeline(generateSquatSequence({ ...seqOpts, seed }), cfg));
  return {
    rmse: rs.reduce((s, r) => s + r.rmse, 0) / rs.length,
    jitter: rs.reduce((s, r) => s + r.jitter, 0) / rs.length,
    repErr: rs.reduce((s, r) => s + Math.abs(r.reps - 6), 0),
  };
}

test('Benchmark: side-view knee RMSE improves >3x over the previous pipeline with no missed reps', () => {
  const before = mean({ yawDeg: 0 }, OLD);
  const after = mean({ yawDeg: 0 }, NEW);
  assert.ok(after.rmse < 5, `RMSE ${after.rmse.toFixed(2)}`);
  assert.ok(after.rmse * 3 < before.rmse, `before ${before.rmse.toFixed(2)} after ${after.rmse.toFixed(2)}`);
  assert.ok(after.jitter < 2, `standing jitter ${after.jitter.toFixed(2)}`);
  assert.equal(after.repErr, 0);
});

test('Benchmark: oblique and frontal views are no worse than before and far less jittery', () => {
  for (const yawDeg of [30, 45, 90]) {
    const before = mean({ yawDeg }, OLD);
    const after = mean({ yawDeg }, NEW);
    assert.ok(after.rmse <= before.rmse + 0.5, `yaw ${yawDeg}: ${after.rmse.toFixed(2)} vs ${before.rmse.toFixed(2)}`);
    assert.ok(after.jitter * 3 < before.jitter, `yaw ${yawDeg}: jitter ${after.jitter.toFixed(2)} vs ${before.jitter.toFixed(2)}`);
    assert.equal(after.repErr, 0, `yaw ${yawDeg}`);
  }
});

test('Smoothing presets adapt to motion speed (beta scaled for normalized coordinates)', () => {
  for (const name of ['low', 'medium', 'high']) {
    assert.ok(SMOOTHING_PRESETS[name].beta >= 10, `${name} beta ${SMOOTHING_PRESETS[name].beta}`);
  }
});

test('WorldPoseFilter reduces depth jitter and restarts after occlusion', () => {
  const wf = new WorldPoseFilter();
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  const raw = [];
  const out = [];
  for (let i = 0; i < 90; i++) {
    const p = pt(0.1, 0.4, 0.1 * rand());
    raw.push(p.z);
    out.push(wf.filter([p], i * 33)[0].z);
  }
  const sd = a => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
  assert.ok(sd(out.slice(30)) < sd(raw.slice(30)) / 2);

  const occluded = wf.filter([pt(0.5, 0.5, 0.5, 0.1)], 90 * 33)[0];
  assert.equal(occluded.z, 0.5, 'low-visibility point passes through untouched');
  const back = wf.filter([pt(0.9, 0.9, 0.9)], 91 * 33)[0];
  assert.equal(back.z, 0.9, 'filter restarts instead of dragging old state');
});

test('selectBilateralAngle prefers the clearly more visible leg and averages otherwise', () => {
  const lm = Array.from({ length: 33 }, () => pt(0.5, 0.5));
  for (const i of [24, 26, 28]) lm[i].visibility = 0.62; // far (right) leg occluded side-on
  const angles = { leftKnee: 100, rightKnee: 140 };
  assert.equal(selectBilateralAngle(angles, lm, 'Knee'), 100);
  assert.equal(bilateralSide(angles, lm, 'Knee'), 'left');
  for (const i of [24, 26, 28]) lm[i].visibility = 0.93;
  assert.equal(selectBilateralAngle(angles, lm, 'Knee'), 120);
  assert.equal(selectBilateralAngle({ leftKnee: null, rightKnee: 140 }, lm, 'Knee'), 140);
  assert.equal(selectBilateralAngle({}, lm, 'Knee'), null);
});

test('ViewOrientationEstimator distinguishes side-on and frontal views and blends angles accordingly', () => {
  const side = new ViewOrientationEstimator();
  const front = new ViewOrientationEstimator();
  const world = (dx, dz) => {
    const w = Array.from({ length: 33 }, () => pt(0, 0, 0));
    w[11] = pt(-dx, -0.5, -dz); w[12] = pt(dx, -0.5, dz);
    return w;
  };
  let ws;
  let wf;
  for (let t = 0; t < 2000; t += 33) {
    ws = side.update(world(0.01, 0.18), t);
    wf = front.update(world(0.18, 0.01), t);
  }
  assert.ok(ws.yawDeg < 10 && ws.sagittal2DWeight > 0.95);
  assert.ok(wf.yawDeg > 80 && wf.sagittal2DWeight < 0.05);

  const a2 = { leftKnee: 100, leftElbow: 90 };
  const a3 = { leftKnee: 110, leftElbow: 95 };
  assert.equal(blendAnglesByView(a2, a3, ws).leftKnee, 100, 'side view uses 2D knee');
  assert.equal(blendAnglesByView(a2, a3, wf).leftKnee, 110, 'frontal view uses 3D knee');
  assert.equal(blendAnglesByView(a2, a3, ws).leftElbow, 95, 'non-sagittal joints keep 3D');
  assert.equal(blendAnglesByView({ leftKnee: 100 }, {}, ws).leftKnee, 100, 'falls back to whichever exists');
});

test('Squat depth cue: hips back at standing height complete a rep even if the knee angle reads low', () => {
  const squat = new SquatExercise();
  let state = squat.createState();
  const frame = (hipY, knee, t) => {
    const lm = Array.from({ length: 33 }, () => pt(0.5, 0.5));
    lm[23] = pt(0.5, hipY); lm[24] = pt(0.5, hipY);
    lm[27] = pt(0.5, 0.9); lm[28] = pt(0.5, 0.9);
    return { angles: { leftKnee: knee, rightKnee: knee }, landmarks: lm, timestampMs: t, trackingQuality: 'good' };
  };
  for (let t = 0; t < 300; t += 33) state = squat.update(state, frame(0.5, 165, t)); // standing, span 0.4
  for (let t = 300; t < 900; t += 33) state = squat.update(state, frame(0.6, 100, t)); // bottom
  // Back at standing height, but correlated depth error holds the knee at 148° (< 155°).
  for (let t = 900; t < 1400; t += 33) state = squat.update(state, frame(0.5, 148, t));
  assert.equal(state.reps, 1);
});

test('Squat depth cue: a low knee reading while hips stay at standing height is not a squat', () => {
  const squat = new SquatExercise();
  let state = squat.createState();
  const lm = hipY => {
    const l = Array.from({ length: 33 }, () => pt(0.5, 0.5));
    l[23] = pt(0.5, hipY); l[24] = pt(0.5, hipY); l[27] = pt(0.5, 0.9); l[28] = pt(0.5, 0.9);
    return l;
  };
  for (let t = 0; t < 2000; t += 33) {
    const knee = t > 600 && t < 1200 ? 110 : 165; // depth-noise dip
    state = squat.update(state, { angles: { leftKnee: knee, rightKnee: knee }, landmarks: lm(0.5), timestampMs: t, trackingQuality: 'good' });
  }
  assert.equal(state.reps, 0);
  assert.equal(state.phase, 'ready');
});

test('hipAnkleSpan / updateDepthRatio track a decaying standing reference', () => {
  const lm = Array.from({ length: 33 }, () => pt(0.5, 0.5));
  lm[23] = pt(0.5, 0.5); lm[24] = pt(0.5, 0.5); lm[27] = pt(0.5, 0.9); lm[28] = pt(0.5, 0.9);
  assert.ok(Math.abs(hipAnkleSpan(lm) - 0.4) < 1e-9);
  let st = {};
  st = { ...st, ...updateDepthRatio(st, 0.4, 0) };
  st = { ...st, ...updateDepthRatio(st, 0.3, 1000) };
  assert.ok(st.depthRatio > 0.75 && st.depthRatio < 0.78);
  assert.equal(updateDepthRatio(st, null, 2000).depthRatio, null);
});

test('AdaptiveController steps down the model ladder one level at a time', () => {
  const ac = new AdaptiveController({ consecutiveViolationsNeeded: 2 });
  ac.setCurrentModel('heavy');
  const slow = { avgLatencyMs: 200, inferenceFps: 5 };
  ac.evaluate(slow);
  const r1 = ac.evaluate(slow);
  assert.equal(r1.shouldFallback, true);
  assert.equal(r1.nextModel, 'full');
  ac.setCurrentModel('full');
  ac.evaluate(slow);
  assert.equal(ac.evaluate(slow).nextModel, 'lite');
  ac.setCurrentModel('lite');
  ac.evaluate(slow);
  assert.equal(ac.evaluate(slow).shouldFallback, false, 'lite is the floor');
  assert.deepEqual(MODEL_LADDER, ['heavy', 'full', 'lite']);
});

test('MediaPipeRunner adaptive resolution keeps the subject at least ~320 px tall', () => {
  const runner = new MediaPipeRunner();
  assert.equal(runner.inferencePreset, 'adaptive');
  assert.equal(runner.adaptiveTargetHeight(720), 480, 'no subject yet: cheap detection pass');
  runner.setSubjectExtent(0.9); // near subject
  assert.equal(runner.adaptiveTargetHeight(720), 384);
  runner.setSubjectExtent(0.4); // far subject needs more pixels
  assert.equal(runner.adaptiveTargetHeight(720), 720);
  runner.setSubjectExtent(0.6);
  const h = runner.adaptiveTargetHeight(1080);
  assert.ok(h * 0.6 >= 320 && h < 1080, `height ${h}`);
});

test('FramingAssistant asks for the recommended camera view once the body is framed', () => {
  const fa = new FramingAssistant();
  const lm = Array.from({ length: 33 }, () => pt(0.5, 0.5));
  const sideEx = fa.evaluate(lm, 'squat', { yawDeg: 70, recommendedView: 'side' });
  assert.equal(sideEx.status, 'partial');
  assert.match(sideEx.guidance, /side-on/);
  assert.equal(fa.evaluate(lm, 'squat', { yawDeg: 10, recommendedView: 'side' }).status, 'optimal');
  assert.match(fa.evaluate(lm, 'shoulder_abduction', { yawDeg: 10, recommendedView: 'front' }).guidance, /Face the camera/);
  assert.equal(fa.evaluate(lm, 'squat').status, 'optimal', 'no view info: unchanged behaviour');
});
