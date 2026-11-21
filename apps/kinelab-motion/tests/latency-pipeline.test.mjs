import test from 'node:test';
import assert from 'node:assert/strict';
import { selectFrameTime, missedFrames, supportsVideoFrameCallback } from '../src/tracking/frame-clock.mjs';
import { isLikelyMobile, resolveModelChoice } from '../src/tracking/device-profile.mjs';
import { PerformanceMonitor, percentile } from '../src/tracking/performance-monitor.mjs';
import { PoseInterpolator } from '../src/filters/pose-interpolator.mjs';
import { validateSettings } from '../src/settings/settings-schema.mjs';
import { SessionRecorder } from '../src/sessions/session-recorder.mjs';

test('Frame time prefers capture, then presentation, then callback time', () => {
  assert.deepEqual(selectFrameTime(1000, { captureTime: 960, presentationTime: 990 }), { timeMs: 960, clock: 'capture' });
  assert.deepEqual(selectFrameTime(1000, { presentationTime: 990 }), { timeMs: 990, clock: 'presentation' });
  assert.deepEqual(selectFrameTime(1000, null), { timeMs: 1000, clock: 'callback' });
  // Clocks from another timeline (e.g. epoch ms) or in the future are ignored.
  assert.equal(selectFrameTime(1000, { captureTime: 1.7e12 }).clock, 'callback');
  assert.equal(selectFrameTime(1000, { captureTime: -5000, presentationTime: 995 }).clock, 'presentation');
});

test('Frame times strictly increase for MediaPipe VIDEO mode', () => {
  const a = selectFrameTime(1000, { captureTime: 960 }, -Infinity);
  const b = selectFrameTime(1001, { captureTime: 960 }, a.timeMs); // same frame clock twice
  assert.ok(b.timeMs > a.timeMs);
  const c = selectFrameTime(1002, { captureTime: 950 }, b.timeMs); // clock went backwards
  assert.ok(c.timeMs > b.timeMs);
});

test('Missed camera frames come from presentedFrames gaps', () => {
  assert.equal(missedFrames(null, 10), 0);
  assert.equal(missedFrames(10, 11), 0);
  assert.equal(missedFrames(10, 14), 3);
  assert.equal(missedFrames(14, 14), 0);
  assert.equal(supportsVideoFrameCallback({ requestVideoFrameCallback() {} }), true);
  assert.equal(supportsVideoFrameCallback({}), false);
});

test('Auto model starts phones on Full and desktops on Heavy; explicit choices win', () => {
  assert.equal(resolveModelChoice('auto', { mobile: true }), 'full');
  assert.equal(resolveModelChoice('auto', { mobile: false }), 'heavy');
  assert.equal(resolveModelChoice('heavy', { mobile: true }), 'heavy');
  assert.equal(resolveModelChoice('lite', { mobile: false }), 'lite');
  const nav = ua => ({ userAgent: ua, maxTouchPoints: 0 });
  assert.equal(isLikelyMobile({ navigator: nav('Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)') }), true);
  assert.equal(isLikelyMobile({ navigator: nav('Mozilla/5.0 (Linux; Android 15; Pixel 9)') }), true);
  assert.equal(isLikelyMobile({ navigator: nav('Mozilla/5.0 (X11; Linux x86_64) Chrome/131') , matchMedia: () => ({ matches: false }) }), false);
  // iPadOS with a desktop UA: touch + coarse pointer.
  assert.equal(isLikelyMobile({
    navigator: { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', maxTouchPoints: 5 },
    matchMedia: () => ({ matches: true }),
  }), true);
  assert.equal(isLikelyMobile({ navigator: { userAgentData: { mobile: true }, userAgent: '' } }), true);
});

test('Settings v4: stored Heavy default moves to Auto; explicit Full/Lite stay; prediction defaults on', () => {
  assert.equal(validateSettings({ version: 3, tracking: { model: 'heavy' } }).tracking.model, 'auto');
  assert.equal(validateSettings({ version: 4, tracking: { model: 'heavy' } }).tracking.model, 'heavy');
  assert.equal(validateSettings({ version: 3, tracking: { model: 'full' } }).tracking.model, 'full');
  assert.equal(validateSettings({ version: 4, tracking: { model: 'auto' } }).tracking.model, 'auto');
  assert.equal(validateSettings(null).overlays.predictiveOverlay, true);
  assert.equal(validateSettings({ overlays: { predictiveOverlay: false } }).overlays.predictiveOverlay, false);
});

test('End-to-end latency percentiles are reported per stage and cleared on model swap', () => {
  const perf = new PerformanceMonitor();
  assert.equal(perf.getMetrics().captureToOverlayP50, null);
  for (let ms = 1; ms <= 100; ms++) perf.recordPipelineLatency('overlay', ms, 'capture');
  perf.recordPipelineLatency('landmarks', 20, 'capture');
  perf.recordPipelineLatency('overlay', NaN);
  perf.recordPipelineLatency('nope', 5);
  const m = perf.getMetrics();
  assert.equal(m.captureToOverlayP50, 50);
  assert.equal(m.captureToOverlayP95, 95);
  assert.equal(m.captureToLandmarksP50, 20);
  assert.equal(m.latencyClock, 'capture');
  perf.resetInferenceStats();
  assert.equal(perf.getMetrics().captureToOverlayP50, null);
  assert.equal(percentile([], 50), null);
  perf.recordDroppedFrame(3);
  perf.recordDroppedFrame(0);
  assert.equal(perf.getMetrics().droppedFrames, 3);
});

function lagAt(interp, speed, frameMs, inferenceMs, renderEvery = 1000 / 60) {
  // Joint moving at constant speed (normalized units per ms). A frame captured at time c
  // reaches the interpolator at c + inferenceMs, stamped with its capture time.
  let worst = 0;
  let nextTarget = 0;
  for (let t = 0; t <= 1000; t += renderEvery) {
    while (nextTarget + inferenceMs <= t) {
      interp.setTarget([{ x: 0.2 + speed * nextTarget, y: 0.5, visibility: 0.9 }], nextTarget);
      nextTarget += frameMs;
    }
    const p = interp.step(t);
    if (t > 300) worst = Math.max(worst, Math.abs(0.2 + speed * t - p[0].x) / speed);
  }
  return worst; // ms behind the true position
}

test('Predictive overlay removes most of the interpolation lag for a moving joint', () => {
  const speed = 0.0005; // 0.5 normalized units/s, a brisk squat
  const plain = lagAt(new PoseInterpolator({ lambda: 32 }), speed, 33.3, 25);
  const predicted = lagAt(new PoseInterpolator({ lambda: 32, predictionMs: 40 }), speed, 33.3, 25);
  assert.ok(plain > 60, `chase alone lags inference + frame age + 1/λ: ${plain.toFixed(1)} ms`);
  assert.ok(predicted < plain / 2, `prediction ${predicted.toFixed(1)} ms vs ${plain.toFixed(1)} ms`);
});

test('Prediction is capped, resets after gaps and ignores snap-sized jumps', () => {
  const interp = new PoseInterpolator({ lambda: 32, predictionMs: 40 });
  interp.setTarget([{ x: 0.1, y: 0.5 }], 0);
  interp.setTarget([{ x: 0.12, y: 0.5 }], 33);
  // Far past the last target the offset stops at 40 ms of motion.
  let p;
  for (let t = 33; t <= 500; t += 16) p = interp.step(t);
  const v = 0.02 / 33;
  assert.ok(p[0].x <= 0.12 + v * 40 + 1e-6, `capped: ${p[0].x}`);
  // A gap longer than 150 ms drops the velocity estimate.
  interp.setTarget([{ x: 0.3, y: 0.5 }], 700);
  assert.equal(interp.velocity, null);
  // A teleport between consecutive frames gives no velocity for that landmark.
  interp.setTarget([{ x: 0.31, y: 0.5 }], 733);
  interp.setTarget([{ x: 0.9, y: 0.5 }], 766);
  assert.equal(interp.velocity[0], null);
  // Prediction off by default: identical to the previous behaviour.
  assert.equal(new PoseInterpolator().predictionMs, 0);
});

test('Recorder clamps frames captured just before recording started to t = 0', () => {
  const recorder = new SessionRecorder({ MediaRecorderClass: null });
  const realNow = performance.now.bind(performance);
  performance.now = () => 1000;
  try { recorder.start({ exerciseId: 'squat' }); } finally { performance.now = realNow; }
  recorder.recordFrame({ landmarks: [], angles: {}, timestampMs: 990, reps: 0 });
  recorder.recordFrame({ landmarks: [], angles: {}, timestampMs: 1033, reps: 0 });
  const frames = recorder.stop().frames;
  assert.equal(frames[0].timeMs, 0);
  assert.equal(frames[1].timeMs, 33);
});

test('Predictive overlay tracks a squat-like oscillation better, reversals included', () => {
  const run = interp => {
    let se = 0;
    let n = 0;
    let next = 0;
    const pos = t => 0.5 + 0.15 * Math.sin(Math.PI * t / 1000); // 0.5 Hz
    for (let t = 0; t <= 5000; t += 1000 / 60) {
      while (next + 25 <= t) { interp.setTarget([{ x: pos(next), y: 0.5 }], next); next += 33.3; }
      const p = interp.step(t);
      if (t > 300) { se += (p[0].x - pos(t)) ** 2; n++; }
    }
    return Math.sqrt(se / n);
  };
  const plain = run(new PoseInterpolator({ lambda: 32 }));
  const predicted = run(new PoseInterpolator({ lambda: 32, predictionMs: 40 }));
  assert.ok(predicted < plain / 2, `RMS ${predicted.toFixed(4)} vs ${plain.toFixed(4)}`);
});

test('Inference cap follows camera cadence, not display-tick callback spacing', async () => {
  const { sampleClockMs, frameDue } = await import('../src/tracking/frame-clock.mjs');
  // 30 fps camera on a 75 Hz display: callbacks land on display ticks 27/40 ms apart.
  const ticks = 1000 / 75;
  let last = -Infinity;
  let sampled = 0;
  let callbackCapped = 0;
  let lastCb = -Infinity;
  for (let f = 0; f < 60; f++) {
    const mediaTime = f / 30; // seconds
    const cbTime = Math.ceil((f * 1000 / 30) / ticks) * ticks;
    const clock = sampleClockMs({ mediaTime }, cbTime);
    if (frameDue(clock, last, 28)) { last = clock; sampled++; }
    if (cbTime - lastCb >= 28) { lastCb = cbTime; callbackCapped++; }
  }
  assert.equal(sampled, 60, 'every 30 fps frame is under the ~35 FPS cap');
  assert.ok(callbackCapped < 50, `callback-spacing cap drops frames: ${callbackCapped}/60`);
  // 60 fps camera: every other frame.
  last = -Infinity; sampled = 0;
  for (let f = 0; f < 60; f++) if (frameDue(f * 1000 / 60, last, 28)) { last = f * 1000 / 60; sampled++; }
  assert.equal(sampled, 30);
  // Without mediaTime the frame time is used; a clock that goes backwards samples.
  assert.equal(sampleClockMs(null, 123), 123);
  assert.equal(frameDue(5, 1000, 28), true);
});

test('A live fallback keeps an Auto preference; explicit models are replaced', async () => {
  const { modelPreferenceAfterFallback, resolveModelChoice } = await import('../src/tracking/device-profile.mjs');
  const saved = modelPreferenceAfterFallback('auto', 'lite');
  assert.equal(saved, 'auto');
  // Next session starts from the device default again, not Lite.
  assert.equal(resolveModelChoice(saved, { mobile: true }), 'full');
  assert.equal(modelPreferenceAfterFallback('heavy', 'full'), 'full');
});

test('Predicted offset is capped by vector length, including diagonal motion', () => {
  const interp = new PoseInterpolator({ lambda: 32, predictionMs: 40 });
  interp.setTarget([{ x: 0.1, y: 0.1 }], 0);
  interp.setTarget([{ x: 0.15, y: 0.15 }], 33);
  const p = interp._predict(0, 40);
  const len = Math.hypot(p.x - 0.15, p.y - 0.15);
  assert.ok(len <= 0.05 + 1e-9, `offset length ${len}`);
  assert.ok(Math.abs((p.x - 0.15) - (p.y - 0.15)) < 1e-9, 'direction preserved');
  // Small offsets are untouched.
  const q = new PoseInterpolator({ predictionMs: 40 });
  q.setTarget([{ x: 0.1, y: 0.1 }], 0);
  q.setTarget([{ x: 0.101, y: 0.1 }], 33);
  assert.ok(Math.abs(q._predict(0, 33).x - 0.102) < 1e-9);
});

test('Latency percentiles never mix frame clocks', () => {
  const perf = new PerformanceMonitor();
  for (let i = 0; i < 100; i++) perf.recordPipelineLatency('landmarks', 80, 'capture');
  perf.recordPipelineLatency('landmarks', 5, 'callback');
  const m = perf.getMetrics();
  assert.equal(m.captureToLandmarksP50, 5);
  assert.equal(m.latencyClock, 'callback');
  perf.recordPipelineLatency('overlay', 90, 'capture');
  assert.equal(perf.getMetrics().overlayLatencyClock, 'capture');
});

test('A stalled mediaTime (e.g. constant 0) does not stop sampling', async () => {
  const { FrameSampler } = await import('../src/tracking/frame-clock.mjs');
  const s = new FrameSampler(28);
  let sampled = 0;
  for (let f = 0; f < 30; f++) if (s.due({ mediaTime: 0 }, 1000 + f * 33.3)) sampled++;
  assert.ok(sampled >= 29, `sampled ${sampled}/30`);
  // Advancing mediaTime still follows camera cadence; reset restores media mode.
  s.reset();
  sampled = 0;
  for (let f = 0; f < 60; f++) if (s.due({ mediaTime: f / 60 }, 5000 + f)) sampled++;
  assert.equal(sampled, 30);
});
