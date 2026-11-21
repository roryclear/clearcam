import test from 'node:test';
import assert from 'node:assert/strict';
import { filtfilt, hampel, fillGaps, smoothSeriesOffline } from '../src/filters/zero-phase.mjs';
import { offlineSessionFrames } from '../src/sessions/offline-analysis.mjs';
import { SessionRecorder } from '../src/sessions/session-recorder.mjs';
import { OneEuroFilter } from '../src/filters/one-euro-filter.mjs';

const fs = 30;
const sine = (f, n, amp = 1, off = 0) => Array.from({ length: n }, (_, i) => off + amp * Math.sin(2 * Math.PI * f * i / fs));

test('filtfilt: unity DC gain, passes slow motion, removes fast noise, no phase lag', () => {
  assert.ok(filtfilt(Array(60).fill(140), fs, 4).every(v => Math.abs(v - 140) < 1e-9));
  const slow = sine(0.5, 300, 40, 140);
  const y = filtfilt(slow, fs, 4);
  const maxErr = Math.max(...y.slice(15, -15).map((v, i) => Math.abs(v - slow[i + 15])));
  assert.ok(maxErr < 0.2, `0.5 Hz passes: ${maxErr}`);
  // Peak position preserved (zero phase).
  const argmin = a => a.indexOf(Math.min(...a));
  assert.equal(argmin(y.slice(30, 90)), argmin(slow.slice(30, 90)));
  const fast = sine(12, 300);
  const amp = Math.max(...filtfilt(fast, fs, 4).slice(15, -15).map(Math.abs));
  assert.ok(amp < 0.01, `12 Hz removed: ${amp}`);
});

test('Hampel removes isolated spikes and keeps real motion', () => {
  const x = sine(0.5, 120, 40, 140);
  const spiked = x.slice();
  spiked[50] = 60; // tracking glitch
  const h = hampel(spiked, 7);
  assert.ok(Math.abs(h[50] - x[50]) < 5, `spike replaced: ${h[50]}`);
  assert.ok(h.every((v, i) => i === 50 || v === x[i]), 'other samples untouched');
});

test('Gaps up to 300 ms are interpolated; longer gaps stay empty and split the series', () => {
  const t = Array.from({ length: 40 }, (_, i) => i * 33);
  const x = Array.from({ length: 40 }, (_, i) => 100 + i);
  for (const i of [5, 6, 7]) x[i] = null; // 132 ms gap
  for (let i = 20; i < 32; i++) x[i] = null; // 429 ms gap
  const f = fillGaps(t, x, 300);
  assert.ok(Math.abs(f[6] - 106) < 1e-9);
  assert.equal(f[25], null);
  const s = smoothSeriesOffline(t, x);
  assert.equal(s[25], null);
  assert.ok(Number.isFinite(s[6]) && Number.isFinite(s[35]));
});

/** Recorded squat: live angles lag (One Euro), raw angles are noisy but unlagged. */
function recordedSquat({ withRaw = true } = {}) {
  const euro = new OneEuroFilter({ minCutoff: 1.0, beta: 0.02 });
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  const frames = [];
  for (let i = 0; i < 300; i++) {
    const t = i * 33.3;
    const truth = 180 - 42.5 * (1 - Math.cos(2 * Math.PI * t / 2400)); // 95° bottom every 2.4 s
    const raw = truth + 3 * rnd();
    const live = euro.filter(raw, t);
    frames.push({ timeMs: Math.round(t), truth, angles: { leftKnee: live, rightKnee: live }, ...(withRaw ? { rawAngles: { leftKnee: raw, rightKnee: raw } } : {}) });
  }
  return frames;
}

test('Offline re-analysis of raw angles tracks the movement better than the live angles', () => {
  const frames = recordedSquat();
  const { frames: out, source } = offlineSessionFrames(frames);
  assert.equal(source, 'raw');
  const rmse = get => Math.sqrt(frames.reduce((s, f, i) => s + (get(i) - f.truth) ** 2, 0) / frames.length);
  const jitter = get => {
    let s = 0;
    for (let i = 2; i < frames.length; i++) s += (get(i) - 2 * get(i - 1) + get(i - 2)) ** 2; // 2nd difference
    return Math.sqrt(s / (frames.length - 2));
  };
  const live = i => frames[i].angles.leftKnee;
  const offline = i => out[i].angles.leftKnee;
  assert.ok(rmse(offline) < rmse(live), `RMSE offline ${rmse(offline).toFixed(2)} vs live ${rmse(live).toFixed(2)}`);
  assert.ok(jitter(offline) < jitter(live) / 2, `jitter offline ${jitter(offline).toFixed(2)} vs live ${jitter(live).toFixed(2)}`);
  assert.equal(out[10].liveAngles, frames[10].angles, 'live angles kept for reference');
});

test('Older recordings without raw angles are smoothed from the live angles', () => {
  const { source, frames } = offlineSessionFrames(recordedSquat({ withRaw: false }));
  assert.equal(source, 'live');
  assert.ok(frames.every(f => Number.isFinite(f.angles.leftKnee)));
  assert.equal(offlineSessionFrames([]).source, 'none');
});

test('Recorder stores raw angles next to the live angles', () => {
  const recorder = new SessionRecorder({ MediaRecorderClass: null });
  const realNow = performance.now.bind(performance);
  performance.now = () => 0;
  try { recorder.start({ exerciseId: 'squat' }); } finally { performance.now = realNow; }
  recorder.recordFrame({ landmarks: [], angles: { leftKnee: 120.04 }, rawAngles: { leftKnee: 118.96, rightKnee: NaN }, timestampMs: 33, reps: 0 });
  recorder.recordFrame({ landmarks: [], angles: { leftKnee: 121 }, timestampMs: 66, reps: 0 });
  const [a, b] = recorder.stop().frames;
  assert.deepEqual(a.rawAngles, { leftKnee: 119, rightKnee: null });
  assert.equal(a.angles.leftKnee, 120);
  assert.equal(b.rawAngles, undefined);
});

test('smoothSeriesOffline: no bridging across frame-time gaps; irregular frames; zero-MAD spikes (PR #16 review)', () => {
  // Two constant runs (180° then 90°) separated by a 4 s capture interruption.
  const t = [];
  const x = [];
  for (let i = 0; i < 30; i++) { t.push(i * 33); x.push(180); }
  for (let i = 0; i < 30; i++) { t.push(5000 + i * 33); x.push(90); }
  const y = smoothSeriesOffline(t, x, { cutoffHz: 4 });
  assert.ok(y.slice(0, 30).every(v => Math.abs(v - 180) < 1e-6), `run 1 ${y[29]}`);
  assert.ok(y.slice(30).every(v => Math.abs(v - 90) < 1e-6), `run 2 ${y[30]}`);

  // Irregular timing (dropped frames): a ramp stays a ramp at its true times.
  const ti = [];
  for (let k = 0, tt = 0; k < 120; k++) { ti.push(tt); tt += k % 5 === 4 ? 66 : 33; }
  const ramp = ti.map(tt => 100 + tt * 0.01);
  const yr = smoothSeriesOffline(ti, ramp, { cutoffHz: 4 });
  const mid = yr.slice(20, 100).map((v, k) => Math.abs(v - ramp[k + 20]));
  assert.ok(Math.max(...mid) < 0.05, `ramp error ${Math.max(...mid)}`);

  // Flat series with one tracking spike: MAD = 0 must not disable rejection.
  const flat = Array(91).fill(180);
  flat[45] = 90;
  const ys = smoothSeriesOffline(flat.map((_, i) => i * 33), flat, { cutoffHz: 4 });
  assert.ok(Math.min(...ys) > 179.9, `spike leaked: ${Math.min(...ys)}`);
  assert.deepEqual(hampel(flat, 7), flat, 'default hampel keeps MAD>0 semantics');
});

test('SessionStorage compact fallback: optional rawAngles never evict older sessions (PR #16 review)', async () => {
  const { SessionStorage } = await import('../src/sessions/session-storage.mjs');
  const { stripRawAngles } = await import('../src/sessions/offline-analysis.mjs');
  class QuotaStorage {
    constructor(max) { this.max = max; this.data = new Map(); }
    size() { let n = 0; for (const [k, v] of this.data) n += k.length + v.length; return n; }
    getItem(k) { return this.data.has(k) ? this.data.get(k) : null; }
    setItem(k, v) {
      const prev = this.data.get(k);
      this.data.set(k, String(v));
      if (this.size() > this.max) {
        if (prev === undefined) this.data.delete(k); else this.data.set(k, prev);
        throw Object.assign(new Error('quota'), { name: 'QuotaExceededError' });
      }
    }
    removeItem(k) { this.data.delete(k); }
  }
  const mk = (id, raw) => ({
    id, recordedAt: '2026-11-21T00:00:00Z', exerciseId: 'squat', frameCount: 60, durationMs: 2000, finalReps: 1,
    frames: Array.from({ length: 60 }, (_, i) => ({ timeMs: i * 33, angles: { leftKnee: 170.1, rightKnee: 169.4 },
      ...(raw ? { rawAngles: { leftKnee: 171.3, rightKnee: 168.8 } } : {}) })),
  });
  const plain = JSON.stringify(mk('x', false)).length;
  const store = new QuotaStorage(plain * 2.3);
  const mgr = new SessionStorage({ storage: store });
  assert.equal(mgr.saveSession(mk('old', false)), true);
  // The full session (with rawAngles) only fits by evicting 'old'; the compact one fits beside it.
  assert.equal(mgr.saveSession(mk('new', true), { compact: stripRawAngles }), true);
  assert.deepEqual(mgr.getIndex().map(s => s.id).sort(), ['new', 'old']);
  assert.equal(mgr.getSession('new').frames[0].rawAngles, undefined);
  // With room, rawAngles are kept.
  const roomy = new SessionStorage({ storage: new QuotaStorage(Infinity) });
  assert.equal(roomy.saveSession(mk('a', true), { compact: stripRawAngles }), true);
  assert.ok(roomy.getSession('a').frames[0].rawAngles);
});

test('smoothSeriesOffline round 2: Hampel stays inside capture runs; irregular runs keep the last frame (PR #16)', () => {
  const t = [];
  const x = [];
  for (let i = 0; i < 60; i++) { t.push(i * 33); x.push(180); }
  for (let i = 0; i < 5; i++) { t.push(6000 + i * 33); x.push(90); }
  const y = smoothSeriesOffline(t, x);
  assert.deepEqual(y.slice(60).map(v => Math.round(v * 1e6) / 1e6), [90, 90, 90, 90, 90]);

  // Irregular run whose end falls between grid points: the final frame is kept.
  const ti = [];
  for (let k = 0; k < 60; k++) ti.push(k * 33 + (k % 7 === 3 ? 12 : 0));
  ti.push(ti[ti.length - 1] + 50);
  const ramp = ti.map(tt => 100 + tt * 0.005);
  const yr = smoothSeriesOffline(ti, ramp, { cutoffHz: 4 });
  assert.ok(Math.abs(yr[yr.length - 1] - ramp[ramp.length - 1]) < 0.05, `last ${yr[yr.length - 1]} vs ${ramp[ramp.length - 1]}`);
});
