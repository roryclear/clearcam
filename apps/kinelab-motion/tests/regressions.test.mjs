import test from 'node:test';
import assert from 'node:assert/strict';
import { SettingsManager } from '../src/settings/settings-manager.mjs';
import { DEFAULT_SETTINGS, createDefaultSettings } from '../src/settings/settings-schema.mjs';
import { SessionStorage } from '../src/sessions/session-storage.mjs';
import { SessionRecorder } from '../src/sessions/session-recorder.mjs';
import { TimelinePlayer } from '../src/sessions/timeline-player.mjs';
import { SitToStandExercise } from '../src/exercises/sit-to-stand-exercise.mjs';
import { MediaPipeRunner } from '../src/tracking/mediapipe-runner.mjs';

class QuotaStorage {
  constructor(maxChars = Infinity) {
    this.maxChars = maxChars;
    this.data = new Map();
  }
  _size() {
    let n = 0;
    for (const [k, v] of this.data) n += k.length + v.length;
    return n;
  }
  getItem(k) { return this.data.has(k) ? this.data.get(k) : null; }
  setItem(k, v) {
    const prev = this.data.get(k);
    this.data.set(k, String(v));
    if (this._size() > this.maxChars) {
      if (prev === undefined) this.data.delete(k); else this.data.set(k, prev);
      const err = new Error('QuotaExceededError');
      err.name = 'QuotaExceededError';
      throw err;
    }
  }
  removeItem(k) { this.data.delete(k); }
}

const makeSession = (id, frameCount = 10) => ({
  id,
  recordedAt: '2026-11-18T00:00:00.000Z',
  exerciseId: 'squat',
  frameCount,
  durationMs: frameCount * 33,
  finalReps: 1,
  hasVideo: true,
  videoBlob: { size: 123 },
  frames: Array.from({ length: frameCount }, (_, i) => ({ timeMs: i * 33, angles: { leftKnee: 170 } })),
});

test('settings defaults are never shared or mutated through the manager', () => {
  const manager = new SettingsManager({ storage: null });
  assert.notEqual(manager.get().exercise, DEFAULT_SETTINGS.exercise);
  assert.throws(() => { DEFAULT_SETTINGS.exercise.activeExercise = 'lunge'; }, TypeError);

  manager.reset();
  manager.get().exercise.activeExercise = 'lunge';
  assert.equal(DEFAULT_SETTINGS.exercise.activeExercise, 'squat');
  assert.equal(createDefaultSettings().exercise.activeExercise, 'squat');
});

test('SessionStorage evicts oldest sessions when quota is exceeded and reports evictions', () => {
  const single = JSON.stringify(makeSession('s0', 40)).length;
  const storage = new QuotaStorage(single * 2.6);
  const manager = new SessionStorage({ storage });
  const evicted = [];

  assert.equal(manager.saveSession(makeSession('s1', 40), { onEvict: id => evicted.push(id) }), true);
  assert.equal(manager.saveSession(makeSession('s2', 40), { onEvict: id => evicted.push(id) }), true);
  assert.equal(manager.saveSession(makeSession('s3', 40), { onEvict: id => evicted.push(id) }), true);

  assert.deepEqual(evicted, ['s1']);
  assert.deepEqual(manager.getIndex().map(s => s.id), ['s3', 's2']);
  for (const item of manager.getIndex()) assert.ok(manager.getSession(item.id), `data for ${item.id}`);
});

test('SessionStorage returns false and keeps index consistent when a session can never fit', () => {
  const storage = new QuotaStorage(200);
  const manager = new SessionStorage({ storage });
  assert.equal(manager.saveSession(makeSession('huge', 200)), false);
  assert.deepEqual(manager.getIndex(), []);
  assert.equal(manager.getSession('huge'), null);
});

test('SessionStorage failed save restores existing sessions and does not report evictions', () => {
  const small = JSON.stringify(makeSession('s0', 5)).length;
  const storage = new QuotaStorage(small * 3);
  const manager = new SessionStorage({ storage });
  assert.equal(manager.saveSession(makeSession('keep1', 5)), true);
  assert.equal(manager.saveSession(makeSession('keep2', 5)), true);
  const evicted = [];

  assert.equal(manager.saveSession(makeSession('huge', 400), { onEvict: id => evicted.push(id) }), false);
  assert.deepEqual(evicted, []);
  assert.deepEqual(manager.getIndex().map(s => s.id), ['keep2', 'keep1']);
  assert.ok(manager.getSession('keep1'));
  assert.ok(manager.getSession('keep2'));
  assert.equal(manager.getSession('huge'), null);
});

test('SessionStorage rolls back when the index write fails after the data write', () => {
  const storage = new QuotaStorage();
  const manager = new SessionStorage({ storage });
  assert.equal(manager.saveSession(makeSession('keep', 5)), true);
  const realSet = storage.setItem.bind(storage);
  storage.setItem = (k, v) => {
    if (k === 'kinelab_sessions_index_v1') throw Object.assign(new Error('quota'), { name: 'QuotaExceededError' });
    realSet(k, v);
  };
  const evicted = [];
  assert.equal(manager.saveSession(makeSession('new', 5), { onEvict: id => evicted.push(id) }), false);
  storage.setItem = realSet;
  assert.deepEqual(evicted, []);
  assert.deepEqual(manager.getIndex().map(s => s.id), ['keep']);
  assert.ok(manager.getSession('keep'));
  assert.equal(manager.getSession('new'), null);
});

test('SessionStorage index records hasVideo and persisted data omits the video blob', () => {
  const manager = new SessionStorage({ storage: new QuotaStorage() });
  manager.saveSession(makeSession('v1'));
  assert.equal(manager.getIndex()[0].hasVideo, true);
  assert.equal('videoBlob' in manager.getSession('v1'), false);
});

test('SessionRecorder keeps video chunks when the MediaRecorder already stopped itself', () => {
  class FakeRecorder {
    constructor() { this.state = 'inactive'; }
    start() { this.state = 'recording'; }
    stop() { this.state = 'inactive'; }
  }
  FakeRecorder.isTypeSupported = () => true;
  const recorder = new SessionRecorder({ MediaRecorderClass: FakeRecorder });
  recorder.start({ exerciseId: 'squat', mediaStream: {} });
  recorder.mediaRecorder.ondataavailable({ data: new Blob(['abc']) });
  recorder.mediaRecorder.state = 'inactive'; // camera track ended -> recorder auto-stopped
  recorder.recordFrame({ landmarks: null, angles: {}, timestampMs: performance.now() });

  const session = recorder.stop();
  assert.equal(session.hasVideo, true);
  assert.equal(session.videoBlob.size, 3);
});

test('TimelinePlayer plays telemetry-only sessions after the video element is detached', async () => {
  const pending = [];
  globalThis.requestAnimationFrame = cb => { pending.push(cb); return pending.length; };
  globalThis.cancelAnimationFrame = () => {};

  const player = new TimelinePlayer();
  player.attachVideoElement({ addEventListener() {}, removeEventListener() {}, playbackRate: 1, paused: true });
  player.detachVideoElement();
  player.loadSession({ frames: [{ timeMs: 0 }, { timeMs: 10 }, { timeMs: 20 }] });

  player.play();
  assert.equal(player.isPlaying, true);
  assert.ok(pending.length > 0, 'RAF loop scheduled');
});

test('TimelinePlayer falls back to the telemetry clock when video playback is rejected', async () => {
  const pending = [];
  globalThis.requestAnimationFrame = cb => { pending.push(cb); return pending.length; };
  globalThis.cancelAnimationFrame = () => {};

  const video = {
    currentTime: 0, playbackRate: 1, paused: true,
    addEventListener() {}, removeEventListener() {},
    play: () => Promise.reject(new Error('NotSupportedError')),
    pause() { this.paused = true; },
  };
  const player = new TimelinePlayer();
  player.attachVideoElement(video);
  player.loadSession({ frames: [{ timeMs: 0 }, { timeMs: 10 }, { timeMs: 20 }] });

  player.play();
  await new Promise(r => setTimeout(r, 0));
  assert.equal(player.isPlaying, true);
  assert.ok(pending.length > 0, 'RAF fallback loop scheduled');
});

test('SitToStandExercise does not count a rep when the user starts standing and sits down', () => {
  const sts = new SitToStandExercise();
  let state = sts.createState();
  const frame = (knee, hip, t) => ({
    angles: { leftKnee: knee, rightKnee: knee, leftHip: hip, rightHip: hip },
    timestampMs: t,
    trackingQuality: 'good',
  });

  for (let t = 0; t < 500; t += 100) state = sts.update(state, frame(170, 170, t));
  for (let t = 500; t < 1000; t += 100) state = sts.update(state, frame(90, 90, t));
  assert.equal(state.reps, 0);

  for (let t = 1000; t < 1400; t += 100) state = sts.update(state, frame(170, 170, t));
  for (let t = 1400; t < 1800; t += 100) state = sts.update(state, frame(90, 90, t));
  assert.equal(state.reps, 1);
});

test('MediaPipeRunner discards a stale initialize() that finishes after a newer one', async () => {
  const gates = [];
  const created = [];
  const PoseLandmarker = {
    createFromOptions: (_fs, opts) => new Promise(resolve => gates.push(() => {
      const lm = { model: opts.baseOptions.modelAssetPath, closed: false, close() { this.closed = true; } };
      created.push(lm);
      resolve(lm);
    })),
  };
  const runner = new MediaPipeRunner({
    loadVision: async () => ({ FilesetResolver: { forVisionTasks: async () => ({}) }, PoseLandmarker }),
  });

  const fallback = runner.initialize('full');        // e.g. auto-fallback still loading
  await new Promise(r => setTimeout(r, 0));
  const restart = runner.initialize('heavy');        // user restarted the camera
  await new Promise(r => setTimeout(r, 0));
  gates[1]();                                        // heavy finishes first
  assert.equal((await restart).model, 'heavy');
  gates[0]();                                        // stale full finishes last
  const stale = await fallback;

  assert.equal(stale.stale, true);
  assert.equal(runner.activeModel, 'heavy');
  assert.match(runner.landmarker.model, /heavy/);
  assert.equal(created.find(l => /full/.test(l.model)).closed, true);
});

test('MediaPipeRunner does not install a landmarker after close()', async () => {
  let release;
  const PoseLandmarker = {
    createFromOptions: () => new Promise(resolve => { release = () => resolve({ close() { this.closed = true; } }); }),
  };
  const runner = new MediaPipeRunner({
    loadVision: async () => ({ FilesetResolver: { forVisionTasks: async () => ({}) }, PoseLandmarker }),
  });
  const pending = runner.initialize('full');
  await new Promise(r => setTimeout(r, 0));
  runner.close();
  release();
  assert.equal((await pending).stale, true);
  assert.equal(runner.landmarker, null);
});
