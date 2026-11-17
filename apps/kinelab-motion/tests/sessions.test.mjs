import test from 'node:test';
import assert from 'node:assert/strict';
import { SessionRecorder } from '../src/sessions/session-recorder.mjs';
import { SessionStorage } from '../src/sessions/session-storage.mjs';
import { exportToCSV, exportToJSON } from '../src/sessions/export-helpers.mjs';
import { TimelinePlayer } from '../src/sessions/timeline-player.mjs';

class MockStorage {
  constructor() {
    this.store = new Map();
  }
  getItem(key) { return this.store.get(key) ?? null; }
  setItem(key, val) { this.store.set(key, String(val)); }
  removeItem(key) { this.store.delete(key); }
  clear() { this.store.clear(); }
}

test('SessionRecorder records frames and generates valid session object', () => {
  const recorder = new SessionRecorder();
  recorder.start({ exerciseId: 'squat' });

  const dummyLandmarks = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.95 }));

  recorder.recordFrame({
    landmarks: dummyLandmarks,
    angles: { leftKnee: 170, rightKnee: 170 },
    timestampMs: 1000,
    phase: 'ready',
    status: 'ready',
    reps: 0,
  });

  recorder.recordFrame({
    landmarks: dummyLandmarks,
    angles: { leftKnee: 95, rightKnee: 95 },
    timestampMs: 1500,
    phase: 'down',
    status: 'rise',
    reps: 0,
  });

  recorder.recordFrame({
    landmarks: dummyLandmarks,
    angles: { leftKnee: 165, rightKnee: 165 },
    timestampMs: 2000,
    phase: 'ready',
    status: 'completed',
    reps: 1,
  });

  const session = recorder.stop();
  assert.ok(session);
  assert.equal(session.exerciseId, 'squat');
  assert.equal(session.frameCount, 3);
  assert.equal(session.finalReps, 1);
  assert.equal(session.frames[0].landmarks.length, 33);
  assert.equal(session.frames[1].angles.leftKnee, 95);
});

test('SessionStorage saves, lists, retrieves, and deletes sessions', () => {
  const storage = new MockStorage();
  const manager = new SessionStorage({ storage });

  const session = {
    id: 'test_session_1',
    recordedAt: new Date().toISOString(),
    exerciseId: 'squat',
    frameCount: 10,
    durationMs: 5000,
    finalReps: 3,
    frames: [],
  };

  const saved = manager.saveSession(session);
  assert.equal(saved, true);

  const list = manager.getIndex();
  assert.equal(list.length, 1);
  assert.equal(list[0].id, 'test_session_1');

  const retrieved = manager.getSession('test_session_1');
  assert.ok(retrieved);
  assert.equal(retrieved.exerciseId, 'squat');

  const deleted = manager.deleteSession('test_session_1');
  assert.equal(deleted, true);
  assert.equal(manager.getIndex().length, 0);
});

test('exportToCSV and exportToJSON format telemetry accurately', () => {
  const session = {
    id: 'test_session_2',
    recordedAt: '2026-11-16T22:00:00.000Z',
    exerciseId: 'squat',
    frameCount: 2,
    durationMs: 500,
    finalReps: 0,
    frames: [
      {
        timeMs: 0,
        landmarks: null,
        angles: { leftKnee: 170, rightKnee: 170 },
        phase: 'ready',
        status: 'ready',
        reps: 0,
      },
      {
        timeMs: 500,
        landmarks: null,
        angles: { leftKnee: 100, rightKnee: 100 },
        phase: 'down',
        status: 'rise',
        reps: 0,
      },
    ],
  };

  const jsonStr = exportToJSON(session);
  assert.ok(jsonStr.includes('"test_session_2"'));

  const csvStr = exportToCSV(session);
  const lines = csvStr.trim().split('\n');
  assert.equal(lines.length, 3);
  assert.ok(lines[0].startsWith('elapsed_ms,elapsed_sec'));
  assert.ok(lines[1].includes('0,0.000,"ready"'));
  assert.ok(lines[2].includes('500,0.500,"down"'));
});

test('TimelinePlayer loads session, seeks, and steps frames', () => {
  const player = new TimelinePlayer();
  const session = {
    id: 's3',
    frames: [
      { timeMs: 0, angles: { leftKnee: 170 }, reps: 0 },
      { timeMs: 33, angles: { leftKnee: 150 }, reps: 0 },
      { timeMs: 66, angles: { leftKnee: 130 }, reps: 0 },
    ],
  };

  player.loadSession(session);
  let frame = player.getCurrentFrame();
  assert.equal(frame.index, 0);
  assert.equal(frame.angles.leftKnee, 170);

  player.seek(2);
  frame = player.getCurrentFrame();
  assert.equal(frame.index, 2);
  assert.equal(frame.angles.leftKnee, 130);

  player.step(-1);
  frame = player.getCurrentFrame();
  assert.equal(frame.index, 1);
});
