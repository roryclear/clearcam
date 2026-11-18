import test from 'node:test';
import assert from 'node:assert/strict';
import { TimelinePlayer } from '../src/sessions/timeline-player.mjs';

class MockVideoElement {
  constructor() {
    this.currentTime = 0;
    this.playbackRate = 1.0;
    this.paused = true;
    this.ended = false;
    this.listeners = new Map();
  }

  addEventListener(type, cb) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(cb);
  }

  removeEventListener(type, cb) {
    if (!this.listeners.has(type)) return;
    this.listeners.set(type, this.listeners.get(type).filter(l => l !== cb));
  }

  _emit(type) {
    const list = this.listeners.get(type) || [];
    for (const cb of list) cb();
  }

  play() {
    this.paused = false;
    this._emit('play');
    return Promise.resolve();
  }

  pause() {
    this.paused = true;
    this._emit('pause');
  }
}

const mockSession = {
  id: 'session_sync_1',
  durationMs: 1000,
  frames: [
    { timeMs: 0, reps: 0, angles: { leftKnee: 180 }, landmarks: null },
    { timeMs: 300, reps: 0, angles: { leftKnee: 140 }, landmarks: null },
    { timeMs: 600, reps: 1, angles: { leftKnee: 90 }, landmarks: null },
    { timeMs: 1000, reps: 1, angles: { leftKnee: 180 }, landmarks: null },
  ],
};

test('TimelinePlayer attaches video element and synchronizes seek', () => {
  const player = new TimelinePlayer();
  const video = new MockVideoElement();

  player.loadSession(mockSession);
  player.attachVideoElement(video);

  assert.equal(player.videoEl, video);

  // Seek to frame index 2 (timeMs: 600 -> 0.6s)
  player.seek(2);
  assert.equal(player.currentFrameIdx, 2);
  assert.equal(video.currentTime, 0.6);

  // Seek to frame index 0
  player.seek(0);
  assert.equal(player.currentFrameIdx, 0);
  assert.equal(video.currentTime, 0.0);
});

test('TimelinePlayer syncs play, pause, and speed with attached video', () => {
  const player = new TimelinePlayer();
  const video = new MockVideoElement();

  player.loadSession(mockSession);
  player.attachVideoElement(video);

  player.setSpeed(2.0);
  assert.equal(video.playbackRate, 2.0);

  player.play();
  assert.equal(video.paused, false);
  assert.equal(player.isPlaying, true);

  player.pause();
  assert.equal(video.paused, true);
  assert.equal(player.isPlaying, false);
});

test('TimelinePlayer updates frame on video timeupdate events', () => {
  const player = new TimelinePlayer();
  const video = new MockVideoElement();

  player.loadSession(mockSession);
  player.attachVideoElement(video);

  let notifiedFrame = null;
  player.subscribe((frame) => {
    notifiedFrame = frame;
  });

  // Advance video time to 0.35s (closest to frame 1 at 300ms)
  video.currentTime = 0.35;
  video._emit('timeupdate');

  assert.ok(notifiedFrame);
  assert.equal(notifiedFrame.index, 1);
  assert.equal(notifiedFrame.timeMs, 300);

  // Advance to 0.95s (frame 3 at 1000ms is closest or current)
  video.currentTime = 0.95;
  video._emit('timeupdate');
  assert.equal(notifiedFrame.index, 2); // 600ms <= 950ms < 1000ms
});
