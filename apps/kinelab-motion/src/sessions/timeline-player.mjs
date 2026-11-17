/**
 * Timeline playback controller for recorded motion sessions.
 */

export class TimelinePlayer {
  constructor() {
    this.session = null;
    this.currentFrameIdx = 0;
    this.isPlaying = false;
    this.playbackSpeed = 1.0;
    this.listeners = new Set();
    this.rafId = 0;
    this.lastTickTime = 0;
    this.playbackTimeMs = 0;
  }

  loadSession(session) {
    this.stop();
    this.session = session;
    this.currentFrameIdx = 0;
    this.playbackTimeMs = 0;
    this.notify();
  }

  setSpeed(speed) {
    if (Number.isFinite(speed) && speed > 0) {
      this.playbackSpeed = speed;
    }
  }

  play() {
    if (!this.session || !this.session.frames.length || this.isPlaying) return;
    this.isPlaying = true;
    this.lastTickTime = performance.now();
    this.loop();
  }

  pause() {
    this.isPlaying = false;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
    this.notify();
  }

  stop() {
    this.pause();
    this.currentFrameIdx = 0;
    this.playbackTimeMs = 0;
    this.notify();
  }

  toggle() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  seek(index) {
    if (!this.session || !this.session.frames.length) return;
    const clamped = Math.max(0, Math.min(this.session.frames.length - 1, Math.floor(index)));
    this.currentFrameIdx = clamped;
    this.playbackTimeMs = this.session.frames[clamped].timeMs;
    this.notify();
  }

  step(direction = 1) {
    this.pause();
    this.seek(this.currentFrameIdx + direction);
  }

  loop() {
    if (!this.isPlaying) return;
    const now = performance.now();
    const dt = (now - this.lastTickTime) * this.playbackSpeed;
    this.lastTickTime = now;

    this.playbackTimeMs += dt;
    const frames = this.session.frames;

    // Find frame matching playbackTimeMs
    while (
      this.currentFrameIdx < frames.length - 1 &&
      frames[this.currentFrameIdx + 1].timeMs <= this.playbackTimeMs
    ) {
      this.currentFrameIdx++;
    }

    if (this.currentFrameIdx >= frames.length - 1) {
      // Reached end of session
      this.currentFrameIdx = frames.length - 1;
      this.pause();
      return;
    }

    this.notify();
    this.rafId = requestAnimationFrame(() => this.loop());
  }

  getCurrentFrame() {
    if (!this.session || !this.session.frames.length) return null;
    const raw = this.session.frames[this.currentFrameIdx];
    if (!raw) return null;

    // Reconstruct landmark objects from compact arrays
    let landmarks = null;
    if (Array.isArray(raw.landmarks)) {
      landmarks = raw.landmarks.map(p => {
        if (!p) return null;
        return {
          x: p[0],
          y: p[1],
          z: p[2] ?? 0,
          visibility: p[3] ?? 1,
          presence: p[3] ?? 1,
        };
      });
    }

    return {
      index: this.currentFrameIdx,
      totalFrames: this.session.frames.length,
      timeMs: raw.timeMs,
      landmarks,
      angles: raw.angles ?? {},
      phase: raw.phase ?? '',
      status: raw.status ?? '',
      reps: raw.reps ?? 0,
    };
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify() {
    const frame = this.getCurrentFrame();
    for (const listener of this.listeners) {
      try {
        listener(frame, this.isPlaying);
      } catch (err) {
        console.error('Error in timeline listener:', err);
      }
    }
  }
}
