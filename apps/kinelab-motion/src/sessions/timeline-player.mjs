/**
 * Timeline playback controller for recorded motion sessions.
 * Supports synchronized HTML5 Video element playback alongside frame-accurate
 * skeletal landmark and joint telemetry telemetry rendering.
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

    this.videoEl = null;
    this._videoHandlers = null;
  }

  attachVideoElement(videoEl) {
    this.detachVideoElement();
    if (!videoEl) return;

    this.videoEl = videoEl;
    this.videoEl.playbackRate = this.playbackSpeed;

    const onTimeUpdate = () => {
      if (!this.session || !this.session.frames || !this.session.frames.length) return;
      const targetMs = (this.videoEl.currentTime || 0) * 1000;
      const frames = this.session.frames;

      // Find frame matching or immediately preceding targetMs
      let bestIdx = 0;
      for (let i = 0; i < frames.length; i++) {
        if (frames[i].timeMs <= targetMs) {
          bestIdx = i;
        } else {
          break;
        }
      }

      this.currentFrameIdx = bestIdx;
      this.playbackTimeMs = targetMs;
      this.notify();
    };

    const onEnded = () => {
      this.isPlaying = false;
      if (this.session && this.session.frames.length) {
        this.currentFrameIdx = this.session.frames.length - 1;
      }
      this.notify();
    };

    const onPlay = () => {
      this.isPlaying = true;
      this.notify();
    };

    const onPause = () => {
      this.isPlaying = false;
      this.notify();
    };

    this._videoHandlers = { onTimeUpdate, onEnded, onPlay, onPause };

    this.videoEl.addEventListener('timeupdate', onTimeUpdate);
    this.videoEl.addEventListener('ended', onEnded);
    this.videoEl.addEventListener('play', onPlay);
    this.videoEl.addEventListener('pause', onPause);
  }

  detachVideoElement() {
    if (!this.videoEl || !this._videoHandlers) return;
    this.videoEl.removeEventListener('timeupdate', this._videoHandlers.onTimeUpdate);
    this.videoEl.removeEventListener('ended', this._videoHandlers.onEnded);
    this.videoEl.removeEventListener('play', this._videoHandlers.onPlay);
    this.videoEl.removeEventListener('pause', this._videoHandlers.onPause);
    this.videoEl = null;
    this._videoHandlers = null;
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
      if (this.videoEl) {
        this.videoEl.playbackRate = speed;
      }
    }
  }

  play() {
    if (!this.session || !this.session.frames.length || this.isPlaying) return;
    this.isPlaying = true;

    if (this.videoEl) {
      this.videoEl.playbackRate = this.playbackSpeed;
      const playPromise = this.videoEl.play();
      if (playPromise !== undefined) {
        playPromise.catch(err => {
          console.warn('[TimelinePlayer] Video play failed, falling back to RAF:', err);
          this.lastTickTime = performance.now();
          this.loop();
        });
      }
      return;
    }

    this.lastTickTime = performance.now();
    this.loop();
  }

  pause() {
    this.isPlaying = false;
    if (this.videoEl && !this.videoEl.paused) {
      this.videoEl.pause();
    }
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
    if (this.videoEl) {
      this.videoEl.currentTime = 0;
    }
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
    const targetMs = this.session.frames[clamped].timeMs;
    this.playbackTimeMs = targetMs;

    if (this.videoEl) {
      this.videoEl.currentTime = targetMs / 1000;
    }

    this.notify();
  }

  step(direction = 1) {
    this.pause();
    this.seek(this.currentFrameIdx + direction);
  }

  loop() {
    if (!this.isPlaying || this.videoEl) return;
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
