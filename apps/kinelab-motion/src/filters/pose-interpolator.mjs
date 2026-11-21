/**
 * Real-Time 60/120 FPS Pose Interpolator
 *
 * Decouples the 25-35 FPS camera pose-inference loop from the native 60/120Hz
 * canvas rendering loop. Performs critically-damped exponential interpolation
 * (LERP with delta-time compensation) on landmark positions and angles so overlays,
 * skeleton lines, arcs, and gauges glide with liquid smoothness without visual stutter.
 *
 * The exponential chase alone trails a moving joint by about 1/lambda (31 ms at 32).
 * With predictionMs > 0 the chase follows the target extrapolated at its recent
 * velocity to the render time, capped at predictionMs past the frame's capture
 * time, so the overlay keeps up with the body. Prediction is reliable only over
 * short horizons (~20–50 ms), and it is for display only: angles and rep logic use
 * the measured landmarks.
 */

/** Chase speed while predicting: the target already leads, so only smooth steps (~11 ms). */
const PREDICTIVE_LAMBDA = 90;
/** Largest predicted offset per landmark, in normalized image units. */
const MAX_PREDICTED_OFFSET = 0.05;
/** Gaps longer than this reset velocity (missed frames, occlusion). */
const MAX_VELOCITY_GAP_MS = 150;

export class PoseInterpolator {
  /**
   * @param {Object} [options]
   * @param {number} [options.lambda=30.0] Convergence speed (higher = faster tracking)
   * @param {number} [options.maxSnapDistance=0.25] Jump distance threshold to snap immediately
   * @param {number} [options.predictionMs=0] Max extrapolation past the target's frame time (0 = off)
   */
  constructor({ lambda = 30.0, maxSnapDistance = 0.25, predictionMs = 0 } = {}) {
    this.lambda = lambda;
    this.maxSnapDistance = maxSnapDistance;
    this.predictionMs = predictionMs;

    this.current = null;
    this.target = null;
    this.targetTime = null;
    this.velocity = null; // per landmark {vx, vy, vz} in units per ms
    this.lastRenderTime = 0;
  }

  setPrediction(predictionMs) {
    this.predictionMs = Number.isFinite(predictionMs) && predictionMs > 0 ? predictionMs : 0;
  }

  /**
   * Supply the latest raw or filtered inference pose target.
   * @param {Array<Object>|null} targetLandmarks
   * @param {number} timestampMs
   */
  setTarget(targetLandmarks, timestampMs) {
    if (!Array.isArray(targetLandmarks) || targetLandmarks.length === 0) {
      this.target = null;
      this.targetTime = null;
      this.velocity = null;
      return;
    }

    this._updateVelocity(targetLandmarks, timestampMs);
    this.target = targetLandmarks;
    this.targetTime = timestampMs;

    if (!this.current) {
      // First frame initialization: clone target
      this.current = targetLandmarks.map(pt => (pt ? { ...pt } : null));
      this.lastRenderTime = timestampMs;
    }
  }

  /**
   * Advance the interpolation by one display animation frame.
   * @param {number} renderTimestampMs
   * @returns {Array<Object>|null}
   */
  step(renderTimestampMs) {
    if (!this.target) {
      return null;
    }

    if (!this.current) {
      this.current = this.target.map(pt => (pt ? { ...pt } : null));
      this.lastRenderTime = renderTimestampMs;
      return this.current;
    }

    const dt = Math.max(0.001, Math.min(0.1, (renderTimestampMs - this.lastRenderTime) / 1000.0));
    this.lastRenderTime = renderTimestampMs;

    const predicting = this.predictionMs > 0 && this.velocity && Number.isFinite(this.targetTime);
    const horizon = predicting ? Math.max(0, Math.min(this.predictionMs, renderTimestampMs - this.targetTime)) : 0;
    // Decay factor: 1 - e^(-lambda * dt)
    const factor = 1.0 - Math.exp(-(predicting ? Math.max(this.lambda, PREDICTIVE_LAMBDA) : this.lambda) * dt);

    for (let i = 0; i < this.target.length; i++) {
      const tgt = horizon > 0 ? this._predict(i, horizon) : this.target[i];
      let cur = this.current[i];

      if (!tgt) {
        this.current[i] = null;
        continue;
      }

      if (!cur) {
        this.current[i] = { ...tgt };
        continue;
      }

      const dx = tgt.x - cur.x;
      const dy = tgt.y - cur.y;
      const dz = (tgt.z ?? 0) - (cur.z ?? 0);
      const dist = Math.hypot(dx, dy, dz);

      // Snap immediately on large teleports (occlusion recovery or camera switch)
      if (dist > this.maxSnapDistance) {
        cur.x = tgt.x;
        cur.y = tgt.y;
        cur.z = tgt.z ?? 0;
      } else {
        cur.x += dx * factor;
        cur.y += dy * factor;
        cur.z = (cur.z ?? 0) + dz * factor;
      }

      cur.visibility = tgt.visibility;
      cur.presence = tgt.presence;
      cur.isFiltered = tgt.isFiltered ?? true;
      cur.isReliable = tgt.isReliable ?? true;
    }

    return this.current;
  }

  /** Target landmark i extrapolated horizonMs ahead at its smoothed velocity. */
  _predict(i, horizonMs) {
    const tgt = this.target[i];
    const v = this.velocity?.[i];
    if (!tgt || !v) return tgt;
    let dx = v.vx * horizonMs;
    let dy = v.vy * horizonMs;
    let dz = v.vz * horizonMs;
    // Cap the offset vector's length (not each axis) so diagonal motion obeys the bound.
    const len = Math.hypot(dx, dy, dz);
    if (len > MAX_PREDICTED_OFFSET) {
      const k = MAX_PREDICTED_OFFSET / len;
      dx *= k; dy *= k; dz *= k;
    }
    return { ...tgt, x: tgt.x + dx, y: tgt.y + dy, z: (tgt.z ?? 0) + dz };
  }

  _updateVelocity(next, timestampMs) {
    const prev = this.target;
    const gap = Number.isFinite(this.targetTime) ? timestampMs - this.targetTime : Infinity;
    if (!prev || !(gap > 0) || gap > MAX_VELOCITY_GAP_MS) {
      this.velocity = null;
      return;
    }
    const old = this.velocity;
    this.velocity = next.map((p, i) => {
      const q = prev[i];
      if (!p || !q) return null;
      const raw = { vx: (p.x - q.x) / gap, vy: (p.y - q.y) / gap, vz: ((p.z ?? 0) - (q.z ?? 0)) / gap };
      // Snap-sized jumps are tracking glitches, not motion.
      if (Math.hypot(p.x - q.x, p.y - q.y) > this.maxSnapDistance) return null;
      const o = old?.[i];
      // Light EMA: keeps a fresh estimate without amplifying frame-to-frame jitter.
      return o ? { vx: 0.5 * (raw.vx + o.vx), vy: 0.5 * (raw.vy + o.vy), vz: 0.5 * (raw.vz + o.vz) } : raw;
    });
  }

  reset() {
    this.current = null;
    this.target = null;
    this.targetTime = null;
    this.velocity = null;
    this.lastRenderTime = 0;
  }
}
