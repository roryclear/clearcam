/**
 * Real-Time 60/120 FPS Pose Interpolator
 *
 * Decouples the 25-35 FPS camera pose-inference loop from the native 60/120Hz
 * canvas rendering loop. Performs critically-damped exponential interpolation
 * (LERP with delta-time compensation) on landmark positions and angles so overlays,
 * skeleton lines, arcs, and gauges glide with liquid smoothness without visual stutter.
 */

export class PoseInterpolator {
  /**
   * @param {Object} [options]
   * @param {number} [options.lambda=30.0] Convergence speed (higher = faster tracking)
   * @param {number} [options.maxSnapDistance=0.25] Jump distance threshold to snap immediately
   */
  constructor({ lambda = 30.0, maxSnapDistance = 0.25 } = {}) {
    this.lambda = lambda;
    this.maxSnapDistance = maxSnapDistance;

    this.current = null;
    this.target = null;
    this.lastRenderTime = 0;
  }

  /**
   * Supply the latest raw or filtered inference pose target.
   * @param {Array<Object>|null} targetLandmarks
   * @param {number} timestampMs
   */
  setTarget(targetLandmarks, timestampMs) {
    if (!Array.isArray(targetLandmarks) || targetLandmarks.length === 0) {
      this.target = null;
      return;
    }

    this.target = targetLandmarks;

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

    // Decay factor: 1 - e^(-lambda * dt)
    const factor = 1.0 - Math.exp(-this.lambda * dt);

    for (let i = 0; i < this.target.length; i++) {
      const tgt = this.target[i];
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

  reset() {
    this.current = null;
    this.target = null;
    this.lastRenderTime = 0;
  }
}
