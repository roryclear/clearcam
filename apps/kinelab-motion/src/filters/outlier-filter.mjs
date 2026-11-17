/**
 * Outlier rejection and jump gating for single-camera 2D pose landmarks.
 * Prevents single-frame detection glitches and unphysical instantaneous teleportation.
 */

export class OutlierFilter {
  /**
   * @param {Object} options
   * @param {number} [options.maxSpeed=4.0] Max allowed landmark speed in normalized screen units per second
   * @param {number} [options.maxInterpolateFrames=3] Max consecutive frames to interpolate through dropped/outlier points
   */
  constructor({ maxSpeed = 4.0, maxInterpolateFrames = 3 } = {}) {
    this.maxSpeed = maxSpeed;
    this.maxInterpolateFrames = maxInterpolateFrames;
    this.previousPoints = new Map(); // index -> { x, y, timestampMs, rejectedCount }
  }

  /**
   * Evaluates if a landmark candidate represents a plausible transition from the previous point.
   * @param {number} index Landmark index (0-32)
   * @param {Object} point Candidate landmark { x, y, visibility, presence }
   * @param {number} timestampMs Current timestamp in ms
   * @returns {{ accepted: boolean, point: Object, wasRejected: boolean }}
   */
  evaluate(index, point, timestampMs) {
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      this.previousPoints.delete(index);
      return { accepted: false, point: null, wasRejected: false };
    }

    const prev = this.previousPoints.get(index);
    if (!prev) {
      // First observation is accepted
      this.previousPoints.set(index, {
        x: point.x,
        y: point.y,
        timestampMs,
        rejectedCount: 0,
      });
      return { accepted: true, point, wasRejected: false };
    }

    const dt = Math.max(1e-4, (timestampMs - prev.timestampMs) / 1000.0);
    const dist = Math.hypot(point.x - prev.x, point.y - prev.y);
    const speed = dist / dt;

    if (speed > this.maxSpeed && prev.rejectedCount < this.maxInterpolateFrames) {
      // Outlier detected: speed is physically impossible for human motion
      prev.rejectedCount++;
      return {
        accepted: true,
        point: {
          ...point,
          x: prev.x, // Hold position during transient glitch
          y: prev.y,
          isInterpolated: true,
        },
        wasRejected: true,
      };
    }

    // Accept new observation and update history
    this.previousPoints.set(index, {
      x: point.x,
      y: point.y,
      timestampMs,
      rejectedCount: 0,
    });
    return { accepted: true, point, wasRejected: false };
  }

  reset(index = null) {
    if (index !== null) {
      this.previousPoints.delete(index);
    } else {
      this.previousPoints.clear();
    }
  }
}
