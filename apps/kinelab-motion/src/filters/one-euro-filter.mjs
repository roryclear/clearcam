/**
 * One Euro Filter for low-latency, jitter-reducing temporal smoothing.
 * Reference: Casiez, G., Roussel, N. and Vogel, F. (2012).
 * "1 € Filter: A Simple Speed-based Low-pass Filter for Angular and Positional Jitter".
 * CHI 2012. https://cristal.univ-lille.fr/~casiez/1euro/
 */

function lowPassAlpha(cutoffHz, dtSec) {
  const r = 2 * Math.PI * cutoffHz * dtSec;
  return r / (r + 1);
}

export class LowPassFilter {
  constructor(alpha = 1) {
    this.alpha = alpha;
    this.hatXPrev = null;
  }

  filter(x, alpha = this.alpha) {
    if (this.hatXPrev === null || !Number.isFinite(this.hatXPrev)) {
      this.hatXPrev = x;
      return x;
    }
    const hatX = alpha * x + (1 - alpha) * this.hatXPrev;
    this.hatXPrev = hatX;
    return hatX;
  }

  hasLast() {
    return this.hatXPrev !== null && Number.isFinite(this.hatXPrev);
  }

  last() {
    return this.hatXPrev;
  }

  reset() {
    this.hatXPrev = null;
  }
}

export class OneEuroFilter {
  /**
   * @param {Object} options
   * @param {number} [options.minCutoff=1.0] Minimum cutoff frequency in Hz (higher = less lag, lower = smoother when slow)
   * @param {number} [options.beta=0.007] Speed coefficient (higher = reduces lag during fast movement)
   * @param {number} [options.dCutoff=1.0] Cutoff frequency for derivative in Hz
   */
  constructor({ minCutoff = 1.0, beta = 0.007, dCutoff = 1.0 } = {}) {
    this.minCutoff = Number.isFinite(minCutoff) && minCutoff > 0 ? minCutoff : 1.0;
    this.beta = Number.isFinite(beta) && beta >= 0 ? beta : 0.007;
    this.dCutoff = Number.isFinite(dCutoff) && dCutoff > 0 ? dCutoff : 1.0;

    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
    this.lastTimeMs = null;
  }

  /**
   * Filter a single scalar value at timestamp.
   * @param {number} x Raw scalar value
   * @param {number} timestampMs Timestamp in milliseconds
   * @returns {number} Filtered value
   */
  filter(x, timestampMs) {
    if (!Number.isFinite(x) || !Number.isFinite(timestampMs)) {
      return x;
    }

    if (this.lastTimeMs === null) {
      this.lastTimeMs = timestampMs;
      return this.xFilter.filter(x);
    }

    const dtSec = Math.max(1e-4, (timestampMs - this.lastTimeMs) / 1000.0);
    this.lastTimeMs = timestampMs;

    // Estimate derivative (velocity)
    const prevX = this.xFilter.hasLast() ? this.xFilter.last() : x;
    const dx = (x - prevX) / dtSec;
    const edx = this.dxFilter.filter(dx, lowPassAlpha(this.dCutoff, dtSec));

    // Adaptive cutoff frequency based on velocity
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(x, lowPassAlpha(cutoff, dtSec));
  }

  reset() {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.lastTimeMs = null;
  }
}

export class OneEuroPoint2D {
  constructor(options = {}) {
    this.fx = new OneEuroFilter(options);
    this.fy = new OneEuroFilter(options);
  }

  filter(x, y, timestampMs) {
    return {
      x: this.fx.filter(x, timestampMs),
      y: this.fy.filter(y, timestampMs),
    };
  }

  reset() {
    this.fx.reset();
    this.fy.reset();
  }
}
