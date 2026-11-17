/**
 * Adaptive controller for automatic model selection and frame throttling
 * based on device capabilities and measured latency.
 */

export class AdaptiveController {
  /**
   * @param {Object} options
   * @param {boolean} [options.autoFallbackEnabled=true] Whether to auto-switch models on sustained high latency
   * @param {number} [options.highLatencyThresholdMs=110] Latency threshold to trigger fallback
   * @param {number} [options.lowFpsThreshold=8] Minimum acceptable inference FPS
   * @param {number} [options.consecutiveViolationsNeeded=4] Number of sample windows before acting
   */
  constructor({
    autoFallbackEnabled = true,
    highLatencyThresholdMs = 110,
    lowFpsThreshold = 8,
    consecutiveViolationsNeeded = 4,
  } = {}) {
    this.autoFallbackEnabled = autoFallbackEnabled;
    this.highLatencyThresholdMs = highLatencyThresholdMs;
    this.lowFpsThreshold = lowFpsThreshold;
    this.consecutiveViolationsNeeded = consecutiveViolationsNeeded;

    this.violationCount = 0;
    this.hasFallenBack = false;
    this.currentModel = 'heavy';
  }

  setCurrentModel(model) {
    this.currentModel = model;
    this.violationCount = 0;
    if (model === 'heavy') {
      this.hasFallenBack = false;
    }
  }

  /**
   * Evaluates current performance metrics.
   * @param {Object} metrics { inferenceFps, avgLatencyMs }
   * @returns {{ shouldFallback: boolean, reason: string|null }}
   */
  evaluate(metrics) {
    if (!this.autoFallbackEnabled || this.currentModel !== 'heavy' || this.hasFallenBack) {
      return { shouldFallback: false, reason: null };
    }

    const isLatencyHigh = metrics.avgLatencyMs > this.highLatencyThresholdMs;
    const isFpsLow = metrics.inferenceFps > 0 && metrics.inferenceFps < this.lowFpsThreshold;

    if (isLatencyHigh || isFpsLow) {
      this.violationCount++;
      if (this.violationCount >= this.consecutiveViolationsNeeded) {
        this.hasFallenBack = true;
        const reason = isLatencyHigh
          ? `Inference latency (${metrics.avgLatencyMs}ms) exceeded ${this.highLatencyThresholdMs}ms.`
          : `Inference frame rate (${metrics.inferenceFps} FPS) dropped below ${this.lowFpsThreshold} FPS.`;
        return { shouldFallback: true, reason };
      }
    } else if (metrics.avgLatencyMs < this.highLatencyThresholdMs * 0.7) {
      this.violationCount = Math.max(0, this.violationCount - 1);
    }

    return { shouldFallback: false, reason: null };
  }

  reset() {
    this.violationCount = 0;
    this.hasFallenBack = false;
  }
}
