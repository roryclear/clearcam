/**
 * Real-time performance monitoring for pose inference and rendering.
 * Tracks inference FPS, render FPS, latency (ms), and frame drops.
 */

export class PerformanceMonitor {
  /**
   * @param {Object} options
   * @param {number} [options.sampleWindowMs=1000] Interval for FPS calculation
   */
  constructor({ sampleWindowMs = 1000 } = {}) {
    this.sampleWindowMs = sampleWindowMs;

    this.inferenceTimes = [];
    this.maxLatencySamples = 30;

    this.lastInferenceTimeMs = 0;
    this.inferenceFrameCount = 0;
    this.lastInferenceFpsCalc = 0;
    this.currentInferenceFps = 0;

    this.renderFrameCount = 0;
    this.lastRenderFpsCalc = 0;
    this.currentRenderFps = 0;

    this.droppedFrames = 0;
    this.lastLatencyMs = 0;
    this.avgLatencyMs = 0;
  }

  /**
   * Records start of an inference pass.
   * @returns {number} Start timestamp (performance.now())
   */
  startInference() {
    return performance.now();
  }

  /**
   * Records completion of an inference pass.
   * @param {number} startTime
   */
  endInference(startTime) {
    const now = performance.now();
    const latency = now - startTime;
    this.lastLatencyMs = latency;

    this.inferenceTimes.push(latency);
    if (this.inferenceTimes.length > this.maxLatencySamples) {
      this.inferenceTimes.shift();
    }

    const sum = this.inferenceTimes.reduce((a, b) => a + b, 0);
    this.avgLatencyMs = Math.round((sum / this.inferenceTimes.length) * 10) / 10;

    this.inferenceFrameCount++;
    if (!this.lastInferenceFpsCalc) this.lastInferenceFpsCalc = now;

    if (now - this.lastInferenceFpsCalc >= this.sampleWindowMs) {
      const elapsed = (now - this.lastInferenceFpsCalc) / 1000;
      this.currentInferenceFps = Math.round(this.inferenceFrameCount / elapsed);
      this.inferenceFrameCount = 0;
      this.lastInferenceFpsCalc = now;
    }
  }

  recordRenderFrame() {
    const now = performance.now();
    this.renderFrameCount++;
    if (!this.lastRenderFpsCalc) this.lastRenderFpsCalc = now;

    if (now - this.lastRenderFpsCalc >= this.sampleWindowMs) {
      const elapsed = (now - this.lastRenderFpsCalc) / 1000;
      this.currentRenderFps = Math.round(this.renderFrameCount / elapsed);
      this.renderFrameCount = 0;
      this.lastRenderFpsCalc = now;
    }
  }

  recordDroppedFrame() {
    this.droppedFrames++;
  }

  getMetrics() {
    return {
      inferenceFps: this.currentInferenceFps,
      renderFps: this.currentRenderFps,
      lastLatencyMs: Math.round(this.lastLatencyMs * 10) / 10,
      avgLatencyMs: this.avgLatencyMs,
      latencySamples: this.inferenceTimes.length,
      latencyWindow: this.maxLatencySamples,
      droppedFrames: this.droppedFrames,
    };
  }

  /**
   * Clears inference latency and FPS history (e.g. after a model swap) so the next
   * measurements describe the new model only. Render and dropped-frame stats stay.
   */
  resetInferenceStats() {
    this.inferenceTimes = [];
    this.inferenceFrameCount = 0;
    this.lastInferenceFpsCalc = 0;
    this.currentInferenceFps = 0;
    this.lastLatencyMs = 0;
    this.avgLatencyMs = 0;
  }

  reset() {
    this.inferenceTimes = [];
    this.inferenceFrameCount = 0;
    this.lastInferenceFpsCalc = 0;
    this.currentInferenceFps = 0;
    this.renderFrameCount = 0;
    this.lastRenderFpsCalc = 0;
    this.currentRenderFps = 0;
    this.droppedFrames = 0;
    this.lastLatencyMs = 0;
    this.avgLatencyMs = 0;
  }
}
