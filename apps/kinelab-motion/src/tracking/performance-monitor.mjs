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

    // End-to-end latency from the frame's capture clock (see frame-clock.mjs).
    this.maxPipelineSamples = 120;
    this.pipeline = { landmarks: [], overlay: [] };
    this.pipelineClock = { landmarks: null, overlay: null };
  }

  /**
   * Records an end-to-end latency sample.
   * @param {'landmarks'|'overlay'} stage capture→landmarks or capture→overlay drawn
   * @param {number} ms
   * @param {string} [clock] which frame clock the sample is measured from
   */
  recordPipelineLatency(stage, ms, clock) {
    if (!this.pipeline[stage] || !Number.isFinite(ms) || ms < 0) return;
    // Samples from different frame clocks aren't comparable: a clock change starts a new history.
    if (clock && clock !== this.pipelineClock[stage]) {
      this.pipeline[stage] = [];
      this.pipelineClock[stage] = clock;
    }
    const buf = this.pipeline[stage];
    buf.push(ms);
    if (buf.length > this.maxPipelineSamples) buf.shift();
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

  recordDroppedFrame(count = 1) {
    if (Number.isFinite(count) && count > 0) this.droppedFrames += count;
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
      captureToLandmarksP50: percentile(this.pipeline.landmarks, 50),
      captureToLandmarksP95: percentile(this.pipeline.landmarks, 95),
      captureToOverlayP50: percentile(this.pipeline.overlay, 50),
      captureToOverlayP95: percentile(this.pipeline.overlay, 95),
      latencyClock: this.pipelineClock.landmarks,
      overlayLatencyClock: this.pipelineClock.overlay,
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
    this.pipeline = { landmarks: [], overlay: [] };
    this.pipelineClock = { landmarks: null, overlay: null };
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
    this.pipeline = { landmarks: [], overlay: [] };
    this.pipelineClock = { landmarks: null, overlay: null };
  }
}

/** Nearest-rank percentile, rounded to 0.1 ms; null when there are no samples. */
export function percentile(samples, p) {
  if (!samples?.length) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return Math.round(sorted[idx] * 10) / 10;
}
