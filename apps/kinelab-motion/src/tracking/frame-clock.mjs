/**
 * Frame timing for the camera pipeline.
 *
 * requestVideoFrameCallback (Chrome 83, Safari 15.4, Firefox 132) fires once per new
 * camera frame and reports when it was captured. Inference runs on that callback, and
 * the capture time is the reference for end-to-end latency: timing only the model call,
 * or stamping at hand-off, under-reports the real delay by 20–50 ms (arXiv 2608.11566).
 */

/** True when the element supports requestVideoFrameCallback. */
export function supportsVideoFrameCallback(video) {
  return Boolean(video && typeof video.requestVideoFrameCallback === 'function');
}

/** How far back a frame clock may lie before it is treated as bogus. */
const MAX_CLOCK_AGE_MS = 1000;

/**
 * Picks the best timestamp for a frame on the performance.now() timeline:
 * captureTime (camera exposure), else presentationTime (handed to the compositor),
 * else the callback time. The result is strictly greater than lastTimeMs, as
 * MediaPipe VIDEO mode requires.
 * @param {number} nowMs callback time (performance.now() timeline)
 * @param {Object|null} metadata VideoFrameCallbackMetadata
 * @param {number} lastTimeMs previous frame time (-Infinity for none)
 * @returns {{ timeMs: number, clock: 'capture'|'presentation'|'callback' }}
 */
export function selectFrameTime(nowMs, metadata, lastTimeMs = -Infinity) {
  const plausible = t => Number.isFinite(t) && t <= nowMs + 1 && t >= nowMs - MAX_CLOCK_AGE_MS;
  let timeMs = nowMs;
  let clock = 'callback';
  if (plausible(metadata?.captureTime)) {
    timeMs = metadata.captureTime;
    clock = 'capture';
  } else if (plausible(metadata?.presentationTime)) {
    timeMs = metadata.presentationTime;
    clock = 'presentation';
  }
  if (Number.isFinite(lastTimeMs) && timeMs <= lastTimeMs) timeMs = lastTimeMs + 0.001;
  return { timeMs, clock };
}

/**
 * Inference-rate cap on the camera's own cadence. The clock is the frame's mediaTime
 * while it advances, so display-tick jitter in callback spacing can't skip frames below
 * the cap (a 30 fps camera on a 75 Hz display has callbacks 27/40 ms apart; capping on
 * callback time at 28 ms dropped every other frame). Live streams may report a constant
 * mediaTime (e.g. 0); once it stops advancing the sampler switches, for the rest of the
 * session, to the frame time, starting a fresh history so the two timelines never mix.
 */
export class FrameSampler {
  constructor(intervalMs) {
    this.intervalMs = intervalMs;
    this.reset();
  }

  reset() {
    this.mode = 'media';
    this.lastMedia = null;
    this.lastSampled = -Infinity;
  }

  /**
   * @param {Object|null} metadata VideoFrameCallbackMetadata
   * @param {number} frameTimeMs frame time on the performance.now() timeline
   * @returns {boolean} true if this frame should go to inference
   */
  due(metadata, frameTimeMs) {
    let clock = frameTimeMs;
    if (this.mode === 'media') {
      const media = metadata?.mediaTime;
      if (Number.isFinite(media) && media >= 0 && (this.lastMedia === null || media > this.lastMedia)) {
        this.lastMedia = media;
        clock = media * 1000;
      } else {
        this.mode = 'frame';
        this.lastSampled = -Infinity;
      }
    }
    if (!frameDue(clock, this.lastSampled, this.intervalMs)) return false;
    this.lastSampled = clock;
    return true;
  }
}

/** mediaTime-based sampling clock in ms, or the fallback when mediaTime is unusable. */
export function sampleClockMs(metadata, fallbackMs) {
  const media = metadata?.mediaTime;
  return Number.isFinite(media) && media >= 0 ? media * 1000 : fallbackMs;
}

/**
 * True when a frame at clockMs is due under an intervalMs cap. A clock that went
 * backwards (stream restart) always samples. Half a millisecond of slack absorbs
 * rounding in media timestamps.
 */
export function frameDue(clockMs, lastSampledMs, intervalMs) {
  if (!Number.isFinite(lastSampledMs) || clockMs < lastSampledMs) return true;
  return clockMs - lastSampledMs >= intervalMs - 0.5;
}

/**
 * Camera frames the page never saw between two callbacks, from presentedFrames.
 * @param {number|null} prevPresented
 * @param {number} presented
 */
export function missedFrames(prevPresented, presented) {
  if (!Number.isFinite(prevPresented) || !Number.isFinite(presented)) return 0;
  return Math.max(0, presented - prevPresented - 1);
}
