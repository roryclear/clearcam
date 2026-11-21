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
 * Camera frames the page never saw between two callbacks, from presentedFrames.
 * @param {number|null} prevPresented
 * @param {number} presented
 */
export function missedFrames(prevPresented, presented) {
  if (!Number.isFinite(prevPresented) || !Number.isFinite(presented)) return 0;
  return Math.max(0, presented - prevPresented - 1);
}
