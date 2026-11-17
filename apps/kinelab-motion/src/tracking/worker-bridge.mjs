/**
 * Web Worker compatibility probe and architecture bridge for MediaPipe Pose Landmarker.
 *
 * Architecture Evaluation:
 * - MediaPipe Tasks-Vision runs WASM with WebGL/WebGPU acceleration.
 * - In browsers, `HTMLVideoElement` cannot be passed directly to a Web Worker.
 * - Processing in a Worker requires either `createImageBitmap(video)` + `postMessage` per frame,
 *   or `MediaStreamTrackProcessor` (WebCodecs).
 * - On iOS Safari and low-end Android browsers, WebGL inside OffscreenCanvas in Web Workers
 *   frequently fails or is disabled due to memory/context limits.
 * - For predictable mobile compatibility, the reliable default is a bounded-frequency
 *   main-thread scheduler with `requestVideoFrameCallback` or throttled rAF.
 */

export class WorkerBridge {
  constructor() {
    this.worker = null;
    this.isSupported = false;
    this.reason = '';
  }

  /**
   * Probes whether the current runtime environment can support pose inference in a Web Worker.
   * @returns {{ supported: boolean, reason: string }}
   */
  static probeFeasibility() {
    if (typeof window === 'undefined') {
      return { supported: false, reason: 'Non-browser execution environment (Node.js/SSR).' };
    }

    if (typeof Worker === 'undefined') {
      return { supported: false, reason: 'Web Workers are not supported in this browser.' };
    }

    if (typeof OffscreenCanvas === 'undefined') {
      return { supported: false, reason: 'OffscreenCanvas is not supported in this browser.' };
    }

    if (typeof createImageBitmap === 'undefined') {
      return { supported: false, reason: 'createImageBitmap is not available for frame extraction.' };
    }

    // Check for iOS / iPadOS WebKit where OffscreenCanvas WebGL in Workers is known to fail
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) ||
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

    if (isIOS) {
      return {
        supported: false,
        reason: 'iOS WebKit has known limitations with WebGL/WASM in Workers; bounded main-thread execution is recommended.',
      };
    }

    return {
      supported: true,
      reason: 'Browser supports Worker, OffscreenCanvas, and ImageBitmap transfer.',
    };
  }
}
