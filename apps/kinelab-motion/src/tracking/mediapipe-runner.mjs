/**
 * MediaPipe Pose Landmarker lifecycle manager and execution runner.
 * Features inference resolution decoupling for high FPS and low latency.
 */

export const SDK_VERSION = '1.2.3';
export const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${SDK_VERSION}`;

export const MODEL_URLS = Object.freeze({
  heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
  full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
});

export const INFERENCE_PRESETS = Object.freeze({
  speed: { width: 360, height: 270, label: 'Speed (360p)' },
  balanced: { width: 480, height: 360, label: 'Balanced (480p)' },
  high: { width: 720, height: 540, label: 'High (720p)' },
  direct: { width: null, height: null, label: 'Direct (Native)' },
});

export class MediaPipeRunner {
  constructor({
    confidenceThreshold = 0.55,
    preferredModel = 'heavy',
    inferencePreset = 'balanced',
  } = {}) {
    this.confidenceThreshold = confidenceThreshold;
    this.preferredModel = preferredModel;

    this.landmarker = null;
    this.activeModel = null;
    this.activeDelegate = null;
    this.isInitializing = false;

    this.inferencePreset = inferencePreset;
    this.targetWidth = 480;
    this.targetHeight = 360;
    this.setInferencePreset(inferencePreset);

    this.offscreenCanvas = null;
    this.offscreenCtx = null;
  }

  setInferencePreset(preset) {
    this.inferencePreset = preset in INFERENCE_PRESETS ? preset : 'balanced';
    const cfg = INFERENCE_PRESETS[this.inferencePreset];
    if (cfg && cfg.width) {
      this.targetWidth = cfg.width;
      this.targetHeight = cfg.height;
    }
  }

  /**
   * Initializes or re-initializes the PoseLandmarker.
   * @param {string} [modelChoice] 'heavy' | 'full'
   * @param {Function} [onStatusUpdate] Callback for loading progress strings
   */
  async initialize(modelChoice = this.preferredModel, onStatusUpdate = () => {}) {
    this.isInitializing = true;
    onStatusUpdate('Loading MediaPipe vision tasks library…');

    const { FilesetResolver, PoseLandmarker } = await import(`${CDN_BASE}/+esm`);
    const fileset = await FilesetResolver.forVisionTasks(`${CDN_BASE}/wasm`);

    const modelsToTry = modelChoice === 'heavy' ? ['heavy', 'full'] : ['full'];
    let lastError = null;

    for (const model of modelsToTry) {
      for (const delegate of ['GPU', 'CPU']) {
        try {
          onStatusUpdate(`Initializing ${model.toUpperCase()} model (${delegate})…`);
          const landmarker = await PoseLandmarker.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: MODEL_URLS[model],
              delegate,
            },
            runningMode: 'VIDEO',
            numPoses: 1,
            minPoseDetectionConfidence: this.confidenceThreshold,
            minPosePresenceConfidence: this.confidenceThreshold,
            minTrackingConfidence: this.confidenceThreshold,
            outputSegmentationMasks: false,
          });

          if (this.landmarker) {
            this.landmarker.close();
          }

          this.landmarker = landmarker;
          this.activeModel = model;
          this.activeDelegate = delegate;
          this.isInitializing = false;

          onStatusUpdate(`Tracker ready: ${model.toUpperCase()} (${delegate})`);
          return {
            success: true,
            model,
            delegate,
          };
        } catch (err) {
          console.warn(`MediaPipe initialization failed for ${model}/${delegate}:`, err);
          lastError = err;
        }
      }
    }

    this.isInitializing = false;
    throw lastError ?? new Error('Failed to initialize any MediaPipe PoseLandmarker model.');
  }

  _getScaledSource(video) {
    if (this.inferencePreset === 'direct' || typeof document === 'undefined') {
      return video;
    }

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const aspect = vw / vh;

    const tw = this.targetWidth;
    const th = Math.round(tw / aspect);

    if (!this.offscreenCanvas) {
      this.offscreenCanvas = document.createElement('canvas');
      this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: false });
    }

    if (this.offscreenCanvas.width !== tw || this.offscreenCanvas.height !== th) {
      this.offscreenCanvas.width = tw;
      this.offscreenCanvas.height = th;
    }

    try {
      this.offscreenCtx.drawImage(video, 0, 0, tw, th);
      return this.offscreenCanvas;
    } catch {
      return video;
    }
  }

  /**
   * Performs synchronous pose detection for a video frame.
   * Feeds the scaled inference canvas to minimize GPU texture transfer overhead.
   * @param {HTMLVideoElement} video
   * @param {number} timestampMs
   * @returns {{ landmarks: Array<Object>|null, worldLandmarks: Array<Object>|null }}
   */
  detectForVideo(video, timestampMs) {
    if (!this.landmarker) return { landmarks: null, worldLandmarks: null };
    const source = this._getScaledSource(video);
    const result = this.landmarker.detectForVideo(source, timestampMs);
    return {
      landmarks: result.landmarks?.[0] ?? null,
      worldLandmarks: result.worldLandmarks?.[0] ?? null,
    };
  }

  close() {
    if (this.landmarker) {
      this.landmarker.close();
      this.landmarker = null;
    }
    this.activeModel = null;
    this.activeDelegate = null;
    this.offscreenCanvas = null;
    this.offscreenCtx = null;
  }
}
