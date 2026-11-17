/**
 * MediaPipe Pose Landmarker lifecycle manager and execution runner.
 */

export const SDK_VERSION = '1.2.3';
export const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${SDK_VERSION}`;

export const MODEL_URLS = Object.freeze({
  heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
  full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
});

export class MediaPipeRunner {
  constructor({
    confidenceThreshold = 0.55,
    preferredModel = 'heavy',
  } = {}) {
    this.confidenceThreshold = confidenceThreshold;
    this.preferredModel = preferredModel;

    this.landmarker = null;
    this.activeModel = null;
    this.activeDelegate = null;
    this.isInitializing = false;
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

  /**
   * Performs synchronous pose detection for a video frame.
   * @param {HTMLVideoElement} video
   * @param {number} timestampMs
   * @returns {{ landmarks: Array<Object>|null, worldLandmarks: Array<Object>|null }}
   */
  detectForVideo(video, timestampMs) {
    if (!this.landmarker) return { landmarks: null, worldLandmarks: null };
    const result = this.landmarker.detectForVideo(video, timestampMs);
    return {
      landmarks: result.landmarks?.[0] ?? null,
      worldLandmarks: result.worldLandmarks?.[0] ?? null,
    };
  }

  close() {
    if (this.landmarker) {
      try {
        this.landmarker.close();
      } catch (err) {
        console.warn('Error closing landmarker:', err);
      }
      this.landmarker = null;
    }
    this.activeModel = null;
    this.activeDelegate = null;
    this.isInitializing = false;
  }
}
