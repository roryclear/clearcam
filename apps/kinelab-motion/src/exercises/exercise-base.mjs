/**
 * Base class and interfaces for modular, testable exercise recognition modules.
 */

export const SENSITIVITY_PRESETS = Object.freeze({
  forgiving: {
    label: 'Forgiving',
    angleToleranceDeg: 12,
    minPhaseDurationMs: 250,
    requiredHoldFrames: 2,
    confidenceThreshold: 0.45,
  },
  normal: {
    label: 'Standard',
    angleToleranceDeg: 0,
    minPhaseDurationMs: 350,
    requiredHoldFrames: 2,
    confidenceThreshold: 0.55,
  },
  strict: {
    label: 'Strict Form',
    angleToleranceDeg: -8,
    minPhaseDurationMs: 450,
    requiredHoldFrames: 3,
    confidenceThreshold: 0.65,
  },
});

export class BaseExercise {
  constructor({
    id,
    name,
    type = 'repetition', // 'repetition' | 'hold'
    recommendedCamera = 'any', // 'side' | 'front' | 'any'
    description = '',
    requiredLandmarks = [],
    sensitivity = 'normal',
  }) {
    this.id = id;
    this.name = name;
    this.type = type;
    this.recommendedCamera = recommendedCamera;
    this.description = description;
    this.requiredLandmarks = requiredLandmarks;
    this.sensitivity = sensitivity;
  }

  getPresetConfig() {
    return SENSITIVITY_PRESETS[this.sensitivity] ?? SENSITIVITY_PRESETS.normal;
  }

  setSensitivity(sensitivity) {
    if (sensitivity in SENSITIVITY_PRESETS) {
      this.sensitivity = sensitivity;
    }
  }

  /**
   * Validates if required landmarks for this exercise are present and reliable.
   * @param {Array<Object>|null} landmarks
   * @param {number} threshold
   * @returns {boolean}
   */
  hasRequiredLandmarks(landmarks, threshold = 0.55) {
    if (!Array.isArray(landmarks)) return false;
    for (const idx of this.requiredLandmarks) {
      const pt = landmarks[idx];
      if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return false;
      if (pt.visibility !== undefined && pt.visibility < threshold) return false;
      if (pt.presence !== undefined && pt.presence < threshold) return false;
    }
    return true;
  }

  /**
   * Creates fresh initial state for this exercise.
   * Subclasses must implement.
   */
  createState() {
    throw new Error('createState() must be implemented by subclass.');
  }

  /**
   * Advances exercise state with a new frame.
   * Subclasses must implement.
   * @param {Object} state
   * @param {Object} frameData { landmarks, angles, width, height, timestampMs, trackingQuality }
   * @returns {Object} nextState
   */
  update(state, frameData) {
    throw new Error('update() must be implemented by subclass.');
  }

  /**
   * Resets counts or timers back to initial state.
   */
  reset(state) {
    return this.createState();
  }

  /**
   * Returns human-readable telemetry items for UI display.
   * @param {Object} state
   * @returns {Array<{ label: string, value: string|number, highlight?: boolean }>}
   */
  getTelemetry(state) {
    return [];
  }
}
