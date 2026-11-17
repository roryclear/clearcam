/**
 * Extensibility contracts and feature flags for interactive capabilities.
 * Ensures future integrations (gestures, avatars, mini-games) plug into
 * the motion pipeline without modifying core pose tracking.
 */

export const FEATURE_FLAGS = Object.freeze({
  AUDIO_CUES: 'audio_cues',
  GESTURE_CONTROLS: 'gesture_controls',
  THREE_VRM_AVATAR: 'three_vrm_avatar',
  MOVEMENT_CHALLENGES: 'movement_challenges',
  MOTION_PAINTING: 'motion_painting',
});

/**
 * Interface contract for gesture recognition extensions.
 * @typedef {Object} GestureRecognizerExtension
 * @property {string} id
 * @property {() => Promise<boolean>} init
 * @property {(frame: ImageData|HTMLVideoElement) => Promise<{ gesture: string, confidence: number }>} recognize
 * @property {() => void} close
 */

/**
 * Interface contract for 3D Avatar (Three-VRM) extensions.
 * @typedef {Object} AvatarExtension
 * @property {string} id
 * @property {(canvas: HTMLCanvasElement) => Promise<void>} attach
 * @property {(landmarks: Array<Object>, worldLandmarks: Array<Object>) => void} updatePose
 * @property {() => void} dispose
 */

/**
 * Interface contract for Movement Challenges / Body-controlled mini-games.
 * @typedef {Object} ChallengeExtension
 * @property {string} id
 * @property {string} title
 * @property {(config: Object) => void} start
 * @property {(landmarks: Array<Object>, angles: Object, dtMs: number) => { score: number, completed: boolean, feedback: string }} step
 * @property {() => void} reset
 */

export class ExtensionManager {
  constructor() {
    this.flags = new Map([
      [FEATURE_FLAGS.AUDIO_CUES, true],
      [FEATURE_FLAGS.GESTURE_CONTROLS, false],
      [FEATURE_FLAGS.THREE_VRM_AVATAR, false],
      [FEATURE_FLAGS.MOVEMENT_CHALLENGES, false],
      [FEATURE_FLAGS.MOTION_PAINTING, false],
    ]);
    this.extensions = new Map();
  }

  isFeatureEnabled(flag) {
    return this.flags.get(flag) ?? false;
  }

  setFeatureEnabled(flag, enabled) {
    this.flags.set(flag, Boolean(enabled));
  }

  registerExtension(type, extension) {
    if (!extension || !extension.id) {
      throw new Error('Extension must provide an id.');
    }
    this.extensions.set(`${type}:${extension.id}`, extension);
  }

  getExtension(type, id) {
    return this.extensions.get(`${type}:${id}`) ?? null;
  }
}

export const defaultExtensionManager = new ExtensionManager();
