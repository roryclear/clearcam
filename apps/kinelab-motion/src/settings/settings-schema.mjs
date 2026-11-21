/**
 * Settings schema, validation rules, and default preferences.
 */

export const SETTINGS_VERSION = 4;
export const STORAGE_KEY = 'kinelab_motion_settings_v2';

function deepFreeze(obj) {
  for (const value of Object.values(obj)) {
    if (value && typeof value === 'object') deepFreeze(value);
  }
  return Object.freeze(obj);
}

export const DEFAULT_SETTINGS = deepFreeze({
  version: SETTINGS_VERSION,
  camera: {
    facingMode: 'environment', // 'environment' | 'user'
    mirroring: 'auto', // 'auto' | 'on' | 'off'
    resolution: '720p', // '480p' | '720p' | '1080p'
  },
  tracking: {
    model: 'auto', // 'auto' (Full on phones, Heavy on desktop) | 'heavy' | 'full' | 'lite'
    autoFallback: true,
    smoothing: 'medium', // 'off' | 'low' | 'medium' | 'high'
    confidenceThreshold: 0.55,
    angleDimension: 'auto', // 'auto' (view-aware 2D/3D blend) | '3d' | '2d'
    levelGuidance: true, // phone-tilt / distance / hip-height setup prompts (gravity sensor)
    tiltCorrection: false, // beta: undo phone roll in trunk/shank angles (not yet device-verified)
  },
  overlays: {
    showSkeleton: true,
    showJointMarkers: true,
    showDegreeLabels: true,
    showAngleArcs: true,
    showMotionTrails: false,
    showDirection: false,
    showSymmetryHUD: true,
    showGhost: false,
    predictiveOverlay: true, // extrapolate the drawn skeleton to hide inference lag (display only)
    angleConvention: 'clinical', // 'clinical' (knee/hip/elbow flexion, 0° = straight) | 'interior' (180° = straight)
  },
  exercise: {
    activeExercise: 'squat', // 'squat' | 'sit_to_stand' | 'lunge' | 'shoulder_abduction' | 'shoulder_flexion' | 'single_leg_balance' | 'free'
    sensitivity: 'normal', // 'forgiving' | 'normal' | 'strict'
    audioCues: true,
    voiceCoach: false,
    baseline: null, // null | { standingKneeDeg, standingHipDeg } captured from a valid standing frame
  },
  privacy: {
    localProcessingOnly: true,
    recordingMode: 'manual', // 'off' | 'manual'
  },
});

/**
 * Returns a fresh, mutable deep copy of the defaults. A shallow spread of
 * DEFAULT_SETTINGS shares its nested objects, so callers could mutate the defaults.
 */
export function createDefaultSettings() {
  return validateSettings(DEFAULT_SETTINGS);
}

export function validateSettings(raw) {
  if (!raw || typeof raw !== 'object') {
    return createDefaultSettings();
  }

  const camera = {
    facingMode: ['environment', 'user'].includes(raw.camera?.facingMode)
      ? raw.camera.facingMode
      : DEFAULT_SETTINGS.camera.facingMode,
    mirroring: ['auto', 'on', 'off'].includes(raw.camera?.mirroring)
      ? raw.camera.mirroring
      : DEFAULT_SETTINGS.camera.mirroring,
    resolution: ['480p', '720p', '1080p'].includes(raw.camera?.resolution)
      ? raw.camera.resolution
      : DEFAULT_SETTINGS.camera.resolution,
  };

  const tracking = {
    // Before v4 every user had 'heavy' stored as the default; move them to 'auto' so
    // phones start on Full. Explicit Full/Lite choices are kept.
    model: (raw.version ?? 0) < 4 && raw.tracking?.model === 'heavy'
      ? 'auto'
      : ['auto', 'heavy', 'full', 'lite'].includes(raw.tracking?.model)
        ? raw.tracking.model
        : DEFAULT_SETTINGS.tracking.model,
    autoFallback: Boolean(raw.tracking?.autoFallback ?? DEFAULT_SETTINGS.tracking.autoFallback),
    smoothing: ['off', 'low', 'medium', 'high'].includes(raw.tracking?.smoothing)
      ? raw.tracking.smoothing
      : DEFAULT_SETTINGS.tracking.smoothing,
    confidenceThreshold:
      typeof raw.tracking?.confidenceThreshold === 'number' &&
      raw.tracking.confidenceThreshold >= 0.3 &&
      raw.tracking.confidenceThreshold <= 0.9
        ? raw.tracking.confidenceThreshold
        : DEFAULT_SETTINGS.tracking.confidenceThreshold,
    // v2 stored '3d' as the default for everyone; move those users to the new 'auto' default.
    angleDimension: (raw.version ?? 0) < 3 && raw.tracking?.angleDimension === '3d'
      ? 'auto'
      : ['auto', '3d', '2d'].includes(raw.tracking?.angleDimension)
        ? raw.tracking.angleDimension
        : DEFAULT_SETTINGS.tracking.angleDimension,
    levelGuidance: Boolean(raw.tracking?.levelGuidance ?? DEFAULT_SETTINGS.tracking.levelGuidance),
    tiltCorrection: Boolean(raw.tracking?.tiltCorrection ?? DEFAULT_SETTINGS.tracking.tiltCorrection),
  };

  const overlays = {
    showSkeleton: Boolean(raw.overlays?.showSkeleton ?? DEFAULT_SETTINGS.overlays.showSkeleton),
    showJointMarkers: Boolean(raw.overlays?.showJointMarkers ?? DEFAULT_SETTINGS.overlays.showJointMarkers),
    showDegreeLabels: Boolean(raw.overlays?.showDegreeLabels ?? DEFAULT_SETTINGS.overlays.showDegreeLabels),
    showAngleArcs: Boolean(raw.overlays?.showAngleArcs ?? DEFAULT_SETTINGS.overlays.showAngleArcs),
    showMotionTrails: Boolean(raw.overlays?.showMotionTrails ?? DEFAULT_SETTINGS.overlays.showMotionTrails),
    showDirection: Boolean(raw.overlays?.showDirection ?? DEFAULT_SETTINGS.overlays.showDirection),
    showSymmetryHUD: Boolean(raw.overlays?.showSymmetryHUD ?? DEFAULT_SETTINGS.overlays.showSymmetryHUD),
    showGhost: Boolean(raw.overlays?.showGhost ?? DEFAULT_SETTINGS.overlays.showGhost),
    predictiveOverlay: Boolean(raw.overlays?.predictiveOverlay ?? DEFAULT_SETTINGS.overlays.predictiveOverlay),
    angleConvention: ['clinical', 'interior'].includes(raw.overlays?.angleConvention)
      ? raw.overlays.angleConvention
      : DEFAULT_SETTINGS.overlays.angleConvention,
  };

  const validExercises = ['squat', 'sit_to_stand', 'lunge', 'shoulder_abduction', 'shoulder_flexion', 'single_leg_balance', 'free'];
  const rawBaseline = raw.exercise?.baseline;
  const baselineValid = rawBaseline && Number.isFinite(rawBaseline.standingKneeDeg) && Number.isFinite(rawBaseline.standingHipDeg) &&
    rawBaseline.standingKneeDeg >= 90 && rawBaseline.standingKneeDeg <= 180 &&
    rawBaseline.standingHipDeg >= 90 && rawBaseline.standingHipDeg <= 180;
  const exercise = {
    activeExercise: validExercises.includes(raw.exercise?.activeExercise)
      ? raw.exercise.activeExercise
      : DEFAULT_SETTINGS.exercise.activeExercise,
    sensitivity: ['forgiving', 'normal', 'strict'].includes(raw.exercise?.sensitivity)
      ? raw.exercise.sensitivity
      : DEFAULT_SETTINGS.exercise.sensitivity,
    audioCues: Boolean(raw.exercise?.audioCues ?? DEFAULT_SETTINGS.exercise.audioCues),
    voiceCoach: Boolean(raw.exercise?.voiceCoach ?? DEFAULT_SETTINGS.exercise.voiceCoach),
    baseline: baselineValid
      ? { standingKneeDeg: rawBaseline.standingKneeDeg, standingHipDeg: rawBaseline.standingHipDeg }
      : null,
  };

  const privacy = {
    localProcessingOnly: true, // Always enforce local
    recordingMode: ['off', 'manual'].includes(raw.privacy?.recordingMode)
      ? raw.privacy.recordingMode
      : DEFAULT_SETTINGS.privacy.recordingMode,
  };

  return {
    version: SETTINGS_VERSION,
    camera,
    tracking,
    overlays,
    exercise,
    privacy,
  };
}
