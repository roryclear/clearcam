/**
 * Backwards-compatible facade re-exporting pure 2D geometry, joint angles,
 * and squat tracking functions for single-camera movement analysis.
 */

export {
  JOINTS,
  SKELETON_CONNECTIONS,
} from './biomechanics/landmark-indices.mjs';

export {
  ANGLE_DEFINITIONS,
  reliable,
  angleAt,
  angleUncertainty,
  getJointAngles,
  calculateTrunkAngle,
  trackingStatus,
  dominantKneeAngle,
} from './biomechanics/angle-calculations.mjs';

export {
  createSquatState,
  advanceSquat,
  SquatExercise,
} from './exercises/squat-exercise.mjs';
