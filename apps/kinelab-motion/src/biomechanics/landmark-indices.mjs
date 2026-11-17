/**
 * Standard MediaPipe 33-landmark indices, groupings, and skeletal topology.
 */

export const LANDMARKS = Object.freeze({
  NOSE: 0,
  LEFT_EYE_INNER: 1, LEFT_EYE: 2, LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4, RIGHT_EYE: 5, RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7, RIGHT_EAR: 8,
  MOUTH_LEFT: 9, MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_PINKY: 17, RIGHT_PINKY: 18,
  LEFT_INDEX: 19, RIGHT_INDEX: 20,
  LEFT_THUMB: 21, RIGHT_THUMB: 22,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
  LEFT_HEEL: 29, RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31, RIGHT_FOOT_INDEX: 32,
});

export const JOINTS = Object.freeze({
  leftShoulder: LANDMARKS.LEFT_SHOULDER,
  rightShoulder: LANDMARKS.RIGHT_SHOULDER,
  leftElbow: LANDMARKS.LEFT_ELBOW,
  rightElbow: LANDMARKS.RIGHT_ELBOW,
  leftWrist: LANDMARKS.LEFT_WRIST,
  rightWrist: LANDMARKS.RIGHT_WRIST,
  leftHip: LANDMARKS.LEFT_HIP,
  rightHip: LANDMARKS.RIGHT_HIP,
  leftKnee: LANDMARKS.LEFT_KNEE,
  rightKnee: LANDMARKS.RIGHT_KNEE,
  leftAnkle: LANDMARKS.LEFT_ANKLE,
  rightAnkle: LANDMARKS.RIGHT_ANKLE,
});

export const SKELETON_CONNECTIONS = Object.freeze([
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Right arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Left leg
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
  // Head / face connections (subtle)
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
]);

// 12 key tracked indices for body presence assessment
export const KEY_TRACKED_INDICES = Object.freeze([
  11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
]);
