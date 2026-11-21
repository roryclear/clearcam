import test from 'node:test';
import assert from 'node:assert/strict';
import {
  angleAt,
  reliable,
  getJointAngles,
  trackingStatus,
  calculateTrunkAngle,
  dominantKneeAngle,
  angleUncertainty,
} from '../src/biomechanics/angle-calculations.mjs';
import { analyzeSymmetry } from '../src/biomechanics/symmetry-analyzer.mjs';

const point = (x, y, visibility = 0.99) => ({ x, y, visibility, presence: 0.99 });
const blankPose = () => Array.from({ length: 33 }, () => point(0.5, 0.5));

test('reliable checks coordinates, visibility and presence', () => {
  assert.equal(reliable(point(0.5, 0.5)), true);
  assert.equal(reliable(point(NaN, 0.5)), false);
  assert.equal(reliable(point(0.5, 0.5, 0.2), 0.55), false);
  assert.equal(reliable(null), false);
});

test('angleAt accurately computes 90 and 180 degree angles', () => {
  // Right angle at (0.5, 0.5)
  const a90 = angleAt(point(0.5, 0.2), point(0.5, 0.5), point(0.8, 0.5), 1000, 1000);
  assert.equal(a90, 90);

  // Straight line 180 degrees
  const a180 = angleAt(point(0.5, 0.2), point(0.5, 0.5), point(0.5, 0.8), 1000, 1000);
  assert.equal(a180, 180);
});

test('angleUncertainty scales with landmark visibility', () => {
  const pHigh = point(0.5, 0.5, 0.98);
  const pLow = point(0.5, 0.5, 0.60);
  const uHigh = angleUncertainty(pHigh, pHigh, pHigh);
  const uLow = angleUncertainty(pLow, pLow, pLow);

  assert.ok(uHigh < 0.1, `High visibility should yield low uncertainty, got ${uHigh}`);
  assert.ok(uLow > 0.3, `Low visibility should yield higher uncertainty, got ${uLow}`);
});

test('calculateTrunkAngle computes vertical torso inclination', () => {
  const pose = blankPose();
  // Upright vertical spine: shoulders directly above hips
  pose[11] = point(0.4, 0.2); // left shoulder
  pose[12] = point(0.6, 0.2); // right shoulder (mid = 0.5, 0.2)
  pose[23] = point(0.4, 0.6); // left hip
  pose[24] = point(0.6, 0.6); // right hip (mid = 0.5, 0.6)

  const uprightAngle = calculateTrunkAngle(pose, 1000, 1000);
  assert.equal(uprightAngle, 0);

  // Leaning forward (shoulders shifted rightwards)
  pose[11] = point(0.6, 0.2);
  pose[12] = point(0.8, 0.2); // mid = 0.7, 0.2 -> dx = 0.2, dy = -0.4
  const leanAngle = calculateTrunkAngle(pose, 1000, 1000);
  assert.ok(leanAngle > 20 && leanAngle < 30);
});

test('analyzeSymmetry detects bilateral limb imbalances', () => {
  const symmetricAngles = {
    leftKnee: 90, rightKnee: 91,
    leftHip: 120, rightHip: 119,
    leftElbow: 150, rightElbow: 152,
    leftShoulder: 45, rightShoulder: 44,
  };
  const symReport = analyzeSymmetry(symmetricAngles);
  assert.equal(symReport.pairs.knee.isSymmetric, true);
  assert.ok(symReport.overallAsymmetryDeg < 2.0);

  const asymmetricAngles = {
    leftKnee: 70, rightKnee: 110, // 40 degree difference
    leftHip: 120, rightHip: 120,
    leftElbow: 150, rightElbow: 150,
    leftShoulder: 45, rightShoulder: 45,
  };
  const asymReport = analyzeSymmetry(asymmetricAngles);
  assert.equal(asymReport.pairs.knee.isSymmetric, false);
  assert.equal(asymReport.pairs.knee.diffDeg, 40);
});

test('trackingStatus evaluates whole body and decouples upper-body exercises', () => {
  const pose = Array.from({ length: 33 }, () => point(0.5, 0.5, 0.1)); // all low visibility
  assert.equal(trackingStatus(pose), 'lost');

  // Upper body joints visible (11, 12, 13, 14, 15, 16, 23, 24)
  [11, 12, 13, 14, 15, 16, 23, 24].forEach(idx => {
    pose[idx] = point(0.5, 0.5, 0.95);
  });

  // Default requires 10 tracked indices for 'good', so 8 indices is 'partial'
  assert.equal(trackingStatus(pose, 0.55), 'partial');

  // For shoulder exercises, upper body visibility qualifies as 'good'
  assert.equal(trackingStatus(pose, 0.55, 'shoulder-abduction'), 'good');
  assert.equal(trackingStatus(pose, 0.55, 'shoulder-flexion'), 'good');
});

test('Clinical convention: knee/hip/elbow as flexion, shoulder unchanged, labels', async () => {
  const { displayAngle, formatAngle, angleLabel, setAngleConvention, getAngleConvention } = await import('../src/biomechanics/angle-convention.mjs');
  assert.equal(displayAngle('leftKnee', 95, 'clinical'), 85);
  assert.equal(displayAngle('rightHip', 170, 'clinical'), 10);
  assert.equal(displayAngle('leftElbow', 180, 'clinical'), 0);
  assert.equal(displayAngle('leftShoulder', 150, 'clinical'), 150);
  assert.equal(displayAngle('Knee', 95, 'interior'), 95);
  assert.equal(displayAngle('leftKnee', NaN), null);
  assert.equal(formatAngle('leftKnee', null), '—');
  assert.equal(angleLabel({ key: 'leftHip', label: 'LEFT HIP' }, 'clinical'), 'LEFT HIP FLEX (TRUNK–THIGH)');
  assert.equal(angleLabel({ key: 'leftKnee', label: 'LEFT KNEE' }, 'interior'), 'LEFT KNEE');
  setAngleConvention('interior');
  assert.equal(getAngleConvention(), 'interior');
  assert.equal(formatAngle('leftKnee', 95.4), '95°');
  setAngleConvention('clinical');
  assert.equal(formatAngle('leftKnee', 95.4), '85°');
});

test('Angle reliability flags the far-side limb side-on and low-confidence joints', async () => {
  const { angleReliability } = await import('../src/biomechanics/angle-reliability.mjs');
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.95 }));
  for (const i of [24, 26, 28]) lm[i] = { x: 0.5, y: 0.5, visibility: 0.6 }; // right leg hidden
  const angles = { leftKnee: 100, rightKnee: 120, leftHip: 90, rightHip: 95, leftShoulder: 20 };
  const side = angleReliability(angles, lm, { sagittal2DWeight: 1 });
  assert.equal(side.rightKnee.low, true);
  assert.match(side.rightKnee.reason, /Far-side/);
  assert.equal(side.leftKnee.low, false);
  // Facing the camera: no far side, but the hidden leg is still low confidence.
  const front = angleReliability(angles, lm, { sagittal2DWeight: 0 });
  assert.match(front.rightKnee.reason, /Low landmark confidence/);
  assert.equal(front.leftKnee.low, false);
  assert.equal(front.rightElbow.low, false, 'missing angle is not flagged');
});
