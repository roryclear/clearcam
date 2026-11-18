import test from 'node:test';
import assert from 'node:assert/strict';
import {
  angle3DAt,
  reliable3D,
  getJointAngles3D,
  calculateTrunkAngle3D,
} from '../src/biomechanics/angle-calculations.mjs';
import { JOINTS } from '../src/biomechanics/landmark-indices.mjs';

test('reliable3D validates finite 3D coordinates and visibility/presence', () => {
  assert.equal(reliable3D({ x: 0, y: 0, z: 0, visibility: 0.9, presence: 0.9 }), true);
  assert.equal(reliable3D({ x: 0.5, y: -0.2, z: 1.2 }), true);
  assert.equal(reliable3D(null), false);
  assert.equal(reliable3D({ x: NaN, y: 0, z: 0 }), false);
  assert.equal(reliable3D({ x: 0, y: 0, z: 0, visibility: 0.3 }, 0.55), false);
  assert.equal(reliable3D({ x: 0, y: 0, z: 0, presence: 0.2 }, 0.55), false);
});

test('angle3DAt computes exact 90 and 180 degree angles in 3D metric space', () => {
  const b = { x: 0, y: 0, z: 0, visibility: 0.9 };
  const a = { x: 1, y: 0, z: 0, visibility: 0.9 };
  const c = { x: 0, y: 1, z: 0, visibility: 0.9 };

  // Perpendicular X and Y axes
  assert.equal(angle3DAt(a, b, c), 90);

  // Collinear opposite X axis
  const d = { x: -1, y: 0, z: 0, visibility: 0.9 };
  assert.equal(angle3DAt(a, b, d), 180);
});

test('angle3DAt accurately resolves depth variation where 2D projection collapses', () => {
  // In 3D space, ray BA points along +Y +Z, ray BC points along +Y -Z.
  // Their dot product is (0*0 + 1*1 + 1*(-1)) = 0 -> exactly 90 degrees apart in 3D.
  // In 2D (X, Y), both vectors project to (0, 1), which would collapse to 0 degrees!
  const b = { x: 0, y: 0, z: 0, visibility: 0.9 };
  const a = { x: 0, y: 1, z: 1, visibility: 0.9 };
  const c = { x: 0, y: 1, z: -1, visibility: 0.9 };

  const angle3D = angle3DAt(a, b, c);
  assert.equal(angle3D, 90);
});

test('angle3DAt safely rejects occluded landmarks or degenerate zero-length limbs', () => {
  const b = { x: 0, y: 0, z: 0, visibility: 0.9 };
  const a = { x: 0, y: 0, z: 0, visibility: 0.9 }; // zero length
  const c = { x: 1, y: 0, z: 0, visibility: 0.9 };
  assert.equal(angle3DAt(a, b, c), null);

  const occluded = { x: 1, y: 0, z: 0, visibility: 0.2 };
  assert.equal(angle3DAt(occluded, b, c, 0.55), null);
});

test('getJointAngles3D computes independent 3D angles across all 8 standard joints', () => {
  const worldLandmarks = Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0, visibility: 0.9 }));

  // Set up a 90 degree left knee in 3D space: Hip at (0, 1, 0), Knee at (0, 0, 0), Ankle at (1, 0, 0)
  worldLandmarks[JOINTS.leftHip] = { x: 0, y: 1, z: 0, visibility: 0.9 };
  worldLandmarks[JOINTS.leftKnee] = { x: 0, y: 0, z: 0, visibility: 0.9 };
  worldLandmarks[JOINTS.leftAnkle] = { x: 1, y: 0, z: 0, visibility: 0.9 };

  const angles3D = getJointAngles3D(worldLandmarks);
  assert.equal(angles3D.leftKnee, 90);
  assert.equal(angles3D.rightKnee, null); // degenerate points return null
});

test('calculateTrunkAngle3D computes upright spine and resolves sagittal forward lean', () => {
  const worldLandmarks = Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0, visibility: 0.9 }));

  // Upright posture: shoulders directly above hips along -Y axis in MediaPipe metric space
  worldLandmarks[JOINTS.leftShoulder] = { x: -0.2, y: -0.5, z: 0, visibility: 0.9 };
  worldLandmarks[JOINTS.rightShoulder] = { x: 0.2, y: -0.5, z: 0, visibility: 0.9 };
  worldLandmarks[JOINTS.leftHip] = { x: -0.15, y: 0, z: 0, visibility: 0.9 };
  worldLandmarks[JOINTS.rightHip] = { x: 0.15, y: 0, z: 0, visibility: 0.9 };

  const uprightAngle = calculateTrunkAngle3D(worldLandmarks);
  assert.equal(uprightAngle, 0);

  // Sagittal forward lean: shoulders lean forward along +Z axis by 0.5m while Y is -0.5m (45 degree lean)
  worldLandmarks[JOINTS.leftShoulder].z = 0.5;
  worldLandmarks[JOINTS.rightShoulder].z = 0.5;

  const leanAngle = calculateTrunkAngle3D(worldLandmarks);
  assert.equal(Math.round(leanAngle), 45);
});
