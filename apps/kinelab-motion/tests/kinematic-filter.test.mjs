import test from 'node:test';
import assert from 'node:assert/strict';
import { KinematicBoneFilter } from '../src/filters/kinematic-filter.mjs';
import { JOINTS } from '../src/biomechanics/landmark-indices.mjs';

test('KinematicBoneFilter establishes calibrated bone lengths and clamps limb elasticity', () => {
  const filter = new KinematicBoneFilter({ minVisibility: 0.6 });

  // Generate 20 frames of consistent standing pose with femur length = 0.42m
  for (let frame = 0; frame < 20; frame++) {
    const landmarks = Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0, visibility: 0.9 }));
    landmarks[JOINTS.leftHip] = { x: -0.1, y: 0.0, z: 0.0, visibility: 0.95 };
    landmarks[JOINTS.leftKnee] = { x: -0.1, y: 0.42, z: 0.0, visibility: 0.95 }; // length = 0.42m
    landmarks[JOINTS.leftAnkle] = { x: -0.1, y: 0.84, z: 0.0, visibility: 0.95 }; // length = 0.42m
    filter.filter(landmarks, frame * 33.3);
  }

  const boneLengths = filter.getCalibratedLengths();
  assert.ok(Math.abs(boneLengths.leftThigh - 0.42) < 0.01, `Expected thigh ~0.42m, got ${boneLengths.leftThigh}`);
  assert.ok(Math.abs(boneLengths.leftShank - 0.42) < 0.01, `Expected shank ~0.42m, got ${boneLengths.leftShank}`);

  // Frame 21: Monocular depth glitch causes knee to jump in Z, stretching femur to 0.60m!
  const noisyLandmarks = Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0, visibility: 0.9 }));
  noisyLandmarks[JOINTS.leftHip] = { x: -0.1, y: 0.0, z: 0.0, visibility: 0.95 };
  noisyLandmarks[JOINTS.leftKnee] = { x: -0.1, y: 0.42, z: 0.43, visibility: 0.95 }; // dist = hypot(0.42, 0.43) ~= 0.60m!

  const filtered = filter.filter(noisyLandmarks, 21 * 33.3);
  const hip = filtered[JOINTS.leftHip];
  const knee = filtered[JOINTS.leftKnee];
  const correctedDist = Math.hypot(knee.x - hip.x, knee.y - hip.y, (knee.z || 0) - (hip.z || 0));

  // Filter must clamp bone length back to ~0.42m (+- 0.02m)
  assert.ok(Math.abs(correctedDist - 0.42) < 0.02, `Corrected distance should be close to 0.42m, got ${correctedDist}`);
  // Direction vector should point in +Y, +Z
  assert.ok(knee.y > hip.y);
  assert.ok((knee.z || 0) > 0);
});

test('KinematicBoneFilter safely handles occluded or missing joints without drifting', () => {
  const filter = new KinematicBoneFilter();
  const landmarks = Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0, visibility: 0.9 }));
  landmarks[JOINTS.leftHip] = { x: 0, y: 0, z: 0, visibility: 0.2 }; // occluded
  landmarks[JOINTS.leftKnee] = { x: 0, y: 0.5, z: 0, visibility: 0.9 };

  const res = filter.filter(landmarks, 100);
  assert.ok(res);
  assert.equal(res[JOINTS.leftHip].visibility, 0.2);
});
