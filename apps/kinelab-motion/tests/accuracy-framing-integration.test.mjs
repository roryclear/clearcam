import test from 'node:test';
import assert from 'node:assert/strict';
import { validateSettings, DEFAULT_SETTINGS } from '../src/settings/settings-schema.mjs';
import { getJointAngles, getJointAngles3D, trackingStatus } from '../src/biomechanics/angle-calculations.mjs';
import { defaultFramingAssistant } from '../src/tracking/framing-assistant.mjs';

const pt2D = (x, y, visibility = 0.99) => ({ x, y, visibility, presence: 0.99 });
const pt3D = (x, y, z, visibility = 0.99) => ({ x, y, z, visibility, presence: 0.99 });

test('Integration: Settings schema defaults to 3D metric angle calculations', () => {
  const settings = validateSettings({});
  assert.equal(settings.tracking.angleDimension, '3d');
});

test('Integration: 3D metric space resolves depth that collapses in 2D image plane', () => {
  // Hip at (0, 0, 0), Knee at (0, 0.4, 0), Ankle foreshortened along Z axis at (0, 0.4, 0.4)
  const worldLandmarks = Array.from({ length: 33 }, () => pt3D(0, 0, 0, 0.1));
  worldLandmarks[23] = pt3D(0, 0, 0, 0.95);     // leftHip
  worldLandmarks[25] = pt3D(0, 0.4, 0, 0.95);   // leftKnee
  worldLandmarks[27] = pt3D(0, 0.4, 0.4, 0.95); // leftAnkle

  const angles3D = getJointAngles3D(worldLandmarks, 0.55);
  // Ray BA is (0, -0.4, 0), ray BC is (0, 0, 0.4) -> perpendicular vectors in 3D -> 90 degrees
  assert.equal(angles3D.leftKnee, 90);

  // In 2D projected image plane, if Z collapses, the ankle (0, 0.4) is coincident with knee (0, 0.4) -> degenerate or 0 length
  const landmarks2D = Array.from({ length: 33 }, () => pt2D(0.5, 0.5, 0.1));
  landmarks2D[23] = pt2D(0.5, 0.3, 0.95);
  landmarks2D[25] = pt2D(0.5, 0.7, 0.95);
  landmarks2D[27] = pt2D(0.5, 0.7, 0.95); // collapsed in 2D
  const angles2D = getJointAngles(landmarks2D, 1000, 1000, 0.55);
  assert.equal(angles2D.leftKnee, null); // safely rejected degenerate joint
});

test('Integration: FramingAssistant and trackingStatus cooperate to guide lower body vs upper body', () => {
  const upperBodyOnlyPose = Array.from({ length: 33 }, () => pt2D(0.5, 0.5, 0.1));
  // Head, shoulders, elbows, wrists, hips visible (sitting at desk)
  [0, 11, 12, 13, 14, 15, 16, 23, 24].forEach(idx => {
    upperBodyOnlyPose[idx] = pt2D(0.5, 0.5, 0.95);
  });

  // For Squats: inadequate framing, guides to step back
  const squatFraming = defaultFramingAssistant.evaluate(upperBodyOnlyPose, 'squat');
  assert.equal(squatFraming.status, 'inadequate');
  assert.ok(squatFraming.missingSegments.includes('lowerBody'));
  assert.ok(squatFraming.guidance.includes('Step back'));
  assert.equal(trackingStatus(upperBodyOnlyPose, 0.55, 'squat'), 'partial');

  // For Shoulder Abduction: optimal framing, does NOT require legs
  const shoulderFraming = defaultFramingAssistant.evaluate(upperBodyOnlyPose, 'shoulder-abduction');
  assert.equal(shoulderFraming.status, 'optimal');
  assert.equal(shoulderFraming.missingSegments.length, 0);
  assert.equal(trackingStatus(upperBodyOnlyPose, 0.55, 'shoulder-abduction'), 'good');
});
