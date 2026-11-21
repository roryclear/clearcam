import test from 'node:test';
import assert from 'node:assert/strict';
import { FramingAssistant } from '../src/tracking/framing-assistant.mjs';
import { JOINTS } from '../src/biomechanics/landmark-indices.mjs';

// Standing person 2–3 m away: head near the top, feet near the bottom, hips mid-frame.
const BODY_Y = {
  [JOINTS.nose]: 0.12, [JOINTS.leftShoulder]: 0.25, [JOINTS.rightShoulder]: 0.25,
  [JOINTS.leftHip]: 0.5, [JOINTS.rightHip]: 0.5, [JOINTS.leftKnee]: 0.7, [JOINTS.rightKnee]: 0.7,
  [JOINTS.leftAnkle]: 0.9, [JOINTS.rightAnkle]: 0.9,
};

function createDummyLandmarks(visibleIndices, { scale = 1, shift = 0 } = {}) {
  const arr = Array(33).fill(null).map(() => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.1 }));
  for (const idx of visibleIndices) {
    const y = 0.5 + ((BODY_Y[idx] ?? 0.5) - 0.5) * scale + shift;
    arr[idx] = { x: 0.5, y, z: 0, visibility: 0.9 };
  }
  return arr;
}

const FULL_BODY = [
  JOINTS.nose, JOINTS.leftShoulder, JOINTS.rightShoulder, JOINTS.leftHip, JOINTS.rightHip,
  JOINTS.leftKnee, JOINTS.rightKnee, JOINTS.leftAnkle, JOINTS.rightAnkle,
];

test('Camera setup guidance: level phone, distance, hip height', () => {
  const assistant = new FramingAssistant();
  const level = { rollDeg: 1, pitchDeg: 2 };
  assert.equal(assistant.evaluate(createDummyLandmarks(FULL_BODY), 'squat', { tilt: level }).status, 'optimal');
  const tilted = assistant.evaluate(createDummyLandmarks(FULL_BODY), 'squat', { tilt: { rollDeg: -7.4, pitchDeg: 2 } });
  assert.equal(tilted.status, 'partial');
  assert.match(tilted.guidance, /Level the phone: tilted 7°/);
  const far = assistant.evaluate(createDummyLandmarks(FULL_BODY, { scale: 0.4 }), 'squat', { tilt: level });
  assert.match(far.guidance, /Move closer/);
  // Hips in the upper part of a level frame: camera below hip height.
  const low = assistant.evaluate(createDummyLandmarks(FULL_BODY, { scale: 0.8, shift: -0.2 }), 'squat', { tilt: level });
  assert.match(low.guidance, /Raise the camera/);
  const high = assistant.evaluate(createDummyLandmarks(FULL_BODY, { scale: 0.8, shift: 0.2 }), 'squat', { tilt: level });
  assert.match(high.guidance, /Lower the camera/);
  // Pitched phone or no sensor: no hip-height advice (can't tell camera height).
  assert.equal(assistant.evaluate(createDummyLandmarks(FULL_BODY, { scale: 0.8, shift: -0.2 }), 'squat', { tilt: { rollDeg: 0, pitchDeg: 20 } }).status, 'optimal');
  assert.equal(assistant.evaluate(createDummyLandmarks(FULL_BODY, { scale: 0.8, shift: -0.2 }), 'squat').status, 'optimal');
  // Upper-body exercises skip the distance and height checks.
  assert.equal(assistant.evaluate(createDummyLandmarks(FULL_BODY, { scale: 0.4 }), 'shoulder_abduction', { tilt: level }).status, 'optimal');
});

test('FramingAssistant flags missing lower body for squat and guides user to step back', () => {
  const assistant = new FramingAssistant();

  // Only upper body visible (head, shoulders, elbows, wrists)
  const upperBodyOnly = createDummyLandmarks([
    JOINTS.nose,
    JOINTS.leftShoulder, JOINTS.rightShoulder,
    JOINTS.leftElbow, JOINTS.rightElbow,
    JOINTS.leftWrist, JOINTS.rightWrist,
    JOINTS.leftHip, JOINTS.rightHip,
  ]);

  const squatResult = assistant.evaluate(upperBodyOnly, 'squat');
  assert.equal(squatResult.status, 'inadequate');
  assert.ok(squatResult.missingSegments.includes('lowerBody'));
  assert.ok(squatResult.guidance.includes('Step back'));
});

test('FramingAssistant permits upper-body framing for shoulder flexion and abduction', () => {
  const assistant = new FramingAssistant();

  // Only upper body visible
  const upperBodyOnly = createDummyLandmarks([
    JOINTS.nose,
    JOINTS.leftShoulder, JOINTS.rightShoulder,
    JOINTS.leftElbow, JOINTS.rightElbow,
    JOINTS.leftWrist, JOINTS.rightWrist,
    JOINTS.leftHip, JOINTS.rightHip,
  ]);

  const shoulderResult = assistant.evaluate(upperBodyOnly, 'shoulder-flexion');
  assert.equal(shoulderResult.status, 'optimal');
  assert.equal(shoulderResult.missingSegments.length, 0);

  const abductionResult = assistant.evaluate(upperBodyOnly, 'shoulder-abduction');
  assert.equal(abductionResult.status, 'optimal');
});

test('FramingAssistant reports optimal when full body is detected for squats', () => {
  const assistant = new FramingAssistant();

  const fullBody = createDummyLandmarks([
    JOINTS.nose,
    JOINTS.leftShoulder, JOINTS.rightShoulder,
    JOINTS.leftHip, JOINTS.rightHip,
    JOINTS.leftKnee, JOINTS.rightKnee,
    JOINTS.leftAnkle, JOINTS.rightAnkle,
  ]);

  const res = assistant.evaluate(fullBody, 'squat');
  assert.equal(res.status, 'optimal');
  assert.equal(res.missingSegments.length, 0);
  assert.ok(res.guidance.includes('Framing optimal'));
});

test('FramingAssistant seamlessly supports underscored and hyphenated exercise IDs', () => {
  const assistant = new FramingAssistant();
  const upperBodyOnly = createDummyLandmarks([
    JOINTS.nose,
    JOINTS.leftShoulder, JOINTS.rightShoulder,
    JOINTS.leftElbow, JOINTS.rightElbow,
    JOINTS.leftWrist, JOINTS.rightWrist,
    JOINTS.leftHip, JOINTS.rightHip,
  ]);

  // sit_to_stand requires lowerBody
  const sitToStand = assistant.evaluate(upperBodyOnly, 'sit_to_stand');
  assert.equal(sitToStand.status, 'inadequate');
  assert.ok(sitToStand.missingSegments.includes('lowerBody'));
  assert.ok(sitToStand.guidance.includes('sit to stand'));

  // shoulder_abduction permits upper body
  const shoulder = assistant.evaluate(upperBodyOnly, 'shoulder_abduction');
  assert.equal(shoulder.status, 'optimal');
  assert.ok(shoulder.guidance.includes('shoulder abduction'));
});

test('Setup guidance: off when disabled; distance/height judged only when standing (PR #16 review)', () => {
  const assistant = new FramingAssistant();
  const level = { rollDeg: 0.5, pitchDeg: 2 };
  // Disabled setting: no distance prompt even without sensor data.
  const far = createDummyLandmarks(FULL_BODY, { scale: 0.4 });
  assert.match(assistant.evaluate(far, 'squat').guidance, /Move closer/);
  assert.equal(assistant.evaluate(far, 'squat', { setupGuidance: false }).status, 'optimal');
  // Squat depth: hips drop low in the frame and the knees bend; no camera-height advice.
  const deep = createDummyLandmarks(FULL_BODY);
  deep[JOINTS.leftHip] = { x: 0.35, y: 0.66, z: 0, visibility: 0.9 };
  deep[JOINTS.rightHip] = { x: 0.35, y: 0.66, z: 0, visibility: 0.9 };
  deep[JOINTS.leftKnee] = { x: 0.6, y: 0.72, z: 0, visibility: 0.9 };
  deep[JOINTS.rightKnee] = { x: 0.6, y: 0.72, z: 0, visibility: 0.9 };
  const res = assistant.evaluate(deep, 'squat', { tilt: level });
  assert.doesNotMatch(res.guidance ?? '', /camera to hip height/);
  assert.equal(res.status, 'optimal');
});

test('Setup guidance: standing check uses pixel proportions (PR #16 review round 2)', () => {
  const assistant = new FramingAssistant();
  const level = { rollDeg: 0, pitchDeg: 2 };
  const lm = createDummyLandmarks(FULL_BODY);
  // 1920×1080: hip (0.4, 0.3), knee (0.5, 0.7), ankle (0.5, 0.9) is a ~156° knee, not standing.
  for (const [h, k, a] of [[JOINTS.leftHip, JOINTS.leftKnee, JOINTS.leftAnkle], [JOINTS.rightHip, JOINTS.rightKnee, JOINTS.rightAnkle]]) {
    lm[h] = { x: 0.4, y: 0.3, z: 0, visibility: 0.9 };
    lm[k] = { x: 0.5, y: 0.7, z: 0, visibility: 0.9 };
    lm[a] = { x: 0.5, y: 0.9, z: 0, visibility: 0.9 };
  }
  lm[JOINTS.nose] = { x: 0.5, y: 0.05, z: 0, visibility: 0.9 };
  const wide = assistant.evaluate(lm, 'squat', { tilt: level, aspect: 1920 / 1080 });
  assert.doesNotMatch(wide.guidance ?? '', /camera to hip height/);
  // Straight legs still count as standing in a portrait frame.
  const tall = createDummyLandmarks(FULL_BODY, { scale: 0.8, shift: -0.2 });
  assert.match(assistant.evaluate(tall, 'squat', { tilt: level, aspect: 9 / 16 }).guidance, /Raise the camera/);
});
