import test from 'node:test';
import assert from 'node:assert/strict';
import { FramingAssistant } from '../src/tracking/framing-assistant.mjs';
import { JOINTS } from '../src/biomechanics/landmark-indices.mjs';

function createDummyLandmarks(visibleIndices) {
  const arr = Array(33).fill(null).map(() => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.1 }));
  for (const idx of visibleIndices) {
    arr[idx] = { x: 0.5, y: 0.5, z: 0, visibility: 0.9 };
  }
  return arr;
}

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
