import test from 'node:test';
import assert from 'node:assert/strict';
import { MediaPipeRunner } from '../src/tracking/mediapipe-runner.mjs';

class MockLandmarker {
  detectForVideo(source, timestampMs) {
    this.lastSource = source;
    this.lastTimestamp = timestampMs;
    return {
      landmarks: [[
        { x: 0.5, y: 0.5, z: 0, visibility: 0.9 },
        { x: 0.6, y: 0.8, z: 0.1, visibility: 0.85 },
      ]],
      worldLandmarks: [[
        { x: 0.05, y: 0.1, z: 0.02, visibility: 0.9 },
        { x: 0.15, y: 0.4, z: 0.05, visibility: 0.85 },
      ]],
    };
  }
}

test('MediaPipeRunner supports inference resolution scaling via offscreen canvas', () => {
  const runner = new MediaPipeRunner();
  runner.landmarker = new MockLandmarker();

  assert.equal(runner.inferencePreset, 'adaptive'); // default preset

  runner.setInferencePreset('speed');
  assert.equal(runner.targetWidth, 360);

  runner.setInferencePreset('balanced');
  assert.equal(runner.targetWidth, 480);

  runner.setInferencePreset('direct');
  assert.equal(runner.inferencePreset, 'direct');
});

test('MediaPipeRunner returns both 2D landmarks and 3D worldLandmarks', () => {
  const runner = new MediaPipeRunner();
  runner.landmarker = new MockLandmarker();

  const mockVideo = { videoWidth: 1280, videoHeight: 720 };
  const res = runner.detectForVideo(mockVideo, 1000);

  assert.ok(Array.isArray(res.landmarks));
  assert.equal(res.landmarks.length, 2);
  assert.equal(res.landmarks[0].x, 0.5);

  assert.ok(Array.isArray(res.worldLandmarks));
  assert.equal(res.worldLandmarks.length, 2);
  assert.equal(res.worldLandmarks[0].x, 0.05);
});
