import test from 'node:test';
import assert from 'node:assert/strict';
import { SessionRecorder } from '../src/sessions/session-recorder.mjs';

// Mock MediaRecorder for Node.js testing environment
class MockMediaStream {}

class MockMediaRecorder {
  static isTypeSupported(type) {
    return type.includes('webm');
  }

  constructor(stream, options = {}) {
    this.stream = stream;
    this.mimeType = options.mimeType || 'video/webm';
    this.state = 'inactive';
    this.ondataavailable = null;
    this.onstop = null;
  }

  start(timeslice) {
    this.state = 'recording';
    this.timeslice = timeslice;
  }

  stop() {
    this.state = 'inactive';
    if (this.ondataavailable) {
      this.ondataavailable({ data: new Blob(['mock-chunk-1'], { type: this.mimeType }) });
      this.ondataavailable({ data: new Blob(['mock-chunk-2'], { type: this.mimeType }) });
    }
    if (this.onstop) {
      this.onstop();
    }
  }
}

test('SessionRecorder captures video chunks when mediaStream and MediaRecorder are present', async () => {
  const recorder = new SessionRecorder({
    MediaRecorderClass: MockMediaRecorder,
  });

  const stream = new MockMediaStream();
  recorder.start({ exerciseId: 'squat', mediaStream: stream });

  assert.equal(recorder.isRecording, true);
  assert.equal(recorder.isRecordingVideo, true);

  // Record a couple of frames
  recorder.recordFrame({
    landmarks: [{ x: 0.5, y: 0.5, z: 0, visibility: 0.9 }],
    angles: { leftKnee: 120 },
    reps: 1,
  });

  const session = await recorder.stop();

  assert.equal(session.exerciseId, 'squat');
  assert.equal(session.hasVideo, true);
  assert.ok(session.videoBlob instanceof Blob);
  assert.ok(session.videoMimeType.includes('webm'));
  assert.equal(session.frameCount, 1);
});

test('SessionRecorder falls back cleanly to telemetry-only when mediaStream is omitted', async () => {
  const recorder = new SessionRecorder({
    MediaRecorderClass: MockMediaRecorder,
  });

  recorder.start({ exerciseId: 'lunge' }); // no mediaStream

  assert.equal(recorder.isRecording, true);
  assert.equal(recorder.isRecordingVideo, false);

  recorder.recordFrame({
    landmarks: null,
    angles: null,
  });

  const session = await recorder.stop();

  assert.equal(session.hasVideo, false);
  assert.equal(session.videoBlob, null);
  assert.equal(session.frameCount, 1);
});
