/**
 * Session Recorder: Local, opt-in recording of movement telemetry and skeletal frames.
 * Reference: Sports2D and Rerun data structures for human kinematics.
 */

export class SessionRecorder {
  constructor() {
    this.isRecording = false;
    this.sessionId = null;
    this.startTimeMs = null;
    this.exerciseId = 'free';
    this.frames = []; // Array of frame objects
  }

  start({ exerciseId = 'free' } = {}) {
    this.isRecording = true;
    this.sessionId = `session_${Date.now()}`;
    this.startTimeMs = performance.now();
    this.exerciseId = exerciseId;
    this.frames = [];
  }

  recordFrame({
    landmarks,
    angles,
    timestampMs = performance.now(),
    phase = 'tracking',
    status = 'active',
    reps = 0,
  }) {
    if (!this.isRecording) return;

    // Compact landmark representation: array of [x, y, z, vis] or null
    let compactLandmarks = null;
    if (Array.isArray(landmarks)) {
      compactLandmarks = landmarks.map(p => {
        if (!p) return null;
        return [
          Math.round(p.x * 10000) / 10000,
          Math.round(p.y * 10000) / 10000,
          Math.round((p.z ?? 0) * 10000) / 10000,
          Math.round((p.visibility ?? 1) * 100) / 100,
        ];
      });
    }

    // Clean angle values
    const cleanAngles = {};
    if (angles && typeof angles === 'object') {
      for (const [k, v] of Object.entries(angles)) {
        cleanAngles[k] = Number.isFinite(v) ? Math.round(v * 10) / 10 : null;
      }
    }

    const elapsedMs = Math.round(timestampMs - this.startTimeMs);

    this.frames.push({
      timeMs: elapsedMs,
      landmarks: compactLandmarks,
      angles: cleanAngles,
      phase,
      status,
      reps,
    });
  }

  stop() {
    if (!this.isRecording) return null;
    this.isRecording = false;

    const totalDurationMs = this.frames.length > 0 ? this.frames[this.frames.length - 1].timeMs : 0;
    const finalReps = this.frames.length > 0 ? this.frames[this.frames.length - 1].reps : 0;

    const sessionData = {
      id: this.sessionId,
      recordedAt: new Date().toISOString(),
      exerciseId: this.exerciseId,
      frameCount: this.frames.length,
      durationMs: totalDurationMs,
      finalReps,
      frames: this.frames,
    };

    return sessionData;
  }
}
