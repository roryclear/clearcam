/**
 * Session Recorder: Local, opt-in recording of movement telemetry, skeletal frames,
 * and synchronized native video stream via MediaRecorder.
 * Reference: Sports2D and Rerun data structures for human kinematics.
 */

import { landmarkConfidence } from '../exercises/rep-analysis.mjs';

export class SessionRecorder {
  constructor({ MediaRecorderClass = typeof MediaRecorder !== 'undefined' ? MediaRecorder : null } = {}) {
    this.MediaRecorderClass = MediaRecorderClass;
    this.isRecording = false;
    this.isRecordingVideo = false;
    this.sessionId = null;
    this.startTimeMs = null;
    this.exerciseId = 'free';
    this.frames = []; // Array of frame objects

    this.mediaRecorder = null;
    this.videoChunks = [];
    this.videoMimeType = null;
  }

  _pickMimeType() {
    if (!this.MediaRecorderClass || typeof this.MediaRecorderClass.isTypeSupported !== 'function') {
      return 'video/webm';
    }
    const candidates = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
      'video/mp4;codecs=avc1',
      'video/mp4',
    ];
    for (const type of candidates) {
      if (this.MediaRecorderClass.isTypeSupported(type)) {
        return type;
      }
    }
    return '';
  }

  start({ exerciseId = 'free', mediaStream = null } = {}) {
    this.isRecording = true;
    this.sessionId = `session_${Date.now()}`;
    this.startTimeMs = performance.now();
    this.exerciseId = exerciseId;
    this.frames = [];
    this.videoChunks = [];
    this.videoMimeType = null;

    if (mediaStream && this.MediaRecorderClass) {
      try {
        const mimeType = this._pickMimeType();
        this.videoMimeType = mimeType || 'video/webm';
        const options = mimeType ? { mimeType } : {};
        this.mediaRecorder = new this.MediaRecorderClass(mediaStream, options);

        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            this.videoChunks.push(event.data);
          }
        };

        this.mediaRecorder.start(250); // Timeslice chunks every 250ms
        this.isRecordingVideo = true;
      } catch (err) {
        console.warn('[SessionRecorder] MediaRecorder initialization failed, falling back to telemetry-only:', err);
        this.mediaRecorder = null;
        this.isRecordingVideo = false;
      }
    } else {
      this.mediaRecorder = null;
      this.isRecordingVideo = false;
    }
  }

  recordFrame({
    landmarks,
    angles,
    rawAngles = null,
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
          // Same quantized min(visibility, presence) the live gate uses.
          landmarkConfidence(p),
        ];
      });
    }

    // Clean angle values
    const clean = src => {
      const out = {};
      if (src && typeof src === 'object') {
        for (const [k, v] of Object.entries(src)) out[k] = Number.isFinite(v) ? Math.round(v * 10) / 10 : null;
      }
      return out;
    };
    const cleanAngles = clean(angles);
    // Angles from unfiltered landmarks, for zero-phase re-analysis (offline-analysis.mjs).
    const cleanRaw = rawAngles ? clean(rawAngles) : null;

    // Frame times come from the camera's capture clock, which can precede start() slightly.
    const elapsedMs = Math.max(0, Math.round(timestampMs - this.startTimeMs));

    this.frames.push({
      timeMs: elapsedMs,
      landmarks: compactLandmarks,
      angles: cleanAngles,
      ...(cleanRaw ? { rawAngles: cleanRaw } : {}),
      phase,
      status,
      reps,
    });
  }

  _buildSessionData(videoBlob = null) {
    const totalDurationMs = this.frames.length > 0 ? this.frames[this.frames.length - 1].timeMs : 0;
    const finalReps = this.frames.length > 0 ? this.frames[this.frames.length - 1].reps : 0;

    return {
      id: this.sessionId,
      recordedAt: new Date().toISOString(),
      exerciseId: this.exerciseId,
      frameCount: this.frames.length,
      durationMs: totalDurationMs,
      finalReps,
      frames: this.frames,
      hasVideo: Boolean(videoBlob && videoBlob.size > 0),
      videoBlob: videoBlob,
      videoMimeType: this.videoMimeType,
    };
  }

  stop() {
    if (!this.isRecording) return null;
    this.isRecording = false;

    // Synchronous path when no video was recorded (preserves backwards compatibility)
    if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
      // The recorder may already have stopped on its own (e.g. the camera track ended);
      // keep whatever video chunks it delivered instead of discarding them.
      const videoBlob = this.videoChunks.length > 0
        ? new Blob(this.videoChunks, { type: this.videoMimeType || 'video/webm' })
        : null;
      this.isRecordingVideo = false;
      this.mediaRecorder = null;
      return this._buildSessionData(videoBlob);
    }

    // Asynchronous path when finalizing video recording
    return new Promise((resolve) => {
      try {
        this.mediaRecorder.onstop = () => {
          let videoBlob = null;
          if (this.videoChunks.length > 0) {
            videoBlob = new Blob(this.videoChunks, { type: this.videoMimeType || 'video/webm' });
          }
          this.isRecordingVideo = false;
          this.mediaRecorder = null;
          resolve(this._buildSessionData(videoBlob));
        };
        this.mediaRecorder.stop();
      } catch (err) {
        console.warn('[SessionRecorder] Error finalizing video recording:', err);
        this.isRecordingVideo = false;
        this.mediaRecorder = null;
        resolve(this._buildSessionData(null));
      }
    });
  }
}
