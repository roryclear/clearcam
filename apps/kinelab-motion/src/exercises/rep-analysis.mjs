/**
 * Per-repetition metrics and quality gating.
 *
 * Each completed repetition gets range of motion, phase timing (tempo) and a
 * confidence verdict. Single-camera validity depends heavily on detection quality:
 * gating out low-confidence frames moved MediaPipe squat knee RMSE from 25.4° to
 * 8.8° (Yue et al., JMIR Rehabil Assist Technol 2026). Reps that fall below the
 * gate are therefore flagged and left out of session averages instead of being
 * reported with false precision.
 *
 * The same functions run live (RepTracker) and on recorded sessions (analyzeReps),
 * so a replay reports the numbers that were shown during the set.
 */

import { bilateralSide } from '../biomechanics/angle-calculations.mjs';
import { JOINTS } from '../biomechanics/landmark-indices.mjs';

/**
 * direction: whether the working phase drives the primary angle to a 'min'
 * (knee flexion) or a 'max' (sit-to-stand knee extension, shoulder elevation).
 * phases: labels for the motion toward and away from that extreme.
 */
export const REP_PROFILES = Object.freeze({
  squat: { joint: 'Knee', direction: 'min', phases: ['down', 'up'] },
  lunge: { joint: 'Knee', direction: 'min', phases: ['down', 'up'], pick: 'min' },
  sit_to_stand: { joint: 'Knee', direction: 'max', phases: ['rise', 'sit'] },
  shoulder_abduction: { joint: 'Shoulder', direction: 'max', phases: ['raise', 'lower'] },
  shoulder_flexion: { joint: 'Shoulder', direction: 'max', phases: ['raise', 'lower'] },
});

export const DEFAULT_QUALITY_GATE = 0.7;
const SETTLE_TIMEOUT_MS = 1500;
const MAX_LOW_FRACTION = 0.25;
const REST_TOLERANCE_DEG = 10;

const JOINT_POINTS = {
  Knee: { left: [JOINTS.leftHip, JOINTS.leftKnee, JOINTS.leftAnkle], right: [JOINTS.rightHip, JOINTS.rightKnee, JOINTS.rightAnkle] },
  Shoulder: { left: [JOINTS.leftElbow, JOINTS.leftShoulder, JOINTS.leftHip], right: [JOINTS.rightElbow, JOINTS.rightShoulder, JOINTS.rightHip] },
};

export function profileFor(exerciseId) {
  return REP_PROFILES[String(exerciseId || '').replace(/-/g, '_')] ?? null;
}

function visibilityOf(p) {
  if (!p) return 0;
  if (Array.isArray(p)) return p[3] ?? 1; // recorded compact form [x, y, z, vis]
  return Math.min(p.visibility ?? 1, p.presence ?? 1);
}

function sideConfidence(landmarks, indices) {
  if (!Array.isArray(landmarks)) return null;
  return indices.reduce((s, i) => s + visibilityOf(landmarks[i]), 0) / indices.length;
}

function asVisibilityLandmarks(landmarks) {
  if (!Array.isArray(landmarks) || !Array.isArray(landmarks[0] ?? null)) return landmarks;
  return landmarks.map(p => (Array.isArray(p) ? { visibility: p[3] ?? 1 } : p));
}

/**
 * Reduces one frame to the primary angle for the exercise plus its confidence.
 * @returns {{ t: number, angle: number|null, conf: number|null, reps: number }}
 */
export function sampleFrame(profile, { angles, landmarks, timestampMs, reps = 0 }) {
  const lm = asVisibilityLandmarks(landmarks);
  const l = angles?.[`left${profile.joint}`];
  const r = angles?.[`right${profile.joint}`];
  const pts = JOINT_POINTS[profile.joint];
  let angle = null;
  let conf = null;

  if (profile.pick === 'min') {
    // Lunge: the working (front) knee is the more flexed one.
    const candidates = [[l, pts.left], [r, pts.right]].filter(([a]) => Number.isFinite(a));
    if (candidates.length) {
      const [a, idx] = candidates.reduce((best, c) => (c[0] < best[0] ? c : best));
      angle = a;
      conf = sideConfidence(lm, idx);
    }
  } else {
    const side = bilateralSide(angles, lm, profile.joint);
    if (side === 'left' || side === 'right') {
      angle = side === 'left' ? l : r;
      conf = sideConfidence(lm, pts[side]);
    } else if (side === 'both') {
      angle = (l + r) / 2;
      const cl = sideConfidence(lm, pts.left);
      const cr = sideConfidence(lm, pts.right);
      conf = cl === null || cr === null ? null : (cl + cr) / 2;
    }
  }
  return { t: timestampMs, angle: Number.isFinite(angle) ? angle : null, conf, reps };
}

const round1 = v => Math.round(v * 10) / 10;

/**
 * Summarizes one repetition from its samples (in time order, rest to rest).
 * @returns {Object|null}
 */
export function summarizeRep(samples, profile, { gate = DEFAULT_QUALITY_GATE } = {}) {
  const valid = samples.map((s, i) => [s, i]).filter(([s]) => s.angle !== null);
  if (valid.length < 3) return null;
  const isMin = profile.direction === 'min';
  const [extSample, extIdx] = valid.reduce((best, cur) => (
    (isMin ? cur[0].angle < best[0].angle : cur[0].angle > best[0].angle) ? cur : best
  ));
  // Start posture = most extended (min-direction) / least elevated point before the
  // extreme; the return is judged against the same posture.
  const before = valid.filter(([, i]) => i <= extIdx).map(([s]) => s.angle);
  const startAngle = isMin ? Math.max(...before) : Math.min(...before);
  const atRest = a => (isMin ? a >= startAngle - REST_TOLERANCE_DEG : a <= startAngle + REST_TOLERANCE_DEG);

  // Movement starts when the angle last left rest before the extreme and ends when it
  // first returns after it, so pauses between reps don't inflate the tempo.
  let startIdx = valid[0][1];
  for (const [s, i] of valid) {
    if (i >= extIdx) break;
    if (atRest(s.angle)) startIdx = i;
  }
  let endIdx = valid[valid.length - 1][1];
  for (const [s, i] of valid) {
    if (i > extIdx && atRest(s.angle)) { endIdx = i; break; }
  }

  const window = samples.slice(startIdx, endIdx + 1);
  const low = window.filter(s => s.angle === null || s.conf === null || s.conf < gate).length;
  const lowFraction = window.length ? low / window.length : 1;
  const extremeConf = extSample.conf;
  const reliable = extremeConf !== null && extremeConf >= gate && lowFraction <= MAX_LOW_FRACTION;

  return {
    startMs: samples[startIdx].t,
    extremeMs: extSample.t,
    endMs: samples[endIdx].t,
    durationMs: Math.round(samples[endIdx].t - samples[startIdx].t),
    phase1Ms: Math.round(extSample.t - samples[startIdx].t),
    phase2Ms: Math.round(samples[endIdx].t - extSample.t),
    extremeAngle: round1(extSample.angle),
    startAngle: round1(startAngle),
    romDeg: round1(Math.abs(startAngle - extSample.angle)),
    extremeConfidence: extremeConf === null ? null : Math.round(extremeConf * 100) / 100,
    lowConfidenceFraction: Math.round(lowFraction * 100) / 100,
    quality: reliable ? 'ok' : 'low-confidence',
  };
}

/**
 * Exercise counters fire before the movement has fully returned (e.g. a squat counts
 * once the knee passes 155°). A rep's window is therefore extended until the angle
 * settles back near its start posture, the next rep is counted, or a timeout passes.
 * @returns {boolean} whether the rep that counted at samples[countIdx] has settled at samples[i]
 */
function hasSettled(samples, segStart, countIdx, i, profile) {
  const s = samples[i];
  if (i > countIdx && s.reps > samples[countIdx].reps) return true;
  if (s.t - samples[countIdx].t >= SETTLE_TIMEOUT_MS) return true;
  if (s.angle === null) return false;
  const angles = samples.slice(segStart, countIdx + 1).map(x => x.angle).filter(a => a !== null);
  if (!angles.length) return true;
  const isMin = profile.direction === 'min';
  const startAngle = isMin ? Math.max(...angles) : Math.min(...angles);
  return isMin ? s.angle >= startAngle - REST_TOLERANCE_DEG / 2 : s.angle <= startAngle + REST_TOLERANCE_DEG / 2;
}

/**
 * Splits samples at rep-counter increments (extended until settled, see hasSettled)
 * and summarizes each completed rep.
 * @param {Array<Object>} frames recorded or live frames { timeMs|timestampMs, angles, landmarks, reps }
 */
export function analyzeReps(frames, exerciseId, options = {}) {
  const profile = profileFor(exerciseId);
  if (!profile || !Array.isArray(frames) || frames.length === 0) return [];
  const tracker = new RepTracker(options);
  tracker.reset(exerciseId);
  for (const f of frames) {
    tracker.update({ angles: f.angles, landmarks: f.landmarks, timestampMs: f.timestampMs ?? f.timeMs, reps: f.reps ?? 0 });
  }
  tracker.flush();
  return tracker.reps;
}

/** Session-level aggregates; averages use reliable reps only. */
export function summarizeSession(reps) {
  const ok = reps.filter(r => r.quality === 'ok');
  const mean = arr => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
  const avgRom = mean(ok.map(r => r.romDeg));
  const sdRom = ok.length > 1
    ? Math.sqrt(ok.reduce((s, r) => s + (r.romDeg - avgRom) ** 2, 0) / (ok.length - 1))
    : null;
  // Cadence from rep-start to rep-start intervals (excludes the last rep's own duration).
  const span = reps.length > 1 ? reps[reps.length - 1].startMs - reps[0].startMs : null;
  return {
    total: reps.length,
    reliable: ok.length,
    lowConfidence: reps.length - ok.length,
    avgRomDeg: avgRom === null ? null : round1(avgRom),
    romCvPct: sdRom === null || !avgRom ? null : round1((sdRom / avgRom) * 100),
    avgPhase1Ms: ok.length ? Math.round(mean(ok.map(r => r.phase1Ms))) : null,
    avgPhase2Ms: ok.length ? Math.round(mean(ok.map(r => r.phase2Ms))) : null,
    repsPerMin: span && span > 0 ? round1(((reps.length - 1) / span) * 60000) : null,
  };
}

/**
 * Incremental analysis: feed every analysed frame. A rep is reported once its
 * movement has settled after the counter increments (usually a fraction of a second).
 * Buffers one small sample per frame (bounded).
 */
export class RepTracker {
  constructor({ gate = DEFAULT_QUALITY_GATE, maxSamples = 1800 } = {}) {
    this.gate = gate;
    this.maxSamples = maxSamples;
    this.reset(null);
  }

  reset(exerciseId = this.exerciseId) {
    this.exerciseId = exerciseId;
    this.profile = profileFor(exerciseId);
    this.samples = [];
    this.pendingCountIdx = null;
    this.reps = [];
  }

  _finalize(endIdx) {
    const countIdx = this.pendingCountIdx;
    const index = this.samples[countIdx].reps;
    const rep = summarizeRep(this.samples.slice(0, endIdx + 1), this.profile, { gate: this.gate });
    this.samples = this.samples.slice(endIdx);
    this.pendingCountIdx = null;
    if (!rep) return null;
    const entry = { index, ...rep };
    this.reps.push(entry);
    return entry;
  }

  /** @returns {Object|null} the rep that just finished settling, if any */
  update({ angles, landmarks, timestampMs, reps }) {
    if (!this.profile) return null;
    const sample = sampleFrame(this.profile, { angles, landmarks, timestampMs, reps: reps ?? 0 });
    const prev = this.samples[this.samples.length - 1];
    if (prev && sample.reps < prev.reps) {
      this.reset(); // counter was reset
    }
    this.samples.push(sample);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
      if (this.pendingCountIdx !== null) this.pendingCountIdx = Math.max(0, this.pendingCountIdx - 1);
    }
    const i = this.samples.length - 1;

    let finished = null;
    if (this.pendingCountIdx !== null && hasSettled(this.samples, 0, this.pendingCountIdx, i, this.profile)) {
      // If the next rep was counted on this very frame, close at the previous frame.
      const nextCounted = this.samples[i].reps > this.samples[this.pendingCountIdx].reps;
      finished = this._finalize(nextCounted ? i - 1 : i);
    }
    const last = this.samples.length - 1;
    if (last > 0 && this.samples[last].reps > this.samples[last - 1].reps && this.pendingCountIdx === null) {
      this.pendingCountIdx = last;
    }
    return finished;
  }

  /** Finalizes a pending rep at the end of a set or recording. */
  flush() {
    if (this.pendingCountIdx === null || !this.samples.length) return null;
    return this._finalize(this.samples.length - 1);
  }

  summary() {
    return summarizeSession(this.reps);
  }
}
