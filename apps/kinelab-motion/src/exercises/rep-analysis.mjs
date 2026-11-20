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
  // Until the seated start is confirmed the person is still setting up (often standing);
  // those frames must not become the first rep's extreme.
  sit_to_stand: { joint: 'Knee', direction: 'max', phases: ['rise', 'sit'], setupStatuses: ['waiting', 'sit-to-begin'] },
  // Shoulder counters follow the most elevated arm, so one-arm raises count; measure that arm.
  shoulder_abduction: { joint: 'Shoulder', direction: 'max', phases: ['raise', 'lower'], pick: 'max' },
  shoulder_flexion: { joint: 'Shoulder', direction: 'max', phases: ['raise', 'lower'], pick: 'max' },
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
  // Recorded frames store [x, y, z, conf] and may hold nulls, even at index 0.
  if (!Array.isArray(landmarks) || !landmarks.some(p => Array.isArray(p))) return landmarks;
  return landmarks.map(p => (Array.isArray(p) ? { visibility: p[3] ?? 1 } : p));
}

/**
 * Reduces one frame to the primary angle for the exercise plus its confidence.
 * @returns {{ t: number, angle: number|null, conf: number|null, reps: number }}
 */
export function sampleFrame(profile, { angles, landmarks, timestampMs, reps = 0, status = null }) {
  const lm = asVisibilityLandmarks(landmarks);
  const l = angles?.[`left${profile.joint}`];
  const r = angles?.[`right${profile.joint}`];
  const pts = JOINT_POINTS[profile.joint];
  let angle = null;
  let conf = null;

  if (profile.pick === 'min' || profile.pick === 'max') {
    // Working limb: the more flexed knee (lunge front leg) or the more elevated arm,
    // matching what the exercise counter follows; confidence comes from that limb.
    const better = profile.pick === 'min' ? (a, b) => a < b : (a, b) => a > b;
    const candidates = [[l, pts.left], [r, pts.right]].filter(([a]) => Number.isFinite(a));
    if (candidates.length) {
      const [a, idx] = candidates.reduce((best, c) => (better(c[0], best[0]) ? c : best));
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
  return {
    t: timestampMs,
    angle: Number.isFinite(angle) ? angle : null,
    conf,
    reps,
    lost: status === 'tracking-lost',
    setup: status !== null && (profile.setupStatuses ?? []).includes(status),
  };
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
 * once the knee passes 155°). After a count, the rep stays open while the limb keeps
 * returning and closes at the return's turning point: when the angle is back near the
 * start posture, when it reverses by REST_TOLERANCE_DEG (the next rep has begun), when
 * the next rep is counted, or after a timeout. Closing at the turning point keeps the
 * next rep's start and descent in the next rep's window.
 */
function returnPeakIndex(samples, countIdx, i, isMin) {
  let peak = countIdx;
  for (let k = countIdx; k <= i; k++) {
    const a = samples[k].angle;
    if (a === null) continue;
    const pa = samples[peak].angle;
    if (pa === null || (isMin ? a > pa : a < pa)) peak = k;
  }
  return peak;
}

function settleIndex(samples, countIdx, i, profile) {
  if (!samples[countIdx] || !samples[i]) return i >= 0 ? i : null; // defensive: never throw in the frame loop
  const isMin = profile.direction === 'min';
  const peak = returnPeakIndex(samples, countIdx, i, isMin);
  const s = samples[i];
  if (i > countIdx && s.reps > samples[countIdx].reps) return peak;
  if (s.t - samples[countIdx].t >= SETTLE_TIMEOUT_MS) return peak;
  if (s.angle === null) return null;
  const pa = samples[peak].angle;
  if (pa !== null && (isMin ? pa - s.angle : s.angle - pa) >= REST_TOLERANCE_DEG) return peak; // next rep began
  const angles = samples.slice(0, countIdx + 1).map(x => x.angle).filter(a => a !== null);
  if (!angles.length) return i;
  const startAngle = isMin ? Math.max(...angles) : Math.min(...angles);
  const back = isMin ? s.angle >= startAngle - REST_TOLERANCE_DEG / 2 : s.angle <= startAngle + REST_TOLERANCE_DEG / 2;
  return back ? i : null;
}

/**
 * Index where the counted movement started: walking back from the count, skip the
 * arrival at rest (counters that fire on return, e.g. sit-to-stand), then the movement
 * itself, and stop at the last rest frame before it. Earlier movements in the window
 * (an aborted raise, setup) are excluded. Returns { idx, found }.
 */
function movementStart(samples, countIdx, endIdx, profile) {
  const isMin = profile.direction === 'min';
  let rest = null;
  for (let k = 0; k <= endIdx; k++) {
    const a = samples[k].angle;
    if (a !== null && (rest === null || (isMin ? a > rest : a < rest))) rest = a;
  }
  if (rest === null) return { idx: 0, found: false };
  const atRest = a => (isMin ? a >= rest - REST_TOLERANCE_DEG : a <= rest + REST_TOLERANCE_DEG);
  let k = countIdx;
  while (k >= 0 && (samples[k].angle === null || atRest(samples[k].angle))) k--;
  while (k >= 0 && (samples[k].angle === null || !atRest(samples[k].angle))) k--;
  if (k < 0) return { idx: 0, found: false };
  // Keep the whole rest plateau so the start posture is measured at its most rested.
  while (k > 0 && (samples[k - 1].angle === null || atRest(samples[k - 1].angle))) k--;
  return { idx: k, found: true };
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
    tracker.update({ angles: f.angles, landmarks: f.landmarks, timestampMs: f.timestampMs ?? f.timeMs, reps: f.reps ?? 0, status: f.status });
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
  // Cadence from rep-start to rep-start intervals within each set (excludes the last
  // rep's own duration and rests between sets).
  let intervals = 0;
  let span = 0;
  for (let k = 1; k < reps.length; k++) {
    if ((reps[k].set ?? 1) !== (reps[k - 1].set ?? 1)) continue;
    intervals++;
    span += reps[k].startMs - reps[k - 1].startMs;
  }
  return {
    total: reps.length,
    reliable: ok.length,
    lowConfidence: reps.length - ok.length,
    avgRomDeg: avgRom === null ? null : round1(avgRom),
    romCvPct: sdRom === null || !avgRom ? null : round1((sdRom / avgRom) * 100),
    avgPhase1Ms: ok.length ? Math.round(mean(ok.map(r => r.phase1Ms))) : null,
    avgPhase2Ms: ok.length ? Math.round(mean(ok.map(r => r.phase2Ms))) : null,
    repsPerMin: intervals && span > 0 ? round1((intervals / span) * 60000) : null,
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
    this.postLoss = false;
    this.partialRep = false;
    // The current window may begin mid-movement (start of tracking, a new set, or
    // history evicted from the buffer).
    this.openStart = true;
    this.set = 1;
    this.lastReps = null;
    this.reps = [];
  }

  /** Clears in-progress segmentation but keeps completed reps (counter reset mid-recording). */
  _startNewSet() {
    this.samples = [];
    this.pendingCountIdx = null;
    this.postLoss = false;
    this.partialRep = false;
    this.openStart = true;
    this.set += 1;
  }

  _finalize(endIdx) {
    const countIdx = this.pendingCountIdx;
    const index = this.samples[countIdx].reps;
    // Consume the recovery flag for this rep whether or not it yields a summary, so it
    // can never carry over to the next, fully observed rep.
    let partial = this.partialRep;
    this.partialRep = false;
    const start = movementStart(this.samples, countIdx, endIdx, this.profile);
    // No rest posture before the counted movement in a window that may have begun
    // mid-movement: its start was not observed, so ROM and tempo are incomplete.
    if (!start.found && this.openStart) partial = true;
    const rep = summarizeRep(this.samples.slice(start.idx, endIdx + 1), this.profile, { gate: this.gate });
    this.samples = this.samples.slice(endIdx);
    this.pendingCountIdx = null;
    this.openStart = false;
    if (!rep) return null;
    const entry = { set: this.set, index, ...rep };
    if (partial) {
      entry.quality = 'low-confidence';
      entry.partial = true;
    }
    this.reps.push(entry);
    return entry;
  }

  /** @returns {Object|null} the rep that just finished settling, if any */
  update({ angles, landmarks, timestampMs, reps, status = null }) {
    if (!this.profile) return null;
    const sample = sampleFrame(this.profile, { angles, landmarks, timestampMs, reps: reps ?? 0, status });
    const resetDetected = this.lastReps !== null && sample.reps < this.lastReps;
    this.lastReps = sample.reps;
    let finished = null;
    // Compared with the last counter value seen, even if the window was just cleared.
    if (resetDetected) {
      // Counter reset (e.g. "Reset count" during a recording): keep earlier sets.
      if (this.pendingCountIdx !== null) finished = this._finalize(this.samples.length - 1);
      this._startNewSet();
    }

    if (sample.lost) {
      // The exercise aborted its movement: close a counted rep, drop any unfinished one
      // so it cannot lend its depth or timing to the next counted rep.
      if (this.pendingCountIdx !== null) finished = this._finalize(this.samples.length - 1);
      this.samples = [];
      this.postLoss = true;
      return finished;
    }

    if (sample.setup && this.pendingCountIdx === null) {
      // Setup posture before the start position is confirmed: not part of any rep.
      this.samples = [];
      return finished;
    }

    if (this.postLoss && (this.pendingCountIdx !== null || sample.reps > (this.samples[this.samples.length - 1]?.reps ?? sample.reps))) {
      // A rep was counted during recovery: stop trimming (that would also invalidate
      // pendingCountIdx). Its movement was only partly observed, so it is flagged.
      this.postLoss = false;
      this.partialRep = true;
    }
    if (this.postLoss && sample.angle !== null) {
      // After an abort the person may still be mid-movement (e.g. at the bottom).
      // Restart the window at the most rested posture seen since, and resume normal
      // tracking once the next movement starts away from it.
      const isMin = this.profile.direction === 'min';
      const rest = this.samples.reduce((best, x) => (x.angle !== null && (best === null || (isMin ? x.angle > best : x.angle < best)) ? x.angle : best), null);
      if (rest === null || (isMin ? sample.angle >= rest : sample.angle <= rest)) {
        this.samples = [];
      } else if (isMin ? rest - sample.angle >= REST_TOLERANCE_DEG : sample.angle - rest >= REST_TOLERANCE_DEG) {
        this.postLoss = false;
      }
    }

    this.samples.push(sample);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
      this.openStart = true;
      if (this.pendingCountIdx !== null) this.pendingCountIdx = Math.max(0, this.pendingCountIdx - 1);
    }
    const i = this.samples.length - 1;

    if (this.pendingCountIdx !== null) {
      const end = settleIndex(this.samples, this.pendingCountIdx, i, this.profile);
      if (end !== null) finished = this._finalize(end);
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
