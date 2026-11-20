import test from 'node:test';
import assert from 'node:assert/strict';
import { generateSquatSequence } from '../bench/synthetic-pose.mjs';
import { PoseFilter } from '../src/filters/pose-filter.mjs';
import { getJointAngles } from '../src/biomechanics/angle-calculations.mjs';
import { SquatExercise } from '../src/exercises/squat-exercise.mjs';
import { analyzeReps, summarizeSession, RepTracker, summarizeRep, REP_PROFILES } from '../src/exercises/rep-analysis.mjs';

function squatFrames({ seed = 1, lowVisReps = [] } = {}) {
  const seq = generateSquatSequence({ seed, yawDeg: 0 });
  const pf = new PoseFilter();
  const squat = new SquatExercise();
  let state = squat.createState();
  return seq.frames.map(f => {
    const rep = Math.floor((f.timeMs - 1500) / 2400) + 1;
    const lm = f.landmarks.map(p => ({ ...p }));
    if (lowVisReps.includes(rep)) for (const i of [23, 24, 25, 26, 27, 28]) lm[i].visibility = 0.5;
    const smooth = pf.filter(lm, f.timeMs);
    const angles = getJointAngles(smooth, f.width, f.height, 0.3);
    state = squat.update(state, { angles, landmarks: smooth, timestampMs: f.timeMs, trackingQuality: 'good' });
    return { timeMs: f.timeMs, angles, landmarks: smooth, reps: state.reps };
  });
}

test('analyzeReps measures ROM and tempo of synthetic squats', () => {
  const reps = analyzeReps(squatFrames(), 'squat');
  assert.equal(reps.length, 6);
  for (const r of reps) {
    assert.ok(r.romDeg > 75 && r.romDeg < 95, `ROM ${r.romDeg}`); // true ROM 85°
    assert.ok(r.extremeAngle > 88 && r.extremeAngle < 104, `bottom ${r.extremeAngle}`); // true 95°
    assert.ok(r.phase1Ms > 700 && r.phase1Ms < 1300, `down ${r.phase1Ms}`); // ~0.93 s
    assert.ok(r.phase2Ms > 700 && r.phase2Ms < 1300, `up ${r.phase2Ms}`);
    assert.equal(r.quality, 'ok');
  }
  const s = summarizeSession(reps);
  assert.equal(s.reliable, 6);
  assert.ok(s.romCvPct < 10, `ROM CV ${s.romCvPct}`);
  assert.ok(Math.abs(s.repsPerMin - 25) < 3, `cadence ${s.repsPerMin}`); // 2.4 s cycle
});

test('reps tracked through low-visibility landmarks are flagged and excluded from averages', () => {
  const reps = analyzeReps(squatFrames({ lowVisReps: [3] }), 'squat');
  assert.equal(reps.length, 6);
  assert.equal(reps[2].quality, 'low-confidence');
  assert.equal(reps.filter(r => r.quality === 'ok').length, 5);
  const s = summarizeSession(reps);
  assert.equal(s.lowConfidence, 1);
  assert.equal(s.reliable, 5);
});

test('RepTracker live results match analyzeReps on the same frames', () => {
  const frames = squatFrames({ seed: 2 });
  const tracker = new RepTracker();
  tracker.reset('squat');
  const live = [];
  for (const f of frames) {
    const rep = tracker.update({ angles: f.angles, landmarks: f.landmarks, timestampMs: f.timeMs, reps: f.reps });
    if (rep) live.push(rep);
  }
  const flushed = tracker.flush();
  if (flushed) live.push(flushed);
  assert.deepEqual(live, analyzeReps(frames, 'squat'));
  assert.equal(tracker.summary().total, 6);
});

test('analyzeReps works on recorded compact landmarks and handles max-direction exercises', () => {
  // Shoulder raise 20° -> 160° -> 20°, recorded format: landmarks as [x, y, z, vis].
  const frames = [];
  for (let t = 0, i = 0; t <= 3000; t += 50, i++) {
    const phase = t < 500 ? 0 : t > 2500 ? 1 : (t - 500) / 2000;
    const a = 20 + 140 * Math.sin(Math.PI * phase);
    frames.push({
      timeMs: t,
      angles: { leftShoulder: a, rightShoulder: a },
      landmarks: Array.from({ length: 33 }, () => [0.5, 0.5, 0, 0.9]),
      reps: t >= 2800 ? 1 : 0,
    });
  }
  const [rep] = analyzeReps(frames, 'shoulder_abduction');
  assert.ok(Math.abs(rep.extremeAngle - 160) < 1);
  assert.ok(Math.abs(rep.romDeg - 140) < 1);
  assert.ok(Math.abs(rep.phase1Ms - rep.phase2Ms) < 150, 'symmetric raise/lower');
  assert.equal(rep.quality, 'ok');
  assert.equal(analyzeReps(frames, 'single_leg_balance').length, 0, 'hold exercises have no reps');
});

test('summarizeRep returns null without enough data and exposes profiles for rep exercises', () => {
  assert.equal(summarizeRep([{ t: 0, angle: null, conf: null }], REP_PROFILES.squat), null);
  assert.deepEqual(Object.keys(REP_PROFILES).sort(), ['lunge', 'shoulder_abduction', 'shoulder_flexion', 'sit_to_stand', 'squat']);
});

test('Recorded sessions round-trip through SessionRecorder and export a per-rep CSV', async () => {
  const { SessionRecorder } = await import('../src/sessions/session-recorder.mjs');
  const { exportRepsToCSV } = await import('../src/sessions/export-helpers.mjs');
  const frames = squatFrames({ seed: 3 });
  const recorder = new SessionRecorder({ MediaRecorderClass: null });
  const realNow = performance.now.bind(performance);
  performance.now = () => 0;
  try {
    recorder.start({ exerciseId: 'squat' });
  } finally {
    performance.now = realNow;
  }
  for (const f of frames) recorder.recordFrame({ landmarks: f.landmarks, angles: f.angles, timestampMs: f.timeMs, reps: f.reps });
  const session = recorder.stop();
  const reps = analyzeReps(session.frames, session.exerciseId);
  assert.equal(reps.length, 6);
  assert.ok(reps.every(r => r.quality === 'ok'));
  const csv = exportRepsToCSV(reps, ['down', 'up']).trim().split('\n');
  assert.equal(csv.length, 7);
  assert.match(csv[0], /^set,rep,start_ms,end_ms,duration_ms,down_ms,up_ms,start_angle,extreme_angle,rom_deg/);
  assert.equal(csv[1].split(',').length, csv[0].split(',').length);
});

// Devin review on #14: regression tests built from the reported examples.
function scripted(segments, { dt = 33, reps = () => 0, status = () => 'active', vis = 0.95, presence = 0.99, joint = 'Knee', other = null } = {}) {
  const frames = [];
  let t = 0;
  let prev = segments[0][0];
  for (const [angle, ms] of segments) {
    for (let k = 0; k < ms; k += dt, t += dt) {
      const a = prev + (angle - prev) * Math.min(1, (k + dt) / Math.max(dt, Math.min(ms, 300)));
      const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: vis, presence }));
      const angles = joint === 'Knee'
        ? { leftKnee: a, rightKnee: a }
        : { leftShoulder: a, rightShoulder: other ?? a };
      frames.push({ timeMs: t, angles, landmarks: lm, reps: reps(t, a), status: status(t) });
    }
    prev = angle;
  }
  return frames;
}

test('One-arm shoulder raises measure the working arm, not the average of both', () => {
  const counted = { n: 0, up: false };
  const repsFn = (t, a) => { if (a > 140) counted.up = true; if (counted.up && a < 50) { counted.n = 1; } return counted.n; };
  for (const exercise of ['shoulder_abduction', 'shoulder_flexion']) {
    counted.n = 0; counted.up = false;
    const frames = scripted([[20, 500], [160, 1000], [20, 1000], [20, 800]], { joint: 'Shoulder', other: 20, reps: repsFn });
    const [rep] = analyzeReps(frames, exercise);
    assert.ok(Math.abs(rep.extremeAngle - 160) < 2, `${exercise} peak ${rep.extremeAngle}`);
    assert.ok(Math.abs(rep.romDeg - 140) < 2, `${exercise} ROM ${rep.romDeg}`);
  }
});

test('Back-to-back squats that never return to full standing get separate windows', () => {
  // 180 -> 100 (hold) -> 160 -> 80 (hold) -> 160: counter fires on each return past 155.
  let count = 0; let down = false;
  const repsFn = (t, a) => { if (a <= 115) down = true; if (down && a >= 155) { count++; down = false; } return count; };
  const frames = scripted([[180, 500], [100, 800], [160, 400], [80, 800], [160, 600], [160, 1800]], { reps: repsFn });
  const reps = analyzeReps(frames, 'squat');
  assert.equal(reps.length, 2);
  assert.ok(Math.abs(reps[0].extremeAngle - 100) < 2, `rep 1 bottom ${reps[0].extremeAngle}`);
  assert.ok(Math.abs(reps[0].romDeg - 80) < 3, `rep 1 ROM ${reps[0].romDeg}`);
  assert.ok(Math.abs(reps[1].extremeAngle - 80) < 2, `rep 2 bottom ${reps[1].extremeAngle}`);
  assert.ok(Math.abs(reps[1].romDeg - 80) < 3, `rep 2 ROM ${reps[1].romDeg}`);
});

test('Tracking loss discards the aborted movement instead of lending it to the next rep', () => {
  // 180 -> 70, tracking lost (rep aborted, not counted), back up, then a real squat to 100.
  let count = 0; let down = false;
  const lostAt = [1300, 1340];
  const statusFn = t => (t >= lostAt[0] && t < lostAt[1] ? 'tracking-lost' : 'active');
  const repsFn = (t, a) => {
    if (t >= lostAt[0] && t < lostAt[1]) { down = false; return count; }
    if (t > 2000 && a <= 115) down = true;
    if (down && a >= 155) { count++; down = false; }
    return count;
  };
  const frames = scripted([[180, 500], [70, 800], [70, 100], [180, 600], [180, 400], [100, 800], [180, 600], [180, 1800]], { reps: repsFn, status: statusFn });
  const reps = analyzeReps(frames, 'squat');
  assert.equal(reps.length, 1);
  assert.ok(Math.abs(reps[0].extremeAngle - 100) < 2, `bottom ${reps[0].extremeAngle}`);
  assert.ok(Math.abs(reps[0].romDeg - 80) < 3, `ROM ${reps[0].romDeg}`);
});

test('Recorded sessions keep the live confidence (min of visibility and presence)', async () => {
  const { SessionRecorder } = await import('../src/sessions/session-recorder.mjs');
  let count = 0; let up = false;
  const repsFn = (t, a) => { if (a > 140) up = true; if (up && a < 50) count = 1; return count; };
  const frames = scripted([[20, 500], [160, 1000], [20, 1000], [20, 800]], { joint: 'Shoulder', reps: repsFn, vis: 0.95, presence: 0.6 });
  const live = analyzeReps(frames, 'shoulder_abduction');
  assert.equal(live[0].quality, 'low-confidence');

  const recorder = new SessionRecorder({ MediaRecorderClass: null });
  const realNow = performance.now.bind(performance);
  performance.now = () => 0;
  try { recorder.start({ exerciseId: 'shoulder_abduction' }); } finally { performance.now = realNow; }
  for (const f of frames) recorder.recordFrame({ landmarks: f.landmarks, angles: f.angles, timestampMs: f.timeMs, reps: f.reps, status: f.status });
  const replay = analyzeReps(recorder.stop().frames, 'shoulder_abduction');
  assert.equal(replay[0].quality, 'low-confidence', 'replay must not upgrade a rejected rep');
});

test('A counter reset mid-recording starts a new set instead of discarding earlier reps', () => {
  const set1 = squatFrames({ seed: 1 });
  const offset = set1[set1.length - 1].timeMs + 33;
  const set2 = squatFrames({ seed: 2 }).map(f => ({ ...f, timeMs: f.timeMs + offset }));
  const reps = analyzeReps([...set1, ...set2], 'squat');
  assert.equal(reps.length, 12);
  assert.deepEqual(reps.map(r => r.set), [...Array(6).fill(1), ...Array(6).fill(2)]);
  assert.deepEqual(reps.slice(6).map(r => r.index), [1, 2, 3, 4, 5, 6]);
  assert.ok(reps.every(r => r.quality === 'ok' && r.romDeg > 75));
});

test('A rep counted while recovering from tracking loss does not crash and is measured', () => {
  // Devin's example: tracking lost, squat reaches 100°, returns through 157°/156° (count
  // fires at 156°), then a 158° frame. Previously this cleared the buffer under a pending rep.
  // Tracking resumes with the person already at the bottom (100°), so recovery is still
  // trimming toward the rest posture when the counter fires at 156°.
  const angles = [180, 180, 150, 120, null, 100, 100, 110, 130, 150, 157, 156, 158, 170, 180, 180, 180, 180];
  const statusAt = i => (angles[i] === null ? 'tracking-lost' : 'active');
  const repsAt = i => (i >= 11 ? 1 : 0);
  const frames = angles.map((a, i) => ({
    timeMs: i * 100,
    angles: { leftKnee: a, rightKnee: a },
    landmarks: Array.from({ length: 33 }, () => ({ visibility: 0.95, presence: 0.99 })),
    reps: repsAt(i),
    status: statusAt(i),
  }));
  const tracker = new RepTracker();
  tracker.reset('squat');
  const live = [];
  assert.doesNotThrow(() => {
    for (const f of frames) {
      const r = tracker.update({ angles: f.angles, landmarks: f.landmarks, timestampMs: f.timeMs, reps: f.reps, status: f.status });
      if (r) live.push(r);
    }
    const r = tracker.flush();
    if (r) live.push(r);
  });
  const offline = analyzeReps(frames, 'squat');
  assert.equal(offline.length, 1);
  assert.deepEqual(live, offline);
  // The descent happened while tracking was lost: flagged, kept out of averages.
  assert.equal(offline[0].quality, 'low-confidence');
  assert.equal(offline[0].partial, true);
});

test('A discarded recovery rep does not taint the next fully observed rep', () => {
  // Devin's sequence on #14, run through the real squat counter.
  const seq = [180, 180, 150, 120, null, 100, 100, 110, 130, 150, 157, 156, 145, 120, 100, 100, 110, 130, 150, 157, 160, 170, 180, 180, 180, 180];
  const squat = new SquatExercise();
  let state = squat.createState();
  const frames = seq.map((a, i) => {
    const angles = { leftKnee: a, rightKnee: a };
    const landmarks = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.95, presence: 0.99 }));
    state = squat.update(state, { angles, timestampMs: i * 100, trackingQuality: a === null ? 'lost' : 'good' });
    return { timeMs: i * 100, angles, landmarks, reps: state.reps, status: state.status };
  });
  const reps = analyzeReps(frames, 'squat');
  const last = reps[reps.length - 1];
  assert.ok(last, 'second rep summarized');
  assert.equal(last.index, 2);
  assert.equal(last.quality, 'ok');
  assert.equal(last.partial, undefined);
});
