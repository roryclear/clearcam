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
  assert.match(csv[0], /^rep,start_ms,end_ms,duration_ms,down_ms,up_ms,start_angle,extreme_angle,rom_deg/);
  assert.equal(csv[1].split(',').length, csv[0].split(',').length);
});
