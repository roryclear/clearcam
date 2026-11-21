/**
 * Accuracy benchmark: runs synthetic squats with known ground truth through
 * candidate angle pipelines and reports knee-angle error, jitter and rep counts.
 * Usage: node bench/accuracy-bench.mjs [--json]
 */
import { generateSquatSequence } from './synthetic-pose.mjs';
import { PoseFilter } from '../src/filters/pose-filter.mjs';
import { KinematicBoneFilter } from '../src/filters/kinematic-filter.mjs';
import { WorldPoseFilter } from '../src/filters/world-pose-filter.mjs';
import { getJointAngles, getJointAngles3D, dominantKneeAngle } from '../src/biomechanics/angle-calculations.mjs';
import { SquatExercise } from '../src/exercises/squat-exercise.mjs';
import * as AC from '../src/biomechanics/angle-calculations.mjs';
import { ViewOrientationEstimator, blendAnglesByView } from '../src/biomechanics/view-orientation.mjs';
import { smoothSeriesOffline } from '../src/filters/zero-phase.mjs';

export const PIPELINES = {
  'current (3D raw + bone filter, avg legs)': { dim: '3d', world: 'off', bone: true, side: 'average' },
  '2D filtered, avg legs': { dim: '2d', world: 'off', bone: false, side: 'average' },
  '2D filtered, best leg': { dim: '2d', world: 'off', bone: false, side: 'best' },
  '3D raw + bone, best leg': { dim: '3d', world: 'off', bone: true, side: 'best' },
  '3D smoothed, best leg': { dim: '3d', world: 'medium', bone: false, side: 'best' },
  '3D smoothed + bone, best leg': { dim: '3d', world: 'medium', bone: true, side: 'best' },
  'view-blend (yaw EMA), best leg': { dim: 'view', world: 'medium', bone: true, side: 'best' }, // = app pipeline
  // Recorded-session analysis: angles from unfiltered landmarks, then zero-phase smoothing.
  'offline: raw view-blend + filtfilt 6 Hz': { dim: 'view', world: 'off', bone: true, side: 'best', raw: true, offlineHz: 6 },
  'offline: raw view-blend + filtfilt 4 Hz': { dim: 'view', world: 'off', bone: true, side: 'best', raw: true, offlineHz: 4 },
  'offline: live angles + filtfilt 6 Hz': { dim: 'view', world: 'medium', bone: true, side: 'best', offlineHz: 6 },
};

/** Mean |error| of each rep's deepest knee angle (truth regions below 150°). */
function peakError(times, est, truth) {
  const errs = [];
  let i = 0;
  while (i < truth.length) {
    if (!(truth[i] < 150)) { i++; continue; }
    let j = i;
    while (j + 1 < truth.length && truth[j + 1] < 150) j++;
    const tMin = Math.min(...truth.slice(i, j + 1));
    const e = est.slice(i, j + 1).filter(Number.isFinite);
    if (e.length) errs.push(Math.abs(Math.min(...e) - tMin));
    i = j + 1;
  }
  return errs.length ? errs.reduce((s, v) => s + v, 0) / errs.length : NaN;
}

function kneeFrom(angles, landmarks, side) {
  if (side === 'best') return AC.selectBilateralAngle(angles, landmarks, 'Knee');
  return dominantKneeAngle(angles);
}

export function runPipeline(seq, cfg) {
  const pf = new PoseFilter();
  const bone = new KinematicBoneFilter();
  const wf = new WorldPoseFilter({ preset: cfg.world === 'off' ? 'off' : cfg.world });
  const view = new ViewOrientationEstimator();
  const squat = new SquatExercise();
  let state = squat.createState();
  const errs = [];
  const diffs = [];
  let prev = null;
  let covered = 0;
  const times = [];
  const kneeSeries = [];
  const truthSeries = [];
  for (const f of seq.frames) {
    const lm = cfg.raw ? f.landmarks : pf.filter(f.landmarks, f.timeMs);
    let world = f.worldLandmarks;
    if (cfg.world !== 'off') world = wf.filter(world, f.timeMs);
    if (cfg.bone) world = bone.filter(world, f.timeMs);
    const a2 = getJointAngles(lm, f.width, f.height);
    const a3 = getJointAngles3D(world);
    let angles;
    if (cfg.dim === '2d') angles = a2;
    else if (cfg.dim === '3d') angles = { ...a2, ...Object.fromEntries(Object.entries(a3).filter(([, v]) => Number.isFinite(v))) };
    else angles = blendAnglesByView(a2, a3, view.update(world, f.timeMs));
    const knee = kneeFrom(angles, lm, cfg.side);
    times.push(f.timeMs);
    kneeSeries.push(Number.isFinite(knee) ? knee : null);
    truthSeries.push(seq.nearSide === 'left' ? f.truth.leftKnee : f.truth.rightKnee);
    if (cfg.offlineHz) continue; // scored after offline smoothing below
    if (Number.isFinite(knee)) {
      covered++;
      const truth = seq.nearSide === 'left' ? f.truth.leftKnee : f.truth.rightKnee;
      errs.push(knee - truth);
      if (prev !== null && f.truth.leftKnee > 179) diffs.push(knee - prev);
      prev = knee;
    } else prev = null;
    state = squat.update(state, { angles: { leftKnee: knee, rightKnee: knee }, landmarks: cfg.depthCue === false ? null : lm, timestampMs: f.timeMs, trackingQuality: 'good' });
  }
  let est = kneeSeries;
  if (cfg.offlineHz) {
    est = smoothSeriesOffline(times, kneeSeries, { cutoffHz: cfg.offlineHz });
    for (let i = 0; i < est.length; i++) {
      const knee = est[i];
      if (Number.isFinite(knee)) {
        covered++;
        errs.push(knee - truthSeries[i]);
        if (prev !== null && truthSeries[i] > 179) diffs.push(knee - prev);
        prev = knee;
      } else prev = null;
      state = squat.update(state, { angles: { leftKnee: knee, rightKnee: knee }, landmarks: null, timestampMs: times[i], trackingQuality: 'good' });
    }
  }
  const mae = errs.reduce((s, e) => s + Math.abs(e), 0) / errs.length;
  const rmse = Math.sqrt(errs.reduce((s, e) => s + e * e, 0) / errs.length);
  const bias = errs.reduce((s, e) => s + e, 0) / errs.length;
  const jitter = Math.sqrt(diffs.reduce((s, d) => s + d * d, 0) / Math.max(1, diffs.length));
  return { mae, rmse, bias, jitter, peakErr: peakError(times, est, truthSeries), reps: state.reps, coverage: covered / seq.frames.length };
}

export function runSuite({ yaws = [0, 30, 45, 60, 90], seeds = [1, 2, 3, 4, 5] } = {}) {
  const out = {};
  for (const [name, cfg] of Object.entries(PIPELINES)) {
    out[name] = {};
    for (const yaw of yaws) {
      const rs = seeds.map(seed => runPipeline(generateSquatSequence({ yawDeg: yaw, seed }), cfg));
      const avg = k => rs.reduce((s, r) => s + r[k], 0) / rs.length;
      out[name][yaw] = {
        rmse: avg('rmse'), bias: avg('bias'), jitter: avg('jitter'), peakErr: avg('peakErr'),
        repErr: rs.reduce((s, r) => s + Math.abs(r.reps - 6), 0), coverage: avg('coverage'),
      };
    }
  }
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const res = runSuite();
  if (process.argv.includes('--json')) { console.log(JSON.stringify(res, null, 1)); process.exit(0); }
  const yaws = Object.keys(Object.values(res)[0]);
  console.log('pipeline'.padEnd(42) + yaws.map(y => `yaw ${y}°: RMSE / bias / jitter / peak / repErr`.padEnd(46)).join(''));
  for (const [name, byYaw] of Object.entries(res)) {
    console.log(name.padEnd(42) + yaws.map(y => {
      const r = byYaw[y];
      return `${r.rmse.toFixed(1)} / ${r.bias >= 0 ? '+' : ''}${r.bias.toFixed(1)} / ${r.jitter.toFixed(2)} / ${r.peakErr.toFixed(1)} / ${r.repErr}`.padEnd(46);
    }).join(''));
  }
}
