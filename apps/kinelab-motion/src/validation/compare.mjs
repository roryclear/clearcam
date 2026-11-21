/**
 * App-vs-reference comparison for one joint: time alignment, frame-level agreement on
 * a common grid, and rep-level agreement of the extreme angle (the clinically reported
 * number). Frame-level pairs are autocorrelated, so use their RMSE/bias descriptively;
 * the CIs and ICC are meaningful for the rep-level pairs.
 */
import { agreement } from './agreement.mjs';
import { estimateLag, resample } from './align.mjs';
import { offlineSessionFrames } from '../sessions/offline-analysis.mjs';
import { analyzeReps, REP_PROFILES } from '../exercises/rep-analysis.mjs';
import { REFERENCE_TRANSFORMS } from './reference-io.mjs';

/**
 * @param {Object} session exported session JSON ({ frames, exerciseId? })
 * @param {{ timeMs: number[], columns: Object }} reference parsed reference
 * @param {Object} spec
 * @param {string} spec.refColumn reference column name
 * @param {string} spec.appKey app angle key, e.g. 'rightKnee'
 * @param {string} [spec.transform] key of REFERENCE_TRANSFORMS
 * @param {'offline'|'live'} [spec.appSource] offline re-analysis (default) or recorded live angles
 * @param {number|null} [spec.lagMs] fixed lag; estimated when null
 * @param {string} [spec.exerciseId] for rep-level comparison (defaults to session.exerciseId)
 * @param {number} [spec.gridMs]
 */
export function compareJoint(session, reference, spec) {
  const { refColumn, appKey, transform = 'interior', appSource = 'offline', lagMs = null, gridMs = 10 } = spec;
  const exerciseId = spec.exerciseId ?? session.exerciseId ?? session.exercise ?? null;
  const refRaw = reference.columns[refColumn];
  if (!refRaw) throw new Error(`reference column not found: ${refColumn} (have: ${Object.keys(reference.columns).join(', ')})`);
  const fn = REFERENCE_TRANSFORMS[transform];
  if (!fn) throw new Error(`unknown transform: ${transform}`);
  const refX = refRaw.map(v => (v === null ? null : fn(v)));
  const refT = reference.timeMs;

  const frames = appSource === 'offline' ? offlineSessionFrames(session.frames).frames : session.frames;
  const appT = frames.map(f => f.timeMs);
  const appX = frames.map(f => (Number.isFinite(f.angles?.[appKey]) ? f.angles[appKey] : null));
  if (appX.filter(v => v !== null).length < 10) throw new Error(`app series ${appKey} has too few values`);

  const lag = lagMs === null ? estimateLag(appT, appX, refT, refX) : { lagMs, r: NaN };
  const refTShifted = refT.map(t => t + lag.lagMs);

  // Frame level on the app's time span.
  const grid = [];
  for (let t = appT[0]; t <= appT[appT.length - 1]; t += gridMs) grid.push(t);
  const a = resample(appT, appX, grid);
  const r = resample(refTShifted, refX, grid);
  const frameLevel = agreement(a, r);

  // Rep level: extreme of each series inside each detected rep window.
  let repLevel = null;
  let reps = [];
  const profile = exerciseId ? REP_PROFILES[exerciseId] : null;
  if (profile && appKey.toLowerCase().endsWith(profile.joint.toLowerCase())) {
    reps = analyzeReps(frames, exerciseId);
    const pick = profile.direction === 'min' ? Math.min : Math.max;
    const pairs = reps.map(rep => {
      const inWin = (t, i, arr) => t >= rep.startMs && t <= rep.endMs && arr[i] !== null;
      const av = a.filter((v, i) => inWin(grid[i], i, a));
      const rv = r.filter((v, i) => inWin(grid[i], i, r));
      return av.length && rv.length ? { app: pick(...av), ref: pick(...rv), quality: rep.quality } : null;
    }).filter(Boolean);
    repLevel = {
      reps: pairs.map(p => ({ app: Math.round(p.app * 10) / 10, ref: Math.round(p.ref * 10) / 10, quality: p.quality })),
      agreement: pairs.length >= 3 ? agreement(pairs.map(p => p.app), pairs.map(p => p.ref)) : null,
    };
  }

  return {
    appKey,
    refColumn,
    transform,
    appSource,
    exerciseId,
    lagMs: lag.lagMs,
    lagCorrelation: lag.r,
    frameLevel,
    repLevel,
  };
}
