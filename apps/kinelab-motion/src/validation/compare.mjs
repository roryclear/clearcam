/**
 * App-vs-reference comparison for one joint: time alignment, frame-level agreement on
 * a common grid, and rep-level agreement of the extreme angle (the clinically reported
 * number). Frame-level pairs are autocorrelated, so use their RMSE/bias descriptively;
 * the CIs and ICC are meaningful for the rep-level pairs.
 */

/** Share of a rep's grid points where both series must exist for a rep-level pair. */
export const MIN_REP_COVERAGE = 0.9;
/**
 * Longest interval between recorded samples (before any interpolation) allowed inside
 * a rep, per series: one dropped 30 fps frame (66 ms) passes, a 100 ms dropout does not.
 */
export const MAX_REP_GAP_MS = 80;

/**
 * Largest interval between consecutive sorted sample times ts that overlaps [t0, t1].
 * Intervals crossing a boundary count in full (the samples just outside the window
 * bound them); without a sample beyond an edge the window edge bounds it.
 */
export function maxSpacing(ts, t0, t1) {
  let prev = null;
  let worst = 0;
  for (const t of ts) {
    if (t <= t0) { prev = t; continue; }
    worst = Math.max(worst, t - (prev ?? t0));
    prev = t;
    if (t >= t1) return worst;
  }
  return Math.max(worst, t1 - (prev ?? t0));
}
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
  // Times of samples the app actually measured (offline re-analysis fills short gaps,
  // so coverage is judged on the series it smoothed).
  const usesRaw = appSource === 'offline' && session.frames.some(f => f?.rawAngles && Object.keys(f.rawAngles).length);
  const appSampleT = session.frames
    .filter(f => Number.isFinite((usesRaw ? f.rawAngles : f.angles)?.[appKey]))
    .map(f => f.timeMs);
  const appT = frames.map(f => f.timeMs);
  const appX = frames.map(f => (Number.isFinite(f.angles?.[appKey]) ? f.angles[appKey] : null));
  if (appX.filter(v => v !== null).length < 10) throw new Error(`app series ${appKey} has too few values`);

  const lag = lagMs === null ? estimateLag(appT, appX, refT, refX) : { lagMs, r: NaN };
  if (lagMs === null && !Number.isFinite(lag.r)) {
    // No lag within the search range gave enough overlap: report, don't score.
    return { appKey, refColumn, transform, appSource, exerciseId, alignment: 'failed', lagMs: null, lagCorrelation: null, frameLevel: null, repLevel: null };
  }
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
    // Extremes come from the same paired samples. A rep is scored only if both series'
    // recorded samples (not interpolated values) span the window with no interval over
    // MAX_REP_GAP_MS, so neither series' own peak (which for bilateral profiles need not
    // coincide with rep.extremeMs) can fall in a dropout; such reps are excluded.
    let excluded = 0;
    const refSampleT = refTShifted.filter((_, i) => refX[i] !== null && Number.isFinite(refX[i])).sort((p, q) => p - q);
    const pairs = reps.map(rep => {
      const idx = [];
      for (let i = 0; i < grid.length; i++) if (grid[i] >= rep.startMs && grid[i] <= rep.endMs) idx.push(i);
      const both = idx.filter(i => a[i] !== null && r[i] !== null);
      const gapOk = maxSpacing(appSampleT, rep.startMs, rep.endMs) <= MAX_REP_GAP_MS
        && maxSpacing(refSampleT, rep.startMs, rep.endMs) <= MAX_REP_GAP_MS;
      if (!idx.length || both.length < MIN_REP_COVERAGE * idx.length || !gapOk) { excluded++; return null; }
      return { app: pick(...both.map(i => a[i])), ref: pick(...both.map(i => r[i])), quality: rep.quality };
    }).filter(Boolean);
    repLevel = {
      excluded,
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
    alignment: lagMs === null ? 'estimated' : 'fixed',
    lagMs: lag.lagMs,
    lagCorrelation: lag.r,
    frameLevel,
    repLevel,
  };
}
