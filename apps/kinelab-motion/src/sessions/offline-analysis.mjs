/**
 * Offline re-analysis of a recorded session.
 *
 * Recorded frames carry the live angles (One Euro smoothed, so they lag) and, from
 * this version on, rawAngles computed from unfiltered landmarks. For results and
 * exports the raw angles are smoothed with a zero-phase filter instead, which removes
 * the causal lag. Synthetic squat benchmark (bench/accuracy-bench.mjs), side view:
 * knee RMSE 3.2° → 2.2°, frame-to-frame jitter 1.12° → 0.50°, rep peak error 1.3° → 1.1°;
 * 45° view: peak error 10.2° → 8.7°.
 * Older recordings without rawAngles get the live angles smoothed at 6 Hz (jitter only).
 */
import { smoothSeriesOffline } from '../filters/zero-phase.mjs';

/**
 * Session without the optional rawAngles (re-analysis then falls back to the live
 * angles); used when storage is too full to keep them.
 */
export function stripRawAngles(session) {
  if (!session?.frames?.some(f => f?.rawAngles)) return session;
  return { ...session, frames: session.frames.map(({ rawAngles, ...f }) => f) };
}

/** Cutoffs: 4 Hz for raw angles (OpenCap uses 4 Hz for squat/STS), 6 Hz on live angles. */
export const OFFLINE_CUTOFF_HZ = { raw: 4, live: 6 };

/**
 * @param {Array<{timeMs: number, angles?: Object, rawAngles?: Object}>} frames
 * @param {Object} [opts]
 * @param {number} [opts.cutoffHz] override the cutoff
 * @returns {{ frames: Array, source: 'raw'|'live'|'none' }} frames with smoothed `angles`
 *   (the recorded live angles stay available as `liveAngles`)
 */
export function offlineSessionFrames(frames, { cutoffHz } = {}) {
  if (!Array.isArray(frames) || frames.length < 3) return { frames: frames ?? [], source: 'none' };
  const hasRaw = frames.some(f => f?.rawAngles && Object.keys(f.rawAngles).length);
  const source = hasRaw ? 'raw' : 'live';
  const pick = f => (hasRaw ? f?.rawAngles : f?.angles) ?? {};
  const keys = new Set();
  for (const f of frames) for (const k of Object.keys(pick(f))) keys.add(k);
  if (!keys.size) return { frames, source: 'none' };
  const t = frames.map(f => f.timeMs);
  const fc = cutoffHz ?? OFFLINE_CUTOFF_HZ[source];
  const smoothed = {};
  for (const k of keys) {
    smoothed[k] = smoothSeriesOffline(t, frames.map(f => pick(f)[k] ?? null), { cutoffHz: fc });
  }
  const out = frames.map((f, i) => {
    const angles = {};
    for (const k of keys) angles[k] = smoothed[k][i] === null ? null : Math.round(smoothed[k][i] * 10) / 10;
    return { ...f, angles, liveAngles: f.angles };
  });
  return { frames: out, source };
}
