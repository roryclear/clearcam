/**
 * Offline (non-causal) smoothing for recorded sessions.
 *
 * Live tracking must use causal filters (One Euro), which lag behind fast motion and
 * under-read peaks. A recording can be filtered forwards and backwards instead, which
 * cancels the lag. Pipeline, as used by Sports2D and OpenCap:
 *   1. Hampel outlier rejection (median ± 3·1.4826·MAD over ~0.5 s)
 *   2. linear gap filling up to ~300 ms; longer gaps stay empty and split the series
 *   3. zero-phase Butterworth low-pass (filtfilt), order 4, 4–6 Hz
 * Refs: Sports2D (Butterworth 4th order, 6 Hz), OpenCap Monocular (4 Hz squat/STS),
 * Fit3D benchmark arXiv 2611.02929 (One Euro slightly raised error, zero-phase/median lowered it).
 */

/** Butterworth 2nd-order section Q values by filter order. */
const BUTTER_Q = { 2: [Math.SQRT1_2], 4: [0.5411961, 1.3065630] };

/** Low-pass biquad coefficients (RBJ cookbook, bilinear transform with prewarp). */
function lowpassBiquad(fs, fc, q) {
  const w0 = (2 * Math.PI * fc) / fs;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: ((1 - cos) / 2) / a0,
    b1: (1 - cos) / a0,
    b2: ((1 - cos) / 2) / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha) / a0,
  };
}

/** One pass of a biquad cascade (direct form II transposed), started at steady state. */
function cascade(x, sections) {
  let y = x.slice();
  for (const s of sections) {
    const out = new Array(y.length);
    const x0 = y[0];
    // Steady state for a constant input x0 (DC gain is 1): avoids a start-up transient.
    let z2 = (s.b2 - s.a2) * x0;
    let z1 = (s.b1 - s.a1) * x0 + z2;
    for (let i = 0; i < y.length; i++) {
      const xi = y[i];
      const yi = s.b0 * xi + z1;
      z1 = s.b1 * xi - s.a1 * yi + z2;
      z2 = s.b2 * xi - s.a2 * yi;
      out[i] = yi;
    }
    y = out;
  }
  return y;
}

/**
 * Zero-phase Butterworth low-pass of a uniformly sampled, gap-free series.
 * Edges use odd reflection (like scipy filtfilt) to limit end effects.
 * @param {number[]} x
 * @param {number} fs sample rate (Hz)
 * @param {number} fc cutoff (Hz)
 * @param {2|4} [order=4]
 */
export function filtfilt(x, fs, fc, order = 4) {
  const n = x.length;
  if (n < 3 || !(fc > 0) || !(fs > 2 * fc)) return x.slice();
  const sections = (BUTTER_Q[order] ?? BUTTER_Q[4]).map(q => lowpassBiquad(fs, fc, q));
  const pad = Math.min(n - 1, 3 * (2 * sections.length + 1));
  const head = [];
  const tail = [];
  for (let i = pad; i >= 1; i--) head.push(2 * x[0] - x[i]);
  for (let i = n - 2; i >= n - 1 - pad; i--) tail.push(2 * x[n - 1] - x[i]);
  const ext = [...head, ...x, ...tail];
  const fwd = cascade(ext, sections);
  const back = cascade(fwd.reverse(), sections).reverse();
  return back.slice(pad, pad + n);
}

function median(a) {
  const s = [...a].sort((p, q) => p - q);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Hampel filter: replaces points more than nSigma robust SDs from the local median.
 * Null/NaN entries are ignored and kept.
 * @param {(number|null)[]} x
 * @param {number} halfWindow samples on each side
 * @param {number} [nSigma=3]
 * @param {number} [minScale=0] floor for the robust SD, so a spike inside a flat or
 *   quantized window (MAD = 0) is still rejected
 */
export function hampel(x, halfWindow, nSigma = 3, minScale = 0) {
  const out = x.slice();
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i])) continue;
    const win = [];
    for (let j = Math.max(0, i - halfWindow); j <= Math.min(x.length - 1, i + halfWindow); j++) {
      if (Number.isFinite(x[j])) win.push(x[j]);
    }
    if (win.length < 3) continue;
    const med = median(win);
    const scale = Math.max(1.4826 * median(win.map(v => Math.abs(v - med))), minScale);
    if (scale > 0 && Math.abs(x[i] - med) > nSigma * scale) out[i] = med;
  }
  return out;
}

/**
 * Linear interpolation across gaps no longer than maxGapMs; longer gaps stay null.
 * @param {number[]} t timestamps (ms)
 * @param {(number|null)[]} x
 */
export function fillGaps(t, x, maxGapMs) {
  const out = x.map(v => (Number.isFinite(v) ? v : null));
  let last = -1;
  for (let i = 0; i < out.length; i++) {
    if (out[i] === null) continue;
    if (last >= 0 && i - last > 1 && t[i] - t[last] <= maxGapMs) {
      for (let k = last + 1; k < i; k++) {
        const f = (t[k] - t[last]) / (t[i] - t[last]);
        out[k] = out[last] + f * (out[i] - out[last]);
      }
    }
    last = i;
  }
  return out;
}

/** Linear interpolation of (xs, ys) at x; xs increasing. */
function interpAt(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo]);
}

/**
 * Zero-phase low-pass of one run. The filter assumes uniform sampling, so a run with
 * irregular frame times (dropped frames, callback jitter) is resampled onto a uniform
 * grid, filtered, and interpolated back to its own timestamps.
 */
function filterRun(t, x, dt, cutoffHz, order) {
  let irregular = false;
  for (let k = 1; k < t.length; k++) if (Math.abs(t[k] - t[k - 1] - dt) > 0.25 * dt) { irregular = true; break; }
  if (!irregular) return filtfilt(x, 1000 / dt, cutoffHz, order);
  // Uniform grid spanning both endpoints, so the first and last frames are grid points.
  const span = t[t.length - 1] - t[0];
  const steps = Math.max(1, Math.round(span / dt));
  const step = span / steps;
  const grid = Array.from({ length: steps + 1 }, (_, k) => t[0] + k * step);
  const sm = filtfilt(grid.map(g => interpAt(t, x, g)), 1000 / step, cutoffHz, order);
  return t.map(tt => interpAt(grid, sm, tt));
}

/**
 * Full offline pipeline for one angle series. Returns a new array; entries that were
 * missing in a gap longer than maxGapMs stay null. Each contiguous run is filtered on
 * its own: runs end at missing values and at frame-time gaps longer than maxGapMs, so
 * the low-pass never bridges a capture interruption.
 * @param {number[]} t timestamps (ms), increasing
 * @param {(number|null)[]} x
 * @param {Object} [opts]
 */
export function smoothSeriesOffline(t, x, {
  cutoffHz = 6, order = 4, maxGapMs = 300, hampelWindowMs = 500, hampelSigma = 3, hampelMinScale = 1,
} = {}) {
  const n = x.length;
  if (n !== t.length || n < 3) return x.map(v => (Number.isFinite(v) ? v : null));
  const dts = [];
  for (let i = 1; i < n; i++) if (t[i] > t[i - 1]) dts.push(t[i] - t[i - 1]);
  if (!dts.length) return x.map(v => (Number.isFinite(v) ? v : null));
  const dt = median(dts);
  const half = Math.max(1, Math.round(hampelWindowMs / 2 / dt));
  const out = x.map(v => (Number.isFinite(v) ? v : null));
  // Capture runs: valid samples whose neighbours are at most maxGapMs apart. Hampel,
  // gap filling and the low-pass all stay inside one run, so no stage mixes postures
  // from either side of an interruption.
  const valid = [];
  for (let i = 0; i < n; i++) if (out[i] !== null) valid.push(i);
  let r = 0;
  while (r < valid.length) {
    let e = r;
    while (e + 1 < valid.length && t[valid[e + 1]] - t[valid[e]] <= maxGapMs) e++;
    const i0 = valid[r];
    const i1 = valid[e];
    const tr = t.slice(i0, i1 + 1);
    const filled = fillGaps(tr, hampel(out.slice(i0, i1 + 1), half, hampelSigma, hampelMinScale), maxGapMs);
    const sm = filled.length >= 2 ? filterRun(tr, filled, dt, cutoffHz, order) : filled;
    for (let k = 0; k < sm.length; k++) out[i0 + k] = sm[k];
    r = e + 1;
  }
  return out;
}
