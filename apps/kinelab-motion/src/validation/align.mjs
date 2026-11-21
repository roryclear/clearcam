/**
 * Time alignment of an app series with a reference series recorded on another clock.
 */

/**
 * Linear resampling of (t, x) at times `grid`; null outside the data or across
 * gaps longer than maxGapMs.
 */
export function resample(t, x, grid, { maxGapMs = 150 } = {}) {
  const pts = [];
  for (let i = 0; i < t.length; i++) if (Number.isFinite(t[i]) && Number.isFinite(x[i])) pts.push([t[i], x[i]]);
  pts.sort((a, b) => a[0] - b[0]);
  const out = new Array(grid.length).fill(null);
  let j = 0;
  for (let g = 0; g < grid.length; g++) {
    const tg = grid[g];
    if (!pts.length || tg < pts[0][0] || tg > pts[pts.length - 1][0]) continue;
    // a = last sample at or before tg, so an exact match is used even right after a gap.
    while (j + 1 < pts.length && pts[j + 1][0] <= tg) j++;
    const a = pts[j];
    if (tg === a[0]) { out[g] = a[1]; continue; }
    const b = pts[j + 1];
    if (b[0] - a[0] > maxGapMs) continue;
    out[g] = a[1] + ((b[1] - a[1]) * (tg - a[0])) / (b[0] - a[0]);
  }
  return out;
}

function pearson(a, b) {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== null && b[i] !== null) { n++; sa += a[i]; sb += b[i]; }
  if (n < 10) return { r: NaN, n };
  const ma = sa / n;
  const mb = sb / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === null || b[i] === null) continue;
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return { r: da > 0 && db > 0 ? num / Math.sqrt(da * db) : NaN, n };
}

/**
 * Finds the lag (ms) to add to the reference clock so it lines up with the app:
 * ref(t − lag) ≈ app(t). Maximises Pearson correlation over ±maxLagMs on a stepMs grid,
 * requiring the overlap to cover at least minOverlap of the shorter series.
 * @returns {{ lagMs: number, r: number }}
 */
export function estimateLag(appT, appX, refT, refX, { maxLagMs = 2000, stepMs = 10, minOverlap = 0.5 } = {}) {
  const t0 = Math.min(appT[0], refT[0]) - maxLagMs;
  const t1 = Math.max(appT[appT.length - 1], refT[refT.length - 1]) + maxLagMs;
  const grid = [];
  for (let t = t0; t <= t1; t += stepMs) grid.push(t);
  const a = resample(appT, appX, grid);
  const r = resample(refT, refX, grid);
  const shortest = Math.min(a.filter(v => v !== null).length, r.filter(v => v !== null).length);
  const maxShift = Math.round(maxLagMs / stepMs);
  let best = { lagMs: 0, r: -Infinity };
  for (let s = -maxShift; s <= maxShift; s++) {
    // Shift reference by s samples: compare a[i] with r[i - s].
    const shifted = grid.map((_, i) => (i - s >= 0 && i - s < r.length ? r[i - s] : null));
    const { r: c, n } = pearson(a, shifted);
    if (n >= shortest * minOverlap && c > best.r) best = { lagMs: s * stepMs, r: c };
  }
  return best;
}
