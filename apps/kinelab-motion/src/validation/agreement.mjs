/**
 * Agreement statistics between the app and a reference method (goniometer, Kinovea,
 * marker mocap / OpenSim). Checked against pingouin 0.7 / scipy 1.18 golden values
 * (tests/validation.test.mjs).
 *
 * - Bland–Altman: bias, SD of differences, 95% limits of agreement (bias ± 1.96·SD),
 *   with 95% CIs for the bias (t·SD/√n) and for each limit (t·√(3·SD²/n)).
 * - ICC(A,1): two-way random effects, absolute agreement, single measures
 *   (McGraw & Wong 1996; Shrout & Fleiss ICC(2,1)), with its 95% CI.
 *   Interpret on the CI (Koo & Li 2016): < 0.5 poor, 0.5–0.75 moderate,
 *   0.75–0.9 good, > 0.9 excellent.
 * - SEM = SD_pooled·√(1 − ICC); MDC95 = 1.96·√2·SEM.
 * Pairs must be independent (e.g. one value per rep, not per frame) for the CIs to hold;
 * with several reps per subject report subject-level means or a repeated-measures model.
 */
import { tCritical, fQuantile } from './stats.mjs';

function mean(a) {
  return a.reduce((s, v) => s + v, 0) / a.length;
}

function sdOf(a) {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
}

/** Keeps pairs where both values are finite. */
export function cleanPairs(app, ref) {
  const a = [];
  const r = [];
  for (let i = 0; i < Math.min(app.length, ref.length); i++) {
    if (Number.isFinite(app[i]) && Number.isFinite(ref[i])) { a.push(app[i]); r.push(ref[i]); }
  }
  return { app: a, ref: r };
}

/**
 * @param {number[]} app
 * @param {number[]} ref
 */
export function blandAltman(app, ref, alpha = 0.05) {
  const p = cleanPairs(app, ref);
  const n = p.app.length;
  if (n < 3) return null;
  const d = p.app.map((v, i) => v - p.ref[i]);
  const bias = mean(d);
  const sd = sdOf(d);
  const t = tCritical(n - 1, alpha);
  const loaHalf = t * Math.sqrt((3 * sd * sd) / n);
  const lower = bias - 1.96 * sd;
  const upper = bias + 1.96 * sd;
  return {
    n,
    bias,
    sd,
    biasCI: [bias - (t * sd) / Math.sqrt(n), bias + (t * sd) / Math.sqrt(n)],
    loa: [lower, upper],
    loaCI: { lower: [lower - loaHalf, lower + loaHalf], upper: [upper - loaHalf, upper + loaHalf] },
    rmse: Math.sqrt(mean(d.map(v => v * v))),
    mae: mean(d.map(Math.abs)),
  };
}

/**
 * ICC(A,1) with a 95% CI (two methods: app and reference).
 * @returns {{ icc: number, ci: [number, number], msr: number, msc: number, mse: number, n: number }|null}
 */
export function iccAgreement(app, ref, alpha = 0.05) {
  const p = cleanPairs(app, ref);
  const n = p.app.length;
  const k = 2;
  if (n < 3) return null;
  const rows = p.app.map((v, i) => [v, p.ref[i]]);
  const gm = mean(rows.flat());
  const rowMeans = rows.map(r => (r[0] + r[1]) / 2);
  const colMeans = [mean(p.app), mean(p.ref)];
  const msr = (k * rowMeans.reduce((s, m) => s + (m - gm) ** 2, 0)) / (n - 1);
  const msc = (n * colMeans.reduce((s, m) => s + (m - gm) ** 2, 0)) / (k - 1);
  let sse = 0;
  rows.forEach((r, i) => r.forEach((y, j) => { sse += (y - rowMeans[i] - colMeans[j] + gm) ** 2; }));
  const mse = sse / ((n - 1) * (k - 1));
  const icc = (msr - mse) / (msr + (k - 1) * mse + (k * (msc - mse)) / n);
  // McGraw & Wong CI with Satterthwaite df (same as pingouin).
  const aa = (k * icc) / (n * (1 - icc));
  const bb = 1 + (k * icc * (n - 1)) / (n * (1 - icc));
  const v = (aa * msc + bb * mse) ** 2 / ((aa * msc) ** 2 / (k - 1) + (bb * mse) ** 2 / ((n - 1) * (k - 1)));
  const fu = fQuantile(1 - alpha / 2, n - 1, v);
  const fl = fQuantile(1 - alpha / 2, v, n - 1);
  const lo = (n * (msr - fu * mse)) / (fu * (k * msc + (k * n - k - n) * mse) + n * msr);
  const hi = (n * (fl * msr - mse)) / (k * msc + (k * n - k - n) * mse + n * fl * msr);
  return { icc, ci: [lo, hi], msr, msc, mse, n };
}

/** Koo & Li (2016) label for an ICC value. */
export function iccLabel(value) {
  if (!Number.isFinite(value)) return 'n/a';
  if (value < 0.5) return 'poor';
  if (value < 0.75) return 'moderate';
  if (value < 0.9) return 'good';
  return 'excellent';
}

/** Full agreement report for paired values. */
export function agreement(app, ref) {
  const ba = blandAltman(app, ref);
  const icc = iccAgreement(app, ref);
  if (!ba || !icc) return null;
  const p = cleanPairs(app, ref);
  const pooledSd = sdOf([...p.app, ...p.ref]);
  const sem = pooledSd * Math.sqrt(Math.max(0, 1 - icc.icc));
  return {
    ...ba,
    icc: icc.icc,
    iccCI: icc.ci,
    iccLabel: iccLabel(icc.ci[0]), // judged on the lower CI bound
    sem,
    mdc95: 1.96 * Math.SQRT2 * sem,
  };
}
