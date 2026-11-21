import test from 'node:test';
import assert from 'node:assert/strict';
import { tCritical, fQuantile, incompleteBeta } from '../src/validation/stats.mjs';
import { blandAltman, iccAgreement, agreement, iccLabel } from '../src/validation/agreement.mjs';
import { parseKinoveaCsv, parseOpenSimMot, parseReference } from '../src/validation/reference-io.mjs';
import { resample, estimateLag } from '../src/validation/align.mjs';
import { compareJoint } from '../src/validation/compare.mjs';
import { runValidation } from '../bench/validate.mjs';
import { SquatExercise } from '../src/exercises/squat-exercise.mjs';
import { mulberry32 } from '../bench/synthetic-pose.mjs';
import { analyzeReps } from '../src/exercises/rep-analysis.mjs';
import { offlineSessionFrames } from '../src/sessions/offline-analysis.mjs';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);

// Golden values: pingouin 0.7.0 intraclass_corr (ICC(A,1) = "ICC2") and scipy 1.18.1.
const A = [96.1, 101.4, 88.9, 110.2, 93.5, 99.8, 105.6, 91.2, 97.7, 102.9, 89.4, 108.1, 95.0, 100.6, 94.3];
const R = [94.0, 103.2, 86.5, 112.9, 92.1, 97.0, 108.4, 90.0, 95.2, 104.8, 88.8, 111.0, 93.9, 98.1, 95.6];
const A2 = [140.0, 128.5, 151.2, 133.9, 146.4, 125.0, 138.8];
const R2 = [131.5, 122.0, 141.9, 129.2, 137.0, 121.4, 130.1];

test('stats: t and F quantiles match scipy', () => {
  close(tCritical(6), 2.4469118511449786, 1e-9, 't(0.975, 6)');
  close(tCritical(40), 2.021075390306273, 1e-9, 't(0.975, 40)');
  close(fQuantile(0.975, 14, 9.3), 3.7164341461551245, 1e-9, 'F(0.975; 14, 9.3)');
  close(incompleteBeta(0.5, 2, 2), 0.5, 1e-12, 'I_0.5(2,2)');
});

test('agreement: Bland–Altman matches golden values', () => {
  const ba = blandAltman(A, R);
  close(ba.bias, 0.2133333333333335, 1e-12, 'bias');
  close(ba.sd, 2.185624025525064, 1e-12, 'sd');
  close(ba.loa[0], -4.070489756695792, 1e-9, 'loa lo');
  close(ba.loa[1], 4.497156423362459, 1e-9, 'loa hi');
  close(ba.biasCI[0], -0.9970249088031488, 1e-9, 'bias CI lo');
  close(ba.biasCI[1], 1.4236915754698158, 1e-9, 'bias CI hi');
  close(ba.loaCI.upper[1] - ba.loa[1], 2.096401970740141, 1e-9, 'LoA CI half-width');
  close(ba.rmse, 2.1222629431811706, 1e-12, 'rmse');
  close(ba.mae, 2.0, 1e-12, 'mae');
  const b2 = blandAltman(A2, R2);
  close(b2.biasCI[0], 5.077942820709957, 1e-9, 'n=7 bias CI lo');
  close(b2.loaCI.upper[1] - b2.loa[1], 3.7497415999924666, 1e-9, 'n=7 LoA CI half-width');
});

test('agreement: ICC(A,1) and its CI match pingouin', () => {
  const i1 = iccAgreement(A, R);
  close(i1.icc, 0.9588932146684781, 1e-12, 'icc');
  close(i1.ci[0], 0.8829678463430741, 1e-9, 'icc CI lo');
  close(i1.ci[1], 0.9859825596170803, 1e-9, 'icc CI hi');
  // Constant offset: consistency is high but absolute agreement is penalised.
  const i2 = iccAgreement(A2, R2);
  close(i2.icc, 0.705626420952721, 1e-12, 'icc biased');
  close(i2.ci[0], -0.04062916049475307, 1e-9, 'icc biased CI lo');
  close(i2.ci[1], 0.9528561930536187, 1e-9, 'icc biased CI hi');
  assert.equal(iccLabel(0.95), 'excellent');
  assert.equal(iccLabel(0.6), 'moderate');
});

test('agreement: SEM/MDC, label on the CI lower bound, nulls dropped, n<3 → null', () => {
  const rep = agreement([...A, null], [...R, 120]);
  assert.equal(rep.n, 15);
  assert.equal(rep.iccLabel, 'good'); // 0.883 lower bound
  close(rep.mdc95, 1.96 * Math.SQRT2 * rep.sem, 1e-12, 'mdc');
  assert.ok(rep.sem > 0);
  assert.equal(agreement([1, 2], [1, 2]), null);
});

test('reference-io: Kinovea CSV variants (semicolon + decimal comma, seconds, timecode)', () => {
  const eu = 'Time (ms);Angle 1;Angle 2\n0;170,5;90\n33;168,0;91,5\n67;;92\n';
  const p = parseKinoveaCsv(eu);
  assert.deepEqual(p.timeMs, [0, 33, 67]);
  assert.deepEqual(p.columns['Angle 1'], [170.5, 168, null]);
  assert.deepEqual(p.columns['Angle 2'], [90, 91.5, 92]);

  const us = 'Knee angle\nTime,Knee\n0.000,175.2\n0.033,174.9\n0.067,173.1\n';
  const q = parseKinoveaCsv(us);
  assert.deepEqual(q.timeMs.map(Math.round), [0, 33, 67]);
  assert.deepEqual(q.columns.Knee, [175.2, 174.9, 173.1]);

  const tc = 'Time\tKnee\n0:00:01.000\t170\n0:00:01.040\t169\n';
  assert.deepEqual(parseKinoveaCsv(tc).timeMs.map(Math.round), [1000, 1040]);
});

test('reference-io: OpenSim .mot (degrees and radians) and auto-detection', () => {
  const mot = 'Coordinates\nversion=1\nnRows=3\nnColumns=3\ninDegrees=yes\nendheader\n'
    + 'time\tknee_angle_r\thip_flexion_r\n0.00\t5.0\t10\n0.01\t6.0\t11\n0.02\t7.5\t12\n';
  const m = parseOpenSimMot(mot);
  assert.deepEqual(m.timeMs, [0, 10, 20]);
  assert.deepEqual(m.columns.knee_angle_r, [5, 6, 7.5]);
  const rad = mot.replace('inDegrees=yes', 'inDegrees=no').replace('5.0\t10', `${Math.PI / 2}\t10`);
  close(parseOpenSimMot(rad).columns.knee_angle_r[0], 90, 1e-9, 'radians → degrees');
  assert.equal(parseReference(mot, 'trial.mot').format, 'opensim');
  assert.equal(parseReference('Time;A\n0;1\n', 'k.csv').format, 'kinovea');
  assert.throws(() => parseOpenSimMot('time a\n0 1\n'), /endheader/);
});

test('align: resample interpolates and leaves gaps null; lag is recovered', () => {
  assert.deepEqual(resample([0, 100], [0, 10], [0, 50, 100, 150]), [0, 5, 10, null]);
  assert.deepEqual(resample([0, 500], [0, 10], [250], { maxGapMs: 150 }), [null]);
  const t = Array.from({ length: 300 }, (_, i) => i * 33.3);
  const sig = tt => 150 + 30 * Math.sin(tt / 400) + 10 * Math.sin(tt / 97);
  const app = t.map(sig);
  // Reference started 420 ms later on its own clock: refClock = appClock − 420.
  const refT = t.map(tt => tt - 420);
  const ref = t.map(sig);
  const { lagMs, r } = estimateLag(t, app, refT, ref);
  close(lagMs, 420, 10, 'lag');
  assert.ok(r > 0.99);
});

function syntheticSquatSession({ offsetMs = 300, noiseDeg = 2, seed = 3 } = {}) {
  const rand = mulberry32(seed);
  const squat = new SquatExercise();
  let state = squat.createState();
  const frames = [];
  const truth = [];
  const fps = 30;
  for (let t = 0; t <= 1500 * 2 + 5 * 2400; t += 1000 / fps) {
    let flex = 0;
    if (t > 1500 && t < 1500 + 5 * 2400) flex = 45 * (1 - Math.cos(((t - 1500) % 2400) / 2400 * 2 * Math.PI));
    const knee = 180 - flex;
    truth.push([t, flex]);
    const noisy = knee + noiseDeg * (rand() * 2 - 1) + 3; // +3° bias
    state = squat.update(state, { angles: { leftKnee: noisy, rightKnee: noisy }, landmarks: null, timestampMs: t, trackingQuality: 'good' });
    frames.push({ timeMs: Math.round(t), angles: { rightKnee: noisy }, rawAngles: { rightKnee: noisy }, landmarks: null, status: state.status, reps: state.reps });
  }
  const mot = 'IK\ninDegrees=yes\nendheader\ntime\tknee_angle_r\n'
    + truth.map(([t, f]) => `${((t - offsetMs) / 1000).toFixed(5)}\t${f.toFixed(3)}`).join('\n');
  return { session: { exerciseId: 'squat', frames }, mot, reps: state.reps };
}

test('compare: synthetic squat vs OpenSim reference recovers lag, bias, per-rep peaks', () => {
  const { session, mot, reps } = syntheticSquatSession();
  assert.equal(reps, 5);
  const res = compareJoint(session, parseReference(mot, 'ik.mot'), { refColumn: 'knee_angle_r', appKey: 'rightKnee', transform: 'flexion' });
  close(res.lagMs, 300, 20, 'lag');
  close(res.frameLevel.bias, 3, 0.6, 'frame bias');
  assert.equal(res.repLevel.reps.length, 5);
  for (const rep of res.repLevel.reps) close(rep.ref, 90, 0.5, 'reference rep peak');
  close(res.repLevel.agreement.bias, 3, 1.5, 'rep bias');
  // With a known clock, a wrong fixed lag degrades the frame-level RMSE.
  const wrong = compareJoint(session, parseReference(mot, 'ik.mot'), { refColumn: 'knee_angle_r', appKey: 'rightKnee', transform: 'flexion', lagMs: 0 });
  assert.ok(wrong.frameLevel.rmse > res.frameLevel.rmse + 2);
  assert.throws(() => compareJoint(session, parseReference(mot, 'ik.mot'), { refColumn: 'nope', appKey: 'rightKnee' }), /column not found/);
});

test('validate CLI: pools rep-level pairs across trials', () => {
  const t1 = syntheticSquatSession({ seed: 3 });
  const t2 = syntheticSquatSession({ seed: 9, offsetMs: -150 });
  const files = { 's1.json': JSON.stringify(t1.session), 'r1.mot': t1.mot, 's2.json': JSON.stringify(t2.session), 'r2.mot': t2.mot };
  const out = runValidation({ session: ['s1.json', 's2.json'], ref: ['r1.mot', 'r2.mot'], joint: ['knee_angle_r:rightKnee'], exercise: null, source: 'offline', lagMs: null }, p => files[p]);
  assert.equal(out.trials.length, 2);
  assert.equal(out.trials[0].transform, 'flexion'); // OpenSim default
  close(out.trials[1].lagMs, -150, 20, 'trial 2 lag');
  const pooled = out.repLevelPooled['knee_angle_r:rightKnee'];
  assert.equal(pooled.reps, 10);
  close(pooled.agreement.bias, 3, 1.5, 'pooled bias');
});

test('review fixes: exact sample after a gap; title row before ;/tab tables; uncovered reps excluded (PR #16)', () => {
  assert.deepEqual(resample([0, 100, 500], [10, 20, 30], [0, 100, 500]), [10, 20, 30]);

  for (const d of [';', '\t']) {
    const p = parseKinoveaCsv(`Knee angle\nTime (ms)${d}Knee\n0${d}175\n33${d}170\n67${d}165\n`);
    assert.deepEqual(p.timeMs, [0, 33, 67], `delimiter ${JSON.stringify(d)}`);
    assert.deepEqual(p.columns.Knee, [175, 170, 165]);
  }

  // Reference stops during the third rep's descent: that rep must not be scored.
  const { session, mot } = syntheticSquatSession({ offsetMs: 0, noiseDeg: 0 });
  const lines = mot.split('\n');
  const cut = lines.filter(l => !/^\d/.test(l) || Number(l.split('\t')[0]) < (1500 + 2 * 2400 + 600) / 1000).join('\n');
  const res = compareJoint(session, parseReference(cut, 'ik.mot'), { refColumn: 'knee_angle_r', appKey: 'rightKnee', transform: 'flexion', lagMs: 0 });
  assert.equal(res.repLevel.reps.length, 2);
  assert.equal(res.repLevel.excluded, 3);
  for (const rep of res.repLevel.reps) close(rep.app - rep.ref, 3, 0.6, 'covered rep');
});

test('review round 2: CLI operands, OpenSim default transforms, quoted CSV, alignment failure, gaps in a rep (PR #16)', async () => {
  const { parseArgs } = await import('../bench/validate.mjs');
  const { defaultTransform } = await import('../src/validation/reference-io.mjs');
  assert.throws(() => parseArgs(['--source', 'offine']), /offline or live/);
  assert.throws(() => parseArgs(['--lag-ms', 'abc']), /finite/);
  assert.throws(() => parseArgs(['--lag-ms']), /finite/);
  assert.throws(() => parseArgs(['--session']), /needs a value/);
  assert.equal(parseArgs(['--lag-ms', '-150']).lagMs, -150);

  assert.equal(defaultTransform('opensim', 'knee_angle_r'), 'flexion');
  assert.equal(defaultTransform('opensim', 'elbow_flex_l'), 'flexion');
  assert.throws(() => defaultTransform('opensim', 'arm_flex_r'), /explicit|pass ref:app/);
  assert.equal(defaultTransform('kinovea', 'Shoulder'), 'interior');

  const q = parseKinoveaCsv('Time (ms),"Knee","Say ""hi"", ok"\n0,170,1\n33,160,2\n');
  assert.deepEqual(q.columns.Knee, [170, 160]);
  assert.deepEqual(q.columns['Say "hi", ok'], [1, 2]);

  // Clocks that never overlap enough: report the failure instead of scoring at lag 0.
  const t = Array.from({ length: 301 }, (_, i) => i * 33.3);
  const sig = tt => 150 + 30 * Math.sin(tt / 400);
  const session = { frames: t.map(tt => ({ timeMs: tt, angles: { rightKnee: sig(tt) } })) };
  const far = { timeMs: t.map(tt => tt + 9000), columns: { k: t.map(tt => sig(tt + 9000)) } };
  const res = compareJoint(session, far, { refColumn: 'k', appKey: 'rightKnee', appSource: 'live' });
  assert.equal(res.alignment, 'failed');
  assert.equal(res.frameLevel, null);

  // A 200 ms reference dropout away from the rep's extreme still excludes that rep.
  const syn = syntheticSquatSession({ offsetMs: 0, noiseDeg: 0 });
  const rows = syn.mot.split('\n');
  const drop = rows.filter(l => {
    if (!/^-?\d/.test(l)) return true;
    const ms = Number(l.split('\t')[0]) * 1000;
    return !(ms > 1500 + 400 && ms < 1500 + 600); // rep 1 descent
  }).join('\n');
  const r2 = compareJoint(syn.session, parseReference(drop, 'ik.mot'), { refColumn: 'knee_angle_r', appKey: 'rightKnee', transform: 'flexion', lagMs: 0 });
  assert.equal(r2.repLevel.excluded, 1);
  assert.equal(r2.repLevel.reps.length, 4);
});

test('review round 3: a 100 ms reference dropout bridged by resampling still excludes the rep (PR #16)', () => {
  const syn = syntheticSquatSession({ offsetMs: 0, noiseDeg: 0 });
  const keep = maxMs => syn.mot.split('\n').filter(l => {
    if (!/^-?\d/.test(l)) return true;
    const ms = Number(l.split('\t')[0]) * 1000;
    return !(ms > 1500 + 3 * 2400 + 500 && ms < 1500 + 3 * 2400 + 500 + maxMs); // rep 4
  }).join('\n');
  const spec = { refColumn: 'knee_angle_r', appKey: 'rightKnee', transform: 'flexion', lagMs: 0 };
  const r100 = compareJoint(syn.session, parseReference(keep(100), 'ik.mot'), spec);
  assert.equal(r100.repLevel.excluded, 1);
  // A fully sampled 30 Hz reference keeps every rep; one dropped app frame does too.
  const dropApp = { ...syn.session, frames: syn.session.frames.filter((_, i) => i !== 300) };
  const r40 = compareJoint(dropApp, parseReference(syn.mot, 'ik.mot'), spec);
  assert.equal(r40.repLevel.excluded, 0);
  assert.equal(r40.repLevel.reps.length, 5);
});

test('maxSpacing: gaps crossing a rep boundary count in full (PR #16 review round 4)', async () => {
  const { maxSpacing } = await import('../src/validation/compare.mjs');
  assert.equal(maxSpacing([0, 33, 66, 100], 0, 100), 34);
  assert.equal(maxSpacing([0, 100, 133, 166, 200], 60, 200), 100); // dropout 0–100 overlaps the start
  assert.equal(maxSpacing([0, 33, 66, 166], 0, 120), 100); // dropout 66–166 overlaps the end
  assert.equal(maxSpacing([40, 73, 106], 0, 100), Infinity); // series starts inside the window
  assert.equal(maxSpacing([0, 33, 66], 0, 100), Infinity); // series ends inside the window
  assert.equal(maxSpacing([], 0, 100), Infinity);
});

test('review round 5: delimiter from the table, not a title or quoted text; reference starting inside a rep (PR #16)', () => {
  const titled = parseKinoveaCsv('Knee angle; trial\nTime (ms),Knee\n0,170\n200,160\n');
  assert.deepEqual(titled.columns.Knee, [170, 160]);
  const quoted = parseKinoveaCsv('Time (ms),"Knee","Note; detail"\n0,170,1\n33,160,2\n');
  assert.deepEqual(quoted.columns.Knee, [170, 160]);
  assert.deepEqual(quoted.columns['Note; detail'], [1, 2]);
  const quotedTab = parseKinoveaCsv('Time (ms);Knee;"a\tb"\n0;170,5;1\n33;160;2\n');
  assert.deepEqual(quotedTab.columns.Knee, [170.5, 160]);
  // Decimal-comma semicolon export still detected as semicolon.
  assert.deepEqual(parseKinoveaCsv('Time (ms);Angle\n0;170,5\n33;168,0\n').columns.Angle, [170.5, 168]);

  // Reference recording starts 40 ms after the first rep starts: that rep is excluded.
  const syn = syntheticSquatSession({ offsetMs: 0, noiseDeg: 0 });
  const spec = { refColumn: 'knee_angle_r', appKey: 'rightKnee', transform: 'flexion', lagMs: 0 };
  const [first] = analyzeReps(offlineSessionFrames(syn.session.frames).frames, 'squat');
  const late = syn.mot.split('\n').filter(l => !/^-?\d/.test(l) || Number(l.split('\t')[0]) * 1000 >= first.startMs + 40).join('\n');
  const r = compareJoint(syn.session, parseReference(late, 'ik.mot'), spec);
  assert.equal(r.repLevel.excluded, 1);
  assert.equal(r.repLevel.reps.length, 4);
});

test('review round 6: quoted tabs in rows, short decimal-comma rows, edge rounding, dropout at a peak (PR #16)', async () => {
  const { maxSpacing } = await import('../src/validation/compare.mjs');
  const qt = parseKinoveaCsv('Time (ms);Knee;"Note\ttext"\n0;170;"a\tb"\n33;160;"c\td"\n');
  assert.deepEqual(qt.timeMs, [0, 33]);
  assert.deepEqual(qt.columns.Knee, [170, 160]);
  const rows = ['Time (ms);Knee;Hip'];
  for (let i = 0; i < 5; i++) rows.push(`${i * 33};170,5;90`);
  for (let i = 5; i < 11; i++) rows.push(`${i * 33};165,5`);
  const short = parseKinoveaCsv(rows.join('\n'));
  assert.equal(short.columns.Knee.length, 11);
  assert.deepEqual(short.columns.Hip.slice(4, 6), [90, null]);

  assert.equal(parseOpenSimMot('x\nendheader\ntime\tk\n16.002\t1\n').timeMs[0], 16002);
  assert.equal(maxSpacing([0, 33, 66, 99.9999], 0, 100), 33.9999);

  // One missing reference sample (≈67 ms gap) excludes its rep, at the peak or anywhere else.
  const syn = syntheticSquatSession({ offsetMs: 0, noiseDeg: 0 });
  const spec = { refColumn: 'knee_angle_r', appKey: 'rightKnee', transform: 'flexion', lagMs: 0 };
  const lines = syn.mot.split('\n');
  const header = lines.filter(l => !/^-?\d/.test(l));
  const data = lines.filter(l => /^-?\d/.test(l));
  const nearest = ms => data.reduce((b, l, i) => (Math.abs(Number(l.split('\t')[0]) * 1000 - ms) < Math.abs(Number(data[b].split('\t')[0]) * 1000 - ms) ? i : b), 0);
  const drop = i => [...header, ...data.filter((_, k) => k !== i)].join('\n');
  const peak2 = 1500 + 2400 + 1200; // second rep's bottom
  const atPeak = compareJoint(syn.session, parseReference(drop(nearest(peak2)), 'ik.mot'), spec);
  assert.equal(atPeak.repLevel.excluded, 1);
  const offPeak = compareJoint(syn.session, parseReference(drop(nearest(peak2 - 600)), 'ik.mot'), spec);
  assert.equal(offPeak.repLevel.excluded, 1);
  assert.equal(offPeak.repLevel.reps.length, 4);
});

test('review round 7: tab + decimal-comma tables; 20 fps phase-offset reference keeps reps (PR #16)', () => {
  const tsv = parseKinoveaCsv('Time (s)\tKnee\n0,000\t170,5\n0,033\t168,0\n');
  assert.deepEqual(tsv.timeMs, [0, 33]);
  assert.deepEqual(tsv.columns.Knee, [170.5, 168]);

  // 20 fps reference in seconds, starting at 0.000010 s: exact 50 ms spacing must pass.
  const syn = syntheticSquatSession({ offsetMs: 0, noiseDeg: 0 });
  const rows = ['Time (s),Knee'];
  for (let k = 0; k * 50 <= 15000; k++) {
    const tMs = k * 50 + 0.01;
    let flex = 0;
    if (tMs > 1500 && tMs < 1500 + 5 * 2400) flex = 45 * (1 - Math.cos(((tMs - 1500) % 2400) / 2400 * 2 * Math.PI));
    rows.push(`${(tMs / 1000).toFixed(6)},${(180 - flex).toFixed(3)}`);
  }
  const ref = parseKinoveaCsv(rows.join('\n'));
  const res = compareJoint(syn.session, ref, { refColumn: 'Knee', appKey: 'rightKnee', transform: 'interior', lagMs: 0 });
  assert.equal(res.repLevel.excluded, 0);
  assert.equal(res.repLevel.reps.length, 5);
});

test('review round 8: rep extremes come from recorded timestamps, matching the app rep summary (PR #16)', () => {
  // V-shaped squats (sharp bottoms) at 30 fps on off-grid times; the bottom frame of each
  // rep is dropped so the best remaining frame falls between 10 ms grid points.
  const squat = new SquatExercise();
  let state = squat.createState();
  const frames = [];
  const mot = ['IK', 'inDegrees=yes', 'endheader', 'time\tknee_angle_r'];
  const flexAt = t => (t > 1500 && t < 1500 + 5 * 2400 ? 90 * (1 - Math.abs(2 * (((t - 1500) % 2400) / 2400) - 1)) : 0);
  for (let k = 0; k * 1000 / 30 <= 15000; k++) {
    const t = Math.round(k * 1000 / 30) + 6; // off the 10 ms grid
    const knee = 180 - flexAt(t);
    state = squat.update(state, { angles: { leftKnee: knee, rightKnee: knee }, landmarks: null, timestampMs: t, trackingQuality: 'good' });
    const bottom = t > 1500 && Math.abs(((t - 1500) % 2400) - 1200) < 17;
    if (!bottom) frames.push({ timeMs: t, angles: { rightKnee: knee }, landmarks: null, status: state.status, reps: state.reps });
  }
  for (let t = 0; t <= 15000; t += 10) mot.push(`${(t / 1000).toFixed(3)}\t${flexAt(t).toFixed(3)}`);
  const session = { exerciseId: 'squat', frames };
  const res = compareJoint(session, parseReference(mot.join('\n'), 'ik.mot'), { refColumn: 'knee_angle_r', appKey: 'rightKnee', transform: 'flexion', appSource: 'live', lagMs: 0 });
  const reps = analyzeReps(frames, 'squat');
  assert.ok(reps.length >= 4);
  assert.equal(res.repLevel.reps.length, reps.length);
  res.repLevel.reps.forEach((r, i) => {
    assert.equal(r.app, reps[i].extremeAngle, `rep ${i + 1} app`);
    assert.equal(r.ref, 90, `rep ${i + 1} ref (sampled bottom)`);
  });
});
