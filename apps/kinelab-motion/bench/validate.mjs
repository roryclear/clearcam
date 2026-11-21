#!/usr/bin/env node
/**
 * Validates app angles against a reference (Kinovea CSV, OpenSim .mot/.sto).
 *
 *   node bench/validate.mjs --session session.json --ref trial.mot \
 *     --joint knee_angle_r:rightKnee[:flexion] [--joint ...] [--exercise squat] \
 *     [--source offline|live] [--lag-ms 0]
 *
 * Several --session/--ref pairs may be given in order (one per trial); rep-level pairs
 * are pooled across trials for the summary. Prints a JSON report to stdout.
 */
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';
import { parseReference, defaultTransform } from '../src/validation/reference-io.mjs';
import { compareJoint } from '../src/validation/compare.mjs';
import { agreement } from '../src/validation/agreement.mjs';

function parseArgs(argv) {
  const out = { session: [], ref: [], joint: [], exercise: null, source: 'offline', lagMs: null };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    const v = argv[i + 1];
    const need = () => { if (v === undefined || v.startsWith('--')) throw new Error(`${k} needs a value`); return v; };
    if (k === '--session') { out.session.push(need()); i++; }
    else if (k === '--ref') { out.ref.push(need()); i++; }
    else if (k === '--joint') { out.joint.push(need()); i++; }
    else if (k === '--exercise') { out.exercise = need(); i++; }
    else if (k === '--source') {
      if (v !== 'offline' && v !== 'live') throw new Error(`--source must be offline or live (got ${v ?? 'nothing'})`);
      out.source = v; i++;
    } else if (k === '--lag-ms') {
      const lag = v === undefined || v.trim() === '' ? NaN : Number(v);
      if (!Number.isFinite(lag)) throw new Error(`--lag-ms needs a finite number of milliseconds (got ${v ?? 'nothing'})`);
      out.lagMs = lag; i++;
    }
    else if (k === '--help' || k === '-h') out.help = true;
    else throw new Error(`unknown argument: ${k}`);
  }
  return out;
}

export function runValidation(args, read = p => readFileSync(p, 'utf8')) {
  if (!args.session.length || args.session.length !== args.ref.length || !args.joint.length) {
    throw new Error('need matching --session/--ref pairs and at least one --joint ref:app[:transform]');
  }
  const trials = [];
  const pooled = {};
  args.session.forEach((sp, ti) => {
    const session = JSON.parse(read(sp));
    const reference = parseReference(read(args.ref[ti]), args.ref[ti]);
    for (const j of args.joint) {
      const [refColumn, appKey, explicit] = j.split(':');
      const transform = explicit ?? defaultTransform(reference.format, refColumn);
      const res = compareJoint(session, reference, {
        refColumn, appKey, transform, appSource: args.source, lagMs: args.lagMs, exerciseId: args.exercise ?? undefined,
      });
      trials.push({ session: basename(sp), reference: basename(args.ref[ti]), format: reference.format, ...res });
      const key = `${refColumn}:${appKey}`;
      if (res.alignment === 'failed') continue; // reported per trial, never pooled
      pooled[key] ??= { app: [], ref: [] };
      for (const r of res.repLevel?.reps ?? []) { pooled[key].app.push(r.app); pooled[key].ref.push(r.ref); }
    }
  });
  const summary = Object.fromEntries(Object.entries(pooled).map(([k, p]) => [k, { reps: p.app.length, agreement: agreement(p.app, p.ref) }]));
  return { trials, repLevelPooled: summary };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (args.help) {
      console.log('usage: node bench/validate.mjs --session s.json --ref r.mot --joint knee_angle_r:rightKnee[:flexion] [--exercise squat] [--source offline|live] [--lag-ms N]');
    } else {
      console.log(JSON.stringify(runValidation(args), (k, v) => (typeof v === 'number' ? Math.round(v * 1000) / 1000 : v), 2));
    }
  } catch (e) {
    console.error(e.message);
    process.exit(1);
  }
}

export { parseArgs };
