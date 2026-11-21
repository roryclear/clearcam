/**
 * Readers for reference joint-angle data:
 * - Kinovea "Export to spreadsheet" CSV/TSV (angle or trajectory tables). Locale
 *   variants: `;` with decimal commas, `,`, or tabs. Time column in ms or s, or as
 *   a timecode h:mm:ss.fff / mm:ss.fff.
 * - OpenSim .mot / .sto storage files (inverse-kinematics output, e.g. OpenCap's
 *   LabValidation marker IK): header up to `endheader`, whitespace columns, time in s,
 *   `inDegrees=yes|no`.
 * All readers return { timeMs: number[], columns: { [name]: (number|null)[] } }.
 */

/**
 * Delimiter of the table, judged on what it produces: the candidate under which the
 * most of the first lines are data rows (≥ 2 quote-aware fields with a parseable time in
 * the first). Title rows, quoted delimiters and short rows (missing trailing cells) do
 * not decide it. Ties go to the more consistent field count, then tab, semicolon, comma.
 */
function detectDelimiter(lines) {
  const sample = lines.slice(0, 20);
  let best = { d: ',', rows: -1, consistency: -1 };
  for (const d of ['\t', ';', ',']) {
    const counts = [];
    for (const l of sample) {
      const f = splitFields(l, d);
      // Decimal commas are possible whenever comma is not the delimiter.
      if (f.length >= 2 && (parseTimecode(f[0]) !== null || parseNumber(f[0], d !== ',') !== null)) counts.push(f.length);
    }
    const freq = new Map();
    for (const c of counts) freq.set(c, (freq.get(c) ?? 0) + 1);
    const consistency = Math.max(0, ...freq.values());
    if (counts.length > best.rows || (counts.length === best.rows && consistency > best.consistency)) {
      best = { d, rows: counts.length, consistency };
    }
  }
  return best.d;
}

/** Splits one delimited line, honouring double quotes ("" = literal quote). */
function splitFields(line, delimiter) {
  const out = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"' && cur.trim() === '') { quoted = true; cur = ''; }
    else if (ch === delimiter) { out.push(cur.trim()); cur = ''; }
    else cur += ch;
  }
  out.push(cur.trim());
  return out;
}

function parseNumber(raw, decimalComma) {
  if (raw === undefined) return null;
  let s = String(raw).trim().replace(/^"|"$/g, '');
  if (s === '' || /^(nan|null|-|n\/a)$/i.test(s)) return null;
  s = s.replace(/\s*(°|deg|ms|s)$/i, '');
  if (decimalComma) s = s.replace(',', '.');
  const v = Number(s);
  return Number.isFinite(v) ? v : null;
}

/** "1:02.500" / "0:00:01.25" → ms. */
function parseTimecode(raw) {
  const s = String(raw).trim().replace(/^"|"$/g, '').replace(',', '.');
  if (!/^\d+(:\d+){1,2}(\.\d+)?$/.test(s)) return null;
  return s.split(':').reduce((acc, part) => acc * 60 + Number(part), 0) * 1000;
}

/**
 * @param {string} text
 * @param {{ timeUnit?: 'auto'|'ms'|'s' }} [opts]
 */
export function parseKinoveaCsv(text, { timeUnit = 'auto' } = {}) {
  const lines = String(text).replace(/^﻿/, '').split(/\r?\n/).filter(l => l.trim() !== '');
  if (!lines.length) throw new Error('empty reference file');
  const delimiter = detectDelimiter(lines);
  const split = l => splitFields(l, delimiter);
  // Header = first row whose first cell is not numeric (Kinovea may prepend a title row).
  let h = 0;
  while (h < lines.length && parseNumber(split(lines[h])[0], false) !== null) h++;
  while (h + 1 < lines.length && split(lines[h + 1])[0] && parseNumber(split(lines[h + 1])[0], delimiter !== ',') === null
    && parseTimecode(split(lines[h + 1])[0]) === null) h++;
  if (h >= lines.length) throw new Error('no header row found');
  const header = split(lines[h]);
  const decimalComma = delimiter === ';' || lines.slice(h + 1, h + 6).some(l => /\d,\d/.test(l) && delimiter !== ',');
  const rows = lines.slice(h + 1).map(split).filter(r => r.length >= 2);
  const timeName = header[0];
  let times = rows.map(r => parseTimecode(r[0]) ?? parseNumber(r[0], decimalComma));
  const isTimecode = rows.some(r => parseTimecode(r[0]) !== null);
  if (!isTimecode) {
    let unit = timeUnit;
    if (unit === 'auto') {
      if (/\(ms\)|\bms\b|millisec/i.test(timeName)) unit = 'ms';
      else if (/\(s\)|\bsec|\bs\b/i.test(timeName)) unit = 's';
      else {
        // Unlabelled: steps under 1 suggest seconds.
        const steps = times.slice(1).map((t, i) => t - times[i]).filter(d => d > 0);
        unit = steps.length && Math.min(...steps) < 1 ? 's' : 'ms';
      }
    }
    // Microsecond rounding removes seconds→ms float noise (0.05 s → 50, not 50.000000000000004).
    if (unit === 's') times = times.map(t => (t === null ? null : Math.round(t * 1e6) / 1e3));
  }
  const columns = {};
  header.slice(1).forEach((name, j) => {
    if (!name) return;
    columns[name] = rows.map(r => parseNumber(r[j + 1], decimalComma));
  });
  const keep = times.map(t => t !== null);
  return {
    timeMs: times.filter((_, i) => keep[i]),
    columns: Object.fromEntries(Object.entries(columns).map(([k, v]) => [k, v.filter((_, i) => keep[i])])),
  };
}

/** @param {string} text OpenSim .mot/.sto */
export function parseOpenSimMot(text) {
  const lines = String(text).split(/\r?\n/);
  const end = lines.findIndex(l => l.trim().toLowerCase() === 'endheader');
  if (end < 0) throw new Error('not an OpenSim storage file (no endheader)');
  const headerLines = lines.slice(0, end);
  const inDegrees = !headerLines.some(l => /^\s*inDegrees\s*=\s*no/i.test(l));
  const names = lines[end + 1].trim().split(/\s+/);
  const timeIdx = names.findIndex(n => n.toLowerCase() === 'time');
  if (timeIdx < 0) throw new Error('no time column');
  const rows = lines.slice(end + 2).filter(l => l.trim()).map(l => l.trim().split(/\s+/).map(Number));
  const columns = {};
  names.forEach((n, j) => {
    if (j === timeIdx) return;
    columns[n] = rows.map(r => (Number.isFinite(r[j]) ? (inDegrees ? r[j] : (r[j] * 180) / Math.PI) : null));
  });
  // Microsecond rounding removes seconds→ms float noise (16.002 s → 16002, not 16001.999…).
  return { timeMs: rows.map(r => Math.round(r[timeIdx] * 1e6) / 1e3), columns, inDegrees };
}

/** Picks a parser from the file name / content. */
export function parseReference(text, fileName = '') {
  if (/\.(mot|sto)$/i.test(fileName) || /^\s*endheader\s*$/im.test(text)) return { ...parseOpenSimMot(text), format: 'opensim' };
  return { ...parseKinoveaCsv(text), format: 'kinovea' };
}

/**
 * Reference → app-convention transforms. App angles are interior (180° = straight limb).
 * - 'interior': reference already measures the interior angle (Kinovea angle tool).
 * - 'flexion': reference is flexion from neutral (OpenSim knee_angle, elbow_flex): 180 − x.
 *   OpenSim knee_angle_r is negative-up in some models; pass 'flexion-abs' to use |x|.
 *   OpenSim hip_flexion is thigh vs pelvis while the app hip is trunk–thigh, so a hip
 *   comparison also contains pelvic tilt/lumbar flexion (expect a bias at deep flexion).
 */
export const REFERENCE_TRANSFORMS = Object.freeze({
  interior: x => x,
  flexion: x => 180 - x,
  'flexion-abs': x => 180 - Math.abs(x),
});

/**
 * Default transform for a reference column. OpenSim flexion coordinates map to the
 * app's interior angle with 180 − x; any other OpenSim coordinate (shoulder elevation,
 * which already matches the app's 0° = arm down, rotations, translations) has no safe
 * default and needs an explicit transform.
 */
export function defaultTransform(format, column = '') {
  if (format !== 'opensim') return 'interior';
  if (/^(knee_angle|elbow_flex|hip_flexion)/i.test(column)) return 'flexion';
  throw new Error(`no default transform for OpenSim column ${column}; pass ref:app:interior|flexion|flexion-abs`);
}
