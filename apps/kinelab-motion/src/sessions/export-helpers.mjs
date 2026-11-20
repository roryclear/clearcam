/**
 * Session Export helpers: CSV, JSON, and download utilities.
 */

export function exportToJSON(session) {
  if (!session) return '';
  return JSON.stringify(session, null, 2);
}

export function exportToCSV(session) {
  if (!session || !Array.isArray(session.frames) || session.frames.length === 0) {
    return 'elapsed_ms,elapsed_sec,phase,status,reps,leftKnee,rightKnee,leftHip,rightHip,leftElbow,rightElbow,leftShoulder,rightShoulder\n';
  }

  const headers = [
    'elapsed_ms',
    'elapsed_sec',
    'phase',
    'status',
    'reps',
    'leftKnee',
    'rightKnee',
    'leftHip',
    'rightHip',
    'leftElbow',
    'rightElbow',
    'leftShoulder',
    'rightShoulder',
  ];

  const rows = [headers.join(',')];

  for (const f of session.frames) {
    const sec = (f.timeMs / 1000).toFixed(3);
    const row = [
      f.timeMs,
      sec,
      `"${f.phase ?? ''}"`,
      `"${f.status ?? ''}"`,
      f.reps ?? 0,
      f.angles?.leftKnee ?? '',
      f.angles?.rightKnee ?? '',
      f.angles?.leftHip ?? '',
      f.angles?.rightHip ?? '',
      f.angles?.leftElbow ?? '',
      f.angles?.rightElbow ?? '',
      f.angles?.leftShoulder ?? '',
      f.angles?.rightShoulder ?? '',
    ];
    rows.push(row.join(','));
  }

  return rows.join('\n');
}

/**
 * Per-repetition table (one row per rep) from analyzeReps output.
 * @param {Array<Object>} reps
 * @param {string[]} [phaseLabels] e.g. ['down', 'up']
 */
export function exportRepsToCSV(reps, phaseLabels = ['phase1', 'phase2']) {
  const [p1, p2] = phaseLabels;
  const header = ['rep', 'start_ms', 'end_ms', 'duration_ms', `${p1}_ms`, `${p2}_ms`, 'start_angle', 'extreme_angle',
    'rom_deg', 'extreme_confidence', 'low_confidence_fraction', 'quality'];
  const rows = (reps || []).map(r => [r.index, Math.round(r.startMs), Math.round(r.endMs), r.durationMs, r.phase1Ms, r.phase2Ms,
    r.startAngle, r.extremeAngle, r.romDeg, r.extremeConfidence ?? '', r.lowConfidenceFraction, r.quality].join(','));
  return [header.join(','), ...rows].join('\n') + '\n';
}

export function downloadFile(content, filename, mimeType = 'text/plain') {
  if (typeof document === 'undefined') return;
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
