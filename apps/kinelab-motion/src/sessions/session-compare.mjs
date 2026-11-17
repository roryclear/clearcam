/**
 * Session comparison: measured deltas between two recorded sessions.
 * Only finite recorded angles contribute; missing data yields null, never 0.
 */

function finiteAngles(session, angleKey) {
  if (!session || !Array.isArray(session.frames)) return [];
  return session.frames.map(f => f?.angles?.[angleKey]).filter(Number.isFinite);
}

export function compareSessions(a, b, angleKey) {
  const va = finiteAngles(a, angleKey);
  const vb = finiteAngles(b, angleKey);
  const paired = Math.min(va.length, vb.length);
  let meanDeltaDeg = null;
  let maxDeltaDeg = null;
  if (paired > 0) {
    let sum = 0;
    let max = 0;
    for (let i = 0; i < paired; i++) {
      const d = Math.abs(vb[i] - va[i]);
      sum += d;
      if (d > max) max = d;
    }
    meanDeltaDeg = Math.round((sum / paired) * 100) / 100;
    maxDeltaDeg = Math.round(max * 100) / 100;
  }
  return {
    angleKey,
    meanDeltaDeg,
    maxDeltaDeg,
    repDelta: (b?.finalReps ?? 0) - (a?.finalReps ?? 0),
    durationDeltaMs: (b?.durationMs ?? 0) - (a?.durationMs ?? 0),
    frameCountA: a?.frames?.length ?? 0,
    frameCountB: b?.frames?.length ?? 0,
  };
}
