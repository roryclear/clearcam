/**
 * Movement direction indicators: per-landmark velocity from measured frames.
 * Pure math only - returns null when there is no valid measurement, so the
 * renderer never invents motion for missing or stale landmarks.
 */

export function computeVelocity(prev, curr, dtMs) {
  if (!prev || !curr || !Number.isFinite(prev.x) || !Number.isFinite(prev.y) ||
      !Number.isFinite(curr.x) || !Number.isFinite(curr.y) ||
      !Number.isFinite(dtMs) || dtMs <= 0) {
    return null;
  }
  const dtSec = dtMs / 1000;
  const vx = (curr.x - prev.x) / dtSec;
  const vy = (curr.y - prev.y) / dtSec;
  return { vx, vy, speed: Math.hypot(vx, vy), angleDeg: Math.atan2(vy, vx) * 180 / Math.PI };
}

export function directionVisible(velocity, minSpeed = 0.15) {
  return Boolean(velocity && Number.isFinite(velocity.speed) && velocity.speed >= minSpeed);
}

export function drawDirectionArrows(ctx, landmarks, prevLandmarks, dtMs, project, { indices = [15, 16, 27, 28], minVisibility = 0.55, scale = 1 } = {}) {
  if (!Array.isArray(landmarks) || !Array.isArray(prevLandmarks)) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(130, 250, 222, 0.85)';
  ctx.fillStyle = 'rgba(130, 250, 222, 0.85)';
  ctx.lineWidth = 2 * scale;
  for (const i of indices) {
    const curr = landmarks[i];
    const prev = prevLandmarks[i];
    if (!curr || (curr.visibility !== undefined && curr.visibility < minVisibility)) continue;
    const v = computeVelocity(prev, curr, dtMs);
    if (!directionVisible(v)) continue;
    const from = project(curr);
    const len = Math.min(46 * scale, v.speed * 30 * scale);
    const ux = v.vx / v.speed;
    const uy = v.vy / v.speed;
    const tx = from.x + ux * len;
    const ty = from.y + uy * len;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(tx, ty, 3 * scale, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
