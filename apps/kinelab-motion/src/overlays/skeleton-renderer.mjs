/**
 * High-legibility, confidence-aware skeleton and joint rendering.
 */

import { SKELETON_CONNECTIONS } from '../biomechanics/landmark-indices.mjs';
import { reliable } from '../biomechanics/angle-calculations.mjs';

export class SkeletonRenderer {
  /**
   * Draws full-body skeletal connections.
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array<Object>} landmarks
   * @param {Function} project Point projection function (x, y) -> { x, y } in canvas pixels
   * @param {Object} options
   */
  static drawConnections(ctx, landmarks, project, {
    scale = 1.0,
    minVisibility = 0.55,
    reducedMotion = false,
  } = {}) {
    if (!Array.isArray(landmarks) || landmarks.length === 0) return;

    ctx.save();
    ctx.lineWidth = Math.max(1.8, 3.0 * scale);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    if (!reducedMotion) {
      ctx.shadowColor = '#65f1d4';
      ctx.shadowBlur = 8 * scale;
    }

    for (const [aIdx, bIdx] of SKELETON_CONNECTIONS) {
      const a = landmarks[aIdx];
      const b = landmarks[bIdx];
      if (!reliable(a, minVisibility) || !reliable(b, minVisibility)) continue;

      const pa = project(a);
      const pb = project(b);

      const avgVis = ((a.visibility ?? 1) + (b.visibility ?? 1)) / 2;
      const alpha = Math.max(0.35, Math.min(1.0, avgVis));

      ctx.strokeStyle = `rgba(125, 246, 222, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    ctx.restore();
  }

  /**
   * Draws circular markers at active body joints.
   */
  static drawJointMarkers(ctx, landmarks, project, {
    scale = 1.0,
    minVisibility = 0.55,
  } = {}) {
    if (!Array.isArray(landmarks) || landmarks.length === 0) return;

    const renderedIndices = new Set(SKELETON_CONNECTIONS.flat());
    const radius = Math.max(3, 5 * scale);

    ctx.save();
    ctx.lineWidth = Math.max(1.5, 2 * scale);
    ctx.strokeStyle = '#081220';

    for (const idx of renderedIndices) {
      const pt = landmarks[idx];
      if (!reliable(pt, minVisibility)) continue;

      const p = project(pt);
      const isCoreJoint = idx >= 11 && idx <= 28;

      ctx.fillStyle = isCoreJoint ? '#b8adff' : '#7af2da';
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    ctx.restore();
  }
}
