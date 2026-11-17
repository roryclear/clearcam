/**
 * Ghost Skeleton overlay renderer for reference movement comparison.
 */

import { SKELETON_CONNECTIONS } from '../biomechanics/landmark-indices.mjs';
import { reliable } from '../biomechanics/angle-calculations.mjs';

export class GhostRenderer {
  /**
   * Draws a faint reference ghost skeleton.
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array<Object>|null} ghostLandmarks
   * @param {Function} project Point projection function
   * @param {Object} options
   */
  static drawGhost(ctx, ghostLandmarks, project, {
    scale = 1.0,
    alpha = 0.35,
  } = {}) {
    if (!Array.isArray(ghostLandmarks) || ghostLandmarks.length === 0) return;

    ctx.save();
    ctx.setLineDash([4 * scale, 4 * scale]);
    ctx.lineWidth = Math.max(1.5, 2.0 * scale);
    ctx.strokeStyle = `rgba(180, 160, 255, ${alpha})`;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (const [aIdx, bIdx] of SKELETON_CONNECTIONS) {
      const a = ghostLandmarks[aIdx];
      const b = ghostLandmarks[bIdx];
      if (!reliable(a, 0.4) || !reliable(b, 0.4)) continue;

      const pa = project(a);
      const pb = project(b);

      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    // Ghost joints
    ctx.setLineDash([]);
    ctx.fillStyle = `rgba(180, 160, 255, ${alpha * 1.2})`;
    const r = Math.max(2, 3 * scale);
    for (const [aIdx] of SKELETON_CONNECTIONS) {
      const a = ghostLandmarks[aIdx];
      if (!reliable(a, 0.4)) continue;
      const pa = project(a);
      ctx.beginPath();
      ctx.arc(pa.x, pa.y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }
}
