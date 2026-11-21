/**
 * Angle Arc and Degree Label renderer for live joints.
 */

import { formatAngle } from '../biomechanics/angle-convention.mjs';
import { ANGLE_DEFINITIONS, reliable } from '../biomechanics/angle-calculations.mjs';

export class AngleArcRenderer {
  /**
   * Draws circular angle sweep arcs at joints with measured angles.
   */
  static drawArcs(ctx, landmarks, angles, project, {
    scale = 1.0,
    minVisibility = 0.55,
  } = {}) {
    if (!landmarks || !angles) return;

    ctx.save();
    const arcRadius = Math.max(16, 26 * scale);

    for (const def of ANGLE_DEFINITIONS) {
      const angle = angles[def.key];
      if (!Number.isFinite(angle)) continue;

      const pFrom = landmarks[def.from];
      const pJoint = landmarks[def.joint];
      const pTo = landmarks[def.to];

      if (!reliable(pFrom, minVisibility) ||
          !reliable(pJoint, minVisibility) ||
          !reliable(pTo, minVisibility)) continue;

      const pj = project(pJoint);
      const pf = project(pFrom);
      const pt = project(pTo);

      const angleFrom = Math.atan2(pf.y - pj.y, pf.x - pj.x);
      const angleTo = Math.atan2(pt.y - pj.y, pt.x - pj.x);

      // Determine shortest sweep direction
      let diff = angleTo - angleFrom;
      while (diff < -Math.PI) diff += 2 * Math.PI;
      while (diff > Math.PI) diff -= 2 * Math.PI;

      const counterClockwise = diff < 0;

      // Draw sector wedge
      ctx.beginPath();
      ctx.moveTo(pj.x, pj.y);
      ctx.arc(pj.x, pj.y, arcRadius, angleFrom, angleFrom + diff, counterClockwise);
      ctx.closePath();
      ctx.fillStyle = 'rgba(128, 246, 221, 0.18)';
      ctx.fill();

      // Draw outer arc line
      ctx.beginPath();
      ctx.arc(pj.x, pj.y, arcRadius, angleFrom, angleFrom + diff, counterClockwise);
      ctx.lineWidth = Math.max(1.5, 2.2 * scale);
      ctx.strokeStyle = '#8efbe4';
      ctx.stroke();
    }

    ctx.restore();
  }

  /**
   * Draws readable degree labels beside tracked joints.
   */
  static drawLabels(ctx, landmarks, angles, project, canvasWidth, canvasHeight, {
    scale = 1.0,
    minVisibility = 0.55,
  } = {}) {
    if (!landmarks || !angles) return;

    ctx.save();
    const fontSize = Math.max(11, 14 * scale);
    ctx.font = `700 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;

    for (const def of ANGLE_DEFINITIONS) {
      const angle = angles[def.key];
      if (!Number.isFinite(angle)) continue;

      const pJoint = landmarks[def.joint];
      if (!reliable(pJoint, minVisibility)) continue;

      const pj = project(pJoint);
      const text = formatAngle(def.key, angle);
      const textMetrics = ctx.measureText(text);
      const padX = 6 * scale;
      const padY = 4 * scale;
      const boxWidth = textMetrics.width + padX * 2;
      const boxHeight = fontSize + padY * 2;

      // Position to side of joint based on left/right limb
      const isLeft = def.key.startsWith('left');
      const offsetX = isLeft ? -boxWidth - 10 * scale : 10 * scale;
      const offsetY = -boxHeight / 2;

      const x = Math.max(6, Math.min(canvasWidth - boxWidth - 6, pj.x + offsetX));
      const y = Math.max(6, Math.min(canvasHeight - boxHeight - 6, pj.y + offsetY));

      // Dark translucent pill badge
      ctx.fillStyle = 'rgba(7, 18, 32, 0.88)';
      ctx.strokeStyle = 'rgba(120, 245, 222, 0.65)';
      ctx.lineWidth = 1;

      ctx.beginPath();
      ctx.roundRect(x, y, boxWidth, boxHeight, 4 * scale);
      ctx.fill();
      ctx.stroke();

      // Text label
      ctx.fillStyle = '#a6ffe9';
      ctx.fillText(text, x + padX, y + boxHeight - padY - 1);
    }

    ctx.restore();
  }
}
