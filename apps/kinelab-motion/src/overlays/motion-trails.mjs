/**
 * Lightweight motion trails for tracking trajectory and velocity of movement extremities.
 */

import { JOINTS } from '../biomechanics/landmark-indices.mjs';
import { reliable } from '../biomechanics/angle-calculations.mjs';

const TRAIL_JOINTS = [
  JOINTS.leftWrist,
  JOINTS.rightWrist,
  JOINTS.leftAnkle,
  JOINTS.rightAnkle,
];

export class MotionTrails {
  constructor({ maxPoints = 6 } = {}) {
    this.maxPoints = maxPoints;
    this.histories = new Map(); // jointIndex -> Array<{ x, y, timestampMs }>
  }

  update(landmarks, timestampMs) {
    if (!Array.isArray(landmarks)) {
      this.clear();
      return;
    }

    for (const joint of TRAIL_JOINTS) {
      const pt = landmarks[joint];
      if (!reliable(pt, 0.55)) {
        this.histories.delete(joint);
        continue;
      }

      let trail = this.histories.get(joint);
      if (!trail) {
        trail = [];
        this.histories.set(joint, trail);
      }

      trail.push({ x: pt.x, y: pt.y, timestampMs });
      if (trail.length > this.maxPoints) {
        trail.shift();
      }
    }
  }

  draw(ctx, project, { scale = 1.0, reducedMotion = false } = {}) {
    if (reducedMotion || this.histories.size === 0) return;

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [joint, trail] of this.histories.entries()) {
      if (trail.length < 2) continue;

      const isWrist = joint === JOINTS.leftWrist || joint === JOINTS.rightWrist;
      const baseRgb = isWrist ? '176, 165, 255' : '129, 251, 224';

      for (let i = 1; i < trail.length; i++) {
        const p1 = project(trail[i - 1]);
        const p2 = project(trail[i]);
        const alpha = (i / trail.length) * 0.45;
        const width = Math.max(1, (i / trail.length) * 3 * scale);

        ctx.strokeStyle = `rgba(${baseRgb}, ${alpha})`;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  clear() {
    this.histories.clear();
  }
}
