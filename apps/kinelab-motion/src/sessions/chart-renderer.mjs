/**
 * Pure HTML5 Canvas joint-angle telemetry and timeline chart renderer.
 */

export class ChartRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  /**
   * Draws a joint-angle timeline graph.
   * @param {Object} options
   * @param {Array<Object>} options.frames Array of session frames
   * @param {string} [options.jointKey='leftKnee'] Key in angles object to plot
   * @param {number} [options.currentFrameIdx=0] Scrubber position
   */
  render({ frames = [], jointKey = 'leftKnee', currentFrameIdx = 0 } = {}) {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    if (!Array.isArray(frames) || frames.length < 2) {
      ctx.fillStyle = '#65748b';
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Record or load a session to view motion graph', width / 2, height / 2);
      return;
    }

    const padLeft = 34;
    const padRight = 14;
    const padTop = 14;
    const padBottom = 22;

    const chartW = width - padLeft - padRight;
    const chartH = height - padTop - padBottom;

    const maxTime = frames[frames.length - 1].timeMs || 1;
    const minAngle = 0; // shoulder angles at rest sit well below 40°
    const maxAngle = 180;

    // Draw horizontal grid lines and degree labels
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#6d7c92';
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'right';

    for (const deg of [0, 45, 90, 135, 180]) {
      const y = padTop + chartH - ((deg - minAngle) / (maxAngle - minAngle)) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.fillText(`${deg}°`, padLeft - 6, y + 3);
    }

    // Time ticks at bottom
    ctx.textAlign = 'center';
    const totalSec = Math.round(maxTime / 1000);
    const tickStep = totalSec > 10 ? 5 : 2;
    for (let s = 0; s <= totalSec; s += tickStep) {
      const x = padLeft + (s * 1000 / maxTime) * chartW;
      ctx.fillText(`${s}s`, x, height - 6);
    }

    // Plot angle points
    const points = [];
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      const val = f.angles?.[jointKey];
      if (Number.isFinite(val)) {
        const x = padLeft + (f.timeMs / maxTime) * chartW;
        const clampedVal = Math.max(minAngle, Math.min(maxAngle, val));
        const y = padTop + chartH - ((clampedVal - minAngle) / (maxAngle - minAngle)) * chartH;
        points.push({ x, y, idx: i });
      }
    }

    if (points.length >= 2) {
      // Area fill
      ctx.beginPath();
      ctx.moveTo(points[0].x, padTop + chartH);
      for (const p of points) ctx.lineTo(p.x, p.y);
      ctx.lineTo(points[points.length - 1].x, padTop + chartH);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + chartH);
      grad.addColorStop(0, 'rgba(125, 246, 222, 0.22)');
      grad.addColorStop(1, 'rgba(125, 246, 222, 0.01)');
      ctx.fillStyle = grad;
      ctx.fill();

      // Curve line
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.strokeStyle = '#81fbe0';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Current scrubber indicator line
    const curTime = frames[currentFrameIdx]?.timeMs ?? 0;
    const curX = padLeft + (curTime / maxTime) * chartW;
    ctx.strokeStyle = '#ff7ea5';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(curX, padTop);
    ctx.lineTo(curX, padTop + chartH);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
