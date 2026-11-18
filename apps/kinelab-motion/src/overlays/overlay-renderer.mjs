/**
 * Master overlay renderer: manages canvas scaling, projection, and configurable visual layers.
 */

import { SkeletonRenderer } from './skeleton-renderer.mjs';
import { AngleArcRenderer } from './angle-arc-renderer.mjs';
import { MotionTrails } from './motion-trails.mjs';
import { GhostRenderer } from './ghost-renderer.mjs';
import { drawDirectionArrows } from './direction-indicators.mjs';

export class OverlayRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.trails = new MotionTrails();
    this.ghostPose = null;
    this.prevLandmarks = null;
    this.prevTimestampMs = null;
  }

  setGhostPose(landmarks) {
    this.ghostPose = Array.isArray(landmarks) ? landmarks : null;
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.trails.clear();
  }

  /**
   * Main render pass.
   * @param {Object} renderData
   * @param {Array<Object>|null} renderData.landmarks
   * @param {Object} renderData.angles
   * @param {number} renderData.timestampMs
   * @param {Object} options Configuration flags
   */
  render(renderData, options = {}) {
    const {
      landmarks = null,
      angles = {},
      timestampMs = performance.now(),
      holdProgress = null, // { current, target } or null
      framing = null, // { status: 'optimal'|'partial'|'inadequate', guidance: string }
    } = renderData;

    const {
      showSkeleton = true,
      showJointMarkers = true,
      showDegreeLabels = true,
      showAngleArcs = true,
      showMotionTrails = false,
      showDirection = false,
      showGhost = false,
      mirrored = false,
      reducedMotion = false,
      minVisibility = 0.55,
    } = options;

    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    if (!landmarks && (!showGhost || !this.ghostPose)) return;

    // Normalizing projection function
    const project = p => ({
      x: (mirrored ? 1 - p.x : p.x) * width,
      y: p.y * height,
    });

    const scale = Math.max(0.7, Math.min(width / 800, height / 500));

    // 1. Ghost reference overlay (if enabled and present)
    if (showGhost && this.ghostPose) {
      GhostRenderer.drawGhost(ctx, this.ghostPose, project, { scale, alpha: 0.35 });
    }

    if (!Array.isArray(landmarks) || landmarks.length === 0) return;

    // 2. Motion trails
    if (showMotionTrails && !reducedMotion) {
      this.trails.update(landmarks, timestampMs);
      this.trails.draw(ctx, project, { scale, reducedMotion });
    }

    // 3. Skeleton connections
    if (showSkeleton) {
      SkeletonRenderer.drawConnections(ctx, landmarks, project, {
        scale,
        minVisibility,
        reducedMotion,
      });
    }

    // 4. Joint markers
    if (showJointMarkers) {
      SkeletonRenderer.drawJointMarkers(ctx, landmarks, project, {
        scale,
        minVisibility,
      });
    }

    // 5. Angle arcs
    if (showAngleArcs) {
      AngleArcRenderer.drawArcs(ctx, landmarks, angles, project, {
        scale,
        minVisibility,
      });
    }

    // 6. Degree labels
    if (showDegreeLabels) {
      AngleArcRenderer.drawLabels(ctx, landmarks, angles, project, width, height, {
        scale,
        minVisibility,
      });
    }

    // 6b. Movement direction arrows (measured velocity only; reduced-motion disables)
    if (showDirection && !reducedMotion && this.prevLandmarks && this.prevTimestampMs !== null) {
      drawDirectionArrows(ctx, landmarks, this.prevLandmarks, timestampMs - this.prevTimestampMs, project, { scale, minVisibility });
    }
    this.prevLandmarks = landmarks;
    this.prevTimestampMs = timestampMs;

    // 7. Optional hold progress ring
    if (holdProgress && Number.isFinite(holdProgress.current) && Number.isFinite(holdProgress.target)) {
      this.drawHoldRing(ctx, width, height, holdProgress, scale);
    }

    // 8. Body framing guidance banner (when framing is non-optimal)
    if (framing && framing.status && framing.status !== 'optimal' && framing.guidance) {
      this.drawFramingBanner(ctx, width, height, framing, scale);
    }
  }

  drawFramingBanner(ctx, width, height, framing, scale) {
    const isWarning = framing.status === 'inadequate';
    const text = framing.guidance;
    const fontSize = Math.max(11, Math.round(13 * scale));
    ctx.save();
    ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    const textMetrics = ctx.measureText(text);
    const paddingX = 14 * scale;
    const pillHeight = 30 * scale;
    const pillWidth = textMetrics.width + paddingX * 2 + 16 * scale;
    const cx = width / 2;
    const topY = 24 * scale;
    const rx = cx - pillWidth / 2;

    // Background pill
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(rx, topY, pillWidth, pillHeight, pillHeight / 2);
    } else {
      ctx.rect(rx, topY, pillWidth, pillHeight);
    }
    ctx.fillStyle = isWarning ? 'rgba(28, 18, 12, 0.88)' : 'rgba(14, 24, 38, 0.88)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = isWarning ? 'rgba(255, 179, 71, 0.6)' : 'rgba(90, 200, 250, 0.6)';
    ctx.stroke();

    // Status dot
    const dotX = rx + paddingX + 4 * scale;
    const dotY = topY + pillHeight / 2;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 4 * scale, 0, Math.PI * 2);
    ctx.fillStyle = isWarning ? '#ffb347' : '#5ac8fa';
    ctx.fill();

    // Guidance text
    ctx.fillStyle = isWarning ? '#ffe3b0' : '#e1f5fe';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, dotX + 10 * scale, dotY);
    ctx.restore();
  }

  drawHoldRing(ctx, width, height, { current, target }, scale) {
    const radius = 28 * scale;
    const cx = width / 2;
    const cy = 48 * scale;
    const progress = Math.min(1.0, current / target);

    ctx.save();
    // Track background
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 5 * scale;
    ctx.stroke();

    // Progress arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
    ctx.strokeStyle = '#82fade';
    ctx.lineWidth = 5 * scale;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Text
    ctx.font = `700 ${12 * scale}px ui-monospace, monospace`;
    ctx.fillStyle = '#a6ffe9';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${current.toFixed(1)}s`, cx, cy);
    ctx.restore();
  }
}
