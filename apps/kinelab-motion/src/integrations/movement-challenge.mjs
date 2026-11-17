/**
 * Movement Challenge & Interactive Gamification Engine.
 * Implements lightweight, interactive movement challenges without degrading tracking performance.
 */

import { JOINTS } from '../biomechanics/landmark-indices.mjs';
import { reliable } from '../biomechanics/angle-calculations.mjs';
import { defaultAudioSynthesizer } from './audio-cues.mjs';

export const CHALLENGE_TYPES = Object.freeze({
  OVERHEAD_REACH: 'overhead_reach',
  SQUAT_HOLD: 'squat_hold',
  BALANCE_CHALLENGE: 'balance_challenge',
});

export class MovementChallenge {
  /**
   * @param {Object} options
   * @param {string} [options.type='overhead_reach']
   * @param {number} [options.targetDurationSec=10]
   */
  constructor({ type = CHALLENGE_TYPES.OVERHEAD_REACH, targetDurationSec = 10 } = {}) {
    this.type = type;
    this.targetDurationSec = targetDurationSec;
    this.isActive = false;
    this.score = 0;
    this.targetHeightNorm = 0.25; // Screen Y coordinate target for overhead reach (0.0 to 1.0)
    this.holdTimeMs = 0;
    this.startTimeMs = null;
    this.isCompleted = false;
  }

  start() {
    this.isActive = true;
    this.score = 0;
    this.holdTimeMs = 0;
    this.startTimeMs = performance.now();
    this.isCompleted = false;
  }

  stop() {
    this.isActive = false;
  }

  reset() {
    this.start();
  }

  /**
   * Evaluates the current frame against active challenge conditions.
   * @param {Array<Object>|null} landmarks 33 MediaPipe landmarks
   * @param {Object} angles Joint angles object
   * @param {number} dtMs Delta time in milliseconds
   * @returns {{ status: string, progress: number, score: number, isCompleted: boolean, feedback: string }}
   */
  update(landmarks, angles, dtMs) {
    if (!this.isActive || !Array.isArray(landmarks)) {
      return {
        status: 'inactive',
        progress: 0,
        score: this.score,
        isCompleted: this.isCompleted,
        feedback: 'Challenge paused.',
      };
    }

    if (this.type === CHALLENGE_TYPES.OVERHEAD_REACH) {
      const lw = landmarks[JOINTS.leftWrist];
      const rw = landmarks[JOINTS.rightWrist];

      if (!reliable(lw) && !reliable(rw)) {
        return {
          status: 'tracking_lost',
          progress: 0,
          score: this.score,
          isCompleted: this.isCompleted,
          feedback: 'Keep wrists visible to reach the target line.',
        };
      }

      const leftAbove = reliable(lw) && lw.y <= this.targetHeightNorm;
      const rightAbove = reliable(rw) && rw.y <= this.targetHeightNorm;

      if (leftAbove && rightAbove) {
        this.holdTimeMs += dtMs;
        const progress = Math.min(1.0, this.holdTimeMs / (this.targetDurationSec * 1000));
        this.score = Math.floor(this.holdTimeMs / 100);

        if (progress >= 1.0 && !this.isCompleted) {
          this.isCompleted = true;
          defaultAudioSynthesizer.playRepCount();
          return {
            status: 'completed',
            progress: 1.0,
            score: this.score,
            isCompleted: true,
            feedback: '🎉 Challenge Complete! Target held successfully.',
          };
        }

        return {
          status: 'holding',
          progress,
          score: this.score,
          isCompleted: this.isCompleted,
          feedback: `Holding target! (${(this.holdTimeMs / 1000).toFixed(1)}s / ${this.targetDurationSec}s)`,
        };
      } else {
        return {
          status: 'reach_higher',
          progress: Math.min(1.0, this.holdTimeMs / (this.targetDurationSec * 1000)),
          score: this.score,
          isCompleted: this.isCompleted,
          feedback: 'Raise both hands above the target line!',
        };
      }
    }

    return {
      status: 'active',
      progress: 0,
      score: this.score,
      isCompleted: false,
      feedback: 'Movement challenge active.',
    };
  }

  /**
   * Draws visual challenge indicators (such as target lines or progress bars) on canvas.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} width
   * @param {number} height
   * @param {number} scale
   */
  draw(ctx, width, height, scale = 1.0) {
    if (!this.isActive) return;

    if (this.type === CHALLENGE_TYPES.OVERHEAD_REACH) {
      const targetY = this.targetHeightNorm * height;

      ctx.save();
      // Draw neon target line
      ctx.beginPath();
      ctx.setLineDash([8 * scale, 6 * scale]);
      ctx.moveTo(0, targetY);
      ctx.lineTo(width, targetY);
      ctx.strokeStyle = this.isCompleted ? '#86f8d8' : '#ffd166';
      ctx.lineWidth = 2 * scale;
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Target Badge
      const badgeText = this.isCompleted ? '★ TARGET ACHIEVED ★' : '▲ REACH ABOVE THIS LINE ▲';
      ctx.font = `700 ${10 * scale}px ui-monospace, monospace`;
      const tw = ctx.measureText(badgeText).width;
      const bx = width / 2 - tw / 2 - 8 * scale;
      const by = targetY - 18 * scale;

      ctx.fillStyle = 'rgba(7, 18, 32, 0.85)';
      ctx.strokeStyle = this.isCompleted ? '#86f8d8' : '#ffd166';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(bx, by, tw + 16 * scale, 16 * scale, 4 * scale);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = this.isCompleted ? '#86f8d8' : '#ffd166';
      ctx.fillText(badgeText, bx + 8 * scale, by + 12 * scale);
      ctx.restore();
    }
  }
}
