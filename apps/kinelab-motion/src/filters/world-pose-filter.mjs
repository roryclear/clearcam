/**
 * Temporal smoothing for MediaPipe 3D world landmarks (meters, hip-centred).
 *
 * Monocular depth is far noisier than the image-plane axes, so z gets a lower
 * cutoff than x/y. Without this, 3D joint angles were computed from raw world
 * landmarks every frame while only the 2D landmarks were smoothed.
 */

import { OneEuroFilter } from './one-euro-filter.mjs';

// Tuned with bench/accuracy-bench.mjs (world units are meters; z is the noisy depth axis).
export const WORLD_SMOOTHING_PRESETS = Object.freeze({
  off: null,
  low: { xy: { minCutoff: 2.0, beta: 20 }, z: { minCutoff: 0.6, beta: 2 } },
  medium: { xy: { minCutoff: 2.0, beta: 20 }, z: { minCutoff: 0.3, beta: 0.5 } },
  high: { xy: { minCutoff: 1.0, beta: 8 }, z: { minCutoff: 0.2, beta: 0.3 } },
});

export class WorldPoseFilter {
  constructor({ preset = 'medium', minVisibility = 0.55 } = {}) {
    this.minVisibility = minVisibility;
    this.setPreset(preset);
  }

  setPreset(preset) {
    this.presetName = preset in WORLD_SMOOTHING_PRESETS ? preset : 'medium';
    this.config = WORLD_SMOOTHING_PRESETS[this.presetName];
    this.reset();
  }

  setVisibilityThreshold(threshold) {
    if (Number.isFinite(threshold) && threshold >= 0 && threshold <= 1) this.minVisibility = threshold;
  }

  _filtersFor(i) {
    let f = this.filters.get(i);
    if (!f) {
      f = {
        x: new OneEuroFilter(this.config.xy),
        y: new OneEuroFilter(this.config.xy),
        z: new OneEuroFilter(this.config.z),
        lastSeen: null,
      };
      this.filters.set(i, f);
    }
    return f;
  }

  /**
   * @param {Array<Object>|null} worldLandmarks
   * @param {number} timestampMs
   * @returns {Array<Object>|null}
   */
  filter(worldLandmarks, timestampMs) {
    if (!Array.isArray(worldLandmarks) || worldLandmarks.length === 0) {
      this.reset();
      return worldLandmarks ?? null;
    }
    if (!this.config) return worldLandmarks;

    return worldLandmarks.map((p, i) => {
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) {
        this.filters.delete(i);
        return p;
      }
      const vis = p.visibility ?? 1;
      const pres = p.presence ?? 1;
      if (vis < this.minVisibility || pres < this.minVisibility) {
        // Do not drag a smoothed estimate through occlusion; restart on reacquisition.
        this.filters.delete(i);
        return { ...p };
      }
      let f = this._filtersFor(i);
      if (f.lastSeen !== null && timestampMs - f.lastSeen > 400) {
        // Long gap: stale state would drag the estimate, so start fresh.
        this.filters.delete(i);
        f = this._filtersFor(i);
      }
      f.lastSeen = timestampMs;
      return {
        ...p,
        x: f.x.filter(p.x, timestampMs),
        y: f.y.filter(p.y, timestampMs),
        z: f.z.filter(p.z ?? 0, timestampMs),
      };
    });
  }

  reset() {
    this.filters = new Map();
  }
}
