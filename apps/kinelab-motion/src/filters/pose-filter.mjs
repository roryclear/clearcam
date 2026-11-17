/**
 * Composite Pose Filter: Temporal smoothing (One Euro), outlier rejection,
 * and confidence-aware landmark management for 33 MediaPipe body landmarks.
 */

import { OneEuroPoint2D } from './one-euro-filter.mjs';
import { OutlierFilter } from './outlier-filter.mjs';

export const SMOOTHING_PRESETS = Object.freeze({
  off: { minCutoff: 30.0, beta: 0.0, dCutoff: 1.0, label: 'Off / Direct' },
  low: { minCutoff: 1.6, beta: 0.015, dCutoff: 1.0, label: 'Low (Fastest)' },
  medium: { minCutoff: 1.0, beta: 0.007, dCutoff: 1.0, label: 'Medium (Balanced)' },
  high: { minCutoff: 0.5, beta: 0.003, dCutoff: 1.0, label: 'High (Smooth)' },
});

export class PoseFilter {
  /**
   * @param {Object} options
   * @param {string} [options.preset='medium'] 'off' | 'low' | 'medium' | 'high'
   * @param {number} [options.minVisibility=0.55] Visibility threshold
   * @param {number} [options.maxSpeed=4.5] Speed outlier limit
   */
  constructor({ preset = 'medium', minVisibility = 0.55, maxSpeed = 4.5 } = {}) {
    this.minVisibility = minVisibility;
    this.presetName = preset in SMOOTHING_PRESETS ? preset : 'medium';
    this.presetConfig = SMOOTHING_PRESETS[this.presetName];

    this.filters = new Map(); // index -> OneEuroPoint2D
    this.outlierFilter = new OutlierFilter({ maxSpeed });
    this.lastSeenTimes = new Map(); // index -> timestampMs
  }

  setPreset(preset) {
    if (preset in SMOOTHING_PRESETS) {
      this.presetName = preset;
      this.presetConfig = SMOOTHING_PRESETS[preset];
      this.reset();
    }
  }

  setVisibilityThreshold(threshold) {
    if (Number.isFinite(threshold) && threshold >= 0 && threshold <= 1) {
      this.minVisibility = threshold;
    }
  }

  /**
   * Filter an array of 33 landmarks from MediaPipe.
   * @param {Array<Object>|null} rawLandmarks
   * @param {number} timestampMs
   * @returns {Array<Object>|null} Filtered landmarks
   */
  filter(rawLandmarks, timestampMs) {
    if (!Array.isArray(rawLandmarks) || rawLandmarks.length === 0) {
      this.reset();
      return null;
    }

    const filtered = new Array(rawLandmarks.length);

    for (let i = 0; i < rawLandmarks.length; i++) {
      const raw = rawLandmarks[i];
      if (!raw || !Number.isFinite(raw.x) || !Number.isFinite(raw.y)) {
        this.resetLandmark(i);
        filtered[i] = raw;
        continue;
      }

      const vis = raw.visibility ?? 1.0;
      const pres = raw.presence ?? 1.0;
      const isReliable = vis >= this.minVisibility && pres >= this.minVisibility;

      if (!isReliable) {
        // Occluded or low-confidence: do not feed into One Euro filter to avoid dragging
        this.resetLandmark(i);
        filtered[i] = { ...raw, isFiltered: false, isReliable: false };
        continue;
      }

      // Check if landmark was missing for more than 400ms. If so, reset filter state to prevent teleport drift.
      const lastSeen = this.lastSeenTimes.get(i);
      if (lastSeen !== undefined && timestampMs - lastSeen > 400) {
        this.resetLandmark(i);
      }
      this.lastSeenTimes.set(i, timestampMs);

      // Outlier rejection
      const { point: outlierCheckedPoint, wasRejected } = this.outlierFilter.evaluate(i, raw, timestampMs);
      const targetPoint = outlierCheckedPoint || raw;

      if (this.presetName === 'off') {
        filtered[i] = {
          ...targetPoint,
          isFiltered: false,
          isReliable: true,
          wasOutlier: wasRejected,
        };
        continue;
      }

      let euro = this.filters.get(i);
      if (!euro) {
        euro = new OneEuroPoint2D(this.presetConfig);
        this.filters.set(i, euro);
      }

      const smoothed = euro.filter(targetPoint.x, targetPoint.y, timestampMs);

      filtered[i] = {
        x: smoothed.x,
        y: smoothed.y,
        z: raw.z ?? 0,
        visibility: vis,
        presence: pres,
        isFiltered: true,
        isReliable: true,
        wasOutlier: wasRejected,
      };
    }

    return filtered;
  }

  resetLandmark(index) {
    this.filters.get(index)?.reset();
    this.filters.delete(index);
    this.outlierFilter.reset(index);
    this.lastSeenTimes.delete(index);
  }

  reset() {
    this.filters.forEach(f => f.reset());
    this.filters.clear();
    this.outlierFilter.reset();
    this.lastSeenTimes.clear();
  }
}
