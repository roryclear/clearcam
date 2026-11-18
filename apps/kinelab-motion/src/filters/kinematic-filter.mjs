/**
 * Kinematic Bone-Length Invariance Filter
 *
 * Implements anthropometric bone-length consistency for 3D monocular landmarks.
 * Monocular depth (Z) is notoriously noisy, causing limb lengths to fluctuate
 * by up to 30-40% during dynamic movement. This filter learns/calibrates the
 * rigid bone lengths of the participant and constrains child joints along their
 * estimated 3D unit direction ray to the invariant bone length, eliminating
 * depth jitter and perspective foreshortening distortion without lag.
 *
 * Reference:
 * ArXiv:2512.06783 "Physics Informed Human Posture Estimation from Monocular RGB-Videos"
 * Sports2D Biomechanical Invariant Model (Pagnon et al., 2024)
 */

import { JOINTS } from '../biomechanics/landmark-indices.mjs';

export const KINEMATIC_SEGMENTS = Object.freeze([
  { key: 'leftTorso', parent: JOINTS.leftShoulder, child: JOINTS.leftHip },
  { key: 'rightTorso', parent: JOINTS.rightShoulder, child: JOINTS.rightHip },
  { key: 'leftThigh', parent: JOINTS.leftHip, child: JOINTS.leftKnee },
  { key: 'rightThigh', parent: JOINTS.rightHip, child: JOINTS.rightKnee },
  { key: 'leftShank', parent: JOINTS.leftKnee, child: JOINTS.leftAnkle },
  { key: 'rightShank', parent: JOINTS.rightKnee, child: JOINTS.rightAnkle },
  { key: 'leftUpperArm', parent: JOINTS.leftShoulder, child: JOINTS.leftElbow },
  { key: 'rightUpperArm', parent: JOINTS.rightShoulder, child: JOINTS.rightElbow },
  { key: 'leftForearm', parent: JOINTS.leftElbow, child: JOINTS.leftWrist },
  { key: 'rightForearm', parent: JOINTS.rightElbow, child: JOINTS.rightWrist },
]);

export class KinematicBoneFilter {
  /**
   * @param {Object} [options]
   * @param {number} [options.minVisibility=0.55]
   * @param {number} [options.calibrationAlpha=0.15] EMA adaptation rate during calibration
   * @param {number} [options.minCalibrationFrames=10]
   */
  constructor({
    minVisibility = 0.55,
    calibrationAlpha = 0.15,
    minCalibrationFrames = 10,
  } = {}) {
    this.minVisibility = minVisibility;
    this.calibrationAlpha = calibrationAlpha;
    this.minCalibrationFrames = minCalibrationFrames;

    // Segment lengths: key -> { length: number, frameCount: number }
    this.calibratedLengths = new Map();
    for (const seg of KINEMATIC_SEGMENTS) {
      this.calibratedLengths.set(seg.key, { length: null, frameCount: 0 });
    }
  }

  isReliable(pt) {
    if (!pt) return false;
    const vis = pt.visibility ?? 1.0;
    const pres = pt.presence ?? 1.0;
    return (
      Number.isFinite(pt.x) &&
      Number.isFinite(pt.y) &&
      Number.isFinite(pt.z ?? 0) &&
      vis >= this.minVisibility &&
      pres >= this.minVisibility
    );
  }

  /**
   * Return copy of current calibrated bone lengths in meters.
   */
  getCalibratedLengths() {
    const res = {};
    for (const [k, v] of this.calibratedLengths.entries()) {
      res[k] = v.length !== null ? Math.round(v.length * 1000) / 1000 : null;
    }
    return res;
  }

  /**
   * Filter 3D world landmarks by enforcing anatomical bone length invariance.
   * @param {Array<Object>|null} worldLandmarks 33 3D landmarks in metric space
   * @param {number} [timestampMs]
   * @returns {Array<Object>|null} Corrected 3D landmarks
   */
  filter(worldLandmarks, timestampMs = 0) {
    if (!Array.isArray(worldLandmarks) || worldLandmarks.length === 0) {
      return worldLandmarks;
    }

    // Clone landmarks to prevent mutating caller's original array
    const out = worldLandmarks.map(pt => (pt ? { ...pt } : null));

    // 1. Update calibration and enforce constraints on kinematic segments
    for (const seg of KINEMATIC_SEGMENTS) {
      const pParent = out[seg.parent];
      const pChild = out[seg.child];

      if (!this.isReliable(pParent) || !this.isReliable(pChild)) {
        continue;
      }

      const dx = pChild.x - pParent.x;
      const dy = pChild.y - pParent.y;
      const dz = (pChild.z ?? 0) - (pParent.z ?? 0);
      const measuredDist = Math.hypot(dx, dy, dz);

      if (measuredDist < 1e-4) continue;

      const segData = this.calibratedLengths.get(seg.key);
      if (segData.length === null) {
        segData.length = measuredDist;
        segData.frameCount = 1;
      } else {
        // Adapt calibration during initial frames or slow continuous tracking
        const alpha = segData.frameCount < this.minCalibrationFrames ? 0.3 : 0.02;
        segData.length = (1 - alpha) * segData.length + alpha * measuredDist;
        segData.frameCount++;
      }

      // If calibrated length is established, enforce bone length along direction ray
      if (segData.frameCount >= 5 && segData.length > 0.05) {
        const targetLen = segData.length;
        const scale = targetLen / measuredDist;

        pChild.x = pParent.x + dx * scale;
        pChild.y = pParent.y + dy * scale;
        pChild.z = (pParent.z ?? 0) + dz * scale;
      }
    }

    return out;
  }

  reset() {
    for (const seg of KINEMATIC_SEGMENTS) {
      this.calibratedLengths.set(seg.key, { length: null, frameCount: 0 });
    }
  }
}
