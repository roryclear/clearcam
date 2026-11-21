/**
 * Body Framing Assistant: Evaluates camera field-of-view, limb presence,
 * and positioning to guide the user into the optimal exercise tracking zone.
 */

import { JOINTS } from '../biomechanics/landmark-indices.mjs';
import { LEVEL_TOLERANCE_DEG, PITCH_TOLERANCE_DEG } from './device-level.mjs';

/** Body height as a fraction of the frame: below this the person is too far away. */
const MIN_BODY_FRACTION = 0.4; // seated sit-to-stand starts are ~60% of standing height
/** Hips this far from the centre row (phone level) mean the camera is not at hip height. */
const HIP_ROW_TOLERANCE = 0.13;
const STANDING_KNEE_DEG = 160; // image-plane knee angle; setup prompts only when standing

export const EXERCISE_REQUIREMENTS = Object.freeze({
  'squat': ['upperBody', 'torso', 'lowerBody'],
  'sit-to-stand': ['upperBody', 'torso', 'lowerBody'],
  'sit_to_stand': ['upperBody', 'torso', 'lowerBody'],
  'lunge': ['upperBody', 'torso', 'lowerBody'],
  'single-leg-balance': ['upperBody', 'torso', 'lowerBody'],
  'single_leg_balance': ['upperBody', 'torso', 'lowerBody'],
  'shoulder-flexion': ['upperBody', 'torso'],
  'shoulder_flexion': ['upperBody', 'torso'],
  'shoulder-abduction': ['upperBody', 'torso'],
  'shoulder_abduction': ['upperBody', 'torso'],
  'free': ['upperBody'],
});

export class FramingAssistant {
  constructor({ minVisibility = 0.5 } = {}) {
    this.minVisibility = minVisibility;
  }

  _isReliable(pt) {
    return Boolean(
      pt &&
      Number.isFinite(pt.x) &&
      Number.isFinite(pt.y) &&
      (pt.visibility === undefined || pt.visibility >= this.minVisibility) &&
      (pt.presence === undefined || pt.presence >= this.minVisibility)
    );
  }

  _checkSegment(landmarks, indices) {
    if (!Array.isArray(landmarks)) return false;
    // Segment is present if at least half of the key indices are reliably detected
    const count = indices.filter(idx => this._isReliable(landmarks[idx])).length;
    return count >= Math.max(1, Math.floor(indices.length / 2));
  }

  /**
   * @param {Array<Object>|null} landmarks
   * @param {string} [exerciseId]
   * @param {Object} [view]
   * @param {number|null} [view.yawDeg] body yaw from ViewOrientationEstimator (0 = side-on)
   * @param {'side'|'front'|'any'} [view.recommendedView] the exercise's camera recommendation
   * @param {{rollDeg:number, pitchDeg:number}|null} [view.tilt] phone tilt (device-level.mjs)
   */
  evaluate(landmarks, exerciseId = 'free', { yawDeg = null, recommendedView = 'any', tilt = null, setupGuidance = true, aspect = 1 } = {}) {
    if (!Array.isArray(landmarks) || landmarks.length === 0) {
      return {
        status: 'inadequate',
        missingSegments: ['all'],
        guidance: 'Step into camera view — No body detected',
        segments: { head: false, upperBody: false, torso: false, lowerBody: false },
      };
    }

    const segments = {
      head: this._checkSegment(landmarks, [JOINTS.nose, JOINTS.leftEar, JOINTS.rightEar]),
      upperBody: this._checkSegment(landmarks, [JOINTS.leftShoulder, JOINTS.rightShoulder]),
      arms: this._checkSegment(landmarks, [JOINTS.leftElbow, JOINTS.rightElbow, JOINTS.leftWrist, JOINTS.rightWrist]),
      torso: this._checkSegment(landmarks, [JOINTS.leftHip, JOINTS.rightHip]),
      lowerBody: this._checkSegment(landmarks, [JOINTS.leftKnee, JOINTS.rightKnee, JOINTS.leftAnkle, JOINTS.rightAnkle]),
    };

    const normalizedKey = String(exerciseId || 'free').toLowerCase();
    const required = EXERCISE_REQUIREMENTS[normalizedKey] ||
      EXERCISE_REQUIREMENTS[normalizedKey.replace(/_/g, '-')] ||
      EXERCISE_REQUIREMENTS.free;
    const missingSegments = required.filter(seg => !segments[seg]);

    const prettyName = normalizedKey.replace(/[_-]/g, ' ');
    let status = 'optimal';
    let guidance = `Framing optimal for ${prettyName}`;

    if (missingSegments.length > 0) {
      if (missingSegments.includes('lowerBody')) {
        status = 'inadequate';
        guidance = `Step back: Knees and feet needed for ${prettyName}`;
      } else if (missingSegments.includes('upperBody') || missingSegments.includes('torso')) {
        status = 'inadequate';
        guidance = `Move into frame: Torso and shoulders needed`;
      } else {
        status = 'partial';
        guidance = `Adjust camera: Some limbs partially out of view`;
      }
    }

    // Camera setup: level phone, whole body at a reasonable size, camera at hip height.
    if (status === 'optimal') {
      const setup = setupGuidance ? this._setupGuidance(landmarks, required, tilt, aspect) : null;
      if (setup) {
        status = 'partial';
        guidance = setup;
      }
    }

    // Camera orientation: single-camera knee error is ~2.5x higher on the far leg in a
    // lateral view and grows quickly as the view turns oblique, so ask for the view
    // each exercise is measured in once the body itself is framed well.
    if (status === 'optimal' && Number.isFinite(yawDeg)) {
      if (recommendedView === 'side' && yawDeg > 40) {
        status = 'partial';
        guidance = `Turn side-on to the camera for accurate ${prettyName} angles`;
      } else if (recommendedView === 'front' && yawDeg < 50) {
        status = 'partial';
        guidance = `Face the camera for accurate ${prettyName} tracking`;
      }
    }

    return {
      status,
      missingSegments,
      guidance,
      segments,
      yawDeg: Number.isFinite(yawDeg) ? yawDeg : null,
    };
  }

  _setupGuidance(landmarks, required, tilt, aspect = 1) {
    if (tilt && Math.abs(tilt.rollDeg) >= LEVEL_TOLERANCE_DEG) {
      return `Level the phone: tilted ${Math.round(Math.abs(tilt.rollDeg))}° sideways`;
    }
    if (!required.includes('lowerBody') || !this._standingUpright(landmarks, aspect)) return null;
    const ys = [JOINTS.nose, JOINTS.leftEar, JOINTS.rightEar, JOINTS.leftAnkle, JOINTS.rightAnkle]
      .map(i => landmarks[i]).filter(p => this._isReliable(p)).map(p => p.y);
    if (ys.length >= 3 && Math.max(...ys) - Math.min(...ys) < MIN_BODY_FRACTION) {
      return 'Move closer: about 2–3 m away, whole body filling most of the frame';
    }
    if (tilt && tilt.pitchDeg < PITCH_TOLERANCE_DEG) {
      const hips = [landmarks[JOINTS.leftHip], landmarks[JOINTS.rightHip]].filter(p => this._isReliable(p));
      if (hips.length) {
        const hipY = hips.reduce((s, p) => s + p.y, 0) / hips.length;
        // Phone upright and level: the centre row is at camera height.
        if (hipY < 0.5 - HIP_ROW_TOLERANCE) return 'Raise the camera to hip height';
        if (hipY > 0.5 + HIP_ROW_TOLERANCE) return 'Lower the camera to hip height';
      }
    }
    return null;
  }

  // Distance and camera height are judged in the standing posture only: at squat depth
  // or seated, body span and hip height reflect the movement, not the camera setup.
  // aspect = frame width / height; normalized x is scaled to pixel proportions.
  _standingUpright(landmarks, aspect = 1) {
    const knees = [
      [JOINTS.leftHip, JOINTS.leftKnee, JOINTS.leftAnkle],
      [JOINTS.rightHip, JOINTS.rightKnee, JOINTS.rightAnkle],
    ].map(ids => ids.map(i => landmarks[i]))
      .filter(pts => pts.every(p => this._isReliable(p)))
      .map(([h, k, a]) => {
        const u = [(h.x - k.x) * aspect, h.y - k.y];
        const v = [(a.x - k.x) * aspect, a.y - k.y];
        const cos = (u[0] * v[0] + u[1] * v[1]) / (Math.hypot(...u) * Math.hypot(...v) || 1);
        return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
      });
    return knees.length > 0 && knees.every(deg => deg >= STANDING_KNEE_DEG);
  }
}

export const defaultFramingAssistant = new FramingAssistant();
