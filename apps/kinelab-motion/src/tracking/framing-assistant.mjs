/**
 * Body Framing Assistant: Evaluates camera field-of-view, limb presence,
 * and positioning to guide the user into the optimal exercise tracking zone.
 */

import { JOINTS } from '../biomechanics/landmark-indices.mjs';

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
   */
  evaluate(landmarks, exerciseId = 'free', { yawDeg = null, recommendedView = 'any' } = {}) {
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
}

export const defaultFramingAssistant = new FramingAssistant();
