/**
 * Body Framing Assistant: Evaluates camera field-of-view, limb presence,
 * and positioning to guide the user into the optimal exercise tracking zone.
 */

import { JOINTS } from '../biomechanics/landmark-indices.mjs';

export const EXERCISE_REQUIREMENTS = Object.freeze({
  'squat': ['upperBody', 'torso', 'lowerBody'],
  'sit-to-stand': ['upperBody', 'torso', 'lowerBody'],
  'lunge': ['upperBody', 'torso', 'lowerBody'],
  'single-leg-balance': ['upperBody', 'torso', 'lowerBody'],
  'shoulder-flexion': ['upperBody', 'torso'],
  'shoulder-abduction': ['upperBody', 'torso'],
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

  evaluate(landmarks, exerciseId = 'free') {
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

    const required = EXERCISE_REQUIREMENTS[exerciseId] || EXERCISE_REQUIREMENTS.free;
    const missingSegments = required.filter(seg => !segments[seg]);

    let status = 'optimal';
    let guidance = `Framing optimal for ${exerciseId.replace('-', ' ')}`;

    if (missingSegments.length > 0) {
      if (missingSegments.includes('lowerBody')) {
        status = 'inadequate';
        guidance = `Step back: Knees and feet needed for ${exerciseId.replace('-', ' ')}`;
      } else if (missingSegments.includes('upperBody') || missingSegments.includes('torso')) {
        status = 'inadequate';
        guidance = `Move into frame: Torso and shoulders needed`;
      } else {
        status = 'partial';
        guidance = `Adjust camera: Some limbs partially out of view`;
      }
    }

    return {
      status,
      missingSegments,
      guidance,
      segments,
    };
  }
}

export const defaultFramingAssistant = new FramingAssistant();
