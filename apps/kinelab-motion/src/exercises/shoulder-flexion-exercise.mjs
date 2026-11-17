/**
 * Shoulder Flexion exercise module (forward arm elevation).
 * Same shoulder angle (elbow-shoulder-hip) and state machine as abduction,
 * but a distinct module: forward raise in the sagittal plane, side camera
 * recommended so elevation is not foreshortened by a front view.
 */
import { ShoulderAbductionExercise } from './shoulder-abduction-exercise.mjs';

export class ShoulderFlexionExercise extends ShoulderAbductionExercise {
  constructor(options = {}) {
    super(options);
    this.id = 'shoulder_flexion';
    this.name = 'Shoulder Flexion';
    this.recommendedCamera = 'side';
    this.description = 'Tracks forward arm elevation toward overhead. Side camera recommended so the raise stays in the image plane.';
  }

  getFeedback(state) {
    if (state.status === 'tracking-lost') {
      return 'Upper body lost. Stand side-on with torso and arms visible.';
    }
    const map = {
      waiting: 'Stand side-on to the camera with arms relaxed at your sides.',
      'arms-down': 'Ready. Raise your arms forward and overhead.',
      raising: 'Raising arms forward... Reach upward as high as comfortable.',
      'peak-reached': 'Peak elevation reached! Lower arms slowly back to sides.',
      lowering: 'Lowering arms in control...',
      completed: 'Repetition completed! Forward range recorded.',
    };
    return map[state.status] ?? 'Maintain upper body visibility.';
  }
}
