/**
 * Exercise registry: enables dynamic discovery and modular configuration
 * of movement modules without touching the core pose estimation engine.
 */

import { SquatExercise } from './squat-exercise.mjs';
import { SitToStandExercise } from './sit-to-stand-exercise.mjs';
import { LungeExercise } from './lunge-exercise.mjs';
import { ShoulderAbductionExercise } from './shoulder-abduction-exercise.mjs';
import { ShoulderFlexionExercise } from './shoulder-flexion-exercise.mjs';
import { SingleLegBalanceExercise } from './balance-exercise.mjs';

export class ExerciseRegistry {
  constructor() {
    this.exercises = new Map();

    // Register 5 core initial exercises
    this.register(new SquatExercise());
    this.register(new SitToStandExercise());
    this.register(new LungeExercise());
    this.register(new ShoulderAbductionExercise());
    this.register(new ShoulderFlexionExercise());
    this.register(new SingleLegBalanceExercise());
  }

  register(exerciseInstance) {
    if (!exerciseInstance || !exerciseInstance.id) {
      throw new Error('Invalid exercise instance: must have an "id" property.');
    }
    this.exercises.set(exerciseInstance.id, exerciseInstance);
  }

  get(id) {
    if (!id) return null;
    return this.exercises.get(id) ??
      this.exercises.get(id.replace(/-/g, '_')) ??
      this.exercises.get(id.replace(/_/g, '-')) ??
      null;
  }

  getAll() {
    return Array.from(this.exercises.values());
  }

  has(id) {
    if (!id) return false;
    return this.exercises.has(id) ||
      this.exercises.has(id.replace(/-/g, '_')) ||
      this.exercises.has(id.replace(/_/g, '-'));
  }
}

// Default singleton registry
export const defaultExerciseRegistry = new ExerciseRegistry();
