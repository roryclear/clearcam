/**
 * Exercise registry: enables dynamic discovery and modular configuration
 * of movement modules without touching the core pose estimation engine.
 */

import { SquatExercise } from './squat-exercise.mjs';
import { SitToStandExercise } from './sit-to-stand-exercise.mjs';
import { LungeExercise } from './lunge-exercise.mjs';
import { ShoulderAbductionExercise } from './shoulder-abduction-exercise.mjs';
import { SingleLegBalanceExercise } from './balance-exercise.mjs';

export class ExerciseRegistry {
  constructor() {
    this.exercises = new Map();

    // Register 5 core initial exercises
    this.register(new SquatExercise());
    this.register(new SitToStandExercise());
    this.register(new LungeExercise());
    this.register(new ShoulderAbductionExercise());
    this.register(new SingleLegBalanceExercise());
  }

  register(exerciseInstance) {
    if (!exerciseInstance || !exerciseInstance.id) {
      throw new Error('Invalid exercise instance: must have an "id" property.');
    }
    this.exercises.set(exerciseInstance.id, exerciseInstance);
  }

  get(id) {
    return this.exercises.get(id) ?? null;
  }

  getAll() {
    return Array.from(this.exercises.values());
  }

  has(id) {
    return this.exercises.has(id);
  }
}

// Default singleton registry
export const defaultExerciseRegistry = new ExerciseRegistry();
