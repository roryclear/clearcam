import test from 'node:test';
import assert from 'node:assert/strict';
import { validateSettings, DEFAULT_SETTINGS } from '../src/settings/settings-schema.mjs';
import { SettingsManager } from '../src/settings/settings-manager.mjs';

class MockStorage {
  constructor() {
    this.store = new Map();
  }
  getItem(key) {
    return this.store.get(key) ?? null;
  }
  setItem(key, val) {
    this.store.set(key, String(val));
  }
  removeItem(key) {
    this.store.delete(key);
  }
  clear() {
    this.store.clear();
  }
}

test('validateSettings returns defaults on null or corrupt input', () => {
  const result = validateSettings(null);
  assert.equal(result.version, DEFAULT_SETTINGS.version);
  assert.equal(result.tracking.model, 'heavy');
  assert.equal(result.exercise.activeExercise, 'squat');

  const corrupt = { camera: { facingMode: 'alien-dimension' }, tracking: { confidenceThreshold: -99 } };
  const sanitized = validateSettings(corrupt);
  assert.equal(sanitized.camera.facingMode, 'environment');
  assert.equal(sanitized.tracking.confidenceThreshold, 0.55);
});

test('SettingsManager loads, updates, persists, and notifies subscribers', () => {
  const storage = new MockStorage();
  const manager = new SettingsManager({ storage });

  let notified = false;
  manager.subscribe(settings => {
    notified = true;
    assert.equal(settings.tracking.smoothing, 'high');
  });

  manager.update({
    tracking: { smoothing: 'high' },
  });

  assert.equal(notified, true);
  assert.equal(manager.get().tracking.smoothing, 'high');

  // New instance reading same storage
  const manager2 = new SettingsManager({ storage });
  assert.equal(manager2.get().tracking.smoothing, 'high');
});

test('validateSettings validates angleDimension defaulting to auto', () => {
  const result = validateSettings({});
  assert.equal(result.tracking.angleDimension, 'auto');

  const custom2D = validateSettings({ tracking: { angleDimension: '2d' } });
  assert.equal(custom2D.tracking.angleDimension, '2d');

  const invalid = validateSettings({ tracking: { angleDimension: '4d' } });
  assert.equal(invalid.tracking.angleDimension, 'auto');
});

test('validateSettings migrates the old v2 default 3d to auto but keeps an explicit v3 choice', () => {
  assert.equal(validateSettings({ version: 2, tracking: { angleDimension: '3d' } }).tracking.angleDimension, 'auto');
  assert.equal(validateSettings({ version: 3, tracking: { angleDimension: '3d' } }).tracking.angleDimension, '3d');
  assert.equal(validateSettings({ version: 2, tracking: { angleDimension: '2d' } }).tracking.angleDimension, '2d');
  assert.equal(validateSettings({ tracking: { model: 'lite' } }).tracking.model, 'lite');
});

test('validateSettings validates voiceCoach option', () => {
  const result = validateSettings({});
  assert.equal(result.exercise.voiceCoach, false);

  const enabled = validateSettings({ exercise: { voiceCoach: true } });
  assert.equal(enabled.exercise.voiceCoach, true);
});
