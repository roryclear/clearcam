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
