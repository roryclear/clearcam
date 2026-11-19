/**
 * Settings persistence, event dispatching, and state management.
 */

import {
  createDefaultSettings,
  STORAGE_KEY,
  validateSettings,
} from './settings-schema.mjs';

export class SettingsManager {
  constructor({ storage = (typeof localStorage !== 'undefined' ? localStorage : null) } = {}) {
    this.storage = storage;
    this.listeners = new Set();
    this.current = this.load();
  }

  load() {
    if (!this.storage) return createDefaultSettings();
    try {
      const serialized = this.storage.getItem(STORAGE_KEY);
      if (!serialized) return createDefaultSettings();
      const parsed = JSON.parse(serialized);
      return validateSettings(parsed);
    } catch (err) {
      console.warn('Failed to parse stored settings, restoring defaults:', err);
      return createDefaultSettings();
    }
  }

  save() {
    if (!this.storage) return;
    try {
      this.storage.setItem(STORAGE_KEY, JSON.stringify(this.current));
    } catch (err) {
      console.warn('Failed to persist settings:', err);
    }
  }

  get() {
    return this.current;
  }

  update(partial) {
    const merged = {
      ...this.current,
      camera: { ...this.current.camera, ...(partial.camera || {}) },
      tracking: { ...this.current.tracking, ...(partial.tracking || {}) },
      overlays: { ...this.current.overlays, ...(partial.overlays || {}) },
      exercise: { ...this.current.exercise, ...(partial.exercise || {}) },
      privacy: { ...this.current.privacy, ...(partial.privacy || {}) },
    };
    this.current = validateSettings(merged);
    this.save();
    this.notify();
    return this.current;
  }

  reset() {
    this.current = createDefaultSettings();
    this.save();
    this.notify();
    return this.current;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify() {
    for (const listener of this.listeners) {
      try {
        listener(this.current);
      } catch (err) {
        console.error('Error in settings listener:', err);
      }
    }
  }
}

export const defaultSettingsManager = new SettingsManager();
