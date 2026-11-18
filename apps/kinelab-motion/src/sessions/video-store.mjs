/**
 * VideoStore: IndexedDB-backed local storage for session video recordings.
 * Allows persisting large binary video blobs (WebM/MP4) without hitting
 * localStorage's 5MB character quota or transmitting data outside the browser.
 */

const DB_NAME = 'kinelab_motion_media_v1';
const DB_VERSION = 1;
const STORE_NAME = 'session_videos';

export class VideoStore {
  constructor({ useMemoryFallback = false } = {}) {
    this.useMemoryFallback = useMemoryFallback || typeof indexedDB === 'undefined';
    this.memoryStore = new Map();
    this._dbPromise = null;
  }

  _getDB() {
    if (this.useMemoryFallback) return Promise.resolve(null);
    if (this._dbPromise) return this._dbPromise;

    this._dbPromise = new Promise((resolve, reject) => {
      try {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME, { keyPath: 'sessionId' });
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => {
          console.warn('[VideoStore] IndexedDB open error, falling back to in-memory store:', req.error);
          this.useMemoryFallback = true;
          resolve(null);
        };
      } catch (err) {
        console.warn('[VideoStore] IndexedDB unavailable, falling back:', err);
        this.useMemoryFallback = true;
        resolve(null);
      }
    });

    return this._dbPromise;
  }

  async saveVideo(sessionId, blob) {
    if (!sessionId || !blob) return false;

    if (this.useMemoryFallback) {
      this.memoryStore.set(sessionId, blob);
      return true;
    }

    const db = await this._getDB();
    if (!db) {
      this.memoryStore.set(sessionId, blob);
      return true;
    }

    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const record = {
          sessionId,
          blob,
          createdAt: Date.now(),
          mimeType: blob.type,
          size: blob.size,
        };
        const req = store.put(record);
        req.onsuccess = () => resolve(true);
        req.onerror = (e) => {
          console.warn('[VideoStore] Error saving video blob:', e.target.error);
          this.memoryStore.set(sessionId, blob);
          resolve(true);
        };
      } catch (err) {
        console.warn('[VideoStore] Transaction failure:', err);
        this.memoryStore.set(sessionId, blob);
        resolve(true);
      }
    });
  }

  async getVideo(sessionId) {
    if (!sessionId) return null;

    if (this.useMemoryFallback || this.memoryStore.has(sessionId)) {
      return this.memoryStore.get(sessionId) || null;
    }

    const db = await this._getDB();
    if (!db) return this.memoryStore.get(sessionId) || null;

    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(sessionId);
        req.onsuccess = () => {
          const res = req.result;
          resolve(res ? res.blob : null);
        };
        req.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }

  async hasVideo(sessionId) {
    if (!sessionId) return false;
    if (this.useMemoryFallback || this.memoryStore.has(sessionId)) {
      return this.memoryStore.has(sessionId);
    }

    const db = await this._getDB();
    if (!db) return this.memoryStore.has(sessionId);

    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req = store.count(sessionId);
        req.onsuccess = () => resolve(req.result > 0);
        req.onerror = () => resolve(false);
      } catch {
        resolve(false);
      }
    });
  }

  async deleteVideo(sessionId) {
    if (!sessionId) return true;
    this.memoryStore.delete(sessionId);

    if (this.useMemoryFallback) return true;

    const db = await this._getDB();
    if (!db) return true;

    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const req = store.delete(sessionId);
        req.onsuccess = () => resolve(true);
        req.onerror = () => resolve(true); // idempotent
      } catch {
        resolve(true);
      }
    });
  }

  async clearAll() {
    this.memoryStore.clear();
    if (this.useMemoryFallback) return true;

    const db = await this._getDB();
    if (!db) return true;

    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const req = store.clear();
        req.onsuccess = () => resolve(true);
        req.onerror = () => resolve(false);
      } catch {
        resolve(false);
      }
    });
  }
}

export const defaultVideoStore = new VideoStore();
