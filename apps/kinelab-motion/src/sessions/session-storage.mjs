/**
 * Local session storage and index management.
 */

const SESSIONS_INDEX_KEY = 'kinelab_sessions_index_v1';
const SESSION_DATA_PREFIX = 'kinelab_session_data_';

export class SessionStorage {
  constructor({ storage = (typeof localStorage !== 'undefined' ? localStorage : null) } = {}) {
    this.storage = storage;
  }

  getIndex() {
    if (!this.storage) return [];
    try {
      const raw = this.storage.getItem(SESSIONS_INDEX_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  saveSession(session) {
    if (!this.storage || !session || !session.id) return false;
    try {
      const summary = {
        id: session.id,
        recordedAt: session.recordedAt,
        exerciseId: session.exerciseId,
        frameCount: session.frameCount,
        durationMs: session.durationMs,
        finalReps: session.finalReps,
      };

      const index = this.getIndex().filter(s => s.id !== session.id);
      index.unshift(summary);

      // Keep up to 20 recorded sessions in index
      if (index.length > 20) {
        const removed = index.pop();
        this.storage.removeItem(`${SESSION_DATA_PREFIX}${removed.id}`);
      }

      this.storage.setItem(SESSIONS_INDEX_KEY, JSON.stringify(index));
      this.storage.setItem(`${SESSION_DATA_PREFIX}${session.id}`, JSON.stringify(session));
      return true;
    } catch (err) {
      console.warn('Failed to save session to local storage:', err);
      return false;
    }
  }

  getSession(id) {
    if (!this.storage || !id) return null;
    try {
      const raw = this.storage.getItem(`${SESSION_DATA_PREFIX}${id}`);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  deleteSession(id) {
    if (!this.storage || !id) return false;
    try {
      const index = this.getIndex().filter(s => s.id !== id);
      this.storage.setItem(SESSIONS_INDEX_KEY, JSON.stringify(index));
      this.storage.removeItem(`${SESSION_DATA_PREFIX}${id}`);
      return true;
    } catch {
      return false;
    }
  }

  clearAll() {
    if (!this.storage) return;
    try {
      const index = this.getIndex();
      for (const item of index) {
        this.storage.removeItem(`${SESSION_DATA_PREFIX}${item.id}`);
      }
      this.storage.removeItem(SESSIONS_INDEX_KEY);
    } catch (err) {
      console.warn('Failed to clear sessions:', err);
    }
  }

  getStorageEstimate() {
    if (!this.storage) return { count: 0, bytes: 0 };
    const index = this.getIndex();
    let totalBytes = 0;
    try {
      for (const item of index) {
        const itemRaw = this.storage.getItem(`${SESSION_DATA_PREFIX}${item.id}`);
        if (itemRaw) totalBytes += itemRaw.length * 2; // rough UTF-16 byte estimate
      }
    } catch {}
    return { count: index.length, bytes: totalBytes };
  }
}

export const defaultSessionStorage = new SessionStorage();
