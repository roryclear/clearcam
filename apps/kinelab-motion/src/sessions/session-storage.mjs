/**
 * Local session storage and index management.
 */

const SESSIONS_INDEX_KEY = 'kinelab_sessions_index_v1';
const SESSION_DATA_PREFIX = 'kinelab_session_data_';
const MAX_SESSIONS = 20;

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

  /**
   * Persists a session and its index entry. Session data is written before the
   * index so the index never references missing data. When localStorage runs out
   * of quota, the oldest sessions are evicted until the new one fits.
   * @param {Object} session
   * @param {Object} [options]
   * @param {(id: string) => void} [options.onEvict] Called for each evicted session id
   *   so callers can delete associated data (e.g. stored videos).
   * @param {(session: Object) => Object} [options.compact] Smaller form of the session
   *   (e.g. without optional fields). When given, the full session is stored only if it
   *   fits without evicting anything; otherwise the compact form is saved with the
   *   normal eviction, so optional data never costs the user older sessions.
   * @returns {boolean} true when the session was stored
   */
  saveSession(session, { onEvict = null, compact = null } = {}) {
    if (typeof compact === 'function') {
      if (this._save(session, { onEvict, evictForSpace: false })) return true;
      return this._save(compact(session) ?? session, { onEvict, evictForSpace: true });
    }
    return this._save(session, { onEvict, evictForSpace: true });
  }

  _save(session, { onEvict, evictForSpace }) {
    if (!this.storage || !session || !session.id) return false;

    const summary = {
      id: session.id,
      recordedAt: session.recordedAt,
      exerciseId: session.exerciseId,
      frameCount: session.frameCount,
      durationMs: session.durationMs,
      finalReps: session.finalReps,
      hasVideo: Boolean(session.hasVideo),
    };
    // Blobs serialize to "{}"; video bytes live in the VideoStore, not here.
    const { videoBlob, ...persisted } = session;
    const serialized = JSON.stringify(persisted);
    const dataKey = `${SESSION_DATA_PREFIX}${session.id}`;

    const originalIndexRaw = this.storage.getItem(SESSIONS_INDEX_KEY);
    const index = this.getIndex().filter(s => s.id !== session.id);
    // Evictions are tentative until the new data and index are both written:
    // removed entries are kept in memory and restored if the save fails, so a
    // recording that can never fit does not destroy existing sessions.
    const evicted = [];
    const evictOldest = () => {
      const item = index.pop();
      const key = `${SESSION_DATA_PREFIX}${item.id}`;
      evicted.push({ item, key, raw: this.storage.getItem(key) });
      this.storage.removeItem(key);
    };
    const rollback = () => {
      this.storage.removeItem(dataKey);
      for (const { key, raw } of evicted.reverse()) {
        if (raw !== null) {
          try { this.storage.setItem(key, raw); } catch (err) { console.warn('Failed to restore session:', err); }
        }
      }
      try {
        if (originalIndexRaw === null) this.storage.removeItem(SESSIONS_INDEX_KEY);
        else this.storage.setItem(SESSIONS_INDEX_KEY, originalIndexRaw);
      } catch (err) {
        console.warn('Failed to restore session index:', err);
      }
    };

    // Keep up to MAX_SESSIONS recorded sessions (including the new one).
    while (index.length >= MAX_SESSIONS) evictOldest();

    let committed = false;
    for (;;) {
      try {
        this.storage.setItem(dataKey, serialized);
        if (this._writeIndex([summary, ...index], { throwOnError: true })) {
          committed = true;
          break;
        }
      } catch (err) {
        if (index.length === 0 || !evictForSpace) {
          if (evictForSpace) console.warn('Failed to save session to local storage:', err);
          break;
        }
        this.storage.removeItem(dataKey);
        evictOldest(); // free space by dropping the oldest session, then retry
      }
    }

    if (!committed) {
      rollback();
      return false;
    }
    if (typeof onEvict === 'function') {
      for (const { item } of evicted) {
        try { onEvict(item.id); } catch (err) { console.warn('onEvict failed:', err); }
      }
    }
    return true;
  }

  _writeIndex(index, { throwOnError = false } = {}) {
    try {
      this.storage.setItem(SESSIONS_INDEX_KEY, JSON.stringify(index));
      return true;
    } catch (err) {
      if (throwOnError) throw err;
      console.warn('Failed to write session index:', err);
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
