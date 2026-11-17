import test from 'node:test';
import assert from 'node:assert/strict';
import { VideoStore, defaultVideoStore } from '../src/sessions/video-store.mjs';

test('VideoStore saves, retrieves, and checks presence of video blobs', async () => {
  const store = new VideoStore({ useMemoryFallback: true });
  const mockBlob = new Blob(['video-binary-data'], { type: 'video/webm' });

  const saved = await store.saveVideo('session_123', mockBlob);
  assert.equal(saved, true);

  const has = await store.hasVideo('session_123');
  assert.equal(has, true);

  const retrieved = await store.getVideo('session_123');
  assert.ok(retrieved instanceof Blob);
  assert.equal(retrieved.type, 'video/webm');
  assert.equal(await retrieved.text(), 'video-binary-data');
});

test('VideoStore deletes a video blob and handles nonexistent keys gracefully', async () => {
  const store = new VideoStore({ useMemoryFallback: true });
  const mockBlob = new Blob(['sample-data'], { type: 'video/mp4' });

  await store.saveVideo('session_del', mockBlob);
  const deleted = await store.deleteVideo('session_del');
  assert.equal(deleted, true);

  const missing = await store.getVideo('session_del');
  assert.equal(missing, null);

  const hasAfterDelete = await store.hasVideo('session_del');
  assert.equal(hasAfterDelete, false);

  // Deleting nonexistent key returns true (idempotent)
  const delMissing = await store.deleteVideo('nonexistent_id');
  assert.equal(delMissing, true);
});

test('VideoStore clears all saved videos', async () => {
  const store = new VideoStore({ useMemoryFallback: true });
  await store.saveVideo('s1', new Blob(['1']));
  await store.saveVideo('s2', new Blob(['2']));

  assert.equal(await store.hasVideo('s1'), true);
  assert.equal(await store.hasVideo('s2'), true);

  await store.clearAll();

  assert.equal(await store.hasVideo('s1'), false);
  assert.equal(await store.hasVideo('s2'), false);
});

test('defaultVideoStore singleton is exported and operational', async () => {
  assert.ok(defaultVideoStore instanceof VideoStore);
});
