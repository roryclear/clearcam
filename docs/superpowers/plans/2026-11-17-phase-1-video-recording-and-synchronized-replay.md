# Phase 1: Real Video Recording & Synchronized Replay Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable client-side camera video recording with MediaRecorder and IndexedDB, automatically load and select newly recorded sessions, display an instant post-workout results card, and provide a synchronized video + skeletal overlay replay player.

**Architecture:** A lightweight IndexedDB store (`VideoStore`) persists raw video blobs locally without exceeding localStorage quotas. `SessionRecorder` is extended to capture `MediaStream` chunks via `MediaRecorder` in parallel with telemetry. `TimelinePlayer` synchronizes frame stepping and seeking with an HTML5 `<video>` element, rendering skeletal overlays and angle charts frame-accurately on top of real recorded footage.

**Tech Stack:** Native HTML5 `MediaRecorder` API, IndexedDB (`idb` pattern without dependencies), Canvas 2D API, Web Audio API, Vanilla ES Modules, Node.js `--test`.

**Spec:** KineLab Motion 2.0 Delivery Report & Mission Phase D (Recording, Replay, and Motion Analytics).

## Global Constraints
- 100% Client-side local processing: video streams and blobs must never leave the browser sandbox.
- Zero external npm runtime dependencies: standard browser APIs (IndexedDB, MediaRecorder, Canvas).
- Backwards compatibility: sessions without video (e.g. existing sessions or unsupported browsers) must still play back telemetry frames cleanly.
- All tests must pass via `node --test apps/kinelab-motion/tests/*.test.mjs`.

## Review Focus
- Browser without MediaRecorder support or restricted MIME type falls back to telemetry-only recording without crashing.
- Corrupted or deleted video blob in IndexedDB gracefully degrades to skeletal-only playback.
- Stopping recording immediately selects the new session, scrolls it into view, and renders the results summary card.
- Seeking and scrubbing on the timeline updates the `<video>` element's `currentTime` and redraws the canvas overlay with zero desync.
- Deleting a session clears both the localStorage telemetry and the IndexedDB video blob.

---

### Task 1: Video Storage Layer (IndexedDB)
**Files:**
- Create: `apps/kinelab-motion/src/sessions/video-store.mjs`
- Test: `apps/kinelab-motion/tests/video-store.test.mjs`

- [ ] **Step 1.1: Write failing tests for VideoStore**
  Create `apps/kinelab-motion/tests/video-store.test.mjs` testing:
  - In-memory / mocked IndexedDB backend storing blobs.
  - `saveVideoBlob(sessionId, blob)` returns true.
  - `getVideoBlob(sessionId)` retrieves the exact blob.
  - `deleteVideoBlob(sessionId)` removes the blob.
  - `clearAllVideos()` wipes store.

- [ ] **Step 1.2: Run test to verify failure**
  Run `node --test apps/kinelab-motion/tests/video-store.test.mjs` and observe failure.

- [ ] **Step 1.3: Implement VideoStore**
  Create `apps/kinelab-motion/src/sessions/video-store.mjs`:
  - Open `kinelab_motion_db` with store `session_videos`.
  - Provide promise-wrapped methods: `saveVideo(id, blob)`, `getVideo(id)`, `deleteVideo(id)`, `clearAll()`.
  - Provide in-memory Map fallback when running in Node.js test environment or if IndexedDB is blocked.

- [ ] **Step 1.4: Run tests to verify pass**
  Verify all tests pass with code 0.

- [ ] **Step 1.5: Commit Task 1**
  `git add apps/kinelab-motion/src/sessions/video-store.mjs apps/kinelab-motion/tests/video-store.test.mjs && git commit -m "feat(kinelab): add IndexedDB video store for recorded sessions"`

---

### Task 2: Session Recorder Video Integration
**Files:**
- Modify: `apps/kinelab-motion/src/sessions/session-recorder.mjs`
- Test: `apps/kinelab-motion/tests/session-recorder-video.test.mjs`

- [ ] **Step 2.1: Write failing tests for video-enabled SessionRecorder**
  Create `apps/kinelab-motion/tests/session-recorder-video.test.mjs`:
  - Mock `MediaRecorder` API.
  - Verify `recorder.start({ exerciseId, mediaStream })` initializes `MediaRecorder` when available.
  - Verify `recorder.stop()` stops recorder, produces a `videoBlob`, and returns `hasVideo: true` in session metadata.
  - Verify graceful fallback when `mediaStream` is null or `MediaRecorder` is undefined.

- [ ] **Step 2.2: Run test to verify failure**
  Run `node --test apps/kinelab-motion/tests/session-recorder-video.test.mjs` and observe failure.

- [ ] **Step 2.3: Implement video capture in SessionRecorder**
  Update `apps/kinelab-motion/src/sessions/session-recorder.mjs`:
  - Detect best supported MIME type (`video/webm;codecs=vp9,opus`, `video/webm;codecs=vp8`, `video/webm`, `video/mp4`).
  - Capture stream chunks in `this.videoChunks = []`.
  - On `stop()`, assemble `new Blob(this.videoChunks, { type: this.mimeType })` and attach to returned result.

- [ ] **Step 2.4: Run tests to verify pass**
  Verify all tests pass with code 0.

- [ ] **Step 2.5: Commit Task 2**
  `git add apps/kinelab-motion/src/sessions/session-recorder.mjs apps/kinelab-motion/tests/session-recorder-video.test.mjs && git commit -m "feat(kinelab): integrate MediaRecorder into SessionRecorder"`

---

### Task 3: Video-Synchronized Timeline Player
**Files:**
- Modify: `apps/kinelab-motion/src/sessions/timeline-player.mjs`
- Test: `apps/kinelab-motion/tests/timeline-player-sync.test.mjs`

- [ ] **Step 3.1: Write failing tests for video synchronization in TimelinePlayer**
  Create `apps/kinelab-motion/tests/timeline-player-sync.test.mjs`:
  - Test `attachVideoElement(videoEl)` binding.
  - Test time mapping from video `currentTime` to nearest frame index.
  - Test seek forward/back synchronizing video playback rate and time.

- [ ] **Step 3.2: Run test to verify failure**
  Run `node --test apps/kinelab-motion/tests/timeline-player-sync.test.mjs`.

- [ ] **Step 3.3: Implement video synchronization in TimelinePlayer**
  Update `apps/kinelab-motion/src/sessions/timeline-player.mjs`:
  - Support optional bound video element.
  - Drive frame dispatch from `timeupdate` or animation frame matched to video `currentTime`.
  - Keep synthetic animation loop for telemetry-only sessions when no video is present.

- [ ] **Step 3.4: Run tests to verify pass**
  Verify all tests pass with code 0.

- [ ] **Step 3.5: Commit Task 3**
  `git commit -m "feat(kinelab): add video synchronization to TimelinePlayer"`

---

### Task 4: UI Stage, Results Summary Card & Auto-Load Integration
**Files:**
- Modify: `apps/kinelab-motion/index.html`
- Modify: `apps/kinelab-motion/styles.css`
- Modify: `apps/kinelab-motion/src/app.mjs`

- [ ] **Step 4.1: Update index.html & styles.css**
  - Add `<div class="replay-stage">` containing `<video id="replay-video">` and `<canvas id="replay-canvas">`.
  - Add `<div class="workout-results-card" id="workout-results-card">` with badges for Reps, Duration, Max Angle, and Bilateral Symmetry.
  - Relocate Saved Sessions selector and counter to the top header of the analytics drawer.

- [ ] **Step 4.2: Wire app.mjs orchestrator**
  - Connect `appState.stream` to `recorder.start({ exerciseId, mediaStream: appState.stream })`.
  - In `stopRecording()`, save video blob to `defaultVideoStore`.
  - Automatically call `loadSessionForPlayback(sessionData.id)`.
  - Automatically unhide `nodes.analyticsDrawer` and scroll it into view.
  - Populate the Workout Results Card with stats from `sessionData`.
  - When loading a session, fetch video blob from `defaultVideoStore` and set `replayVideo.src = URL.createObjectURL(blob)`.

- [ ] **Step 4.3: Verify via test suite & browser agent**
  - Run `node --test apps/kinelab-motion/tests/*.test.mjs` to ensure no regressions.
  - Test in browser to verify record, stop, instant results card, and synchronized video playback.

- [ ] **Step 4.4: Commit Task 4**
  `git commit -m "feat(kinelab): wire synchronized video replay, results summary, and auto-load"`
