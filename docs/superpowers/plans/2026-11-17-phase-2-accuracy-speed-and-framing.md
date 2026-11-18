# Phase 2: Accuracy, High-Throughput Speed, Latency & Smart Framing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate KineLab Motion into a high-precision, low-latency motion intelligence engine by adding perspective-invariant 3D metric joint angles from MediaPipe `worldLandmarks`, decoupling inference resolution for 30–60 FPS throughput, and implementing an intelligent Body Framing Radar and upper-body relaxation.

**Architecture:** 
1. `angle-calculations.mjs` is extended with 3D metric vector dot-product geometry operating on real-world metric coordinates (`worldLandmarks` in meters).
2. `mediapipe-runner.mjs` uses an optimized offscreen downscaling canvas (`480p/360p`) so heavy video decoding does not choke GPU texture transfers, dropping latency from ~67ms to ~20ms.
3. `FramingAssistant` evaluates landmark bounding envelopes and limb presence against the active exercise's requirements, outputting actionable placement guidance on the live canvas.
4. Upper-body exercises (shoulder flexion/abduction) are decoupled from lower-limb requirements so desk/webcam users can track exercises seamlessly.

**Tech Stack:** MediaPipe Tasks Vision (`worldLandmarks` 3D meters), HTML5 Canvas 2D, Vector Biomechanics, Node.js `--test`.

## Global Constraints
- 100% Client-side local execution: no frame data ever leaves the device.
- Zero npm runtime dependencies.
- Backwards compatibility: 2D angle fallback is preserved and toggleable.
- All tests must pass via `node --test apps/kinelab-motion/tests/*.test.mjs`.

## Review Focus
- Oblique/perspective camera angles (e.g. 45° to user) maintain accurate 3D joint angle measurements without foreshortening collapse.
- 3D angles with occluded or missing `worldLandmarks` return null without producing NaN or fabricated values.
- Resolution downscaling preserves correct normalized landmark coordinates on the display canvas.
- Missing feet/ankles during upper-body exercises do not trigger `NO POSE` tracking drops.
- Framing radar correctly alerts when feet/knees leave frame during squats.

---

### Task 1: 3D Metric Vector Joint Angles via `worldLandmarks`
**Files:**
- Modify: `apps/kinelab-motion/src/biomechanics/angle-calculations.mjs`
- Test: `apps/kinelab-motion/tests/angle-calculations-3d.test.mjs`

- [ ] **Step 1.1: Write failing tests for 3D angle calculations**
  Test `angle3DAt(a, b, c, minVisibility)` with metric 3D vectors:
  - Perpendicular 3D vectors `[1, 0, 0]`, `[0, 0, 0]`, `[0, 1, 0]` yield exactly 90.0°.
  - Straight 3D vectors `[1, 0, 0]`, `[0, 0, 0]`, `[-1, 0, 0]` yield 180.0°.
  - Oblique depth vectors `[0, 1, 1]`, `[0, 0, 0]`, `[0, -1, 1]` compute true 3D spatial angle rather than 2D projected angle.
  - Test `getJointAngles3D(worldLandmarks, threshold)`.
  - Missing/occluded points return null without errors.

- [ ] **Step 1.2: Run test to verify failure**
  Run `node --test apps/kinelab-motion/tests/angle-calculations-3d.test.mjs`.

- [ ] **Step 1.3: Implement 3D angle calculations**
  Update `apps/kinelab-motion/src/biomechanics/angle-calculations.mjs`:
  - Implement `reliable3D(point, minVisibility)`.
  - Implement `angle3DAt(a, b, c, minVisibility)` using 3D dot product:
    $u = a - b, v = c - b$
    $\cos \theta = (u \cdot v) / (\|u\| \|v\|)$
  - Implement `getJointAngles3D(worldLandmarks, threshold)`.
  - Export `angle3DAt` and `getJointAngles3D`.

- [ ] **Step 1.4: Run tests to verify pass**
  Run `node --test apps/kinelab-motion/tests/angle-calculations-3d.test.mjs`.

- [ ] **Step 1.5: Commit Task 1**
  `git add apps/kinelab-motion/src/biomechanics/angle-calculations.mjs apps/kinelab-motion/tests/angle-calculations-3d.test.mjs && git commit -m "feat(kinelab): add perspective-invariant 3D metric joint angle calculations"`

---

### Task 2: High-Throughput Inference & Resolution Decoupling
**Files:**
- Modify: `apps/kinelab-motion/src/tracking/mediapipe-runner.mjs`
- Modify: `apps/kinelab-motion/src/app.mjs`
- Test: `apps/kinelab-motion/tests/mediapipe-scaling.test.mjs`

- [ ] **Step 2.1: Write failing tests for downscaling pipeline**
  Create `apps/kinelab-motion/tests/mediapipe-scaling.test.mjs`:
  - Test resolution scaler logic mapping between display coordinates and inference coordinates.
  - Verify aspect ratio and landmark coordinate integrity.

- [ ] **Step 2.2: Implement inference resolution decoupling in mediapipe-runner**
  - Add offscreen scaled canvas in `mediapipe-runner.mjs`.
  - Provide `setInferenceResolution(width, height)` (default: 480×360 or 640×360).
  - Extract both `landmarks` and `worldLandmarks` from `PoseLandmarkerResult`.
  - In `app.mjs`, reduce `FRAME_INTERVAL_MS` to 25ms (~40 FPS target) with adaptive throttling.

- [ ] **Step 2.3: Run tests to verify pass**
  Run `node --test apps/kinelab-motion/tests/mediapipe-scaling.test.mjs`.

- [ ] **Step 2.4: Commit Task 2**
  `git commit -m "perf(kinelab): decouple inference resolution for high-FPS low-latency tracking"`

---

### Task 3: Body Framing Radar & Upper-Body Decoupling
**Files:**
- Create: `apps/kinelab-motion/src/tracking/framing-assistant.mjs`
- Modify: `apps/kinelab-motion/src/exercises/exercise-base.mjs`
- Modify: `apps/kinelab-motion/src/exercises/shoulder-flexion-exercise.mjs`
- Modify: `apps/kinelab-motion/src/exercises/shoulder-abduction-exercise.mjs`
- Modify: `apps/kinelab-motion/src/overlays/overlay-renderer.mjs`
- Test: `apps/kinelab-motion/tests/framing-assistant.test.mjs`

- [ ] **Step 3.1: Write failing tests for FramingAssistant**
  Create `apps/kinelab-motion/tests/framing-assistant.test.mjs`:
  - Upper-body only in frame with squat exercise returns `missing_lower_body` + message "Step back: knees and feet needed".
  - Upper-body only in frame with shoulder abduction returns `good` framing.
  - Full body in frame returns `optimal` framing.

- [ ] **Step 3.2: Implement FramingAssistant & Upper-Body relaxation**
  - Create `src/tracking/framing-assistant.mjs`.
  - Add `requiredSegments` to exercises (`['upper', 'lower']` for squats, `['upper']` for shoulders).
  - Add visual radar / banner to `overlay-renderer.mjs` when framing is inadequate.

- [ ] **Step 3.3: Run tests to verify pass**
  Run `node --test apps/kinelab-motion/tests/framing-assistant.test.mjs`.

- [ ] **Step 3.4: Commit Task 3**
  `git commit -m "feat(kinelab): add smart body framing assistant and upper-body decoupling"`

---

### Task 4: UI & Telemetry Integration
**Files:**
- Modify: `apps/kinelab-motion/src/app.mjs`
- Modify: `apps/kinelab-motion/index.html`
- Modify: `apps/kinelab-motion/styles.css`
- Modify: `apps/kinelab-motion/src/settings/settings-schema.mjs`

- [ ] **Step 4.1: Wire 3D vs 2D angle toggle in settings and HUD**
  - Add `angleDimension: '3d'` ('3d' or '2d') in `settings-schema.mjs`.
  - Add toggle badge `3D METRIC` vs `2D SCREEN` in the telemetry panel.
  - Display cadence (reps/min) and trunk inclination angle.

- [ ] **Step 4.2: Full regression and browser verification**
  - Run `node --test apps/kinelab-motion/tests/*.test.mjs`.
  - Verify via browser agent at `http://localhost:4173`.

- [ ] **Step 4.3: Commit Task 4**
  `git commit -m "feat(kinelab): wire 3D angle mode toggle, cadence metrics, and framing HUD"`
