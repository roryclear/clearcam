# KineLab Motion Intelligence 2.0 — Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining gaps between the merged KineLab Motion 2.0 build and the Motion Intelligence 2.0 spec, without rebuilding or removing any existing functionality.

**Architecture:** The merged build already provides the modular engine (tracking/filters/biomechanics/exercises/overlays/sessions/settings/integrations) with 37 passing Node tests. This plan adds only the missing pieces — shoulder flexion mode, movement-direction indicators, personalized baseline, session comparison stats, full performance HUD, and an honest dataset-validation report — each as a small TDD task inside the existing module boundaries.

**Tech Stack:** Vanilla ES modules (`.mjs`), MediaPipe Tasks Vision 1.2.3 (Heavy→Full), Node `node:test`, HTML5 Canvas, localStorage. No new dependencies.

**Spec:** User mission "PROJECT UPGRADE — KineLab Motion Intelligence 2.0" (Phases A–E, Verification §12, Deliverables §13).

## Global Constraints

- Preserve all existing Clearcam and KineLab behavior; existing 37 tests must stay green.
- Single camera only; no backend; no silent upload; no raw video recording by default.
- Invalid/missing landmarks never produce fabricated angles (`—`, not numbers).
- No definitive medical correctness/safety claims; feedback stays heuristic and quality-gated.
- Test command: `node --test apps/kinelab-motion/tests/*.test.mjs` from repo root.
- Branch: `feat/kinelab-motion-2.0-gap-closure`; PR only, never auto-merge.

## Review Focus

- Tracking lost mid-rep → rep discarded, never counted (pinned in Task 2 tests).
- Landmark visibility below threshold → angle reads as unavailable, not estimated (Task 1/2).
- Corrupt/outdated persisted settings → safe defaults restored (already covered; re-verified Task 4).
- Camera permission denied / model load failure → recoverable error state (verified by code inspection + existing handling, Task 6 QA).
- Recording left on unintentionally → recording stays opt-in/manual only (Task 4 test).

---

### Task 1: Shoulder flexion mode

**Files:**
- Modify: `apps/kinelab-motion/src/exercises/shoulder-abduction-exercise.mjs` (add plane support) or create `apps/kinelab-motion/src/exercises/shoulder-flexion-exercise.mjs`
- Modify: `apps/kinelab-motion/src/exercises/exercise-registry.mjs`, `apps/kinelab-motion/src/settings/settings-schema.mjs` (valid exercise list)
- Test: `apps/kinelab-motion/tests/exercises.test.mjs`

**Interfaces:**
- Consumes: `BaseExercise`, `SENSITIVITY_PRESETS`, shoulder/elbow/hip landmark indices.
- Produces: exercise id `shoulder_flexion` usable via registry + settings validation, with phase state machine (down → raising → overhead → lowering) and rep counting gated on required-landmark confidence.

- [ ] Write failing test: synthetic forward-raise trajectory counts 1 rep; tracking loss mid-raise counts 0.
- [ ] Run: `node --test apps/kinelab-motion/tests/exercises.test.mjs` — expect FAIL (unknown exercise).
- [ ] Implement flexion exercise + registry + settings entry, mirroring abduction thresholds/hysteresis.
- [ ] Run full suite — expect PASS. Commit.

### Task 2: Movement direction indicators

**Files:**
- Create: `apps/kinelab-motion/src/overlays/direction-indicators.mjs`
- Modify: `apps/kinelab-motion/src/overlays/overlay-renderer.mjs` (optional flag), `apps/kinelab-motion/src/settings/settings-schema.mjs` (`showDirection` overlay toggle, default false)
- Test: `apps/kinelab-motion/tests/filters.test.mjs` or new `tests/overlays.test.mjs`

**Interfaces:**
- Produces: `computeJointVelocity(prev, curr, dtMs)` → `{vx, vy, speed}` and `shouldShowDirection(speed, threshold)`; renderer draws a short arrow only above speed threshold. No invented motion: zero/low-confidence frames yield no arrow.

- [ ] Write failing test: velocity sign/magnitude on known points; below-threshold → hidden.
- [ ] Run test — expect FAIL. Implement module + wiring + settings toggle.
- [ ] Run full suite — expect PASS. Commit.

### Task 3: Session comparison stats

**Files:**
- Create: `apps/kinelab-motion/src/sessions/session-compare.mjs`
- Modify: `apps/kinelab-motion/src/sessions/export-helpers.mjs` only if needed for shared angle keys
- Test: `apps/kinelab-motion/tests/sessions.test.mjs`

**Interfaces:**
- Produces: `compareSessions(a, b, angleKey)` → `{ meanDeltaDeg, maxDeltaDeg, repDelta, durationDeltaMs, frameCountA, frameCountB }`, all computed only from recorded finite angles; missing angle data yields `null` fields, never zeros-as-measurements.

- [ ] Write failing test on two synthetic sessions with known angle offsets.
- [ ] Run — expect FAIL. Implement. Run full suite — expect PASS. Commit.

### Task 4: Personalized baseline + full performance HUD

**Files:**
- Modify: `apps/kinelab-motion/src/settings/settings-schema.mjs` (`exercise.baseline`: `{ standingKneeDeg, standingHipDeg } | null`, validated finite 90–180 or null)
- Modify: `apps/kinelab-motion/src/app.mjs` + `apps/kinelab-motion/index.html` (HUD shows render FPS + dropped frames; baseline capture button sets baseline from current valid angles only)
- Test: `apps/kinelab-motion/tests/settings.test.mjs`

**Interfaces:**
- Consumes: `PerformanceMonitor.getMetrics()` (already returns renderFps/droppedFrames).
- Produces: validated `baseline` in persisted settings; HUD fields updated each metrics tick.

- [ ] Write failing settings test: valid baseline kept, out-of-range/non-finite baseline → null.
- [ ] Run — expect FAIL. Implement schema + HUD wiring.
- [ ] Run full suite — expect PASS. Commit.

### Task 5: Dataset validation report (docs only)

**Files:**
- Create: `apps/kinelab-motion/docs/validation.md`

**Interfaces:** Documents UI-PRMD / RepCount / public benchmark availability, license, modality, and suitability for single-phone-camera evaluation; states synthetic-benchmark results actually measured by `tests/synthetic-benchmarks.test.mjs`; explicitly disclaims clinical-grade precision and mocap transfer.

- [ ] Write report from verified facts only (dataset homepages/licenses checked; no invented error numbers).
- [ ] Commit.

### Task 6: QA, review, PR

- [ ] Run `node --test apps/kinelab-motion/tests/*.test.mjs` — record exit code and counts.
- [ ] Headless Playwright (isolated `agent-workspace`) desktop + mobile viewport screenshots of the app shell; camera-denied state captured; no physical-device camera claim.
- [ ] Self-review full diff for correctness/privacy/licensing; fix findings; re-run tests.
- [ ] Push branch, open PR with architecture map, test evidence, limitations, next steps. Do not merge.
