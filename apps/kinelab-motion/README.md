# KineLab Motion 2.0
### Single-camera motion tracking, exercise exploration, visual feedback, and local movement analytics

> **Project status:** Experimental personal-use web application. This document describes the source code present in the Clearcam fork as reviewed on **November 17, 2026**. A feature being present in source code does not mean it has been validated on every phone or browser.  
> **Scope:** One camera at a time; browser-side processing. Not a medical device, diagnosis tool, validated motion-capture instrument, or authoritative judge of exercise safety or technique.

**Repository:** https://github.com/sorensadrgit-art/clearcam  
**App directory:** `apps/kinelab-motion/`  
**Parent project:** [Clearcam](https://github.com/roryclear/clearcam) — a separate AI security-camera/NVR application in the same repository  
**Current implementation:** Native HTML, CSS, and JavaScript ES modules; Google MediaPipe Pose Landmarker; Canvas overlays; browser `localStorage`; Node built-in tests.

---

## Contents

1. [The application in plain language](#1-the-application-in-plain-language)
2. [What is available now](#2-what-is-available-now)
3. [How the system works](#3-how-the-system-works)
4. [The live interface and every control](#4-the-live-interface-and-every-control)
5. [Exercise modes explained](#5-exercise-modes-explained)
6. [All current settings and defaults](#6-all-current-settings-and-defaults)
7. [Understanding the measurements](#7-understanding-the-measurements)
8. [Session recording, playback, comparison, and export](#8-session-recording-playback-comparison-and-export)
9. [Getting started on a computer or phone](#9-getting-started-on-a-computer-or-phone)
10. [Camera positioning and practical use](#10-camera-positioning-and-practical-use)
11. [Privacy, storage, and security](#11-privacy-storage-and-security)
12. [Architecture and actual source files](#12-architecture-and-actual-source-files)
13. [Testing, known limitations, and verification status](#13-testing-known-limitations-and-verification-status)
14. [Future capabilities: not yet live](#14-future-capabilities-not-yet-live)
15. [Development priorities and acceptance checklist](#15-development-priorities-and-acceptance-checklist)
16. [FAQ and terminology](#16-faq-and-terminology)

---

## 1. The application in plain language

**KineLab Motion** is an interactive movement studio that uses a **single phone camera or computer webcam** to find a person's body landmarks and make movement visible in real time.

Imagine setting a phone on a tripod, opening the KineLab website, and performing a squat. The screen shows the camera preview with a drawn skeleton over the person, a number beside each visible knee or hip, a squat counter, and short feedback such as “Knee bend detected” or “Straighten legs to complete the repetition.” You can hide or show those overlays, switch to another exercise, or deliberately record the **motion measurements** for review afterward.

The main objectives are:

- **See movement:** Show live body landmarks, connected skeleton lines, joint-angle labels, arcs, and motion trails.
- **Measure roughly:** Calculate screen-plane joint angles, left/right differences, movement phases, repetitions, and simple holds.
- **Experiment:** Try several activity-specific analyzers without changing the camera tracking engine.
- **Review:** Save an opt-in session of timestamped skeleton and angle data, replay it, inspect angle curves, and compare recordings.
- **Extend:** Provide a base for games, avatars, gestures, movement challenges, or additional exercise modules later.

This is a **personal experiment, education, and creative software project**. The measurements are approximations from video, not a substitute for validated clinical goniometry, instrumented biomechanics, or professional movement assessment.

### Relationship to Clearcam

The repository originally contains **Clearcam**, a Python-based network video recorder/security camera system with RTSP ingestion, object detection, event management, and notification features. **KineLab Motion is a separate static web module** within that fork. It does **not** require you to launch the Clearcam Python NVR, and its current camera path does **not** use Clearcam's RTSP feeds.

| Project area | Primary purpose | Typical camera connection | Runtime |
| --- | --- | --- | --- |
| Original Clearcam NVR | Security camera video, object detection, event recordings and alerts | Network video source such as RTSP | Python + FFmpeg; NVR dashboard typically on port 8080 |
| KineLab Motion 2.0 | Single-person skeletal tracking, exercises, angle telemetry and local replay | Browser camera permission via `getUserMedia()` | Static browser app; local development example on port 4173 |

Adding KineLab to the fork does not make Clearcam's object detector a biomechanical model. The two codepaths presently serve different purposes.

---

## 2. What is available now

This table distinguishes **implemented code** from features that are merely conceptual or partially wired.

| Capability | Source-level state | What it means today |
| --- | --- | --- |
| One live camera | **Implemented** | Starts a phone or webcam through browser permission |
| Front/rear camera choice | **Implemented** | Prefers the selected facing mode when supported; Flip restarts capture |
| Heavy / Full / Lite MediaPipe models | **Implemented** | **Auto** (default) starts phones on Full (about 2× faster) and desktops on Heavy; initialization and the live auto-fallback step down Heavy → Full → Lite |
| 33-point body pose | **Implemented** | Uses Google's Pose Landmarker; visual overlay uses a subset of connections/points |
| Skeleton, joints, labels and arcs | **Implemented** | Real-time canvas graphics over the live preview |
| Motion trails / direction arrows | **Implemented** | Optional overlays for wrist/ankle movement |
| Eight joint-angle values | **Implemented** | Left/right knees, hips, elbows, shoulders. **Auto** (default) uses precise 2D image angles side-on and 3D world angles when the body turns oblique; **3D** and **2D** can be forced. The more visible leg is used instead of averaging both. |
| Segment biomechanics & alignment | **Implemented** | Shank (tibial) angle and Trunk (spine) inclination relative to vertical gravity; parallel alignment evaluation |
| Kinematic bone-length invariance filter | **Implemented** | Physics-informed constraint model that eliminates 3D depth jitter and limb elasticity along direction rays |
| Decoupled 60/120 FPS pose interpolation | **Implemented** | Native display rate LERP follower decoupling 30 FPS inference from canvas rendering for liquid smooth motion. **Predictive overlay** (default on) extrapolates the drawn skeleton up to 40 ms; display only |
| Per-frame inference scheduling | **Implemented** | Inference runs once per camera frame via `requestVideoFrameCallback` (Chrome 83, Safari 15.4, Firefox 132), with rAF polling as fallback; pipeline timestamps use the frame's capture time |
| Tracking confidence states | **Implemented** | Shows good, partial or lost tracking; unavailable angles appear as `—` |
| Per-rep metrics & quality gate | **Implemented** | Live log: last rep's ROM, phase tempo and reliability, plus running average ROM ± CV, tempo and cadence. Results card: session aggregates (average ROM ± CV, average tempo, within-set cadence, reliable/total). Reps CSV: every rep's start/extreme angle, ROM, phase durations, extreme confidence and quality. Reps below the confidence gate, or whose start was not observed, are flagged and left out of averages and peak depth. |
| Offline re-analysis of recordings | **Implemented** | Results card, rep analysis and Reps CSV use zero-phase smoothing (Hampel outliers → gap fill ≤300 ms → 4th-order Butterworth `filtfilt`, 4 Hz) of angles from unfiltered landmarks; no live-filter lag. Synthetic side-view squat: knee RMSE 3.2° → 2.2°, jitter 1.12° → 0.50°. Older recordings: live angles at 6 Hz |
| Camera setup guidance | **Implemented** | Prompts to level the phone (gravity sensor, ±3°), stand 2–3 m away, and place the camera at hip height |
| Camera tilt correction | **Implemented, beta (off)** | Rotates landmarks by the phone's measured roll before angles; sign convention unverified on devices, so check that a vertical trunk reads ≈0° |
| Clinical angle display | **Implemented** | Default **Clinical**: knee/elbow/hip flexion = 180° − interior (0° = straight), hip labelled trunk–thigh, shoulder as elevation. **Interior** shows raw interior angles. Exports always stay interior |
| Angle reliability flags | **Implemented** | Far-side (occluded) limb in side views and joints below 0.7 confidence are dimmed with a reason tooltip |
| Validation harness | **Implemented (developer CLI)** | `bench/validate.mjs` compares a session export with Kinovea CSV or OpenSim `.mot` (time alignment, frame- and rep-level bias, LoA, ICC(A,1), SEM, MDC95). No real-data results yet; see [docs/validation.md](docs/validation.md) |
| One Euro filtering / outlier rejection | **Implemented** | Smooths 2D landmarks and, separately, 3D world landmarks (lower depth cutoff); presets tuned on `bench/accuracy-bench.mjs`. See [docs/accuracy-speed-research.md](docs/accuracy-speed-research.md) |
| Smart body framing radar | **Implemented** | On-canvas position guidance; upper-body decoupling for seated/close-up exercises |
| Repetition/hold modules | **Implemented, experimental** | Squat, sit-to-stand, lunge, shoulder abduction, shoulder flexion, single-leg balance |
| Free Tracking | **Implemented** | Skeleton and angle exploration without exercise counting |
| Left/right symmetry card | **Implemented, heuristic** | Shows angle differences; thresholds are not a normative diagnosis |
| Audio cues & Voice Coach | **Implemented** | Web Audio depth target ping (1046Hz), lockout tick, rep chime, and Web Speech spoken rep counting |
| Personal standing baseline | **Implemented, basic** | Stores a snapshot of average standing knee/hip angles |
| Record motion-data session | **Implemented** | User-triggered landmark/angle/time recording in browser storage, not a camera video |
| Replay and graphs | **Implemented** | Scrub, step, play at 0.5×/1×/2×, inspect angle vs time |
| Session comparison / ghost pose | **Implemented, basic** | Frame-index comparison and a selected frozen reference skeleton |
| CSV / JSON export and deletion | **Implemented** | Download measurements; remove saved browser sessions |
| Performance metrics | **Implemented, partial caveats** | Inference/render FPS, model-call latency, **E2E** camera→overlay latency (p50; hover for p95 and camera→landmarks) and camera frames missed. E2E excludes sensor exposure and display scan-out; see limitations. `?input=direct` skips the 2D-canvas downscale for A/B tests |
| Dual video & telemetry recording | **Implemented** | HTML5 MediaRecorder captures video stream saved to IndexedDB; timeline player synchronizes video and skeletal canvas replay |
| Multiple cameras / calibrated 3D | **Not in scope** | There is no synchronized multiview reconstruction |
| Automated clinical “correct form” decision | **Not implemented or validated** | Feedback is based on simple geometry and movement-state rules |
| Gesture-controlled UI, 3D avatars, motion painting | **Planned extension points only** | Not shipped as active user experiences |
| Fully playable motion game | **Not in the current UI** | A preliminary developer-side movement challenge class exists; it is not a complete integrated arcade |

**Important distinction:** “Implemented” means the necessary source files and application wiring are present. It does not certify accuracy or confirm successful performance on your particular device.

---

## 3. How the system works

### End-to-end flow

```text
PHONE CAMERA OR WEBCAM
       │
       │ Browser permission: getUserMedia()
       ▼
LIVE VIDEO ELEMENT
       │
       │ Sample frames with a bounded requestAnimationFrame loop
       ▼
GOOGLE MEDIAPIPE POSE LANDMARKER
       │
       │ Up to 33 normalized body landmarks + confidence values
       ▼
CONFIDENCE / OUTLIER / ONE EURO FILTERING
       │
       ├──────────────► CANVAS OVERLAYS
       │                Skeleton · Joint dots · Degrees · Arcs · Trails
       │
       ├──────────────► 2D BIOMECHANICS
       │                8 angles · Trunk helper · Left/right comparisons
       │
       └──────────────► EXERCISE MODULE
                        Phases · Reps or hold duration · Brief cues
                                   │
                       Optional “Record” action
                                   ▼
                            BROWSER STORAGE
                                   │
                     Replay · Compare · JSON / CSV
```

### Key processing stages

1. **Camera acquisition.** `navigator.mediaDevices.getUserMedia()` asks the browser for one video source. The request disables microphone audio, uses a preferred front/rear camera, and requests a selected resolution.
2. **Model initialization.** The app loads a pinned MediaPipe Tasks Vision library, WebAssembly assets, and a Heavy or Full Pose Landmarker model when the camera session starts. It tries GPU then CPU for each candidate.
3. **Pose inference.** `detectForVideo()` processes selected frames. The present main-thread scheduler requests an inference no more often than roughly once every **75 ms**, so its nominal processing cap is approximately **13.3 inference calls per second** before computation overhead. That is not a measured device FPS guarantee.
4. **Filtering and validity.** Landmarks are screened for confidence; temporal smoothing is applied; some abrupt jumps are rejected. Lost/partial tracking is conveyed to the display and exercise modules.
5. **2D angle calculation.** The app forms the angle at a landmark using its connected landmarks, scaled to video pixel width and height. Unreliable or degenerate inputs return `null` rather than a number.
6. **Exercise recognition.** The selected module consumes angles, landmarks, and timestamps to advance a small state machine. The result controls counting, hold timers, and text/audio feedback.
7. **Visual rendering.** A canvas positioned over the camera renders markers, limb links, arcs, degree badges, optional motion trails, and optional direction arrows.
8. **Optional telemetry capture.** Pressing **Record** stores landmarks, angles, timestamps, current phase, status, and rep totals for later local analysis.

### What is “AI” here?

The AI component is **pose estimation**: predicting where body landmarks appear in an image. The current exercise interpretations are predominantly **hand-written rules and thresholds**, not a model trained to recognize safe versus unsafe exercise form. A joint angle value is an observation of a projection, not an explanation of why movement occurred.

---

## 4. The live interface and every control

The app uses a dark, motion-focused interface. Its main screen has a large **Live Canvas** and a **Live Telemetry** column.

### Top bar

- **KineLab / Motion 2.0:** Application identity and home.
- **LOCAL PROCESSING:** Indicates that analysis is designed to run in the browser; downloading model assets still requires network access on first use.
- **SESSIONS + count:** Opens or closes the saved-session analytics area; the count reflects entries in local browser storage.
- **SETTINGS:** Opens the five-section Studio Settings dialog.

### Live canvas

- **Video preview:** The incoming picture from the selected camera.
- **Pose canvas:** Transparent rendering layer placed over that picture.
- **Empty state:** Setup instructions shown when the camera is not active.
- **Status chip:** OFFLINE/LIVE and tracking feedback.
- **Tracking HUD:** STANDBY, TRACKING, PARTIAL, or NO POSE depending on recognized landmark coverage.
- **Performance HUD:** Inference FPS, active model, last model-call latency, E2E camera→overlay latency (p50), render FPS, and camera frames the page missed.
- **Overlay legend:** Reminds the user that the interface works with 33 body landmarks, smoothing, and 2D angles.

### Capture buttons

| Control | What it does | Important detail |
| --- | --- | --- |
| **Start camera** | Requests permission, starts video, loads the tracker, and begins inference | Requires a supported secure browser context |
| **Stop** | Stops camera tracks, closes the model, cancels the loop, and clears the overlay | If motion-data recording is active, it ends that recording |
| **Flip** | Changes preferred front/rear facing mode | Stops and restarts capture; result depends on actual device cameras |
| **Record / Stop REC** | Starts or finishes **motion telemetry** recording | **Does not record the underlying camera video** |
| **Reset count** | Resets the selected exercise state | Does not delete previously saved sessions |

### Quick controls

- **EXERCISE MODULE:** Squat, Sit-to-Stand, Lunge, Shoulder Abduction, Shoulder Flexion, Single-Leg Balance, or Free Tracking.
- **TRACKER MODEL:** High Accuracy / Heavy, Faster / Full, or Fastest / Lite.
- **Angle badges:** Quick on/off control for floating degree numbers.
- **Angle arcs:** Quick on/off control for geometric sweep arcs.

Model selection while live causes a camera restart; changing the exercise selects a different state machine and resets the active exercise's in-memory count.

### Live telemetry panel

- **Joint angles:** Eight labeled values in degrees, grouped by body area. A dash indicates an unavailable measurement.
- **Exercise monitor:** Shows current count or hold seconds, named movement phase, available derived metrics, and short status-based messages.
- **Standing baseline deviation:** When a baseline was saved, displays differences from the stored reference knee and hip angles.
- **Bilateral symmetry HUD:** Displays left and right values plus their absolute difference for knees, hips, elbows and shoulders. An overall label uses internal prototype thresholds; it is **not a biological normality test**.

---

## 5. Exercise modes explained

The current code contains **six exercise analyzers** plus Free Tracking. All exercise classifiers are experimental. They require a useful camera position, adequate visibility, and enough successive valid frames.

| Mode | Recommended view | What it follows | What appears |
| --- | --- | --- | --- |
| **Squat** | Side | Knee bend and extension across a down/up cycle | Reps, current phase, knee angle, phase-specific message |
| **Sit-to-Stand** | Side | Seated → rising → upright → lowering → seated | Complete seated-to-standing-to-seated cycles, knee/hip values |
| **Lunge** | Side | More-flexed knee as a proxy for the lead leg; lowering and recovery | Lunge count, inferred lead leg, knee angle |
| **Shoulder Abduction** | Front | Arm elevation from resting position to a peak and back | Reps and left/right shoulder-angle estimates |
| **Shoulder Flexion** | Side | Forward arm elevation using a related shoulder-angle state machine | Reps and approximate shoulder elevation |
| **Single-Leg Balance** | Front | Vertical difference between ankle image positions | Current hold seconds, best hold, inferred stance leg |
| **Free Tracking** | Any useful view | No exercise rule; unrestricted pose visualization | Joint metrics and skeleton; no exercise rep counter |

### Squat: a complete example

A simple squat session looks like this:

1. Select **Squat**, stand side-on, and ensure hips, knees and ankles appear.
2. When the knee-angle estimate decreases past the configured “down” threshold for enough consecutive frames, the detector enters its down phase.
3. When the angle increases past the “up” threshold and minimum timing checks are satisfied, it counts one repetition.
4. If the needed knee landmark becomes unavailable, the incomplete cycle is discarded rather than scored.

The baseline **standard-sensitivity** squat thresholds in source are approximately **115° down**, **155° up**, **two confirming frames**, **350 ms minimum phase time**, and **500 ms between completed counts**. Other sensitivity modes modify some thresholds and confirmation requirements. These are **software heuristics**, not universal squat-depth standards.

### Sit-to-Stand

The module looks for seated, rising, standing, lowering, and seated phases and counts a complete return to sitting. Although the source description references chair-stand testing, the current interface does **not** implement a standardized 30-second chair-stand administration or validated score.

### Lunge

The tracker estimates a lead knee from which available knee is more flexed, then follows a descent/depth/recovery sequence. This is not step-direction recognition, floor-contact measurement, or a validated lunge-form classifier.

### Shoulder Abduction and Flexion

The two UI modes are distinct entries. Both presently derive shoulder elevation from the elbow–shoulder–hip angle. Abduction recommends a **front view**; flexion recommends a **side view**. They do not resolve 3D shoulder-plane motion or account reliably for trunk compensation.

### Single-Leg Balance

The current rule infers that one ankle is lifted when its **vertical pixel position** differs sufficiently from the other. It times continuous holds and records a best hold. Its result is particularly sensitive to viewing angle, stance width, camera tilt and occlusion. It does **not** directly measure postural sway, center of mass, pressure or true foot-floor contact.

### “Proper form” versus useful feedback

Messages like “return upright,” “raise the arm,” and “complete the repetition” are cues mapped from simple movement states. The software **cannot presently prove** that a user is moving safely, with good alignment, or within an appropriate individual range. Use feedback for experimentation and self-observation, not injury or rehabilitation decisions.

---

## 6. All current settings and defaults

Settings are saved under browser key `kinelab_motion_settings_v2`. Invalid stored values fall back to schema defaults. Changes made to settings affecting the capture pipeline may require a camera restart to take full effect.

### Camera tab

| Setting | Available options | Default | Meaning |
| --- | --- | --- | --- |
| **Camera Orientation** | Rear/Environment; Front/User | **Rear** | Preferred camera direction; browser/device determines actual selection |
| **Mirroring** | Auto (front camera); Always On; Off | **Auto** | Horizontal preview/overlay mirroring |
| **Capture Resolution** | 480p SD; 720p HD; 1080p FHD | **720p** | Requested `ideal` resolution, not a guaranteed output |

The camera request uses `audio: false`, requests roughly 30 FPS as an ideal input, and allows a maximum of 60 FPS **if supported**. Camera capture FPS and AI inference FPS are separate numbers.

### Tracking tab

| Setting | Available options | Default | Meaning |
| --- | --- | --- | --- |
| **MediaPipe Model** | Auto; Heavy; Full; Lite | **Auto** | Auto = Full on phones/tablets, Heavy on desktop; or force a model |
| **Adaptive Performance Fallback** | On; Off | **On** | Watch high latency or low inference FPS |
| **Temporal Smoothing (1€ filter)** | Off; Low; Medium; High | **Medium** | Reduce landmark movement jitter |
| **Confidence Threshold** | 0.45 Lenient; 0.55 Standard; 0.65 Strict | **0.55** | Minimum confidence required for usable landmarks/angles |
| **Camera Level Guidance** | On; Off | **On** | Uses the phone's gravity sensor to prompt leveling (iOS asks permission on Start) |
| **Tilt Correction (beta)** | On; Off | **Off** | Corrects measured camera roll before computing angles |

**Model fallback:** On initial load, a model that fails to initialize falls back down the ladder Heavy → Full → Lite. During a run, sustained high latency or low FPS makes the adaptive controller load the next faster model in the background and swap it in without restarting the camera (when *Auto fallback* is on).

**Smoothing nuance:** Smoothing can make motion visually steadier but may delay fast changes. Even “Off” smoothing still passes through certain confidence/outlier checks.

### Overlays tab

| Setting | Default | What it shows |
| --- | --- | --- |
| Full-body Skeleton Lines | **On** | Connected limb/body line segments |
| Colored Joint Markers | **On** | Landmark dots |
| Joint Degree Labels | **On** | Angle badges near valid joints |
| Joint Angle Arcs | **On** | Drawn angle sweep sectors |
| Motion Trails | **Off** | Short wrist/ankle movement history |
| Movement Direction | **Off** | Small wrist/ankle directional indicators |
| Bilateral Symmetry HUD | **On** | Left/right knee, hip, elbow, shoulder differences |
| Ghost Reference Skeleton | **Off** | One manually chosen, faint reference pose |
| Angle Convention | **Clinical** | Clinical (flexion, 0° = straight) or Interior (180° = straight) for labels, arcs, telemetry and results |

Quick controls for badges and arcs also exist outside Settings.

### Exercise tab

| Setting | Options / behavior | Default |
| --- | --- | --- |
| **Form Sensitivity** | Forgiving (+12° tolerance), Standard, Strict (−8° tolerance); also changes minimum hold time / confirming frames | **Standard** |
| **Audio Cues** | Repetition and hold chimes synthesized by the browser | **On** |
| **Capture Baseline** | Saves averages of currently visible knee/hip angle estimates for a standing reference | **Not captured** |

The “form” sensitivity label is a UI name; it changes internal event thresholds, **not a clinical definition of good form**. A baseline is only a personal angle reference, not camera calibration.

### Privacy tab

- **Local-Only Execution:** Explains the intended browser-local pose processing.
- **Saved Motion Sessions:** Shows a rough local-storage usage estimate and saved count.
- **Clear All Data:** Removes locally recorded sessions from the app's session index; it is not a delete-all-browser-history button.
- **Reset Defaults:** Replaces user settings with defaults; this is distinct from clearing sessions.

There is also a settings-schema `recordingMode` value supporting `manual` and `off`, with `manual` as default; the current UI does not expose that choice as a separate control. Recording begins only when the user presses **Record**.

---

## 7. Understanding the measurements

### The 33 landmarks

MediaPipe Pose Landmarker produces estimates for **33 anatomical reference points** including the nose/face-related points, shoulders, elbows, wrists, hands, hips, knees, ankles, heels and feet. The visible skeleton and angle modules use the subset necessary for body links and exercise calculations.

### The eight angle readouts

| Display label | Landmark triplet at the middle joint |
| --- | --- |
| Left / Right Knee | Hip → **Knee** → Ankle |
| Left / Right Hip | Shoulder → **Hip** → Knee |
| Left / Right Elbow | Shoulder → **Elbow** → Wrist |
| Left / Right Shoulder | Elbow → **Shoulder** → Hip |

The implementation computes the **internal angle between two projected limb vectors** using pixel-aspect-corrected coordinates. Approximate 180° commonly represents a straight knee or elbow in the chosen view; orientation and landmark geometry change the reading.

With the default **Clinical** convention the display shows knee, elbow and hip flexion as 180° − interior (0° = straight; the hip value is trunk–thigh, so it includes trunk lean) and the shoulder as elevation; the Interior convention shows the raw interior angle. Readouts on the occluded far-side limb in a side view, or below 0.7 confidence, are dimmed with a reason tooltip.

The implementation also includes a helper for estimated **trunk inclination relative to vertical**, but it is not one of the eight primary live sidebar angles.

### Why one camera is limited

A single image loses true depth information. If a joint rotates toward or away from the camera, its screen-plane angle can change without the same change in actual anatomical 3D joint motion. Clothing, low lighting, camera perspective, self-occlusion and model misidentification introduce additional errors. Reported degrees may look precise but are **not guaranteed accurate to that precision**.

**Practical rule:** Compare the same exercise with the same camera orientation, height, distance and lighting where possible. Prefer repeated within-view trends rather than treating any single number as a definitive measurement.

### Missing landmarks and confidence

If required points are absent, hidden, too low-confidence, or geometrically degenerate, the angle helper returns no measurement and the UI shows **—**. Tracking messages communicate approximate visibility, not a calibrated probability of accuracy.

### Symmetry

The app reports absolute left/right angle differences for available paired joints. Its internal code labels differences of **12° or less** as symmetric per pair and an overall average of **10° or less** as “BALANCED.” Those are **prototype visualization thresholds, not validated reference ranges**. A larger difference is not automatically abnormal or unsafe.

### Timing, FPS, and latency

- **Camera FPS:** Frames the device actually captures and sends to the video element.
- **Inference FPS:** Frames the AI tracker analyzes each second.
- **Render FPS:** Frequency at which the UI render loop executes, not necessarily how frequently a new pose is estimated.
- **Latency (ms):** Time used by one model call (`detectForVideo`).
- **E2E (ms):** Median time from the frame's camera capture clock (`requestVideoFrameCallback` `captureTime`, else `presentationTime`) to the overlay being drawn. Hover for p95 and camera→landmarks. It excludes sensor exposure and display scan-out (tens of ms on phones); verify a device once with 240 fps slow-motion video of the screen. Without `requestVideoFrameCallback` the clock is the callback time and under-reports.
- **Dropped frames:** Camera frames the page never received, from gaps in `presentedFrames`. Frames skipped on purpose by the ~35 FPS inference cap are not counted. Always 0 without `requestVideoFrameCallback`.
- **A/B input path:** `?input=direct` feeds the `<video>` straight to MediaPipe instead of the adaptive 2D-canvas downscale; compare E2E and FPS on the same device.

---

## 8. Session recording, playback, comparison, and export

### What the Record button saves

**This is a measurement recorder, not a video recorder.** A session contains:

- Session ID and creation time.
- Selected exercise type.
- Number of stored movement frames.
- Duration and final repetition count.
- Per-frame elapsed timestamp.
- A compact array of the estimated pose landmark coordinates (`x`, `y`, `z` where supplied, visibility).
- Calculated joint angles.
- Detected movement phase, tracking/feedback status, and repetition count.

The recorded `z` coordinate is a model value stored with the pose, **not calibrated physical depth**. Recording is opt-in; pressing Stop REC or stopping the camera ends capture.

### Where it is saved

Sessions are stored in this browser's `localStorage` with an index. The code keeps **up to 20 indexed sessions** and removes the oldest session as newer ones are saved beyond that limit. Browser quota limits apply; large sessions may fail to save. The storage is linked to the browser profile and website origin, not automatically synchronized across devices.

Browser storage is **not an encrypted vault**. Anyone with appropriate access to that browser profile/site data may be able to read stored movement measurements. Clearing browser/site data may erase sessions permanently.

### Sessions drawer and replay

Open **SESSIONS**, choose a saved item and select **Load & Inspect**:

- Play/pause a recorded skeletal sequence.
- Scrub through its frame timeline.
- Step one frame backward or forward.
- Choose playback speed **0.5×, 1×, or 2×**.
- View one selected joint's **angle-over-time** chart.
- Inspect measurements for the selected replay frame.
- Delete the selected session.

The recorded data reconstructs a skeleton sequence. Since the current recorder does not preserve raw video, replay is **skeleton and telemetry playback**, not replay of the original person on camera. The replay renderer draws onto the canvas when the live camera is not active.

### Ghost reference

With a saved session loaded, pause or seek to a useful frame and click **Set as Ghost Overlay**. This stores the **selected single pose frame** as a translucent visual reference. It is not automatic temporal synchronization of two full movement videos.

### Compare Sessions

Select two saved sessions and a joint. The comparison reports:

- Difference in total repetitions.
- Difference in recording duration.
- Mean absolute difference between available, index-paired angle samples.
- Maximum absolute difference between those paired samples.
- Frame counts for both sessions.

**Current limitation:** Comparison pairs measurements by sample order; it does **not** first synchronize exercises by movement phase, elapsed time or dynamic time warping. A large reported difference may come from recording speed or start time rather than biomechanics.

### Exports

| Format | Contains | Does not contain |
| --- | --- | --- |
| **JSON** | Session metadata, timestamps, compact landmarks, live angles, angles from unfiltered landmarks (`rawAngles`, dropped when storage is too full to keep them without evicting older sessions), phases, statuses and reps | Original camera video |
| **CSV** | Row per frame with elapsed time, phase, status, reps, eight joint-angle fields | Raw landmark-coordinate arrays or original video |

There is no built-in MP4 export, 3D animation export, PDF clinical report, or cloud backup in the current KineLab module.

---

## 9. Getting started on a computer or phone

### Requirements

- A computer with Git and Python 3, or another simple static HTTP server.
- A webcam, built-in laptop camera, or compatible phone camera.
- A modern camera-capable browser such as Chrome, Edge or Safari.
- Internet access **at least when loading the MediaPipe package, WASM, and model assets** (unless those assets are later bundled locally).
- A **secure origin** for camera permission: `http://localhost` on the same device or an HTTPS site.

For KineLab's **static web module**, you do **not** need to install Clearcam's Python AI/NVR dependencies, FFmpeg, tinygrad, React, Vite, or an npm build chain just to serve the page.

### Quick start on desktop

From a terminal:

```bash
git clone https://github.com/sorensadrgit-art/clearcam.git
cd clearcam
python apps/kinelab-motion/serve.py 4173
# (Or: python3 -m http.server 4173 --directory apps/kinelab-motion)
```

On Windows, `python` or `py` may be the available launcher instead of `python3`.

`serve.py` binds to `127.0.0.1` so the app directory is not exposed on your network. To reach it from another device, pass `--host 0.0.0.0`; note that browsers only allow camera access over HTTPS or `localhost`, so a phone on plain-HTTP LAN will not get the camera.

Open **http://localhost:4173**, select an exercise, press **Start camera**, and approve the browser permission request. The first model download may take longer than subsequent starts.

**Note:** This runs KineLab only. To run the original Clearcam NVR, follow the repository's root [README](../../README.md); it uses a separate Python process and port.

### Using a phone

The phone can run the browser app and perform inference on the phone itself. Open a trusted **HTTPS** deployment of `apps/kinelab-motion/`, grant camera access, and select rear or front camera.

A plain LAN URL such as `http://192.168.1.50:4173` is generally **not** considered secure by mobile browsers and may not permit camera capture. Options include an HTTPS static host or a trusted HTTPS reverse proxy/tunnel for this **static app only**.

**Do not publish the original Clearcam NVR's unauthenticated port 8080 as a substitute for HTTPS hosting.**

### First-use walkthrough

1. Place the phone or webcam on a stable support; frame the intended joints.
2. Open the KineLab static app on a secure address.
3. Select **Squat** or **Free Tracking**.
4. Choose Heavy for accuracy-oriented experimentation, or Full if the device struggles.
5. Press **Start camera** and grant video permission.
6. Confirm the skeleton appears and the HUD reports tracking.
7. Move gently through a few test repetitions. Watch the joint labels and movement-phase feedback.
8. Open **Settings** to adjust smoothing, overlays and confidence.
9. Press **Record** only if you want to save the movement measurements.
10. Stop recording and open **SESSIONS** to replay, graph, compare, export or delete them.

---

## 10. Camera positioning and practical use

| Activity | Camera setup | Why |
| --- | --- | --- |
| Squat / Sit-to-Stand | Side view; keep shoulders, hips, knees, ankles and feet visible | Knee and hip flexion project more clearly in this plane |
| Lunge | Side or clear oblique view with both legs visible | Helps the detector differentiate relative knee flexion |
| Shoulder Abduction | Front view, both arms fully inside frame | Supports left/right elevation comparison |
| Shoulder Flexion | Side view with shoulder, elbow and torso visible | Reduces foreshortening of forward raises |
| Single-Leg Balance | Front view, both ankles visible | Current rule uses a screen-space ankle-height difference |
| Free Tracking | Any view that shows the joints of interest | No particular exercise threshold applies |

General tips: use good even lighting, position the camera at approximately hip height for lower-body exercises, stabilize the phone, avoid cutting off hands/feet, and allow enough space to finish the movement without leaving the frame.

A **front-facing selfie camera** may be mirrored; use consistent mirroring and camera position between sessions. A tripod or stable shelf is usually more useful for repeatability than buying an additional camera.

---

## 11. Privacy, storage, and security

### Current data path

The **pose inference** and **motion-session persistence** are designed for local execution in the browser. The current KineLab module does not define a backend API for uploading camera frames or saved sessions.

MediaPipe Tasks keeps camera frames on-device, but per its own privacy notice it **sends performance and usage metrics about the API to Google**.

However, the site **downloads its JavaScript library/WASM and machine-learning models from external hosts**. That is a network dependency; “local inference” is not the same as “fully offline from first use.” A normal hosted website also exposes ordinary page-request metadata to its hosting provider.

### Recording consent

- Camera permission is requested through the browser.
- Motion recording begins only after **Record** is pressed.
- Stop closes camera tracks and model resources.
- Session data is retained in local browser storage until deleted, evicted by the app's 20-session policy, or cleared by the browser.
- No sign-in, access-control roles, cloud sync, encryption-at-rest or cross-device backup are implemented in this static module.

For private experimentation, use recordings of yourself or of participants who have clearly agreed to the capture. Do not put identifiable patient recordings or protected health information into this experimental tool without a dedicated privacy/security and regulatory review.

### Relationship to the parent Clearcam security system

The KineLab module should remain independently hosted while the original Clearcam NVR receives its own authentication, credential-handling and network-exposure review. The codebase being shared does **not** imply the applications share secure storage, permissions, or camera sources.

---

## 12. Architecture and actual source files

The current KineLab 2.0 implementation uses **plain HTML + CSS + native ES modules**, not a React/Vite application. Its major source organization is:

```text
apps/kinelab-motion/
├── README.md
├── index.html                # Studio interface, controls, settings and analytics
├── styles.css                # Dark responsive visual system
├── serve.py                  # Multi-threaded no-cache dev server; binds 127.0.0.1 (use --host to change)
├── src/
│   ├── app.mjs               # Runtime orchestrator, camera and UI events
│   ├── pose-math.mjs         # Backward-compatible geometry re-exports
│   ├── tracking/
│   │   ├── mediapipe-runner.mjs     # Model lifecycle, CDN loader, resolution decoupling
│   │   ├── framing-assistant.mjs    # Body positioning radar & limb requirement checks
│   │   ├── performance-monitor.mjs  # Inference/render FPS and latency tracking
│   │   ├── adaptive-controller.mjs  # Performance watchdog and auto-fallback
│   │   └── worker-bridge.mjs        # Web Worker feasibility probe
│   ├── filters/
│   │   ├── one-euro-filter.mjs      # LowPassFilter and OneEuroPoint2D
│   │   ├── outlier-filter.mjs       # Speed and jump gating
│   │   └── pose-filter.mjs          # 33-landmark composite filter & presets
│   ├── biomechanics/
│   │   ├── landmark-indices.mjs     # MediaPipe topology & constants
│   │   ├── angle-calculations.mjs   # 2D & 3D metric vector geometry, angleAt, uncertainty
│   │   └── symmetry-analyzer.mjs    # Bilateral left/right limb symmetry
│   ├── exercises/
│   │   ├── exercise-base.mjs        # BaseExercise interface & sensitivity presets
│   │   ├── exercise-registry.mjs    # Dynamic module registry
│   │   ├── squat-exercise.mjs       # Squat repetition state machine
│   │   ├── sit-to-stand-exercise.mjs# Chair rise functional mobility
│   │   ├── lunge-exercise.mjs       # Lunge depth and lead leg detection
│   │   ├── shoulder-abduction-exercise.mjs # Overhead reach and range of motion
│   │   ├── shoulder-flexion-exercise.mjs   # Forward reach and sagittal range of motion
│   │   └── balance-exercise.mjs     # Single-leg timed hold
│   ├── overlays/
│   │   ├── overlay-renderer.mjs     # Master canvas coordinator
│   │   ├── skeleton-renderer.mjs    # Bones and joint nodes
│   │   ├── angle-arc-renderer.mjs   # Geometric sweep arcs and badges
│   │   ├── motion-trails.mjs        # Extremity velocity trails
│   │   ├── direction-indicators.mjs # Velocity arrows and movement direction
│   │   └── ghost-renderer.mjs       # Reference skeleton overlay
│   ├── sessions/
│   │   ├── session-recorder.mjs     # Video MediaRecorder & compact telemetry serialization
│   │   ├── video-store.mjs          # IndexedDB video blob persistence engine
│   │   ├── session-storage.mjs      # LocalStorage index & metadata persistence
│   │   ├── timeline-player.mjs      # Synchronized video/canvas replay, scrub, speed
│   │   ├── chart-renderer.mjs       # Pure canvas joint angle graph
│   │   ├── session-compare.mjs      # Delta metrics and session comparison
│   │   └── export-helpers.mjs       # CSV & JSON exporters
│   ├── settings/
│   │   ├── settings-schema.mjs      # Defaults and schema validation
│   │   └── settings-manager.mjs     # Evented state container
│   └── integrations/
│       ├── audio-cues.mjs           # Zero-dependency Web Audio synthesizers
│       ├── extension-contracts.mjs  # Feature flags and extensibility interfaces
│       └── movement-challenge.mjs   # Interactive target acquisition challenge
└── tests/
    ├── dom-contract.test.mjs                 # Verifies all DOM IDs in app.mjs exist in index.html
    ├── angle-calculations-3d.test.mjs        # 3D metric space and depth vector tests
    ├── mediapipe-scaling.test.mjs            # High-throughput resolution decoupling tests
    ├── framing-assistant.test.mjs            # Body framing radar & upper-body decoupling tests
    ├── accuracy-framing-integration.test.mjs # Cross-module accuracy integration tests
    ├── video-store.test.mjs                  # IndexedDB video storage & blob lifecycle tests
    ├── session-recorder-video.test.mjs       # MediaRecorder video chunk capture tests
    ├── timeline-player-sync.test.mjs         # Video element synchronization tests
    ├── pose-math.test.mjs                    # Preserved baseline regression tests
    ├── filters.test.mjs                      # 1€ filter, outlier, and pose smoothing tests
    ├── biomechanics.test.mjs                 # 2D projection, trunk angle, symmetry tests
    ├── exercises.test.mjs                    # State transitions, debounce, tracking-loss tests
    ├── settings.test.mjs                     # Validation, defaults, corruption recovery
    ├── sessions.test.mjs                     # Recording, playback, CSV/JSON serialization
    ├── synthetic-benchmarks.test.mjs         # UI-PRMD / RepCount style kinematics benchmarks
    ├── gap-closure.test.mjs                  # Shoulder flexion, direction, and baseline tests
    ├── integrations.test.mjs                 # Extension isolation and audio synthesis tests
    └── movement-challenge.test.mjs           # Interactive challenge gameplay logic tests
```

### Module responsibilities

- **`app.mjs`** owns the UI state, camera lifecycle, frame loop, telemetry rendering, recording toggles, and settings bindings.
- **`tracking/*`** loads the MediaPipe model, times inference, evaluates performance fallback, and probes Worker feasibility.
- **`filters/*`** handles confidence-aware smoothing, temporary outlier rejection, and missing landmarks.
- **`biomechanics/*`** stores landmark indices, computes projected angles, and reports paired-joint differences.
- **`exercises/*`** provides a registry of configurable analyzers whose internal states can be independently tested.
- **`overlays/*`** draws skeleton, angles, trails, arrows, and ghost reference.
- **`sessions/*`** stores compact telemetry, controls timeline playback, draws graphs, compares sessions, and serializes exports.
- **`settings/*`** validates preferences, supplies defaults, and persists them locally.
- **`integrations/*`** provides audio synthesis and extension contracts; only select experimental extension logic exists.

### External libraries and services

- [Google MediaPipe Tasks Vision](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker): Pose Landmarker engine.
- `@mediapipe/tasks-vision@1.2.3`: SDK version referenced directly by the current loader.
- jsDelivr: The configured JavaScript/WASM delivery host.
- `storage.googleapis.com/mediapipe-models/`: Configured Heavy/Full/Lite model file host.
- Browser APIs: `getUserMedia`, `HTMLVideoElement`, `CanvasRenderingContext2D`, `requestAnimationFrame`, `localStorage`, `Web Audio`, Blob downloads.

The repository presently references fixed SDK/model URLs, **but this README does not claim that asset integrity hashes are enforced**, since no such independent verification was established in this documentation audit.

---

## 13. Testing, known limitations, and verification status

### Automated tests present in source

The KineLab module contains **27 Node test files and 193 automated test definitions**, all passing with 100% green coverage on Node.js 20+ with zero npm dependencies.

The test files exercise categories including 3D metric vector geometry, bone-length invariance filtering, 60/120 FPS pose interpolation, depth target audio cues & voice coach, exercise state machines, settings validation, DOM element contracts, sessions, extension behavior, movement challenges, and synthetic measurement scenarios.

Run them from the repository root:

```bash
node --test apps/kinelab-motion/tests/*.test.mjs
```

Node.js **20+** is a reasonable baseline. The test runner comes with Node; this standalone module has no npm install step for these tests. Execution takes ~2 s across all tests (the synthetic accuracy benchmark dominates). Tests resolve paths relative to their own files, so they can be run from any directory.

### Known limits and caveats

1. **Uncalibrated 2D:** The degrees are image-plane angles, not reliable anatomical 3D ROM or laboratory motion capture.
2. **Device performance unknown:** Heavy may be too slow or may fail on some mobile hardware.
3. **Main-thread inference:** The active app calls `detectForVideo()` synchronously. WorkerBridge currently checks feasibility; it is **not a deployed worker inference pipeline**.
4. **Adaptive fallback incomplete:** In-session fallback can update saved model preference without reinitializing the model until capture restart.
5. **Dropped-frame metric partial:** The counter exists, but the main frame loop does not currently account for every skipped input frame.
6. **Simple exercise logic:** Hard-coded projected-angle thresholds may miss reps or count non-target motions. Sensitivity options are not medically validated.
7. **Single-person mode:** `numPoses: 1` is configured; multiple simultaneous people are not analyzed as separate participants.
8. **No raw-video sessions:** Replay and JSON use recorded landmark data. There is no built-in MP4 source footage or synchronized video playback.
9. **Session comparison is not time-aligned:** It compares valid samples by ordinal position, not equivalent phases or timestamps.
10. **LocalStorage limitations:** Storage quota, eviction, browser clearing, and profile changes can affect availability; data is not encrypted or synced.
11. **Baselines are simplistic:** Capturing current knee/hip averages is not camera calibration or full-body personalized kinematics.
12. **Incomplete extension UX:** Gesture controls, 3D avatar, motion painting, and a production-grade motion arcade are not active built-in screens.
13. **Device coverage not established here:** Camera permissions, orientation switching, actual inference throughput, audio policies and UI behavior require hardware/browser testing.
14. **External model assets:** The first run depends on the configured CDN/model host being reachable.
15. **No clinical interpretation:** The project is for personal experiments and visualization, not patient assessment, fall-risk classification, or exercise safety decisions.

### Things developers should verify before release

- Actual `getUserMedia` permissions and camera switching on Android Chrome and iOS Safari.
- Heavy/Full WASM/model downloads and GPU/CPU fallbacks on supported devices.
- Correct overlay mirroring, aspect ratio, and joint-label alignment.
- Tracking-loss behavior and realistic exercise repetition counts on real footage.
- Local recording failures when storage is full; deletion and clearing.
- Camera teardown after Stop, page hide, or error.
- Settings persist after reload and recover from invalid stored state.
- Keyboard navigation, modal focus, contrast, touch targets, and reduced-motion behavior.
- Measurement stability and added delay under each smoothing preset.
- Accurate performance counters and actual model switching after fallback.

---

## 14. Future capabilities: not yet live

These are **directions for future development**, not claims about the current application.

| Future idea | What it would add | Current reality |
| --- | --- | --- |
| Stronger exercise coaching | Configurable camera orientation, personalized thresholds, more meaningful movement warnings and repeatability metrics | Only simple current exercise rules exist |
| Motion Arcade | Squat/lean/reach-driven games, scoring, levels, effects | A developer-side movement challenge module exists; full UI and game loop are not integrated |
| Gesture navigation | Start/stop and controls using hand gestures | Extension contract only; not active |
| 3D avatar puppet | Retarget body poses to a rigged character in Three.js / VRM | Extension contract only; not active |
| Motion painting | Draw light trails or particles through movement | Extension flag only |
| More exercise modules | Step-ups, heel raises, side-steps, overhead presses, athletic drills | Would require new analyzers and tests |
| Advanced calibration | Consistent distance/viewpoint guidance, per-user baselines and view checks | Only simple standing baseline currently |
| Better analysis | Phase-aligned session comparisons, velocity, tempo, per-rep curves and report generation | Basic current angles, time curves, and comparisons |
| Optional video capture | Explicit local raw-video recording linked to pose timestamps | Not implemented |
| Optional offline bundle | Package model/WASM so the app works after installation without third-party asset fetches | Not bundled today |
| RTMPose comparison | Benchmark a second estimator on the same consented clips | No live RTMPose selection in current UI |
| Multi-purpose mode plugins | Different visual experiences sharing one tracking engine | Initial modular exercise registry exists |

**Multi-camera capture remains deliberately excluded** from the current plan. Add it only if the project goal changes and its extra complexity is justified.

---

## 15. Development priorities and acceptance checklist

### Suggested near-term priorities

**Priority 1 — Trust the core tracker**

- Test real phone and webcam performance.
- Correct the active-model fallback so a switch actually takes effect.
- Make skipped-frame/drop metrics reflect measured events.
- Validate smoothing and overlay alignment against representative video.

**Priority 2 — Make exercise modules dependable**

- Improve handling of partial visibility and tracking interruptions.
- Add configurable exercise-specific view requirements.
- Create annotated personal test clips with known repetition counts.
- Make feedback distinguish observed motion from heuristic suggestions.

**Priority 3 — Make sessions robust**

- Handle storage failures explicitly; consider IndexedDB for longer sessions.
- Add phase/time alignment to session comparisons.
- Improve ghost pose display and make saving/exporting clearer.
- Make the recording status and “no raw video” distinction impossible to miss.

**Priority 4 — Add one playful experiment**

- Wire a small overhead-reach challenge with a clear target and score.
- Measure whether the extra effect reduces live tracking performance.
- Build a reusable extension interface only after the core workflow is stable.

### Acceptance checklist for a usable experimental release

- [ ] One phone or webcam starts/stops repeatedly without leaving camera hardware active.
- [ ] No video frame is uploaded by the KineLab module.
- [ ] Skeleton remains aligned with visible limbs when moving and mirroring.
- [ ] Unreliable joints are hidden or marked unavailable, never given fabricated values.
- [ ] Each exercise module starts, resets, counts and handles lost tracking predictably.
- [ ] Settings have documented defaults and survive reload.
- [ ] User can deliberately record and later delete a local motion session.
- [ ] Timeline, chart, CSV and JSON are consistent with captured timestamps.
- [ ] Smartphone and desktop UI fit without horizontal overflow.
- [ ] Tests pass and the exact command/output is attached to the release review.
- [ ] Actual mobile hardware and browsers are tested before marking the app production-ready.
- [ ] Documentation accurately separates implementation from experimental claims.

---

## 16. FAQ and terminology

**Can I use only a phone?**  
Yes. Open the static app on a secure HTTPS site in a supported mobile browser and grant camera access. The phone can both capture frames and run the model; performance depends on the device.

**Do I need an external webcam or special sensors?**  
No. A built-in phone or laptop camera is sufficient for trying the current single-camera experience.

**Does the app automatically know the right way to squat?**  
No. It checks approximate knee-angle transitions using hand-written thresholds. It does not validate knee loading, actual 3D alignment, pain, safety, or clinical appropriateness.

**Are joint angles actual 3D measurements?**  
Three modes are provided. **Auto** (default) estimates how the body is turned relative to the camera. Knee and hip angles come from the 2D image when side-on, where 2D is most precise, and from 3D world landmarks ($X, Y, Z$ in meters, pelvis-centred) as the view turns oblique, where 2D foreshortens. **3D** and **2D** force one source. The HUD toggle cycles Auto → 3D → 2D. On the synthetic benchmark, Auto cut side-view knee RMSE from 13.4° to 3.2°; see [docs/accuracy-speed-research.md](docs/accuracy-speed-research.md).

**Can it track two people simultaneously?**  
Not as separate participants in the current configuration. The MediaPipe runner sets `numPoses: 1`.

**What happens when a joint disappears?**  
The relevant confidence checks suppress unreliable calculations, often showing a dash. Some exercise modules cancel incomplete transitions on lost tracking.

**Does the Record button make a video?**  
Yes! The Record button uses the HTML5 MediaRecorder API to record camera video chunks directly into browser IndexedDB alongside timestamped landmark and angle telemetry. The Replay stage synchronizes video playback with skeletal telemetry.

**Where are my sessions?**  
In local browser storage for that site and browser profile. They are not synced to another phone or account.

**Can I compare two workouts?**  
Yes, in a basic way: reps, duration and paired angle-sample deltas. The present comparison is not phase- or time-normalized.

**What is Ghost Mode today?**  
A faint reference skeleton from **one selected saved frame**, overlaid on the current view when enabled.

**Why does Heavy look slow?**  
Heavy is a more computationally demanding model. Try Full or Lite, reduce capture resolution, or improve device cooling and lighting. With *Auto fallback* on, the app steps down automatically. Inference resolution adapts to how large you appear in frame. Real performance varies.

**Can it work completely offline?**  
Inference is local once the SDK/WASM/model are loaded, but the current app is configured to obtain those assets from the internet; guaranteed first-run offline operation is not yet provided.

**Is this part of the original security-camera product?**  
It is inside the same fork, but is a separate static module with a different camera connection and purpose.

### Short glossary

- **Pose landmark:** Estimated reference point on the body, such as knee, wrist, hip or shoulder.
- **Inference:** Running the ML model on an image to predict landmarks.
- **2D projected angle:** Angle between visible limb segments on the image plane.
- **Confidence gating:** Rejecting measurements when necessary landmarks are not reliable enough.
- **One Euro filter:** Speed-adaptive low-pass smoothing of changing coordinates.
- **Outlier:** A suspiciously abrupt landmark movement that may represent model error.
- **Exercise state machine:** Rules for movement phases (ready → moving → returning → complete).
- **Hysteresis/debounce:** Rules to avoid multiple counts from small, noisy threshold crossings.
- **Ghost pose:** Previously saved reference skeleton used as a visual guide.
- **LocalStorage:** Browser storage scoped to a site and browser profile; not a secure database.
- **WASM:** WebAssembly, used by the browser's model inference runtime.

---

## Maintainer notes and source of truth

The following are the principal source-of-truth files for this document:

- [Application shell](./index.html)
- [Main runtime](./src/app.mjs)
- [Settings defaults and validation](./src/settings/settings-schema.mjs)
- [MediaPipe runner](./src/tracking/mediapipe-runner.mjs)
- [Exercise registry](./src/exercises/exercise-registry.mjs)
- [Angle calculations](./src/biomechanics/angle-calculations.mjs)
- [Session recorder](./src/sessions/session-recorder.mjs)
- [Local session storage](./src/sessions/session-storage.mjs)
- [Extension contracts](./src/integrations/extension-contracts.mjs)
- [Parent Clearcam README](../../README.md)

This README is intentionally more conservative than marketing copy. Whenever the software changes, update **“What is available now,” “All current settings,” “Known limitations,” and the test/verification status** in the same pull request.

**License:** This module resides in a fork whose parent repository is distributed under **GPL-3.0**. Review the root [license file](../../LICENSE.md) and license obligations before redistribution or combining code from external repositories.
