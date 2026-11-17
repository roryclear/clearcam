# KineLab Motion 2.0 — Modular Motion Intelligence Studio

High-performance, modular, **single-camera** motion analysis and interactive movement studio built into the Clearcam project.

Maintains complete privacy: all inference, smoothing, exercise counting, and session recording occurs 100% locally in the browser sandbox. No video frames, landmarks, or biomechanical data are ever transmitted to external servers. Multiple-camera capture and clinical medical diagnosis are explicitly out of scope.

---

## 1. Key Capabilities

### A. Motion Tracking Engine
- **MediaPipe Pose Landmarker**: Pinned official release `@mediapipe/tasks-vision@1.2.3`, running Heavy model by default with automatic GPU/CPU and Full model fallbacks.
- **One Euro Temporal Filter**: CHI 2012 speed-based adaptive low-pass filter eliminates resting joint jitter without introducing perceptible latency during explosive movements.
- **Outlier Rejection**: Rejects unphysical landmark teleportation and single-frame detection glitches.
- **Missing Landmark Handling**: Graceful coasting and re-initialization when limbs are occluded or leave the frame. Unavailable joints show `—` instead of fabricated angles.
- **Adaptive Performance Fallback**: Automatically monitors frame rate and inference latency; recommends or switches to Full model if latency exceeds 110ms.
- **Performance HUD**: Real-time inference FPS, render FPS, latency (ms), and model status.

### B. Modular Exercise Recognition Engine
Modular state-machine architecture with configurable thresholds, hysteresis, and debounce logic:
1. **Squat**: Preserves exact baseline behavior, tracking knee flexion, inflection depth, and return to standing extension.
2. **Sit-to-Stand**: Evaluates 30-second chair-stand functional mobility transitions (seated $\to$ rising $\to$ standing $\to$ seated).
3. **Lunge**: Detects lead-leg vs rear-leg depth and upright recovery.
4. **Shoulder Flexion / Abduction**: Evaluates lateral and forward arm elevation range of motion and bilateral symmetry.
5. **Single-Leg Balance**: Timed continuous single-leg hold duration counter with stance leg detection and touchdown tracking.
6. **Free Tracking**: Unconstrained exploration with live 33-landmark skeleton and joint telemetry.

*Safety Rule: Tracking loss immediately halts phase progression and discards incomplete repetitions.*

### C. Real-Time Visual Intelligence
- Neon bone connections with confidence-based transparency.
- High-contrast joint markers.
- Live angle degree badges with pixel aspect ratio preservation.
- Circular joint angle sweep arcs showing movement depth.
- Optional motion trails on extremities (wrists, ankles).
- Bilateral symmetry HUD comparing left vs right limb angles and asymmetry delta ($\Delta^\circ$).
- Ghost reference skeleton overlay from recorded sessions.
- Full respect for `prefers-reduced-motion`.

### D. Recording, Replay, and Motion Analytics
- **Local Recording**: Opt-in recording of compact landmark frames, timestamps, movement phases, and joint angles.
- **Timeline Replay**: Scrub bar, frame stepping, Play/Pause, and playback speeds (0.5x, 1x, 2x).
- **Interactive Chart**: Pure HTML5 Canvas joint-angle graph over time with scrubber indicator line.
- **Ghost Comparison**: Save any playback frame as a ghost overlay for movement comparison.
- **Data Export**: One-click export to CSV and JSON.
- **Storage Management**: Local browser storage management with size estimation and clear controls.

### E. Intelligent Settings Panel
- **Camera**: Facing mode (Rear/Front), mirroring mode, resolution preference (720p, 1080p, 480p).
- **Tracking**: Heavy/Full model, auto-fallback, 1€ smoothing preset (Off, Low, Medium, High), confidence threshold.
- **Overlays**: Independent toggles for skeleton, joints, badges, arcs, trails, symmetry HUD, and ghost.
- **Exercise**: Sensitivity adjustment (Forgiving, Standard, Strict), procedural Web Audio API cues.
- **Privacy**: Local processing status, local session data controls, reset to factory defaults.

---

## 2. Architecture & Module Map

```
apps/kinelab-motion/
├── index.html                 # Semantic dark cinematic studio shell
├── styles.css                 # Responsive CSS grid/flexbox with neon accents
├── README.md                  # System documentation & developer guide
├── src/
│   ├── app.mjs                # Application orchestrator
│   ├── pose-math.mjs          # Backwards-compatible facade re-exporting core math
│   ├── tracking/
│   │   ├── mediapipe-runner.mjs     # Model lifecycle, CDN loader, delegate fallback
│   │   ├── performance-monitor.mjs  # Inference/render FPS and latency tracking
│   │   ├── adaptive-controller.mjs  # Performance watchdog and auto-fallback
│   │   └── worker-bridge.mjs        # Web Worker feasibility check & architecture probe
│   ├── filters/
│   │   ├── one-euro-filter.mjs      # LowPassFilter and OneEuroPoint2D
│   │   ├── outlier-filter.mjs       # Speed and jump gating
│   │   └── pose-filter.mjs          # 33-landmark composite filter & presets
│   ├── biomechanics/
│   │   ├── landmark-indices.mjs     # MediaPipe topology & constants
│   │   ├── angle-calculations.mjs   # 2D geometry, angleAt, uncertainty metrics
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
│   │   ├── direction-indicators.mjs # Velocity arrows and movement direction
│   │   ├── motion-trails.mjs        # Extremity velocity trails
│   │   └── ghost-renderer.mjs       # Reference skeleton overlay
│   ├── sessions/
│   │   ├── session-recorder.mjs     # Compact telemetry serialization
│   │   ├── session-storage.mjs      # LocalStorage index & persistence
│   │   ├── timeline-player.mjs      # Replay loop, scrubbing, speed control
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
    ├── pose-math.test.mjs           # Preserved baseline regression tests
    ├── filters.test.mjs             # 1€ filter, outlier, and pose smoothing tests
    ├── biomechanics.test.mjs        # 2D projection, trunk angle, symmetry tests
    ├── exercises.test.mjs           # State transitions, debounce, tracking-loss tests
    ├── settings.test.mjs            # Validation, defaults, corruption recovery
    ├── sessions.test.mjs            # Recording, playback, CSV/JSON serialization
    ├── synthetic-benchmarks.test.mjs# UI-PRMD / RepCount style kinematics benchmarks
    ├── gap-closure.test.mjs         # Shoulder flexion, direction, and baseline tests
    ├── integrations.test.mjs        # Extension isolation and audio synthesis tests
    └── movement-challenge.test.mjs  # Interactive challenge gameplay logic tests
```

---

## 3. Run Locally

From the repository root:

```bash
python3 -m http.server 4173 --directory apps/kinelab-motion
```

Open <http://localhost:4173> in Google Chrome, Microsoft Edge, or Safari.
Click **Start camera**, grant browser camera permission, and position your body inside the frame.

> **Mobile Note**: Accessing the server from a phone over LAN (`http://<IP>:4173`) requires an HTTPS origin or localhost port forwarding via Chrome DevTools (`chrome://inspect`) because browsers restrict `navigator.mediaDevices.getUserMedia` to secure contexts.

---

## 4. Test Suite

Run the complete test suite:

```bash
node --test apps/kinelab-motion/tests/*.test.mjs
```

All 47 automated tests run natively on Node.js 20+ without external npm dependencies.

---

## 5. Privacy & Security

- **Zero Remote Video Transmission**: Video frames never leave the device.
- **Zero Cloud Databases**: Saved sessions remain in the browser's `localStorage` until cleared.
- **No Clearcam NVR Exposure**: Does not expose or modify Clearcam's backend RTSP or NVR endpoints.
- **Client-Side Only**: Third-party assets are pinned by version and hash from trusted CDNs.
