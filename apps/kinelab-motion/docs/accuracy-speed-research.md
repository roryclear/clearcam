# KineLab accuracy & speed: research, changes, roadmap

_2026-11-18 · applies to `apps/kinelab-motion` after PR "accuracy & speed pass"._

## 1. Result

Knee angle vs ground truth on the synthetic squat benchmark (`node bench/accuracy-bench.mjs`; 5 seeds × 6 reps per view; RMSE / standing frame-to-frame jitter / rep-count errors):

| Camera view (body yaw) | Before | After |
|---|---|---|
| side-on 0° | 13.4° / 10.8° / 0 | **3.2° / 1.1° / 0** |
| 30° | 13.4° / 10.7° / 0 | **6.7° / 1.1° / 0** |
| 45° | 13.6° / 10.7° / 0 | **10.6° / 1.1° / 0** |
| 60° | 14.0° / 10.7° / 0 | **13.4° / 1.6° / 0** |
| frontal 90° | 11.7° / 6.7° / 0 | **10.5° / 1.2° / 0** |

Smoothing-lag effect on reported squat depth, measured separately (`medium` preset): peak-depth error 8.5° → 4.6°; `high` preset 15.4° → 4.8°.

**Caveat:** the numbers are synthetic. The noise model is taken from published MediaPipe error characteristics (§3), but this is not a validation against motion capture. `tests/accuracy-pipeline.test.mjs` runs the benchmark as a regression gate.

## 2. Changes

| # | Change | Why (evidence) | Files |
|---|---|---|---|
| 1 | Re-tuned 2D One Euro presets (beta 0.007 → 20–60) | Landmark speeds are ~0.1–1 normalized units/s. At beta 0.007 the cutoff never rose with motion, so the filter lagged ~150 ms and under-read depth by 6–15°. | `filters/pose-filter.mjs` |
| 2 | New temporal filter for 3D world landmarks, with a separate depth (z) cutoff | 3D angles came from raw world landmarks: 10.8° standing jitter. Monocular z is the noisiest axis [Dill 2024]. | `filters/world-pose-filter.mjs` |
| 3 | Best-visible-leg selection replaces averaging both legs | Side-on, the far knee has 25.1° RMSE vs 10.7° for the near knee [Dill 2024]. MediaPipe still reports the far leg as ~0.6–0.75 visible, so it passed the 0.55 gate and was averaged in. | `biomechanics/angle-calculations.mjs` (`selectBilateralAngle`), squat, sit-to-stand, telemetry |
| 4 | **Auto** angle mode (new default): body yaw is estimated from 3D shoulder/hip lines (EMA, 600 ms). Knee and hip angles use 2D when side-on and blend to 3D as the view turns oblique. | 2D image angles are precise but foreshorten off-plane; 3D avoids foreshortening but is noisy. Measured crossover is near 45°. A per-limb plane estimate was tried first and was too noisy. | `biomechanics/view-orientation.mjs`, `app.mjs` |
| 5 | Squat hip-height cue: hip-above-ankle span ÷ standing span. It confirms standing (≥0.97) and gates depth (≤0.92). | Vertical image position is unaffected by yaw and has no depth term. Correlated depth error held smoothed 3D knees below the 155° standing threshold for >0.5 s, which missed reps. | `exercises/squat-exercise.mjs` |
| 6 | Camera-view guidance ("Turn side-on…" / "Face the camera…") | Validity depends strongly on viewing angle [Dill 2024; JMIR 2026]. | `tracking/framing-assistant.mjs` |
| 7 | Adaptive inference resolution (new default): the person stays ≥320 px tall in the input | The landmark model crops the person to 256×256. Fixed 480×360 downscaling starves a distant subject; near subjects run at lower resolution. Full resolution skips the copy entirely. | `tracking/mediapipe-runner.mjs` |
| 8 | Lite model plus model ladder: auto-fallback heavy → full → lite | Gives low-end phones a usable tier; fallback previously stopped at Full. | runner, `adaptive-controller.mjs`, settings, UI |
| 9 | DOM telemetry throttled to ~12 Hz; pose math still runs every frame | Symmetry/telemetry DOM was rebuilt at inference rate (~35 Hz). | `app.mjs` |
| 10 | Settings v3 migration: the stored v2 default `3d` becomes `auto`; explicit choices are kept | v2 persisted `3d` for every user, so nobody would get the new default. | `settings-schema.mjs` |
| 11 | Privacy text corrected | MediaPipe Tasks keeps frames on-device but **sends usage metrics to Google** (MediaPipe README, June 2026). | `index.html` |

## 3. Evidence base

- **Yue, Gu, Yu 2026, JMIR Rehabil Assist Technol 13:e102399.** BlazePose Heavy, lateral iPhone video, 15 adults.
  - Static RMSE 15.5°, with angle-dependent bias: +11° at 30° flexion, −10.6° at 120°.
  - Squat RMSE 8.8° (ICC 0.77) and sit-to-stand 7.6° (ICC 0.63) once low-confidence frames are gated out (target-frame confidence ≥0.3). Without the gate, squat RMSE was 25.4°.
  - Single-leg stance ICC was only 0.14.
  - https://rehab.jmir.org/2026/1/e102399
- **Dill et al. 2024, Sensors 24:7772.** MediaPipe vs Vicon.
  - Monocular z suffers high noise.
  - Lateral view: near knee 10.7°, far knee 25.1° median RMSE. The frontal view is worse than the lateral view.
  - Stereo fusion of two cameras: 56 → 30 mm position RMSE.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC11644880/
- **MediaPipe Pose Landmarker docs:** world landmarks are in meters, hip-centred, with visibility and presence. https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- **MediaPipe Tasks web README** (privacy notice, last modified June 5, 2026): input stays on device; usage metrics go to Google. https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md
- **RTMPose / RTMW (OpenMMLab).** Higher 2D accuracy than BlazePose; RTMPose-s reaches 72.2 AP at 70+ FPS on a Snapdragon 865 (arXiv:2303.07399). Runs in the browser through ONNX Runtime Web. RTMW: https://arxiv.org/html/2407.08634v1

## 4. Roadmap (ranked by value per effort)

| Rank | Addition | Value | Effort | Notes |
|---|---|---|---|---|
| 1 | ✅ **Done.** Per-rep quality gate and confidence badge: reps whose extreme frame or more than 25% of frames fall below 0.7 visibility are flagged and excluded from averages. | High: gating moved squat RMSE from 25° to 9° [JMIR 2026] | S | `exercises/rep-analysis.mjs` |
| 2 | ✅ **Done.** Per-rep metrics: ROM, phase tempo (down/up, rise/sit, raise/lower), extreme angle and confidence, shown live, in the results card and in the reps CSV (one row per rep, with set number). Set-level average ROM ± CV and cadence are shown live and in the results card. All of it is recomputed from recorded frames, so older sessions get it too. Synthetic check: ROM 85° ±1% (truth 85°), tempo 0.9 s / 0.9 s (truth 0.93 s). | High for PT use | S–M | `exercises/rep-analysis.mjs` |
| 3 | **Clinical test modes:** 30-s chair stand (count + cycle times), 5×STS timer, TUG. | High: standard outcome measures; smartphone pose versions validated [Bertrand 2026 PMC12773808; Sher 2026 PMC13091807; Hsu 2026 PMC13326458] | M | Countdown + auto-stop; 30s-STS = sit-to-stand module + timer. |
| 4 | **Frontal-plane knee valgus (FPPA)** for single-leg squat / step-down in a front view. | High for ACL/PFP screening; 2D FPPA is a reliable clinical measure [Munro 2012 PMID 22104115; Nagahori 2026 PMC13214613] | M | Pure 2D hip–knee–ankle angle in the frontal plane; reuses view guidance with `recommendedView: 'front'`. |
| 5 | **Real-data validation harness:** import video + reference angles (goniometer / Kinovea CSV); report RMSE, bias and Bland-Altman per joint. | Turns synthetic claims into measured ones | M | Feed recorded sessions through the same pipeline offline. |
| 6 | **Session telemetry to IndexedDB** (videos are already there). | Removes the ~5 MB localStorage ceiling (a few minutes of recording) | S | Carried over from the previous audit. |
| 7 | **Angle-bias calibration:** optional correction curve once #5 has data. | Medium: published bias is ±10° and angle-dependent | S after #5 | Don't ship a curve fitted to someone else's camera. |
| 8 | **Sit-to-stand power / velocity** (hip vertical velocity × body mass). | Medium: proposed as a strength proxy [Hwang 2026 PMC12834695] | S | Hip-height cue from change #5 already gives normalized vertical displacement. |
| 9 | **Experimental RTMPose/RTMW backend** (ONNX Runtime Web, WebGPU) behind a flag. | Possibly better 2D keypoints and foot points; no world landmarks | L | Keep MediaPipe for 3D; benchmark both on #5 data before switching. |
| 10 | **Optional second phone (stereo).** | Large 3D gain (56 → 30 mm) [Dill 2024] | L | Needs sync + calibration UX. |
| 11 | ✅ **Partly done** (see §6): `requestVideoFrameCallback` scheduling, capture-clock E2E latency, predictive overlay, Auto model. MediaPipe has no WebGPU delegate for vision tasks and none is planned ([#5826](https://github.com/google-ai-edge/mediapipe/issues/5826)), so that part is dropped. Worker inference is still open; the iOS exclusion is outdated (MediaPipe uses OffscreenCanvas WebGL on Safari ≥17, [#5292](https://github.com/google-ai-edge/mediapipe/issues/5292)). | Medium | M | Measure on real phones first; headless CPU numbers here are not representative. |

## 5. Reproduce

```bash
node bench/accuracy-bench.mjs           # table of all candidate pipelines
node bench/accuracy-bench.mjs --json    # machine-readable
node --test tests/*.test.mjs            # includes the benchmark regression gate
```

## 6. Research round 2 (Nov 2026): resources for accuracy, speed and latency

Four parallel literature/tool searches (Firecrawl). Evidence numbers are from the cited sources; "unverified" items were not confirmed on a primary page.

### 6.1 Implemented in this change

| Change | Evidence | Files |
|---|---|---|
| Inference scheduled per camera frame with `requestVideoFrameCallback`; rAF polling fallback | Fires once per new frame; removes rAF polling offset (≈0–16 ms, estimate) and gives exact frame times ([web.dev](https://web.dev/articles/requestvideoframecallback-rvfc)) | `app.mjs` `onVideoFrame`/`processFrame`, `tracking/frame-clock.mjs` |
| Pipeline time = frame `captureTime` (fallback `presentationTime`, then callback time), strictly increasing | Naive emit-time stamping under-reports latency by 20–50 ms ([arXiv 2608.11566](https://arxiv.org/abs/2608.11566)) | `tracking/frame-clock.mjs` |
| E2E latency in the HUD: camera→landmarks and camera→overlay p50/p95; frames the page never saw (`presentedFrames` gaps) | Same; excludes exposure and scan-out (Android preview alone is 57–106 ms glass-to-glass, [inovex](https://www.inovex.de/de/blog/the-glass-to-glass-latency-on-android/)) | `tracking/performance-monitor.mjs`, `index.html` |
| Predictive overlay: chase the target extrapolated at its velocity, ≤40 ms past capture, offset ≤0.05; display only | The λ=32 chase trails by ~1/λ = 31 ms plus frame age and inference. Synthetic 0.5 Hz squat, 25 ms inference: overlay RMS error 22 → 5 (×10⁻³ image units); constant speed lag 66 → 10 ms. Prediction degrades past ~20–50 ms ([DAQRI](https://medium.com/@DAQRI/motion-to-photon-latency-in-mobile-ar-and-vr-99f82c480926)) | `filters/pose-interpolator.mjs`, setting `overlays.predictiveOverlay` |
| Model setting **Auto** (default, settings v4): Full on phones, Heavy on desktop; stored v≤3 `heavy` default migrates to Auto | Heavy ≈2× slower than Full (BlazePose MediaPipe runtime: Pixel 5 32/22 FPS lite/full; MacBook Pro 2019 92/81/38) ([tfjs-models](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/blazepose_mediapipe/README.md)) | `tracking/device-profile.mjs`, `settings/settings-schema.mjs` |
| `?input=direct` A/B switch: feed `<video>` straight to the GPU delegate | The video→canvas copy cost ~7 ms in a TF.js case ([tfjs#5948](https://github.com/tensorflow/tfjs/issues/5948)); unverified for MediaPipe, so measure with the E2E figure before changing the default | `app.mjs` |

Headless Chrome (software GL) check: capture clock is reported for getUserMedia frames; fallback path reports `callback`. Headless numbers (400–1400 ms) only show the plumbing works; measure on real phones.

### 6.2 Done in this round (items 1–4 of the previous ranking)

| Change | Measured / verified | Where |
|---|---|---|
| Offline zero-phase pipeline: Hampel (±3 MAD, 0.5 s) → linear gap fill ≤300 ms → 4th-order Butterworth `filtfilt` (odd-reflection padding), 4 Hz on angles from unfiltered landmarks, 6 Hz on live angles of older sessions | Synthetic squat RMSE / rep-peak error (yaw 0°, 45°, 90°): live view-blend 3.2/1.3°, 10.6/10.2°, 10.5/3.3° → offline 2.2/1.1°, 10.8/8.7°, 10.7/3.5°; side-view jitter 1.12 → 0.50°. Gain is mainly lag and jitter; peaks change little. The bench's repErr for offline rows (1–3 of 30 at 45°/60°) comes from running the live counter on smoothed angles, which the app never does: counts and statuses stay live, offline angles feed analysis only | `filters/zero-phase.mjs`, `sessions/offline-analysis.mjs`, `bench/accuracy-bench.mjs` |
| Setup prompts (level ±3°, 2–3 m, hip height) and gravity roll correction (beta, off) | Tilt from `DeviceMotionEvent.accelerationIncludingGravity` + screen angle. Roll sign for front/rear cameras is a model assumption, unverified on devices | `tracking/device-level.mjs`, `tracking/framing-assistant.mjs` |
| Clinical convention display (exports stay interior) and reliability flags (far side in side view, confidence < 0.7) | Unit tests | `biomechanics/angle-convention.mjs`, `biomechanics/angle-reliability.mjs` |
| Validation harness: Kinovea CSV / OpenSim `.mot` readers, cross-correlation lag, frame- and rep-level Bland–Altman (bias, LoA, CIs), ICC(A,1) + CI, SEM, MDC95 | ICC, CIs, LoA match pingouin 0.7 / scipy 1.18 to 1e-9; synthetic end-to-end recovers a 300 ms clock offset and a +3° bias. No real-data numbers yet | `validation/*.mjs`, `bench/validate.mjs`, `docs/validation.md` |

### 6.3 Next, ranked by value per effort

| # | Addition | Evidence | Effort |
|---|---|---|---|
| 1 | Run the harness on OpenCap LabValidation (squat, STS; mocap/OpenSim GT; OpenCap Monocular 4.8° MAE baseline) and REHAB24-6 (shoulder abduction, lunge; non-commercial), then an in-house Kinovea/goniometer set, n ≥ 30 | [SimTK](https://simtk.org/frs/?group_id=2385), [REHAB24-6](https://zenodo.org/records/13305826); thresholds [Koo & Li 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/), COSMIN | M (data access) |
| 2 | Real-phone speed matrix: BrowserStack camera video injection + Chrome fake camera in CI, logging the new E2E p50/p95 | [BrowserStack](https://www.browserstack.com/docs/automate/selenium/camera-injection) | M |
| 3 | Replace `KinematicBoneFilter` (fixes length, not direction) with 2D-ray + bone-ratio fusion | BlazePose angle error 12.3° → 10.6°, bone-length variance −94% ([arXiv 2512.06783](https://arxiv.org/abs/2512.06783)) | M |
| 4 | MotionAGFormer-XS 2D→3D lifter on BlazePose 2D, offline first | Most accurate method in an exercise study: 2D knee MAE 8.5° ([Rode 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12589393/)); 2.2M params, Apache-2.0 ([repo](https://github.com/TaatiTeam/MotionAGFormer)); ONNX export unverified | M |
| 5 | Worker inference (MediaStreamTrackProcessor on Chrome/Safari 18, ImageBitmap/VideoFrame on Firefox) | Frees the main thread; gain unproven | M–L |
| 6 | Angle-dependent linear bias correction, fitted per camera setup after #1 | MoveNet side view: RMSE 11.6 → 4.0° hip, 10.9 → 3.9° knee ([PMC13347614](https://pmc.ncbi.nlm.nih.gov/articles/PMC13347614/)) | S after #1 |

Later: RTMPose-m Halpe26 via ONNX Runtime Web behind a flag (small 2D gain, foot points; checkpoint licence [mmpose#3275](https://github.com/open-mmlab/mmpose/issues/3275) open); stereo + OpenCap marker enhancer (4.34° RMSE).

Skip: MediaPipe WebGPU delegate (not planned), WebNN (not shipped by default anywhere), int8 pose models (only float16 hosted), YOLO-pose (AGPL; worst lifter input), Sapiens / VideoPose3D / SmoothNet (non-commercial), ViTPose live (too heavy for phones), LiteRT.js / ORT WebGPU backends for speed (no proven browser pose speedup).

Open questions: `captureTime` coverage on Safari/Firefox; 60 fps front camera on iOS Safari; whether the 7 ms canvas-copy saving applies to MediaPipe.

