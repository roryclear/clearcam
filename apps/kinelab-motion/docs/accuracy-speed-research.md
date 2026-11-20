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
| 2 | ✅ **Done.** Per-rep metrics: ROM, phase tempo (down/up, rise/sit, raise/lower), ROM variability (CV), cadence. Shown live, in the results card and as a reps CSV export; recomputed from recorded frames, so older sessions get them too. Synthetic check: ROM 85° ±1% (truth 85°), tempo 0.9 s / 0.9 s (truth 0.93 s). | High for PT use | S–M | `exercises/rep-analysis.mjs` |
| 3 | **Clinical test modes:** 30-s chair stand (count + cycle times), 5×STS timer, TUG. | High: standard outcome measures; smartphone pose versions validated [Bertrand 2026 PMC12773808; Sher 2026 PMC13091807; Hsu 2026 PMC13326458] | M | Countdown + auto-stop; 30s-STS = sit-to-stand module + timer. |
| 4 | **Frontal-plane knee valgus (FPPA)** for single-leg squat / step-down in a front view. | High for ACL/PFP screening; 2D FPPA is a reliable clinical measure [Munro 2012 PMID 22104115; Nagahori 2026 PMC13214613] | M | Pure 2D hip–knee–ankle angle in the frontal plane; reuses view guidance with `recommendedView: 'front'`. |
| 5 | **Real-data validation harness:** import video + reference angles (goniometer / Kinovea CSV); report RMSE, bias and Bland-Altman per joint. | Turns synthetic claims into measured ones | M | Feed recorded sessions through the same pipeline offline. |
| 6 | **Session telemetry to IndexedDB** (videos are already there). | Removes the ~5 MB localStorage ceiling (a few minutes of recording) | S | Carried over from the previous audit. |
| 7 | **Angle-bias calibration:** optional correction curve once #5 has data. | Medium: published bias is ±10° and angle-dependent | S after #5 | Don't ship a curve fitted to someone else's camera. |
| 8 | **Sit-to-stand power / velocity** (hip vertical velocity × body mass). | Medium: proposed as a strength proxy [Hwang 2026 PMC12834695] | S | Hip-height cue from change #5 already gives normalized vertical displacement. |
| 9 | **Experimental RTMPose/RTMW backend** (ONNX Runtime Web, WebGPU) behind a flag. | Possibly better 2D keypoints and foot points; no world landmarks | L | Keep MediaPipe for 3D; benchmark both on #5 data before switching. |
| 10 | **Optional second phone (stereo).** | Large 3D gain (56 → 30 mm) [Dill 2024] | L | Needs sync + calibration UX. |
| 11 | **Speed:** `requestVideoFrameCallback` scheduling; WebGPU delegate when it ships for Tasks; Worker inference on non-iOS. | Medium | M | Measure on real phones first; headless CPU numbers here are not representative. |

## 5. Reproduce

```bash
node bench/accuracy-bench.mjs           # table of all candidate pipelines
node bench/accuracy-bench.mjs --json    # machine-readable
node --test tests/*.test.mjs            # includes the benchmark regression gate
```
