# External Validation — Datasets and Measured Results

This document records what was actually verified about external datasets and
what this project has actually measured. It claims no clinical-grade precision.

## Datasets evaluated

| Dataset | Verified facts | License | Suitability for single-phone-camera evaluation |
|---|---|---|---|
| UI-PRMD (Univ. of Idaho) | 10 rehabilitation exercises × 10 healthy subjects × 10 reps; captured with Vicon motion capture + Kinect skeleton tracking. Source: https://webpages.uidaho.edu/ui-prmd/ | Open Data Commons Public Domain Dedication and License (PDDL) v1.0 — free public use | Useful as a *trajectory-shape* reference (joint-angle patterns over a rep). Its ground truth comes from calibrated mocap/Kinect, which does **not** transfer directly to uncalibrated monocular phone video; use for rep-segmentation and shape comparison, not absolute angle accuracy claims. |
| RepCount (SVIP Lab) | 1,451 in-the-wild exercise videos, ~19,280 frame-level repetition start/end annotations. Source: https://svip-lab.github.io/dataset/RepCount_dataset.html | License listed as "Unknown" (Papers With Code) at time of writing — treat as research-evaluation only; do not redistribute videos or annotations until terms are confirmed with the authors | Good fit in principle: ordinary monocular videos like a phone camera. Blocked on license confirmation before any bundled use. |
| Public pose benchmarks (e.g. COCO keypoints, MPII) | General 2D keypoint benchmarks, not exercise-repetition corpora | Varies per dataset | Measure detector keypoint quality, not repetition counting or movement classification; not used as exercise ground truth here. |

## What this project has actually measured

- `tests/synthetic-benchmarks.test.mjs` generates UI-PRMD-*style* synthetic
  joint-angle trajectories (not dataset recordings) and runs them through the
  real exercise state machines: 10/10 synthetic squat reps counted, 5/5
  sit-to-stand reps, tracking-dropout mid-rep discards the partial rep, and
  jittered input produces no false reps. These are deterministic logic tests
  of counting behaviour, **not** accuracy measurements against real people.
- Phase-sequence segmentation (added on `feat/kinelab-phase2-validation`):
  squat visits `ready > down > ready` per rep and is bit-identical across two
  runs (2/2 reps); sit-to-stand visits `seated > rising > standing > sitting`
  in order. Note the squat machine has no separate bottom/up phases — depth
  is tracked inside `down`; an earlier draft of this test wrongly assumed
  `bottom`/`up` phases and failed, which is how this was pinned down.
- Full suite at time of writing: 44 tests, all passing
  (`node --test apps/kinelab-motion/tests/*.test.mjs`, exit 0).

## Explicitly not claimed

- No accuracy figures against UI-PRMD, RepCount, or any recorded human data:
  neither dataset has been downloaded or run through this pipeline yet.
- No clinical, medical, or rehabilitation efficacy claim. Angles are
  camera-relative 2D projections from a single uncalibrated camera and vary
  with camera placement; they are movement feedback, not measurement-grade
  biomechanics.
- Motion-capture ground truth (Vicon) does not transfer directly to
  monocular video; any future error report must state the recording modality
  beside every number.

## Validation harness (developer CLI)

`bench/validate.mjs` compares an exported session (JSON) with a reference recording:

```bash
node apps/kinelab-motion/bench/validate.mjs \
  --session squat-trial1.json --ref squat-trial1_ik.mot \
  --joint knee_angle_r:rightKnee --joint knee_angle_l:leftKnee \
  --exercise squat            # optional; defaults to the session's exercise
  # --source live             # compare recorded live angles instead of offline re-analysis
  # --lag-ms 0                # fixed clock offset instead of estimation
```

- Readers: Kinovea "Export to spreadsheet" CSV/TSV (`;` + decimal comma, `,`, tab; time in ms, s or timecode) and OpenSim `.mot`/`.sto` (degrees or radians).
- Joint spec `refColumn:appKey[:transform]`: `interior` (default for Kinovea angle tool), `flexion` = 180 − x (default for OpenSim `knee_angle_*`, `elbow_flex_*`, `hip_flexion_*`), `flexion-abs`. Other OpenSim columns (e.g. shoulder elevation, which already matches the app's 0° = arm down) need an explicit transform. OpenSim `hip_flexion` is thigh vs pelvis while the app hip is trunk–thigh: a hip comparison includes pelvic tilt and lumbar flexion.
- Clocks are aligned by maximising cross-correlation over ±2 s (10 ms grid).
- Output (JSON): per trial and joint, frame-level agreement (descriptive only: frames are autocorrelated) and rep-level agreement of each rep's extreme angle; rep pairs are pooled across trials. A rep is scored only when both series cover ≥90% of its window with recorded app samples never more than 80 ms apart (one dropped 30 fps frame is tolerated: it is part of what the app reports), recorded reference samples never more than 50 ms apart, and both series starting before and ending after the rep; others are counted in `excluded`. When no lag within ±2 s gives enough overlap the trial reports `alignment: "failed"` and is not pooled (pass `--lag-ms` to force one). Statistics: bias, SD, 95% LoA with CIs (bias: t·SD/√n; limits: t·√(3SD²/n)), RMSE, MAE, ICC(A,1) with McGraw–Wong CI, Koo & Li label on the CI lower bound, SEM = SD·√(1 − ICC), MDC95 = 1.96·√2·SEM.
- Checked against pingouin 0.7.0 / scipy 1.18.1 to 1e-9 (`tests/validation.test.mjs`). Pooling several reps per subject violates independence; for subject-level claims report one value per subject or use a repeated-measures model.

Candidate reference data (not yet downloaded or run): OpenCap LabValidation (SimTK; squat, sit-to-stand; 5 video views with marker mocap and OpenSim IK; requires a SimTK login), REHAB24-6 (Zenodo; shoulder abduction, lunge; non-commercial licence), and an in-house set recorded with Kinovea or a goniometer.

## Next validation steps (in order)

0. Run `bench/validate.mjs` on OpenCap LabValidation squat/sit-to-stand trials (side and 45° views) and publish the rep-level table here with view, model and n.
1. Confirm RepCount usage terms with SVIP Lab; if cleared, sample a small
   set of squat/sit-to-stand videos, run MediaPipe + this engine offline,
   and report rep-count error (counted vs annotated) per video.
2. Compare UI-PRMD Kinect skeleton angle trajectories against this engine's
   state machines for phase-segmentation agreement only.
3. Publish the resulting error table here with modality, sample size, and
   failure cases — including videos where counting failed.

A sampled reference can miss a true extreme by up to ½·|θ''|·(g/2)² for sample spacing g: ≤ 0.6° at g = 50 ms when |θ''| ≤ 1800°/s² (a 90° movement repeated every second). This is an assumption about the movement, not checked by the tool.
