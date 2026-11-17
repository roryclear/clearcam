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

## Next validation steps (in order)

1. Confirm RepCount usage terms with SVIP Lab; if cleared, sample a small
   set of squat/sit-to-stand videos, run MediaPipe + this engine offline,
   and report rep-count error (counted vs annotated) per video.
2. Compare UI-PRMD Kinect skeleton angle trajectories against this engine's
   state machines for phase-segmentation agreement only.
3. Publish the resulting error table here with modality, sample size, and
   failure cases — including videos where counting failed.
