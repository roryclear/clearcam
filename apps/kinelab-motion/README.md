# KineLab Motion — single-camera live tracking (experimental)

Standalone, **single-camera** browser prototype inside the Clearcam fork. It does not change the original Clearcam streaming, cameras, or NVR code. This milestone intentionally excludes multi-camera capture, calibrated 3D, recording, uploads, and clinical interpretation.

## Features

- One phone or webcam using `getUserMedia`, front/back camera switch when supported.
- Google MediaPipe Pose Landmarker (Heavy by default, GPU/CPU fallback, then Full if needed).
- 33 body landmarks, real-time skeleton connections, and live knee/hip/elbow/shoulder angles (left and right).
- Confidence-gated joints: display `—` instead of a made-up number when a landmark is lost.
- Free Tracking and an **approximate** squat-repetition prototype based on 2D knee angles.
- Responsive dashboard; no recording, server-side inference, upload, accounts, or database.

## Run locally

From the repository root:

```bash
python3 -m http.server 4173 --directory apps/kinelab-motion
```

Open <http://localhost:4173> in Chrome, Edge, or Safari. Click **Start camera**, grant permission, keep your full body visible, and choose **Free tracking** or **Squat prototype**. The first use needs internet access to fetch the pinned MediaPipe SDK/WASM and the model from Google. The app runs pose inference in the browser after loading.

> Opening a page on a phone at `http://YOUR_PC_IP:4173` generally **won't permit camera access**: mobile devices need an HTTPS origin. Serve this static folder behind an HTTPS-enabled local tunnel or trusted static HTTPS host. Don't expose the Clearcam Python NVR's port 8080 to the public internet.

## Tests

```bash
node --test apps/kinelab-motion/tests/*.test.mjs
```

Node 20+ is recommended for the built-in test runner. No npm install is required for this isolated web prototype.

## Limitations

- Shown angles are **2D projected image-plane angles**, *not* calibrated 3D kinematics. They depend on camera orientation, perspective, joint visibility, and where the user stands. They are not clinical measurements.
- The squat tracker applies simple fixed knee-angle thresholds with hysteresis and a minimum phase duration. It does **not** determine correct or safe form and is not suitable as an exercise prescription.
- Web MediaPipe `detectForVideo()` is synchronous. We throttle inference to approximately 10 FPS to reduce UI blocking; displayed FPS is inference FPS, not camera capture FPS. On slower devices use the Full model.
- Default mode uses the rear camera when available. Camera capture occurs only after user permission, and tracks are stopped on Stop / page exit.
- Google MediaPipe SDK and model load from remote CDNs on first use. Frames are analyzed locally in the browser. Consult MediaPipe's privacy notice for SDK metrics details.
- iOS support and device-level performance must be tested on real hardware; browser camera support and model delegate performance vary.

## Future modular additions

Exercise-specific analyzers can build on `src/pose-math.mjs` and the live pose result loop. Add new modes as separate tested pure functions; do not bake diagnosis or claims of "proper form" into generic landmark detection. Support recording or remote streaming **only after an explicit consent, retention, and access-control design**.
