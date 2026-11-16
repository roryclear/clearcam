import {
  ANGLE_DEFINITIONS, SKELETON_CONNECTIONS,
  reliable, getJointAngles, trackingStatus,
  createSquatState, advanceSquat, dominantKneeAngle,
} from './pose-math.mjs';

// Pinned production release; no unpinned @latest import. The model is downloaded only
// after Start Camera is clicked. Camera images are never transmitted to these hosts.
const SDK_VERSION = '1.2.3';
const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${SDK_VERSION}`;
const MODEL_URLS = {
  heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
  full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
};
const FRAME_INTERVAL_MS = 95; // Limit main-thread work for phone responsiveness.

const el = id => document.getElementById(id);
const nodes = {
  start: el('start-button'), stop: el('stop-button'), flip: el('switch-button'),
  video: el('camera-video'), canvas: el('pose-canvas'), stage: el('video-stage'), empty: el('stage-empty'),
  modelChoice: el('model-choice'), modelName: el('model-name'), mode: el('analysis-mode'),
  showAngles: el('show-angles'), angleList: el('angle-list'), feedback: el('squat-feedback'),
  repCount: el('rep-count'), resetReps: el('reset-reps'), message: el('app-message'),
  fps: el('inference-fps'), hudState: el('hud-state'), captureLabel: el('capture-label'),
  trackingPill: el('tracking-pill'),
};
const ctx = nodes.canvas.getContext('2d');
const state = {
  stream: null, poseLandmarker: null, session: 0, active: false, starting: false,
  facingMode: 'environment', model: 'heavy', lastVideoTime: -1,
  lastInferenceAt: -Infinity, lastFpsAt: 0, framesSinceFps: 0,
  raf: 0, squat: createSquatState(),
};
const angleRows = new Map();

for (const def of ANGLE_DEFINITIONS) {
  const row = document.createElement('div');
  row.className = 'angle-row';
  const label = document.createElement('span');
  label.className = 'angle-label'; label.textContent = def.label;
  const value = document.createElement('span');
  value.className = 'angle-value unavailable'; value.textContent = '—';
  row.append(label, value);
  nodes.angleList.append(row);
  angleRows.set(def.key, value);
}

function announce(message, isError = false) {
  nodes.message.textContent = message;
  nodes.message.classList.toggle('is-error', isError);
}

function showTracking(status) {
  const text = status === 'good' ? 'Body visible · Tracking stable'
    : status === 'partial' ? 'Partial body · Angles may be missing'
      : status === 'lost' ? 'Move into frame · Full body needed'
        : 'Waiting for camera';
  nodes.trackingPill.lastChild.textContent = text;
  nodes.trackingPill.classList.toggle('is-good', status === 'good');
  nodes.trackingPill.classList.toggle('is-warning', status === 'partial' || status === 'lost');
  nodes.hudState.textContent = status === 'good' ? 'TRACKING' : status === 'partial' ? 'PARTIAL' : status === 'lost' ? 'NO POSE' : 'STANDBY';
}

function resetSquats() {
  state.squat = createSquatState();
  nodes.repCount.textContent = '00';
  nodes.feedback.textContent = nodes.mode.value === 'squat'
    ? 'Face sideways, keep hips, knees and ankles visible, then bend and return to standing.'
    : 'Select “Squat prototype” to activate approximate repetition tracking.';
}

function clearMetrics() {
  for (const value of angleRows.values()) { value.textContent = '—'; value.classList.add('unavailable'); }
  nodes.fps.textContent = '--';
  showTracking('waiting');
}

function renderMetrics(angles) {
  for (const def of ANGLE_DEFINITIONS) {
    const value = angleRows.get(def.key);
    const measured = angles[def.key];
    value.textContent = Number.isFinite(measured) ? `${Math.round(measured)}°` : '—';
    value.classList.toggle('unavailable', !Number.isFinite(measured));
  }
}

function drawPose(landmarks, angles) {
  const width = nodes.canvas.width;
  const height = nodes.canvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!Array.isArray(landmarks) || !landmarks.length) return;
  const mirrored = state.facingMode === 'user';
  const px = p => ({ x: (mirrored ? 1 - p.x : p.x) * width, y: p.y * height });
  const scale = Math.max(1, Math.min(width / 800, height / 500));
  ctx.save();
  ctx.lineWidth = 3 * scale;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.shadowColor = '#7af6de';
  ctx.shadowBlur = 9 * scale;
  for (const [a, b] of SKELETON_CONNECTIONS) {
    if (!reliable(landmarks[a]) || !reliable(landmarks[b])) continue;
    const p = px(landmarks[a]), q = px(landmarks[b]);
    ctx.strokeStyle = '#81fbe0';
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
  }
  ctx.shadowBlur = 0;
  for (const index of new Set(SKELETON_CONNECTIONS.flat())) {
    const point = landmarks[index];
    if (!reliable(point)) continue;
    const p = px(point);
    ctx.fillStyle = '#b0a5ff';
    ctx.strokeStyle = '#0f1c31';
    ctx.lineWidth = 2 * scale;
    ctx.beginPath(); ctx.arc(p.x, p.y, 5 * scale, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
  if (nodes.showAngles.checked) {
    const fontSize = Math.max(12, 15 * scale);
    ctx.font = `700 ${fontSize}px ui-monospace, SFMono-Regular, monospace`;
    for (const def of ANGLE_DEFINITIONS) {
      if (!Number.isFinite(angles[def.key]) || !reliable(landmarks[def.joint])) continue;
      const p = px(landmarks[def.joint]);
      const text = `${Math.round(angles[def.key])}°`;
      const labelWidth = ctx.measureText(text).width + 15 * scale;
      const x = Math.max(5, Math.min(width - labelWidth - 5, p.x + (def.key.startsWith('left') ? -labelWidth - 9 * scale : 9 * scale)));
      const y = Math.max(8, Math.min(height - fontSize - 12, p.y - fontSize));
      ctx.fillStyle = '#07151fe0';
      ctx.fillRect(x, y, labelWidth, fontSize + 10 * scale);
      ctx.strokeStyle = '#6ae6d9b0';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, labelWidth, fontSize + 10 * scale);
      ctx.fillStyle = '#a2ffe9';
      ctx.fillText(text, x + 7 * scale, y + fontSize + 1);
    }
  }
  ctx.restore();
}

function movementUpdate(angles, confidence, timestampMs) {
  if (nodes.mode.value !== 'squat') return;
  const kneeAngle = dominantKneeAngle(angles);
  state.squat = advanceSquat(state.squat, kneeAngle, timestampMs);
  nodes.repCount.textContent = String(state.squat.reps).padStart(2, '0');
  if (confidence === 'lost' || !Number.isFinite(kneeAngle)) {
    nodes.feedback.textContent = 'Knee landmarks unavailable. Move your whole body into the frame.';
    return;
  }
  const map = {
    ready: 'Ready. Bend your knees to start a repetition (side view recommended).',
    lowering: 'Knee bend detected. Continue moving while staying visible.',
    rise: 'Bend threshold reached. Return toward standing.',
    finishing: 'Straighten to complete the repetition.',
    completed: 'Repetition counted. Ready for the next one.',
  };
  nodes.feedback.textContent = map[state.squat.status] ?? 'Maintain full-body visibility.';
}

function frameLoop() {
  if (!state.active || !state.poseLandmarker) return;
  state.raf = window.requestAnimationFrame(frameLoop);
  const video = nodes.video;
  const now = performance.now();
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.currentTime === state.lastVideoTime
      || now - state.lastInferenceAt < FRAME_INTERVAL_MS) return;
  state.lastVideoTime = video.currentTime;
  state.lastInferenceAt = now;
  try {
    const result = state.poseLandmarker.detectForVideo(video, now);
    const landmarks = result.landmarks?.[0] ?? null;
    const quality = trackingStatus(landmarks);
    const angles = getJointAngles(landmarks, video.videoWidth, video.videoHeight);
    drawPose(landmarks, angles);
    renderMetrics(angles);
    showTracking(quality);
    movementUpdate(angles, quality, now);
    state.framesSinceFps++;
    if (!state.lastFpsAt) state.lastFpsAt = now;
    if (now - state.lastFpsAt >= 1000) {
      nodes.fps.textContent = String(Math.round(state.framesSinceFps * 1000 / (now - state.lastFpsAt)));
      state.lastFpsAt = now; state.framesSinceFps = 0;
    }
  } catch (error) {
    console.error('Pose inference failed', error);
    stopCamera();
    announce('Pose analysis stopped because of an inference error. Retry or select the Full model.', true);
  }
}

async function loadLandmarker(session) {
  announce('Loading Google MediaPipe. The first download can take a moment…');
  const { FilesetResolver, PoseLandmarker } = await import(`${CDN_BASE}/+esm`);
  const fileset = await FilesetResolver.forVisionTasks(`${CDN_BASE}/wasm`);
  if (session !== state.session) return null;
  const choices = state.model === 'heavy' ? ['heavy', 'full'] : ['full'];
  let finalError;
  for (const model of choices) {
    for (const delegate of ['GPU', 'CPU']) {
      if (session !== state.session) return null;
      try {
        const landmarkModel = await PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URLS[model], delegate },
          runningMode: 'VIDEO', numPoses: 1,
          minPoseDetectionConfidence: 0.55,
          minPosePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
          outputSegmentationMasks: false,
        });
        if (session !== state.session) { landmarkModel.close(); return null; }
        nodes.modelName.textContent = model.toUpperCase();
        announce(model === state.model
          ? `Live tracking ready · ${model.toUpperCase()} model · ${delegate}`
          : `Heavy model unavailable. Live tracking using Full · ${delegate}.`);
        return landmarkModel;
      } catch (error) {
        finalError = error;
        console.warn(`MediaPipe ${model}/${delegate} unavailable`, error);
      }
    }
  }
  throw finalError ?? new Error('No MediaPipe model could initialize.');
}

function setLive(isLive) {
  nodes.start.disabled = isLive || state.starting;
  nodes.stop.disabled = !isLive && !state.starting;
  nodes.flip.disabled = !isLive;
  nodes.modelChoice.disabled = isLive || state.starting;
  nodes.captureLabel.classList.toggle('is-live', isLive);
  nodes.captureLabel.lastElementChild.textContent = isLive ? 'LIVE' : 'OFFLINE';
  nodes.empty.hidden = isLive;
}

function stopCamera() {
  state.session++;
  state.active = false;
  state.starting = false;
  window.cancelAnimationFrame(state.raf);
  state.raf = 0;
  for (const track of state.stream?.getTracks() ?? []) track.stop();
  state.stream = null;
  nodes.video.pause();
  nodes.video.srcObject = null;
  if (state.poseLandmarker) { state.poseLandmarker.close(); state.poseLandmarker = null; }
  ctx.clearRect(0, 0, nodes.canvas.width, nodes.canvas.height);
  state.lastVideoTime = -1;
  state.lastInferenceAt = -Infinity;
  state.lastFpsAt = 0;
  state.framesSinceFps = 0;
  clearMetrics();
  setLive(false);
}

async function startCamera() {
  if (state.starting || state.active) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    announce('Camera access requires HTTPS or localhost and a browser supporting getUserMedia.', true);
    return;
  }
  state.starting = true;
  const currentSession = ++state.session;
  setLive(false);
  announce('Requesting camera permission…');
  try {
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: state.facingMode }, width: { ideal: 1280 },
        height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 },
      },
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    if (state.session !== currentSession) { stream.getTracks().forEach(track => track.stop()); return; }
    state.stream = stream;
    nodes.video.srcObject = stream;
    nodes.video.style.transform = state.facingMode === 'user' ? 'scaleX(-1)' : 'none';
    await nodes.video.play();
    if (state.session !== currentSession) return;
    nodes.stage.style.aspectRatio = `${nodes.video.videoWidth || 16} / ${nodes.video.videoHeight || 9}`;
    nodes.canvas.width = nodes.video.videoWidth || 1280;
    nodes.canvas.height = nodes.video.videoHeight || 720;
    state.model = nodes.modelChoice.value;
    state.poseLandmarker = await loadLandmarker(currentSession);
    if (state.session !== currentSession || !state.poseLandmarker) return;
    state.starting = false;
    state.active = true;
    setLive(true);
    resetSquats();
    frameLoop();
  } catch (error) {
    if (currentSession !== state.session) return;
    console.error('Camera could not start', error);
    stopCamera();
    const name = error?.name;
    announce(name === 'NotAllowedError' || name === 'PermissionDeniedError'
      ? 'Camera permission denied. Enable camera access in browser settings and retry.'
      : name === 'NotFoundError' ? 'No camera was found on this device.'
        : 'Unable to start. Check camera availability, network access for the model, then retry.', true);
  }
}

nodes.start.addEventListener('click', startCamera);
nodes.stop.addEventListener('click', () => { stopCamera(); announce('Camera stopped. No video was stored.'); });
nodes.flip.addEventListener('click', () => {
  state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
  stopCamera();
  startCamera();
});
nodes.resetReps.addEventListener('click', resetSquats);
nodes.mode.addEventListener('change', resetSquats);
nodes.showAngles.addEventListener('change', () => {
  announce(nodes.showAngles.checked ? 'Angle labels enabled.' : 'Angle labels hidden; joint telemetry remains active.');
});
window.addEventListener('pagehide', stopCamera);
window.addEventListener('beforeunload', stopCamera);
clearMetrics();
resetSquats();
setLive(false);
