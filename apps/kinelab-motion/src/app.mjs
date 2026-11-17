/**
 * KineLab Motion 2.0 — Motion Intelligence Studio
 * Main application orchestrator connecting tracking, filtering, biomechanics,
 * modular exercise recognition, real-time overlays, settings, and session analytics.
 */

import { ANGLE_DEFINITIONS, SKELETON_CONNECTIONS } from './pose-math.mjs';
import { getJointAngles, trackingStatus } from './biomechanics/angle-calculations.mjs';
import { analyzeSymmetry } from './biomechanics/symmetry-analyzer.mjs';
import { PoseFilter } from './filters/pose-filter.mjs';
import { PerformanceMonitor } from './tracking/performance-monitor.mjs';
import { AdaptiveController } from './tracking/adaptive-controller.mjs';
import { MediaPipeRunner } from './tracking/mediapipe-runner.mjs';
import { OverlayRenderer } from './overlays/overlay-renderer.mjs';
import { defaultExerciseRegistry } from './exercises/exercise-registry.mjs';
import { defaultSettingsManager } from './settings/settings-manager.mjs';
import { SessionRecorder } from './sessions/session-recorder.mjs';
import { defaultSessionStorage } from './sessions/session-storage.mjs';
import { TimelinePlayer } from './sessions/timeline-player.mjs';
import { ChartRenderer } from './sessions/chart-renderer.mjs';
import { exportToCSV, exportToJSON, downloadFile } from './sessions/export-helpers.mjs';
import { defaultAudioSynthesizer } from './integrations/audio-cues.mjs';

const el = id => document.getElementById(id);

// DOM Elements
const nodes = {
  // Stage & Video
  video: el('camera-video'),
  canvas: el('pose-canvas'),
  stage: el('video-stage'),
  empty: el('stage-empty'),
  startBtn: el('start-button'),
  stopBtn: el('stop-button'),
  flipBtn: el('switch-button'),
  recordBtn: el('record-button'),
  recIndicator: el('recording-indicator'),
  recTimer: el('rec-timer'),
  captureLabel: el('capture-label'),
  trackingPill: el('tracking-pill'),
  message: el('app-message'),

  // HUD
  hudState: el('hud-state'),
  inferenceFps: el('inference-fps'),
  modelName: el('model-name'),
  latencyDisplay: el('latency-display'),

  // Quick Controls
  analysisMode: el('analysis-mode'),
  modelChoice: el('model-choice'),
  showAngles: el('show-angles'),
  showArcs: el('show-arcs'),

  // Telemetry & Insight
  angleList: el('angle-list'),
  insightTitle: el('insight-title'),
  exerciseBadge: el('exercise-badge'),
  repCount: el('rep-count'),
  repUnitLabel: el('rep-unit-label'),
  dynamicTelemetry: el('dynamic-telemetry'),
  feedback: el('squat-feedback'),
  resetReps: el('reset-reps'),
  symmetryGrid: el('symmetry-grid'),
  overallSymBadge: el('overall-sym-badge'),

  // Settings Modal
  openSettingsBtn: el('open-settings'),
  closeSettingsBtn: el('close-settings'),
  settingsDialog: el('settings-dialog'),
  resetDefaultsBtn: el('reset-defaults-btn'),
  clearAllSessionsBtn: el('clear-all-sessions-btn'),
  storageEstimateLabel: el('storage-estimate-label'),

  // Settings Inputs
  cfgCameraFacing: el('cfg-camera-facing'),
  cfgCameraMirror: el('cfg-camera-mirror'),
  cfgCameraRes: el('cfg-camera-res'),
  cfgTrackModel: el('cfg-track-model'),
  cfgTrackAutoFallback: el('cfg-track-autofallback'),
  cfgTrackSmoothing: el('cfg-track-smoothing'),
  cfgTrackConfidence: el('cfg-track-confidence'),
  cfgOverlaySkeleton: el('cfg-overlay-skeleton'),
  cfgOverlayJoints: el('cfg-overlay-joints'),
  cfgOverlayLabels: el('cfg-overlay-labels'),
  cfgOverlayArcs: el('cfg-overlay-arcs'),
  cfgOverlayTrails: el('cfg-overlay-trails'),
  cfgOverlaySymHud: el('cfg-overlay-symhud'),
  cfgOverlayGhost: el('cfg-overlay-ghost'),
  cfgExerciseSensitivity: el('cfg-exercise-sensitivity'),
  cfgExerciseAudio: el('cfg-exercise-audio'),

  // Analytics & Replay Drawer
  openRecordingsBtn: el('open-recordings'),
  closeAnalyticsBtn: el('close-analytics'),
  analyticsDrawer: el('analytics-drawer'),
  sessionCountBadge: el('session-count-badge'),
  activeSessionTitle: el('active-session-title'),
  playPauseBtn: el('play-pause-btn'),
  stepBackBtn: el('step-back-btn'),
  stepFwdBtn: el('step-fwd-btn'),
  timelineTime: el('timeline-time'),
  timelineScrubber: el('timeline-scrubber'),
  chartCanvas: el('analytics-chart'),
  chartJointSelect: el('chart-joint'),
  savedSessionsList: el('saved-sessions-list'),
  exportJsonBtn: el('export-json-btn'),
  exportCsvBtn: el('export-csv-btn'),
  setGhostBtn: el('set-ghost-btn'),
  deleteSessionBtn: el('delete-session-btn'),
};

// Subsystem Instances
const settings = defaultSettingsManager;
const filter = new PoseFilter();
const perf = new PerformanceMonitor();
const adaptive = new AdaptiveController();
const runner = new MediaPipeRunner();
const overlay = new OverlayRenderer(nodes.canvas);
const recorder = new SessionRecorder();
const storage = defaultSessionStorage;
const player = new TimelinePlayer();
const chart = new ChartRenderer(nodes.chartCanvas);
const audio = defaultAudioSynthesizer;

// Application State
const appState = {
  stream: null,
  active: false,
  starting: false,
  sessionToken: 0,
  raf: 0,
  lastVideoTime: -1,
  lastInferenceAt: -Infinity,
  FRAME_INTERVAL_MS: 75, // ~13.3 FPS inference throttling for smooth mobile performance
  activeExerciseInstance: null,
  exerciseState: null,
  currentSessionViewing: null,
  recStartTime: 0,
  recTimerInterval: 0,
};

// UI Angle Row Cache
const angleRows = new Map();
for (const def of ANGLE_DEFINITIONS) {
  const row = document.createElement('div');
  row.className = 'angle-row';
  const label = document.createElement('span');
  label.className = 'angle-label';
  label.textContent = def.label;
  const value = document.createElement('span');
  value.className = 'angle-value unavailable';
  value.textContent = '—';
  row.append(label, value);
  nodes.angleList.append(row);
  angleRows.set(def.key, value);
}

function announce(message, isError = false) {
  nodes.message.textContent = message;
  nodes.message.classList.toggle('is-error', isError);
}

function updateTrackingHUD(status) {
  const text = status === 'good' ? 'Body visible · Tracking stable'
    : status === 'partial' ? 'Partial body · Angles may be missing'
    : status === 'lost' ? 'Move into frame · Full body needed'
    : 'Waiting for camera';

  nodes.trackingPill.lastChild.textContent = text;
  nodes.trackingPill.classList.toggle('is-good', status === 'good');
  nodes.trackingPill.classList.toggle('is-warning', status === 'partial' || status === 'lost');
  nodes.hudState.textContent = status === 'good' ? 'TRACKING' : status === 'partial' ? 'PARTIAL' : status === 'lost' ? 'NO POSE' : 'STANDBY';
}

function renderTelemetry(angles) {
  for (const def of ANGLE_DEFINITIONS) {
    const valueEl = angleRows.get(def.key);
    const measured = angles?.[def.key];
    valueEl.textContent = Number.isFinite(measured) ? `${Math.round(measured)}°` : '—';
    valueEl.classList.toggle('unavailable', !Number.isFinite(measured));
  }
}

function renderSymmetry(angles) {
  const cfg = settings.get();
  if (!cfg.overlays.showSymmetryHUD) return;

  const report = analyzeSymmetry(angles);
  nodes.symmetryGrid.innerHTML = '';

  for (const [id, data] of Object.entries(report.pairs)) {
    const row = document.createElement('div');
    row.className = 'sym-row';
    const label = document.createElement('span');
    label.className = 'sym-label';
    label.textContent = data.label;

    const valGroup = document.createElement('span');
    valGroup.className = 'sym-vals';

    const lText = Number.isFinite(data.left) ? `${Math.round(data.left)}°` : '—';
    const rText = Number.isFinite(data.right) ? `${Math.round(data.right)}°` : '—';
    valGroup.textContent = `L: ${lText} / R: ${rText}`;

    if (Number.isFinite(data.diffDeg)) {
      const diffSpan = document.createElement('span');
      diffSpan.className = `sym-diff ${data.isSymmetric ? '' : 'asym'}`;
      diffSpan.textContent = `Δ${data.diffDeg}°`;
      valGroup.appendChild(diffSpan);
    }

    row.append(label, valGroup);
    nodes.symmetryGrid.append(row);
  }

  if (report.overallAsymmetryDeg !== null) {
    const isGood = report.overallAsymmetryDeg <= 10.0;
    nodes.overallSymBadge.textContent = isGood ? 'BALANCED' : `ASYM ±${report.overallAsymmetryDeg}°`;
    nodes.overallSymBadge.classList.toggle('is-asym', !isGood);
  } else {
    nodes.overallSymBadge.textContent = 'NO DATA';
    nodes.overallSymBadge.classList.remove('is-asym');
  }
}

function renderDynamicTelemetry(items) {
  nodes.dynamicTelemetry.innerHTML = '';
  for (const item of items) {
    const div = document.createElement('div');
    div.className = 'dyn-item';
    const lbl = document.createElement('span');
    lbl.className = 'dyn-label';
    lbl.textContent = item.label;
    const val = document.createElement('span');
    val.className = 'dyn-val';
    val.textContent = item.value;
    div.append(lbl, val);
    nodes.dynamicTelemetry.append(div);
  }
}

function switchExercise(exerciseId) {
  const cfg = settings.get();
  cfg.exercise.activeExercise = exerciseId;
  settings.update({ exercise: { activeExercise: exerciseId } });

  if (exerciseId === 'free') {
    appState.activeExerciseInstance = null;
    appState.exerciseState = null;
    nodes.insightTitle.textContent = 'Free Tracking';
    nodes.exerciseBadge.textContent = 'EXPLORATION';
    nodes.repCount.textContent = '—';
    nodes.repUnitLabel.textContent = 'FREE MODE';
    nodes.feedback.textContent = 'Free tracking active. Joint angles and skeleton stream live.';
    nodes.dynamicTelemetry.innerHTML = '';
    return;
  }

  const ex = defaultExerciseRegistry.get(exerciseId);
  if (ex) {
    ex.setSensitivity(cfg.exercise.sensitivity);
    appState.activeExerciseInstance = ex;
    appState.exerciseState = ex.createState();
    nodes.insightTitle.textContent = `${ex.name} Monitor`;
    nodes.exerciseBadge.textContent = ex.type.toUpperCase();
    nodes.repCount.textContent = '00';
    nodes.repUnitLabel.textContent = ex.type === 'hold' ? 'SECONDS' : 'REPETITIONS';
    nodes.feedback.textContent = ex.getFeedback(appState.exerciseState);
    renderDynamicTelemetry(ex.getTelemetry(appState.exerciseState));
  }
}

function handleExerciseFrame(landmarks, angles, width, height, timestampMs, trackingQuality) {
  if (!appState.activeExerciseInstance) return;

  const prevState = appState.exerciseState;
  const nextState = appState.activeExerciseInstance.update(prevState, {
    landmarks,
    angles,
    width,
    height,
    timestampMs,
    trackingQuality,
  });
  appState.exerciseState = nextState;

  // Sound cues on rep completion or milestone
  if (settings.get().exercise.audioCues) {
    if (nextState.reps > (prevState?.reps ?? 0)) {
      audio.playRepCount();
    } else if (nextState.phase === 'holding' && Math.floor(nextState.currentHoldSec) > Math.floor(prevState?.currentHoldSec ?? 0)) {
      audio.playHoldTick();
    }
  }

  // Update Telemetry Display
  if (appState.activeExerciseInstance.type === 'hold') {
    nodes.repCount.textContent = String(Math.floor(nextState.currentHoldSec)).padStart(2, '0');
  } else {
    nodes.repCount.textContent = String(nextState.reps ?? 0).padStart(2, '0');
  }

  nodes.feedback.textContent = appState.activeExerciseInstance.getFeedback(nextState);
  renderDynamicTelemetry(appState.activeExerciseInstance.getTelemetry(nextState));
}

// Main RequestAnimationFrame Loop
function frameLoop() {
  if (!appState.active) return;
  appState.raf = window.requestAnimationFrame(frameLoop);
  perf.recordRenderFrame();

  const video = nodes.video;
  const now = performance.now();

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.currentTime === appState.lastVideoTime ||
      now - appState.lastInferenceAt < appState.FRAME_INTERVAL_MS) {
    return;
  }

  appState.lastVideoTime = video.currentTime;
  appState.lastInferenceAt = now;

  const tStart = perf.startInference();
  try {
    const rawResult = runner.detectForVideo(video, now);
    perf.endInference(tStart);

    const metrics = perf.getMetrics();
    nodes.inferenceFps.textContent = String(metrics.inferenceFps);
    nodes.latencyDisplay.textContent = `${Math.round(metrics.lastLatencyMs)}ms`;

    // Adaptive Performance Check
    const adaptEval = adaptive.evaluate(metrics);
    if (adaptEval.shouldFallback) {
      announce(`Performance notice: ${adaptEval.reason} Switching to Full model for responsiveness.`);
      settings.update({ tracking: { model: 'full' } });
      nodes.modelChoice.value = 'full';
    }

    const cfg = settings.get();
    // 1 Euro and Outlier Filtering
    filter.setVisibilityThreshold(cfg.tracking.confidenceThreshold);
    const smoothedLandmarks = filter.filter(rawResult.landmarks, now);

    const quality = trackingStatus(smoothedLandmarks, cfg.tracking.confidenceThreshold);
    const angles = getJointAngles(smoothedLandmarks, video.videoWidth, video.videoHeight, cfg.tracking.confidenceThreshold);

    // Render Master Overlays
    const isMirrored = cfg.camera.mirroring === 'on' ||
      (cfg.camera.mirroring === 'auto' && cfg.camera.facingMode === 'user');

    overlay.render({
      landmarks: smoothedLandmarks,
      angles,
      timestampMs: now,
    }, {
      showSkeleton: cfg.overlays.showSkeleton,
      showJointMarkers: cfg.overlays.showJointMarkers,
      showDegreeLabels: cfg.overlays.showDegreeLabels && nodes.showAngles.checked,
      showAngleArcs: cfg.overlays.showAngleArcs && nodes.showArcs.checked,
      showMotionTrails: cfg.overlays.showMotionTrails,
      showGhost: cfg.overlays.showGhost,
      mirrored: isMirrored,
      minVisibility: cfg.tracking.confidenceThreshold,
    });

    renderTelemetry(angles);
    renderSymmetry(angles);
    updateTrackingHUD(quality);
    handleExerciseFrame(smoothedLandmarks, angles, video.videoWidth, video.videoHeight, now, quality);

    // Record telemetry frame if recording is active
    if (recorder.isRecording) {
      recorder.recordFrame({
        landmarks: smoothedLandmarks,
        angles,
        timestampMs: now,
        phase: appState.exerciseState?.phase ?? 'tracking',
        status: appState.exerciseState?.status ?? 'active',
        reps: appState.exerciseState?.reps ?? 0,
      });
    }
  } catch (error) {
    console.error('Pose inference failed:', error);
    stopCamera();
    announce('Inference stopped due to an error. Select Full model and restart.', true);
  }
}

// Camera Lifecycle
function setLiveUI(isLive) {
  nodes.startBtn.disabled = isLive || appState.starting;
  nodes.stopBtn.disabled = !isLive && !appState.starting;
  nodes.flipBtn.disabled = !isLive;
  nodes.recordBtn.disabled = !isLive;
  nodes.modelChoice.disabled = isLive || appState.starting;
  nodes.captureLabel.classList.toggle('is-live', isLive);
  nodes.captureLabel.lastElementChild.textContent = isLive ? 'LIVE' : 'OFFLINE';
  nodes.empty.hidden = isLive;
}

function clearMetricsUI() {
  for (const value of angleRows.values()) {
    value.textContent = '—';
    value.classList.add('unavailable');
  }
  nodes.inferenceFps.textContent = '--';
  nodes.latencyDisplay.textContent = '--ms';
  updateTrackingHUD('waiting');
}

function stopCamera() {
  appState.sessionToken++;
  appState.active = false;
  appState.starting = false;
  window.cancelAnimationFrame(appState.raf);
  appState.raf = 0;

  if (recorder.isRecording) {
    stopRecording();
  }

  for (const track of appState.stream?.getTracks() ?? []) {
    track.stop();
  }
  appState.stream = null;

  nodes.video.pause();
  nodes.video.srcObject = null;
  runner.close();
  overlay.clear();
  filter.reset();
  perf.reset();
  adaptive.reset();

  clearMetricsUI();
  setLiveUI(false);
}

async function startCamera() {
  if (appState.starting || appState.active) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    announce('Camera access requires HTTPS or localhost and a supported browser.', true);
    return;
  }

  appState.starting = true;
  const currentToken = ++appState.sessionToken;
  setLiveUI(false);
  announce('Requesting camera permission…');

  const cfg = settings.get();
  const resolutionMap = {
    '480p': { width: { ideal: 640 }, height: { ideal: 480 } },
    '720p': { width: { ideal: 1280 }, height: { ideal: 720 } },
    '1080p': { width: { ideal: 1920 }, height: { ideal: 1080 } },
  };
  const res = resolutionMap[cfg.camera.resolution] || resolutionMap['720p'];

  try {
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: cfg.camera.facingMode },
        ...res,
        frameRate: { ideal: 30, max: 60 },
      },
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    if (appState.sessionToken !== currentToken) {
      stream.getTracks().forEach(t => t.stop());
      return;
    }

    appState.stream = stream;
    nodes.video.srcObject = stream;

    const isMirrored = cfg.camera.mirroring === 'on' ||
      (cfg.camera.mirroring === 'auto' && cfg.camera.facingMode === 'user');
    nodes.video.style.transform = isMirrored ? 'scaleX(-1)' : 'none';

    await nodes.video.play();
    if (appState.sessionToken !== currentToken) return;

    nodes.stage.style.aspectRatio = `${nodes.video.videoWidth || 16} / ${nodes.video.videoHeight || 9}`;
    nodes.canvas.width = nodes.video.videoWidth || 1280;
    nodes.canvas.height = nodes.video.videoHeight || 720;

    filter.setPreset(cfg.tracking.smoothing);
    filter.setVisibilityThreshold(cfg.tracking.confidenceThreshold);
    adaptive.setCurrentModel(cfg.tracking.model);

    await runner.initialize(cfg.tracking.model, statusStr => announce(statusStr));
    if (appState.sessionToken !== currentToken) return;

    nodes.modelName.textContent = runner.activeModel.toUpperCase();
    appState.starting = false;
    appState.active = true;
    setLiveUI(true);

    switchExercise(cfg.exercise.activeExercise);
    frameLoop();
  } catch (error) {
    if (currentToken !== appState.sessionToken) return;
    console.error('Camera startup failed:', error);
    stopCamera();
    const name = error?.name;
    announce(
      name === 'NotAllowedError' || name === 'PermissionDeniedError'
        ? 'Camera permission denied. Enable camera access in browser settings and retry.'
        : name === 'NotFoundError'
          ? 'No camera found on this device.'
          : 'Unable to start. Verify camera connection, network access for model, then retry.',
      true
    );
  }
}

// Recording & Sessions
function startRecording() {
  if (!appState.active) return;
  const cfg = settings.get();
  recorder.start({ exerciseId: cfg.exercise.activeExercise });
  nodes.recordBtn.classList.add('is-recording');
  nodes.recordBtn.querySelector('span:last-child').textContent = 'Stop REC';
  nodes.recIndicator.hidden = false;

  appState.recStartTime = performance.now();
  nodes.recTimer.textContent = '00:00';
  appState.recTimerInterval = setInterval(() => {
    const elapsedSec = Math.floor((performance.now() - appState.recStartTime) / 1000);
    const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
    const s = String(elapsedSec % 60).padStart(2, '0');
    nodes.recTimer.textContent = `${m}:${s}`;
  }, 500);

  announce('Session recording started. Frames and joint telemetry stored locally.');
}

function stopRecording() {
  clearInterval(appState.recTimerInterval);
  nodes.recIndicator.hidden = true;
  nodes.recordBtn.classList.remove('is-recording');
  nodes.recordBtn.querySelector('span:last-child').textContent = 'Record';

  const sessionData = recorder.stop();
  if (sessionData && sessionData.frameCount > 0) {
    storage.saveSession(sessionData);
    updateSessionsList();
    announce(`Session recorded: ${sessionData.frameCount} frames, ${sessionData.finalReps} reps saved.`);
  }
}

function updateSessionsList() {
  const index = storage.getIndex();
  nodes.sessionCountBadge.textContent = String(index.length);
  nodes.savedSessionsList.innerHTML = '';

  if (index.length === 0) {
    nodes.savedSessionsList.innerHTML = '<div style="color: #6d7d93; font-size: 11px; padding: 10px;">No saved sessions yet. Click "Record" during a movement to save a session.</div>';
    return;
  }

  for (const item of index) {
    const div = document.createElement('div');
    div.className = `session-item ${appState.currentSessionViewing?.id === item.id ? 'active' : ''}`;
    const info = document.createElement('div');
    info.className = 'session-info';

    const title = document.createElement('span');
    title.className = 'session-title';
    title.textContent = `${item.exerciseId.toUpperCase()} (${item.finalReps} reps)`;

    const meta = document.createElement('span');
    meta.className = 'session-meta';
    const durSec = (item.durationMs / 1000).toFixed(1);
    meta.textContent = `${durSec}s · ${item.frameCount} frames`;

    info.append(title, meta);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-ghost';
    btn.textContent = 'Load & Inspect';
    btn.addEventListener('click', () => loadSessionForPlayback(item.id));

    div.append(info, btn);
    nodes.savedSessionsList.append(div);
  }
}

function loadSessionForPlayback(sessionId) {
  const fullSession = storage.getSession(sessionId);
  if (!fullSession) return;

  appState.currentSessionViewing = fullSession;
  player.loadSession(fullSession);
  nodes.activeSessionTitle.textContent = `${fullSession.exerciseId.toUpperCase()} · ${fullSession.frameCount} frames`;
  nodes.timelineScrubber.max = String(Math.max(0, fullSession.frameCount - 1));
  nodes.timelineScrubber.value = '0';

  chart.render({
    frames: fullSession.frames,
    jointKey: nodes.chartJointSelect.value,
    currentFrameIdx: 0,
  });

  updateSessionsList();
  nodes.analyticsDrawer.hidden = false;
  nodes.analyticsDrawer.scrollIntoView({ behavior: 'smooth' });
}

// Timeline Player subscription
player.subscribe((frame, isPlaying) => {
  nodes.playPauseBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
  if (!frame) return;

  nodes.timelineScrubber.value = String(frame.index);
  const curSec = (frame.timeMs / 1000).toFixed(1);
  const totSec = (appState.currentSessionViewing?.durationMs / 1000 || 0).toFixed(1);
  nodes.timelineTime.textContent = `${curSec}s / ${totSec}s [f:${frame.index}]`;

  // Draw frame on canvas if live camera is not running
  if (!appState.active) {
    const cfg = settings.get();
    overlay.render({
      landmarks: frame.landmarks,
      angles: frame.angles,
      timestampMs: frame.timeMs,
    }, {
      showSkeleton: cfg.overlays.showSkeleton,
      showJointMarkers: cfg.overlays.showJointMarkers,
      showDegreeLabels: cfg.overlays.showDegreeLabels,
      showAngleArcs: cfg.overlays.showAngleArcs,
      showGhost: cfg.overlays.showGhost,
      mirrored: false,
    });
    renderTelemetry(frame.angles);
    renderSymmetry(frame.angles);
  }

  // Update chart scrubber line
  if (appState.currentSessionViewing) {
    chart.render({
      frames: appState.currentSessionViewing.frames,
      jointKey: nodes.chartJointSelect.value,
      currentFrameIdx: frame.index,
    });
  }
});

// UI Event Listeners
nodes.startBtn.addEventListener('click', startCamera);
nodes.stopBtn.addEventListener('click', () => {
  stopCamera();
  announce('Camera stopped. Frames and tracks cleared from memory.');
});
nodes.flipBtn.addEventListener('click', () => {
  const cfg = settings.get();
  const nextMode = cfg.camera.facingMode === 'environment' ? 'user' : 'environment';
  settings.update({ camera: { facingMode: nextMode } });
  stopCamera();
  startCamera();
});

nodes.recordBtn.addEventListener('click', () => {
  if (recorder.isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

nodes.resetReps.addEventListener('click', () => {
  if (appState.activeExerciseInstance) {
    appState.exerciseState = appState.activeExerciseInstance.createState();
    nodes.repCount.textContent = '00';
    nodes.feedback.textContent = appState.activeExerciseInstance.getFeedback(appState.exerciseState);
    renderDynamicTelemetry(appState.activeExerciseInstance.getTelemetry(appState.exerciseState));
    announce('Repetition counter reset.');
  }
});

nodes.analysisMode.addEventListener('change', () => {
  switchExercise(nodes.analysisMode.value);
});

nodes.modelChoice.addEventListener('change', () => {
  const chosen = nodes.modelChoice.value;
  settings.update({ tracking: { model: chosen } });
  if (appState.active) {
    stopCamera();
    startCamera();
  }
});

nodes.showAngles.addEventListener('change', () => {
  settings.update({ overlays: { showDegreeLabels: nodes.showAngles.checked } });
});

nodes.showArcs.addEventListener('change', () => {
  settings.update({ overlays: { showAngleArcs: nodes.showArcs.checked } });
});

// Analytics Drawer Listeners
nodes.openRecordingsBtn.addEventListener('click', () => {
  nodes.analyticsDrawer.hidden = !nodes.analyticsDrawer.hidden;
  if (!nodes.analyticsDrawer.hidden) {
    updateSessionsList();
    nodes.analyticsDrawer.scrollIntoView({ behavior: 'smooth' });
  }
});

nodes.closeAnalyticsBtn.addEventListener('click', () => {
  nodes.analyticsDrawer.hidden = true;
  player.pause();
});

nodes.playPauseBtn.addEventListener('click', () => player.toggle());
nodes.stepBackBtn.addEventListener('click', () => player.step(-1));
nodes.stepFwdBtn.addEventListener('click', () => player.step(1));
nodes.timelineScrubber.addEventListener('input', e => player.seek(Number(e.target.value)));

document.querySelectorAll('.speed-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    player.setSpeed(Number(btn.dataset.speed));
  });
});

nodes.chartJointSelect.addEventListener('change', () => {
  if (appState.currentSessionViewing) {
    const cur = player.getCurrentFrame()?.index ?? 0;
    chart.render({
      frames: appState.currentSessionViewing.frames,
      jointKey: nodes.chartJointSelect.value,
      currentFrameIdx: cur,
    });
  }
});

nodes.exportJsonBtn.addEventListener('click', () => {
  if (!appState.currentSessionViewing) return;
  const jsonStr = exportToJSON(appState.currentSessionViewing);
  downloadFile(jsonStr, `${appState.currentSessionViewing.id}.json`, 'application/json');
});

nodes.exportCsvBtn.addEventListener('click', () => {
  if (!appState.currentSessionViewing) return;
  const csvStr = exportToCSV(appState.currentSessionViewing);
  downloadFile(csvStr, `${appState.currentSessionViewing.id}.csv`, 'text/csv');
});

nodes.setGhostBtn.addEventListener('click', () => {
  const curFrame = player.getCurrentFrame();
  if (curFrame?.landmarks) {
    overlay.setGhostPose(curFrame.landmarks);
    settings.update({ overlays: { showGhost: true } });
    nodes.cfgOverlayGhost.checked = true;
    announce('Selected frame set as ghost reference skeleton overlay.');
  }
});

nodes.deleteSessionBtn.addEventListener('click', () => {
  if (!appState.currentSessionViewing) return;
  const id = appState.currentSessionViewing.id;
  storage.deleteSession(id);
  appState.currentSessionViewing = null;
  player.stop();
  updateSessionsList();
  nodes.activeSessionTitle.textContent = 'Session deleted.';
  chart.render({ frames: [] });
  announce('Session deleted from local storage.');
});

// Settings Modal Listeners
nodes.openSettingsBtn.addEventListener('click', () => {
  populateSettingsForm();
  nodes.settingsDialog.showModal();
});

nodes.closeSettingsBtn.addEventListener('click', () => {
  nodes.settingsDialog.close();
});

document.querySelectorAll('.tab-btn').forEach(tabBtn => {
  tabBtn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    tabBtn.classList.add('active');
    tabBtn.setAttribute('aria-selected', 'true');
    const targetContent = el(tabBtn.dataset.tab);
    if (targetContent) targetContent.classList.add('active');
  });
});

function populateSettingsForm() {
  const cfg = settings.get();
  nodes.cfgCameraFacing.value = cfg.camera.facingMode;
  nodes.cfgCameraMirror.value = cfg.camera.mirroring;
  nodes.cfgCameraRes.value = cfg.camera.resolution;

  nodes.cfgTrackModel.value = cfg.tracking.model;
  nodes.cfgTrackAutoFallback.checked = cfg.tracking.autoFallback;
  nodes.cfgTrackSmoothing.value = cfg.tracking.smoothing;
  nodes.cfgTrackConfidence.value = String(cfg.tracking.confidenceThreshold);

  nodes.cfgOverlaySkeleton.checked = cfg.overlays.showSkeleton;
  nodes.cfgOverlayJoints.checked = cfg.overlays.showJointMarkers;
  nodes.cfgOverlayLabels.checked = cfg.overlays.showDegreeLabels;
  nodes.cfgOverlayArcs.checked = cfg.overlays.showAngleArcs;
  nodes.cfgOverlayTrails.checked = cfg.overlays.showMotionTrails;
  nodes.cfgOverlaySymHud.checked = cfg.overlays.showSymmetryHUD;
  nodes.cfgOverlayGhost.checked = cfg.overlays.showGhost;

  nodes.cfgExerciseSensitivity.value = cfg.exercise.sensitivity;
  nodes.cfgExerciseAudio.checked = cfg.exercise.audioCues;

  const usage = storage.getStorageEstimate();
  const kb = Math.round(usage.bytes / 1024);
  nodes.storageEstimateLabel.textContent = `${usage.count} sessions stored (~${kb} KB locally)`;
}

// Bind Settings Form Changes
function bindSettingsInputs() {
  const save = () => {
    settings.update({
      camera: {
        facingMode: nodes.cfgCameraFacing.value,
        mirroring: nodes.cfgCameraMirror.value,
        resolution: nodes.cfgCameraRes.value,
      },
      tracking: {
        model: nodes.cfgTrackModel.value,
        autoFallback: nodes.cfgTrackAutoFallback.checked,
        smoothing: nodes.cfgTrackSmoothing.value,
        confidenceThreshold: Number(nodes.cfgTrackConfidence.value),
      },
      overlays: {
        showSkeleton: nodes.cfgOverlaySkeleton.checked,
        showJointMarkers: nodes.cfgOverlayJoints.checked,
        showDegreeLabels: nodes.cfgOverlayLabels.checked,
        showAngleArcs: nodes.cfgOverlayArcs.checked,
        showMotionTrails: nodes.cfgOverlayTrails.checked,
        showSymmetryHUD: nodes.cfgOverlaySymHud.checked,
        showGhost: nodes.cfgOverlayGhost.checked,
      },
      exercise: {
        sensitivity: nodes.cfgExerciseSensitivity.value,
        audioCues: nodes.cfgExerciseAudio.checked,
      },
    });

    // Update Quick Controls
    nodes.showAngles.checked = nodes.cfgOverlayLabels.checked;
    nodes.showArcs.checked = nodes.cfgOverlayArcs.checked;
    nodes.modelChoice.value = nodes.cfgTrackModel.value;

    audio.setEnabled(nodes.cfgExerciseAudio.checked);
  };

  const inputs = [
    nodes.cfgCameraFacing, nodes.cfgCameraMirror, nodes.cfgCameraRes,
    nodes.cfgTrackModel, nodes.cfgTrackAutoFallback, nodes.cfgTrackSmoothing,
    nodes.cfgTrackConfidence, nodes.cfgOverlaySkeleton, nodes.cfgOverlayJoints,
    nodes.cfgOverlayLabels, nodes.cfgOverlayArcs, nodes.cfgOverlayTrails,
    nodes.cfgOverlaySymHud, nodes.cfgOverlayGhost,
    nodes.cfgExerciseSensitivity, nodes.cfgExerciseAudio,
  ];

  for (const input of inputs) {
    input.addEventListener('change', save);
  }
}

nodes.resetDefaultsBtn.addEventListener('click', () => {
  settings.reset();
  populateSettingsForm();
  announce('Settings reset to defaults.');
});

nodes.clearAllSessionsBtn.addEventListener('click', () => {
  storage.clearAll();
  updateSessionsList();
  populateSettingsForm();
  announce('All local sessions cleared.');
});

window.addEventListener('pagehide', stopCamera);
window.addEventListener('beforeunload', stopCamera);

// Initialize application on load
bindSettingsInputs();
clearMetricsUI();
setLiveUI(false);
updateSessionsList();
switchExercise(settings.get().exercise.activeExercise);
