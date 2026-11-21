/**
 * KineLab Motion 2.0 — Motion Intelligence Studio
 * Main application orchestrator connecting tracking, filtering, biomechanics,
 * modular exercise recognition, real-time overlays, settings, and session analytics.
 */

import { ANGLE_DEFINITIONS, SKELETON_CONNECTIONS } from './pose-math.mjs';
import {
  getJointAngles,
  getJointAngles3D,
  trackingStatus,
  calculateTrunkAngle,
  calculateTrunkAngle3D,
  calculateShankAngle,
  calculateShankAngle3D,
  evaluateSquatBiomechanics,
  selectBilateralAngle,
  bilateralSide,
} from './biomechanics/angle-calculations.mjs';
import { ViewOrientationEstimator, blendAnglesByView } from './biomechanics/view-orientation.mjs';
import { WorldPoseFilter } from './filters/world-pose-filter.mjs';
import { defaultFramingAssistant } from './tracking/framing-assistant.mjs';
import { analyzeSymmetry } from './biomechanics/symmetry-analyzer.mjs';
import { PoseFilter } from './filters/pose-filter.mjs';
import { KinematicBoneFilter } from './filters/kinematic-filter.mjs';
import { PoseInterpolator } from './filters/pose-interpolator.mjs';
import { PerformanceMonitor } from './tracking/performance-monitor.mjs';
import { supportsVideoFrameCallback, selectFrameTime, missedFrames, FrameSampler } from './tracking/frame-clock.mjs';
import { isLikelyMobile, resolveModelChoice, modelPreferenceAfterFallback } from './tracking/device-profile.mjs';
import { DeviceLevel, correctImageRoll, correctWorldRoll } from './tracking/device-level.mjs';
import { setAngleConvention, getAngleConvention, formatAngle, angleLabel } from './biomechanics/angle-convention.mjs';
import { angleReliability } from './biomechanics/angle-reliability.mjs';
import { AdaptiveController } from './tracking/adaptive-controller.mjs';
import { MediaPipeRunner } from './tracking/mediapipe-runner.mjs';
import { OverlayRenderer } from './overlays/overlay-renderer.mjs';
import { defaultExerciseRegistry } from './exercises/exercise-registry.mjs';
import { defaultSettingsManager } from './settings/settings-manager.mjs';
import { SessionRecorder } from './sessions/session-recorder.mjs';
import { defaultSessionStorage } from './sessions/session-storage.mjs';
import { TimelinePlayer } from './sessions/timeline-player.mjs';
import { ChartRenderer } from './sessions/chart-renderer.mjs';
import { exportToCSV, exportToJSON, exportRepsToCSV, downloadFile } from './sessions/export-helpers.mjs';
import { RepTracker, analyzeReps, summarizeSession, profileFor, countedReps } from './exercises/rep-analysis.mjs';
import { offlineSessionFrames, stripRawAngles } from './sessions/offline-analysis.mjs';
import { compareSessions } from './sessions/session-compare.mjs';
import { defaultAudioSynthesizer } from './integrations/audio-cues.mjs';
import { defaultVideoStore } from './sessions/video-store.mjs';

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
  hudAngleMode: el('hud-angle-mode'),
  inferenceFps: el('inference-fps'),
  modelName: el('model-name'),
  latencyDisplay: el('latency-display'),
  renderFps: el('render-fps'),
  droppedFrames: el('dropped-frames'),
  e2eLatency: el('e2e-latency'),

  // Quick Controls
  analysisMode: el('analysis-mode'),
  modelChoice: el('model-choice'),
  showAngles: el('show-angles'),
  showArcs: el('show-arcs'),

  // Telemetry & Insight
  angleDimToggle: el('angle-dim-toggle'),
  dimAuto: el('dim-auto'),
  dim3d: el('dim-3d'),
  dim2d: el('dim-2d'),
  telemetryNote: el('telemetry-note'),
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
  cfgTrackAngleDim: el('cfg-track-angledim'),
  cfgTrackLevel: el('cfg-track-level'),
  cfgTrackTiltFix: el('cfg-track-tiltfix'),
  cfgOverlaySkeleton: el('cfg-overlay-skeleton'),
  cfgOverlayJoints: el('cfg-overlay-joints'),
  cfgOverlayLabels: el('cfg-overlay-labels'),
  cfgOverlayArcs: el('cfg-overlay-arcs'),
  cfgOverlayTrails: el('cfg-overlay-trails'),
  cfgOverlayPredict: el('cfg-overlay-predict'),
  cfgOverlayConvention: el('cfg-overlay-convention'),
  cfgOverlayDirection: el('cfg-overlay-direction'),
  cfgOverlaySymHud: el('cfg-overlay-symhud'),
  cfgOverlayGhost: el('cfg-overlay-ghost'),
  cfgExerciseSensitivity: el('cfg-exercise-sensitivity'),
  cfgExerciseAudio: el('cfg-exercise-audio'),
  cfgExerciseVoice: el('cfg-exercise-voice'),

  // Segment Metrics
  valTrunkLean: el('val-trunk-lean'),
  valShankTilt: el('val-shank-tilt'),
  valSquatAlign: el('val-squat-align'),

  // Analytics & Replay Drawer
  openRecordingsBtn: el('open-recordings'),
  closeAnalyticsBtn: el('close-analytics'),
  analyticsDrawer: el('analytics-drawer'),
  sessionCountBadge: el('session-count-badge'),
  activeSessionTitle: el('active-session-title'),
  sessionSelectTop: el('session-select-top'),
  workoutResultsCard: el('workout-results-card'),
  resExerciseTag: el('res-exercise-tag'),
  resTimestamp: el('res-timestamp'),
  resRepCount: el('res-rep-count'),
  resDuration: el('res-duration'),
  resPeakAngle: el('res-peak-angle'),
  resSymmetry: el('res-symmetry'),
  replayVideo: el('replay-video'),
  replayCanvas: el('replay-canvas'),
  replayNoVideo: el('replay-no-video'),
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
  exportRepsBtn: el('export-reps-btn'),
  repLog: el('rep-log'),
  repLogLast: el('rep-log-last'),
  repLogSummary: el('rep-log-summary'),
  resAvgRom: el('res-avg-rom'),
  resTempo: el('res-tempo'),
  resTempoLabel: el('res-tempo-label'),
  resQuality: el('res-quality'),
  setGhostBtn: el('set-ghost-btn'),
  deleteSessionBtn: el('delete-session-btn'),
};

// Subsystem Instances
const settings = defaultSettingsManager;
const filter = new PoseFilter();
const worldFilter = new WorldPoseFilter();
const kinematicFilter = new KinematicBoneFilter();
// Separate bone-length state for the unfiltered angles recorded for offline re-analysis.
const rawKinematicFilter = new KinematicBoneFilter();
const viewEstimator = new ViewOrientationEstimator();
const repTracker = new RepTracker();
const interpolator = new PoseInterpolator({ lambda: 32.0 });
const perf = new PerformanceMonitor();
const adaptive = new AdaptiveController();
const runner = new MediaPipeRunner();
const deviceLevel = new DeviceLevel();
// A/B switch for the inference input: ?input=direct feeds the <video> straight to the GPU
// delegate (no 2D-canvas downscale); compare the E2E figure in the HUD against the default.
{
  const inputPreset = new URLSearchParams(globalThis.location?.search ?? '').get('input');
  if (inputPreset) runner.setInferencePreset(inputPreset);
}
const overlay = new OverlayRenderer(nodes.canvas);
const replayOverlay = new OverlayRenderer(nodes.replayCanvas);
const recorder = new SessionRecorder();
const storage = defaultSessionStorage;
const videoStore = defaultVideoStore;
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
  FRAME_INTERVAL_MS: 28, // High-throughput ~35 FPS real-time motion tracking
  lastAngles: null,
  lastFraming: null,
  lastTilt: null, // { rollDeg, pitchDeg } from the gravity sensor, or null
  lastWorldLandmarks: null,
  lastSmoothedLandmarks: null,
  activeExerciseInstance: null,
  exerciseState: null,
  currentSessionViewing: null,
  recStartTime: 0,
  recTimerInterval: 0,
  stoppingRecording: null,
  fallbackSessionToken: null, // camera session whose faster-model fallback is loading
  lastUiAt: -Infinity, // DOM telemetry is throttled; pose math still runs every frame
  lastView: null,
  useVideoFrameCallback: false, // inference runs per camera frame via requestVideoFrameCallback
  vfcHandle: 0,
  lastPresentedFrames: null,
  frameSampler: null, // FrameSampler: inference cap on camera cadence (set in startCamera)
  lastFrameTime: -Infinity, // MediaPipe VIDEO timestamps must strictly increase, across sessions too
  pendingOverlay: null, // { timeMs, clock } of the newest landmarks not yet drawn
};

/** Overlay extrapolation cap; prediction is reliable only over short horizons. */
const OVERLAY_PREDICTION_MS = 40;

const KNEE_EXERCISES = new Set(['squat', 'sit_to_stand', 'lunge']);
const SHOULDER_EXERCISES = new Set(['shoulder_abduction', 'shoulder_flexion']);
const normalizeExerciseId = id => String(id || '').replace(/-/g, '_');
const UI_INTERVAL_MS = 80; // ~12 Hz is plenty for numbers; avoids rebuilding DOM at inference rate

/** Vertical extent of the reliably detected body, in normalized image units. */
function subjectExtent(landmarks, minVisibility) {
  if (!Array.isArray(landmarks)) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of landmarks) {
    if (!p || !Number.isFinite(p.y) || (p.visibility ?? 1) < minVisibility) continue;
    if (p.y < lo) lo = p.y;
    if (p.y > hi) hi = p.y;
  }
  return hi > lo ? hi - lo : null;
}

// UI Angle Row Cache
const angleRows = new Map();
const angleRowLabels = new Map();
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
  angleRowLabels.set(def.key, label);
}

/**
 * Applies the angle display convention: row labels and the open results card's peak are
 * reformatted now. A paused replay re-emits its current frame (without seeking, so the
 * video position is kept), so the replay canvas and telemetry redraw together from the
 * selected recording; live values and a running replay follow on their next frame.
 */
function applyAngleConvention(convention) {
  setAngleConvention(convention);
  for (const def of ANGLE_DEFINITIONS) angleRowLabels.get(def.key).textContent = angleLabel(def);
  const peak = appState.resultsPeak;
  if (peak && nodes.resPeakAngle) nodes.resPeakAngle.textContent = peak.deg !== null ? formatAngle(peak.joint, peak.deg) : '—';
  if (player.session && !player.isPlaying) player.notify();
}

/** Recorded [x, y, z, conf] landmarks → objects the reliability checks read. */
function recordedLandmarks(landmarks) {
  if (!Array.isArray(landmarks)) return null;
  return landmarks.map(p => (Array.isArray(p) ? { x: p[0], y: p[1], z: p[2], visibility: p[3] ?? 1 } : p));
}

/** Replay telemetry: recorded confidence feeds the reliability flags; no live view estimate. */
function renderReplayTelemetry(frame) {
  renderTelemetry(frame.angles, null, null, true, 640, 480, {
    reliabilityLandmarks: recordedLandmarks(frame.landmarks),
    view: null,
  });
  renderSymmetry(frame.angles);
}
applyAngleConvention(settings.get().overlays.angleConvention);
settings.subscribe(cfg => {
  if (cfg?.overlays?.angleConvention && cfg.overlays.angleConvention !== getAngleConvention()) {
    applyAngleConvention(cfg.overlays.angleConvention);
  }
});

function announce(message, isError = false) {
  nodes.message.textContent = message;
  nodes.message.classList.toggle('is-error', isError);
}

function updateTrackingHUD(status, framing = null) {
  let text = status === 'good' ? 'Body visible · Tracking stable'
    : status === 'partial' ? 'Partial body · Angles may be missing'
    : status === 'lost' ? 'Move into frame · Full body needed'
    : 'Waiting for camera';

  let hudText = status === 'good' ? 'TRACKING' : status === 'partial' ? 'PARTIAL' : status === 'lost' ? 'NO POSE' : 'STANDBY';

  if (framing && framing.status && framing.status !== 'optimal') {
    text = framing.guidance;
    if (status === 'good') {
      hudText = 'ADJUST';
    }
  }

  nodes.trackingPill.lastChild.textContent = text;
  nodes.trackingPill.classList.toggle('is-good', status === 'good' && (!framing || framing.status === 'optimal'));
  nodes.trackingPill.classList.toggle('is-warning', status === 'partial' || status === 'lost' || Boolean(framing && framing.status !== 'optimal'));
  nodes.hudState.textContent = hudText;
}

const ANGLE_MODES = ['auto', '3d', '2d'];

function updateAngleDimUI(dimension = 'auto', view = null) {
  const mode = ANGLE_MODES.includes(dimension) ? dimension : 'auto';
  if (nodes.hudAngleMode) {
    nodes.hudAngleMode.textContent = mode === 'auto' ? 'AUTO VIEW' : mode === '3d' ? '3D METRIC' : '2D ANGLES';
  }
  nodes.dimAuto?.classList.toggle('active', mode === 'auto');
  nodes.dim3d?.classList.toggle('active', mode === '3d');
  nodes.dim2d?.classList.toggle('active', mode === '2d');
  if (nodes.telemetryNote) {
    if (mode === 'auto') {
      const yaw = view?.yawDeg;
      const source = !Number.isFinite(yaw) ? '' : view.sagittal2DWeight >= 0.5
        ? ` · body ~${Math.round(yaw)}° from side-on → knee/hip from precise 2D image`
        : ` · body ~${Math.round(yaw)}° from side-on → knee/hip from 3D world`;
      nodes.telemetryNote.textContent = `Auto: 2D image angles in side views, 3D world angles when turned oblique${source}.`;
    } else {
      nodes.telemetryNote.textContent = mode === '3d'
        ? '3D world angles from pelvis-centred coordinates. No foreshortening, but monocular depth is noisy.'
        : 'Projected 2D angles on the camera plane. Most precise side-on; foreshortened when oblique.';
    }
  }
  if (nodes.cfgTrackAngleDim) {
    nodes.cfgTrackAngleDim.value = mode;
  }
}

function renderTelemetry(angles, smoothedLandmarks = null, worldLandmarks = null, use3DSegments = true, width = 640, height = 480, {
  reliabilityLandmarks = smoothedLandmarks,
  view = appState.lastView,
} = {}) {
  const reliability = angleReliability(angles, reliabilityLandmarks, view);
  for (const def of ANGLE_DEFINITIONS) {
    const valueEl = angleRows.get(def.key);
    const measured = angles?.[def.key];
    valueEl.textContent = formatAngle(def.key, measured);
    valueEl.classList.toggle('unavailable', !Number.isFinite(measured));
    const rel = reliability[def.key];
    valueEl.parentElement.classList.toggle('is-uncertain', Boolean(rel?.low));
    valueEl.parentElement.title = rel?.reason ?? '';
  }

  // Segment angles (Trunk & Shank) and Squat Alignment Evaluation. Phone roll adds 1:1 to
  // angles measured against the image vertical; optionally undo it (beta).
  const cfg = settings.get();
  const roll = cfg.tracking.tiltCorrection ? appState.lastTilt?.rollDeg : null;
  if (Number.isFinite(roll)) {
    smoothedLandmarks = correctImageRoll(smoothedLandmarks, roll, width, height, cfg.camera.facingMode);
    worldLandmarks = correctWorldRoll(worldLandmarks, roll, cfg.camera.facingMode);
  }
  const trunk = use3DSegments && worldLandmarks
    ? calculateTrunkAngle3D(worldLandmarks)
    : calculateTrunkAngle(smoothedLandmarks, width, height);
  // Measure the shank on the more visible side; the occluded leg is unreliable side-on.
  const shankSide = bilateralSide(angles, smoothedLandmarks, 'Knee') === 'right' ? 'right' : 'left';
  const shank = use3DSegments && worldLandmarks
    ? calculateShankAngle3D(worldLandmarks, shankSide)
    : calculateShankAngle(smoothedLandmarks, width, height, shankSide);

  if (nodes.valTrunkLean) {
    nodes.valTrunkLean.textContent = Number.isFinite(trunk) ? `${Math.round(trunk)}°` : '—';
  }
  if (nodes.valShankTilt) {
    nodes.valShankTilt.textContent = Number.isFinite(shank) ? `${Math.round(shank)}°` : '—';
  }
  if (nodes.valSquatAlign) {
    const knee = selectBilateralAngle(angles, smoothedLandmarks, 'Knee');
    const evalResult = evaluateSquatBiomechanics({ kneeAngle: knee, trunkAngle: trunk, shankAngle: shank });
    nodes.valSquatAlign.textContent = evalResult.alignmentScore;
    nodes.valSquatAlign.classList.toggle('warn', evalResult.alignmentScore === 'TORSO LEAN');
  }
  const baselineEl = document.getElementById('baseline-deviation');
  if (baselineEl) {
    const baseline = settings.get().exercise.baseline;
    const knee = [angles?.leftKnee, angles?.rightKnee].filter(Number.isFinite);
    const hip = [angles?.leftHip, angles?.rightHip].filter(Number.isFinite);
    if (baseline && knee.length && hip.length) {
      // Flexion runs opposite to the interior angle, so its deltas flip sign.
      const k = getAngleConvention() === 'clinical' ? -1 : 1;
      const dKnee = Math.round(k * (knee.reduce((s, v) => s + v, 0) / knee.length - baseline.standingKneeDeg) * 10) / 10;
      const dHip = Math.round(k * (hip.reduce((s, v) => s + v, 0) / hip.length - baseline.standingHipDeg) * 10) / 10;
      baselineEl.textContent = `Baseline Δ knee ${dKnee > 0 ? '+' : ''}${dKnee}° · hip ${dHip > 0 ? '+' : ''}${dHip}°`;
    } else {
      baselineEl.textContent = baseline ? 'Baseline Δ — (angles unavailable)' : '';
    }
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

    const lText = formatAngle(id, data.left);
    const rText = formatAngle(id, data.right);
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

const fmtSec = ms => (Number.isFinite(ms) ? `${(ms / 1000).toFixed(1)}s` : '--');

function resetRepLog(exerciseId) {
  repTracker.reset(exerciseId);
  if (nodes.repLog) {
    nodes.repLog.hidden = true;
    nodes.repLogLast.textContent = '';
    nodes.repLogSummary.textContent = '';
  }
}

/** Live per-rep line: ROM, tempo and confidence of the rep that just finished. */
function renderRepLog(rep) {
  const profile = repTracker.profile;
  if (!nodes.repLog || !profile || !rep) return;
  const [p1, p2] = profile.phases;
  nodes.repLogLast.replaceChildren();
  nodes.repLogLast.append(`Rep ${rep.index} · ROM ${Math.round(rep.romDeg)}° · ${p1} ${fmtSec(rep.phase1Ms)} / ${p2} ${fmtSec(rep.phase2Ms)} `);
  const flag = document.createElement('span');
  flag.className = rep.quality === 'ok' ? 'rep-ok' : 'rep-flag';
  flag.textContent = rep.quality === 'ok' ? '✓' : '⚠ low confidence';
  nodes.repLogLast.append(flag);
  const s = repTracker.summary();
  nodes.repLogSummary.textContent = s.reliable
    ? `Avg ROM ${Math.round(s.avgRomDeg)}°${s.romCvPct !== null ? ` (±${Math.round(s.romCvPct)}%)` : ''} · ${p1} ${fmtSec(s.avgPhase1Ms)} / ${p2} ${fmtSec(s.avgPhase2Ms)}${s.repsPerMin !== null ? ` · ${Math.round(s.repsPerMin)}/min` : ''} · ${s.reliable}/${s.total} reliable`
    : `${s.lowConfidence} low-confidence rep${s.lowConfidence === 1 ? '' : 's'} excluded · improve lighting or framing`;
  nodes.repLog.hidden = false;
}

function switchExercise(exerciseId) {
  const cfg = settings.update({ exercise: { activeExercise: exerciseId } });
  resetRepLog(exerciseId);
  if (nodes.analysisMode.value !== exerciseId) nodes.analysisMode.value = exerciseId;

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

  const finishedRep = repTracker.update({ angles, landmarks, timestampMs, reps: nextState.reps, status: nextState.status });
  if (finishedRep) renderRepLog(finishedRep);

  // Sound cues on rep completion, depth target, or milestone
  if (settings.get().exercise.audioCues) {
    if (nextState.reps > (prevState?.reps ?? 0)) {
      audio.playRepCount();
      audio.speakRep(nextState.reps);
    } else if (nextState.justHitDepth) {
      audio.playDepthTarget();
    } else if (nextState.phase === 'holding' && Math.floor(nextState.currentHoldSec) > Math.floor(prevState?.currentHoldSec ?? 0)) {
      audio.playHoldTick();
    }
  }

  // Update Telemetry Display (throttled, but always on a rep change)
  const repChanged = nextState.reps !== prevState?.reps;
  if (!appState.uiDue && !repChanged) return;
  if (appState.activeExerciseInstance.type === 'hold') {
    nodes.repCount.textContent = String(Math.floor(nextState.currentHoldSec)).padStart(2, '0');
  } else {
    nodes.repCount.textContent = String(nextState.reps ?? 0).padStart(2, '0');
  }

  nodes.feedback.textContent = appState.activeExerciseInstance.getFeedback(nextState);
  renderDynamicTelemetry(appState.activeExerciseInstance.getTelemetry(nextState));
}

// Display loop (requestAnimationFrame): draws the overlay at the display rate. Inference
// runs once per camera frame in onVideoFrame(), or here by polling when the browser
// lacks requestVideoFrameCallback.
function frameLoop() {
  if (!appState.active) return;
  appState.raf = window.requestAnimationFrame(frameLoop);
  perf.recordRenderFrame();

  const video = nodes.video;
  const now = performance.now();
  const cfg = settings.get();

  // 1. Decoupled 60/120 FPS High-Fluidity Overlay Rendering
  interpolator.setPrediction(cfg.overlays.predictiveOverlay ? OVERLAY_PREDICTION_MS : 0);
  const renderedLandmarks = interpolator.step(now);
  if (!renderedLandmarks) {
    // No pose in view: clear the last skeleton instead of leaving it frozen on screen.
    overlay.render({ landmarks: null }, {
      showGhost: cfg.overlays.showGhost,
      mirrored: cfg.camera.mirroring === 'on' || (cfg.camera.mirroring === 'auto' && cfg.camera.facingMode === 'user'),
    });
  } else if (appState.lastAngles) {
    const isMirrored = cfg.camera.mirroring === 'on' ||
      (cfg.camera.mirroring === 'auto' && cfg.camera.facingMode === 'user');

    overlay.render({
      landmarks: renderedLandmarks,
      angles: appState.lastAngles,
      timestampMs: now,
      framing: appState.lastFraming,
    }, {
      showSkeleton: cfg.overlays.showSkeleton,
      showJointMarkers: cfg.overlays.showJointMarkers,
      showDegreeLabels: cfg.overlays.showDegreeLabels && nodes.showAngles.checked,
      showAngleArcs: cfg.overlays.showAngleArcs && nodes.showArcs.checked,
      showMotionTrails: cfg.overlays.showMotionTrails,
      showDirection: cfg.overlays.showDirection,
      showGhost: cfg.overlays.showGhost,
      mirrored: isMirrored,
      minVisibility: cfg.tracking.confidenceThreshold,
    });
  }
  if (renderedLandmarks && appState.pendingOverlay) {
    // Capture → overlay drawn (the canvas is composited with this display frame).
    perf.recordPipelineLatency('overlay', performance.now() - appState.pendingOverlay.timeMs, appState.pendingOverlay.clock);
    appState.pendingOverlay = null;
  }

  // 2. Fallback inference scheduling: poll for a new video frame (bounded at ~35 FPS).
  if (appState.useVideoFrameCallback) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.currentTime === appState.lastVideoTime ||
      now - appState.lastInferenceAt < appState.FRAME_INTERVAL_MS) {
    return;
  }
  appState.lastVideoTime = video.currentTime;
  processFrame(video, now, null);
}

/** requestVideoFrameCallback handler: runs once per new camera frame. */
function onVideoFrame(token, metadata) {
  // The token stops a callback chain from an earlier camera session.
  if (!appState.active || !appState.useVideoFrameCallback || token !== appState.sessionToken) return;
  const video = nodes.video;
  appState.vfcHandle = video.requestVideoFrameCallback((_, m) => onVideoFrame(token, m));
  perf.recordDroppedFrame(missedFrames(appState.lastPresentedFrames, metadata?.presentedFrames));
  appState.lastPresentedFrames = metadata?.presentedFrames ?? null;
  const now = performance.now();
  // Cap on the camera's own cadence (60 fps cameras: every other frame), not callback spacing.
  if (!appState.frameSampler.due(metadata, selectFrameTime(now, metadata).timeMs)) return;
  processFrame(video, now, metadata);
}

/**
 * Runs pose inference and analysis on the current video frame.
 * @param {HTMLVideoElement} video
 * @param {number} callNow performance.now() when processing started (UI throttling)
 * @param {Object|null} metadata VideoFrameCallbackMetadata, or null when polling
 */
function processFrame(video, callNow, metadata) {
  const cfg = settings.get();
  appState.lastInferenceAt = callNow;
  // Pipeline time is the frame's capture time when the browser reports it, so filter dt,
  // rep timing and latency all refer to when the movement happened.
  const frame = selectFrameTime(callNow, metadata, appState.lastFrameTime);
  appState.lastFrameTime = frame.timeMs;
  const now = frame.timeMs;

  const tStart = perf.startInference();
  try {
    const rawResult = runner.detectForVideo(video, now);
    perf.endInference(tStart);
    perf.recordPipelineLatency('landmarks', performance.now() - now, frame.clock);

    const metrics = perf.getMetrics();
    appState.uiDue = callNow - appState.lastUiAt >= UI_INTERVAL_MS;
    if (appState.uiDue) {
      appState.lastUiAt = callNow;
      nodes.inferenceFps.textContent = String(metrics.inferenceFps);
      nodes.latencyDisplay.textContent = `${Math.round(metrics.lastLatencyMs)}ms`;
      if (nodes.e2eLatency && metrics.captureToOverlayP50 !== null) {
        nodes.e2eLatency.textContent = `${Math.round(metrics.captureToOverlayP50)}ms`;
        nodes.e2eLatency.title = `Camera → overlay p50 ${metrics.captureToOverlayP50} ms, p95 ${metrics.captureToOverlayP95} ms; ` +
          `camera → landmarks p50 ${metrics.captureToLandmarksP50} ms (clock: ${metrics.latencyClock}). ` +
          'Excludes sensor exposure and display scan-out.';
      }
      if (nodes.renderFps) nodes.renderFps.textContent = String(metrics.renderFps);
      if (nodes.droppedFrames) nodes.droppedFrames.textContent = String(metrics.droppedFrames);
    }

    // Adaptive Performance Check
    adaptive.autoFallbackEnabled = cfg.tracking.autoFallback;
    // Don't let evaluate() consume its one-shot fallback while this session's switch is loading.
    if (appState.fallbackSessionToken !== appState.sessionToken) {
      const adaptEval = adaptive.evaluate(metrics);
      if (adaptEval.shouldFallback) {
        switchToFallbackModel(adaptEval.reason, adaptEval.nextModel);
      }
    }

    // 1 Euro and Outlier Filtering on 2D image coordinates
    filter.setVisibilityThreshold(cfg.tracking.confidenceThreshold);
    const smoothedLandmarks = filter.filter(rawResult.landmarks, now);

    runner.setSubjectExtent(subjectExtent(smoothedLandmarks, cfg.tracking.confidenceThreshold));

    // 3D world landmarks: temporal smoothing (depth is the noisiest axis), then
    // the bone-length constraint.
    worldFilter.setVisibilityThreshold(cfg.tracking.confidenceThreshold);
    let constrainedWorldLandmarks = rawResult.worldLandmarks;
    if (rawResult.worldLandmarks) {
      constrainedWorldLandmarks = kinematicFilter.filter(worldFilter.filter(rawResult.worldLandmarks, now), now);
    } else {
      worldFilter.reset();
    }
    const view = viewEstimator.update(constrainedWorldLandmarks, now);
    appState.lastView = view;

    const activeExerciseId = cfg.exercise.activeExercise || 'squat';
    const quality = trackingStatus(smoothedLandmarks, cfg.tracking.confidenceThreshold, activeExerciseId);
    const tilt = deviceLevel.tilt();
    appState.lastTilt = tilt;
    const framing = defaultFramingAssistant.evaluate(smoothedLandmarks, activeExerciseId, {
      yawDeg: view.yawDeg,
      recommendedView: appState.activeExerciseInstance?.recommendedCamera,
      tilt: cfg.tracking.levelGuidance ? tilt : null,
      setupGuidance: cfg.tracking.levelGuidance,
      aspect: video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 1,
    });
    appState.lastFraming = framing;

    // Angles: 'auto' blends precise 2D image angles (side views) with 3D world angles
    // (oblique views) per joint; '3d' and '2d' force one source.
    const angles2D = getJointAngles(smoothedLandmarks, video.videoWidth, video.videoHeight, cfg.tracking.confidenceThreshold);
    const mode = cfg.tracking.angleDimension;
    let angles = angles2D;
    if (mode !== '2d' && constrainedWorldLandmarks) {
      const angles3D = getJointAngles3D(constrainedWorldLandmarks, cfg.tracking.confidenceThreshold);
      if (mode === 'auto') {
        angles = blendAnglesByView(angles2D, angles3D, view);
      } else {
        angles = { ...angles2D };
        for (const [k, v] of Object.entries(angles3D)) {
          if (Number.isFinite(v)) angles[k] = v;
        }
      }
    }
    const use3DSegments = mode === '3d' || (mode === 'auto' && view.sagittal2DWeight < 0.5);

    // Supply target pose to 60/120 FPS interpolator (null clears it when the person leaves frame)
    interpolator.setTarget(smoothedLandmarks, now);
    appState.pendingOverlay = smoothedLandmarks ? { timeMs: now, clock: frame.clock } : null;

    appState.lastAngles = angles;
    if (appState.uiDue) {
      renderTelemetry(angles, smoothedLandmarks, constrainedWorldLandmarks, use3DSegments, video.videoWidth, video.videoHeight);
      renderSymmetry(angles);
      updateTrackingHUD(quality, framing);
      if (mode === 'auto') updateAngleDimUI('auto', view);
    }
    handleExerciseFrame(smoothedLandmarks, angles, video.videoWidth, video.videoHeight, now, quality);

    // Record telemetry frame if recording is active
    if (recorder.isRecording) {
      recorder.recordFrame({
        landmarks: smoothedLandmarks,
        angles,
        rawAngles: rawJointAngles(rawResult, video, cfg, view, now),
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

function switchToFallbackModel(reason, nextModel = 'full') {
  const token = appState.sessionToken;
  if (appState.fallbackSessionToken === token) return;
  appState.fallbackSessionToken = token;
  const label = nextModel.charAt(0).toUpperCase() + nextModel.slice(1);
  announce(`Performance notice: ${reason} Switching to ${label} model for responsiveness.`);
  // The current landmarker keeps running until the faster model is ready, then is replaced.
  runner.initialize(nextModel, () => {})
    .then(result => {
      // The runner discards a stale load itself; also ignore results from an older camera session.
      if (result?.stale || token !== appState.sessionToken) return;
      adaptive.setCurrentModel(runner.activeModel);
      perf.resetInferenceStats(); // measure the new model on its own samples
      // An Auto preference stays Auto; the HUD shows the model actually running.
      const saved = modelPreferenceAfterFallback(settings.get().tracking.model, runner.activeModel);
      settings.update({ tracking: { model: saved } });
      nodes.modelChoice.value = saved;
      nodes.modelName.textContent = runner.activeModel.toUpperCase();
      announce(`Switched to ${runner.activeModel.toUpperCase()} model for responsiveness.`);
    })
    .catch(error => {
      console.warn('Fallback model initialization failed:', error);
    })
    .finally(() => {
      if (appState.fallbackSessionToken === token) appState.fallbackSessionToken = null;
    });
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
  // Report a rep that was still settling when the camera stopped.
  const pendingRep = repTracker.flush();
  if (pendingRep) renderRepLog(pendingRep);
  appState.sessionToken++;
  appState.active = false;
  appState.starting = false;
  window.cancelAnimationFrame(appState.raf);
  appState.raf = 0;
  if (appState.vfcHandle && typeof nodes.video.cancelVideoFrameCallback === 'function') {
    nodes.video.cancelVideoFrameCallback(appState.vfcHandle);
  }
  appState.vfcHandle = 0;
  appState.useVideoFrameCallback = false;
  appState.lastPresentedFrames = null;
  appState.pendingOverlay = null;

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
  kinematicFilter.reset();
  worldFilter.reset();
  viewEstimator.reset();
  runner.setSubjectExtent(null);
  interpolator.reset();
  perf.reset();
  adaptive.reset();
  appState.lastAngles = null;
  appState.lastFraming = null;
  appState.lastTilt = null;
  deviceLevel.stop();
  appState.lastVideoTime = -1;
  appState.lastInferenceAt = -Infinity;
  appState.fallbackSessionToken = null;

  clearMetricsUI();
  setLiveUI(false);
}

async function startCamera() {
  if (appState.starting || appState.active) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    announce('Camera access requires HTTPS or localhost and a supported browser.', true);
    return;
  }

  // Create/resume the AudioContext inside the click gesture; browsers (notably iOS
  // Safari) keep contexts created later from the render loop suspended.
  if (settings.get().exercise.audioCues) audio.ensureContext();
  // iOS only grants motion sensors from a user gesture, so ask before the first await.
  if (settings.get().tracking.levelGuidance || settings.get().tracking.tiltCorrection) deviceLevel.start();

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
    await runner.initialize(resolveModelChoice(cfg.tracking.model, { mobile: isLikelyMobile() }), statusStr => announce(statusStr));
    if (appState.sessionToken !== currentToken) return;
    // initialize() may itself fall back from Heavy to Full if Heavy fails to load.
    adaptive.setCurrentModel(runner.activeModel);
    adaptive.autoFallbackEnabled = cfg.tracking.autoFallback;

    nodes.modelName.textContent = runner.activeModel.toUpperCase();
    appState.starting = false;
    appState.active = true;
    setLiveUI(true);

    switchExercise(cfg.exercise.activeExercise);
    appState.useVideoFrameCallback = supportsVideoFrameCallback(nodes.video);
    if (appState.useVideoFrameCallback) {
      const token = appState.sessionToken;
      appState.frameSampler = new FrameSampler(appState.FRAME_INTERVAL_MS);
      appState.vfcHandle = nodes.video.requestVideoFrameCallback((_, m) => onVideoFrame(token, m));
    }
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
/** Recorded frames re-smoothed offline for results and exports (cached per session). */
const analysisCache = new WeakMap();
function analysisFrames(session) {
  if (!session?.frames) return [];
  if (!analysisCache.has(session)) analysisCache.set(session, offlineSessionFrames(session.frames).frames);
  return analysisCache.get(session);
}

/**
 * Joint angles from unfiltered landmarks (same 2D/3D/view blend as the live angles,
 * no temporal smoothing), recorded so the session can be re-analysed with a
 * zero-phase filter that has no lag.
 */
function rawJointAngles(rawResult, video, cfg, view, now) {
  if (!rawResult?.landmarks) return null;
  const thr = cfg.tracking.confidenceThreshold;
  const a2 = getJointAngles(rawResult.landmarks, video.videoWidth, video.videoHeight, thr);
  const mode = cfg.tracking.angleDimension;
  if (mode === '2d' || !rawResult.worldLandmarks) return a2;
  const a3 = getJointAngles3D(rawKinematicFilter.filter(rawResult.worldLandmarks, now), thr);
  if (mode === 'auto') return blendAnglesByView(a2, a3, view);
  const out = { ...a2 };
  for (const [k, v] of Object.entries(a3)) if (Number.isFinite(v)) out[k] = v;
  return out;
}

function startRecording() {
  rawKinematicFilter.reset();
  // Wait for the previous recording to finish saving; otherwise its pending stop
  // would swallow the stop request for the new recording.
  if (!appState.active || appState.stoppingRecording || recorder.isRecording) return;
  const cfg = settings.get();
  recorder.start({ exerciseId: cfg.exercise.activeExercise, mediaStream: appState.stream });
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

  announce('Session recording started. Video and joint telemetry storing locally.');
}

function stopRecording() {
  if (appState.stoppingRecording) return appState.stoppingRecording;
  nodes.recordBtn.disabled = true;
  appState.stoppingRecording = finishRecording().finally(() => {
    appState.stoppingRecording = null;
    nodes.recordBtn.disabled = !appState.active;
  });
  return appState.stoppingRecording;
}

async function finishRecording() {
  clearInterval(appState.recTimerInterval);
  nodes.recIndicator.hidden = true;
  nodes.recordBtn.classList.remove('is-recording');
  nodes.recordBtn.querySelector('span:last-child').textContent = 'Record';

  const sessionData = await recorder.stop();
  if (!sessionData || sessionData.frameCount === 0) {
    announce('Recording stopped. No frames were captured, so nothing was saved.');
    return;
  }

  const saved = storage.saveSession(sessionData, {
    onEvict: id => { videoStore.deleteVideo(id); },
    compact: stripRawAngles,
  });
  if (!saved) {
    announce('Session could not be saved: browser storage is full. Delete older sessions or record a shorter clip.', true);
    return;
  }
  if (sessionData.hasVideo && sessionData.videoBlob) {
    await videoStore.saveVideo(sessionData.id, sessionData.videoBlob);
  }
  updateSessionsList();
  announce(`Session recorded: ${sessionData.frameCount} frames, ${sessionData.finalReps} reps saved.`);
  await loadSessionForPlayback(sessionData.id);
}

function fillSelect(select, items, placeholder = null) {
  select.replaceChildren();
  if (placeholder !== null) select.append(new Option(placeholder, ''));
  for (const [value, label] of items) select.append(new Option(label, value));
}

function updateSessionsList() {
  const index = storage.getIndex();
  nodes.sessionCountBadge.textContent = String(index.length);
  nodes.savedSessionsList.innerHTML = '';

  if (nodes.sessionSelectTop) {
    const curId = appState.currentSessionViewing?.id || '';
    fillSelect(nodes.sessionSelectTop, index.map(item => [
      item.id,
      `${String(item.exerciseId).toUpperCase()} · ${item.recordedAt ? new Date(item.recordedAt).toLocaleTimeString() : item.id} (${item.finalReps} reps)`,
    ]), '-- Choose a session --');
    if (curId && index.some(item => item.id === curId)) {
      nodes.sessionSelectTop.value = curId;
    }
  }

  for (const selId of ['compare-a', 'compare-b']) {
    const sel = document.getElementById(selId);
    if (sel) {
      const prev = sel.value;
      fillSelect(sel, index.map(item => [
        item.id,
        `${String(item.exerciseId).toUpperCase()} · ${item.recordedAt ? new Date(item.recordedAt).toLocaleString() : item.id}`,
      ]));
      if (index.some(item => item.id === prev)) sel.value = prev;
    }
  }

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
    title.textContent = `${String(item.exerciseId).toUpperCase()} (${item.finalReps} reps)`;

    const meta = document.createElement('span');
    meta.className = 'session-meta';
    const durSec = (item.durationMs / 1000).toFixed(1);
    meta.textContent = `${durSec}s · ${item.frameCount} frames ${item.hasVideo ? '📹 Video' : ''}`;

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

function releaseReplayVideo() {
  const src = nodes.replayVideo.getAttribute('src');
  if (src && src.startsWith('blob:')) URL.revokeObjectURL(src);
  // Assigning '' would resolve to the page URL and make the video element fetch it.
  nodes.replayVideo.removeAttribute('src');
  nodes.replayVideo.load();
}

function resetReplayView(title) {
  appState.currentSessionViewing = null;
  player.stop();
  player.detachVideoElement();
  player.loadSession(null);
  releaseReplayVideo();
  nodes.replayVideo.hidden = true;
  nodes.replayNoVideo.hidden = false;
  replayOverlay.clear();
  nodes.activeSessionTitle.textContent = title;
  if (nodes.workoutResultsCard) {
    nodes.resRepCount.textContent = '--';
    nodes.resDuration.textContent = '--:--';
    nodes.resPeakAngle.textContent = '--°';
    appState.resultsPeak = null;
    nodes.resSymmetry.textContent = '--';
    nodes.resTimestamp.textContent = 'No session selected';
    if (nodes.resAvgRom) {
      nodes.resAvgRom.textContent = '--';
      nodes.resTempo.textContent = '--';
      nodes.resQuality.textContent = '--';
    }
  }
  chart.render({ frames: [] });
  updateSessionsList();
}

function renderRepResults(reps, profile) {
  if (!nodes.resAvgRom) return;
  const s = summarizeSession(reps);
  const [p1, p2] = profile?.phases ?? ['phase 1', 'phase 2'];
  nodes.resTempoLabel.textContent = `TEMPO (${p1.toUpperCase()} / ${p2.toUpperCase()})`;
  if (!s.total) {
    nodes.resAvgRom.textContent = '—';
    nodes.resTempo.textContent = '—';
    nodes.resQuality.textContent = profile ? '0 reps' : 'n/a';
    return;
  }
  nodes.resAvgRom.textContent = s.avgRomDeg !== null
    ? `${Math.round(s.avgRomDeg)}°${s.romCvPct !== null ? ` ±${Math.round(s.romCvPct)}%` : ''}`
    : '—';
  nodes.resTempo.textContent = s.avgPhase1Ms !== null
    ? `${fmtSec(s.avgPhase1Ms)} / ${fmtSec(s.avgPhase2Ms)}${s.repsPerMin !== null ? ` · ${Math.round(s.repsPerMin)}/min` : ''}`
    : '—';
  nodes.resQuality.textContent = `${s.reliable}/${s.total}`;
}

async function loadSessionForPlayback(sessionId) {
  const fullSession = storage.getSession(sessionId);
  if (!fullSession) return;

  appState.currentSessionViewing = fullSession;
  player.loadSession(fullSession);
  nodes.activeSessionTitle.textContent = `${fullSession.exerciseId.toUpperCase()} · ${fullSession.frameCount} frames`;
  nodes.timelineScrubber.max = String(Math.max(0, fullSession.frameCount - 1));
  nodes.timelineScrubber.value = '0';

  if (nodes.sessionSelectTop) {
    nodes.sessionSelectTop.value = sessionId;
  }

  // Populate Workout Results Summary Card
  if (nodes.workoutResultsCard) {
    nodes.resExerciseTag.textContent = `${fullSession.exerciseId.toUpperCase()} SESSION RESULTS`;
    nodes.resTimestamp.textContent = fullSession.recordedAt ? new Date(fullSession.recordedAt).toLocaleString() : 'Recorded session';
    nodes.resRepCount.textContent = String(fullSession.finalReps ?? 0).padStart(2, '0');
    const durSec = (fullSession.durationMs / 1000).toFixed(1);
    const m = Math.floor(fullSession.durationMs / 60000);
    const s = Math.floor((fullSession.durationMs % 60000) / 1000);
    nodes.resDuration.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')} (${durSec}s)`;

    // Calculate peak depth / angle from frames
    let peakDeg = null;
    let symSum = 0;
    let symCount = 0;
    for (const f of analysisFrames(fullSession)) {
      const a = f.angles || {};
      const knee = [a.leftKnee, a.rightKnee].filter(Number.isFinite);
      const shoulder = [a.leftShoulder, a.rightShoulder].filter(Number.isFinite);
      const exerciseKey = normalizeExerciseId(fullSession.exerciseId);
      if (KNEE_EXERCISES.has(exerciseKey)) {
        if (knee.length) {
          const minKnee = Math.min(...knee);
          if (peakDeg === null || minKnee < peakDeg) peakDeg = minKnee;
        }
      } else if (SHOULDER_EXERCISES.has(exerciseKey) && shoulder.length) {
        const maxArm = Math.max(...shoulder);
        if (peakDeg === null || maxArm > peakDeg) peakDeg = maxArm;
      }
      if (Number.isFinite(a.leftKnee) && Number.isFinite(a.rightKnee)) {
        symSum += Math.abs(a.leftKnee - a.rightKnee);
        symCount++;
      }
    }
    // Per-rep analysis, recomputed from recorded frames with zero-phase smoothing
    // (no causal lag), so older sessions get it too.
    const reps = analyzeReps(analysisFrames(fullSession), fullSession.exerciseId);
    const repProfile = profileFor(fullSession.exerciseId);
    const reliableReps = reps.filter(r => r.quality === 'ok');
    // Peak from reliable reps where the rep extreme is the "peak" (squat/lunge depth,
    // shoulder elevation); per-frame minima can come from an occluded limb.
    if (repProfile && normalizeExerciseId(fullSession.exerciseId) !== 'sit_to_stand') {
      // No reliable rep means no trustworthy peak: show "—" rather than an ungated
      // frame minimum that may come from an occluded limb.
      const extremes = reliableReps.map(r => r.extremeAngle);
      peakDeg = extremes.length ? (repProfile.direction === 'min' ? Math.min(...extremes) : Math.max(...extremes)) : null;
    }
    const peakJoint = repProfile?.joint ?? (/shoulder/.test(normalizeExerciseId(fullSession.exerciseId)) ? 'Shoulder' : 'Knee');
    appState.resultsPeak = { joint: peakJoint, deg: peakDeg };
    nodes.resPeakAngle.textContent = peakDeg !== null ? formatAngle(peakJoint, peakDeg) : '—';
    // A counter reset mid-recording starts a new set; show reps across all sets.
    if (repProfile) nodes.resRepCount.textContent = String(countedReps(fullSession.frames)).padStart(2, '0');
    renderRepResults(reps, repProfile);
    if (symCount > 0) {
      const avgSym = Math.round((symSum / symCount) * 10) / 10;
      nodes.resSymmetry.textContent = avgSym <= 10 ? `Balanced (±${avgSym}°)` : `Asymmetric (±${avgSym}°)`;
    } else {
      nodes.resSymmetry.textContent = 'Stable';
    }
  }

  // Load video if present in videoStore
  const videoBlob = await videoStore.getVideo(sessionId);
  if (appState.currentSessionViewing?.id !== sessionId) return; // a newer load started meanwhile
  releaseReplayVideo();
  if (videoBlob) {
    nodes.replayVideo.src = URL.createObjectURL(videoBlob);
    nodes.replayVideo.hidden = false;
    nodes.replayNoVideo.hidden = true;
    nodes.replayVideo.onloadedmetadata = () => {
      nodes.replayCanvas.width = nodes.replayVideo.videoWidth || 640;
      nodes.replayCanvas.height = nodes.replayVideo.videoHeight || 480;
    };
    player.attachVideoElement(nodes.replayVideo);
  } else {
    // Telemetry-only session: the player must drive playback from its own clock.
    player.detachVideoElement();
    nodes.replayVideo.hidden = true;
    nodes.replayNoVideo.hidden = false;
  }

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

  const cfg = settings.get();

  // Render on replay canvas overlay
  replayOverlay.render({
    landmarks: frame.landmarks,
    angles: frame.angles,
    timestampMs: frame.timeMs,
  }, {
    showSkeleton: cfg.overlays.showSkeleton,
    showJointMarkers: cfg.overlays.showJointMarkers,
    showDegreeLabels: cfg.overlays.showDegreeLabels,
    showAngleArcs: cfg.overlays.showAngleArcs,
    mirrored: false,
  });

  // Also draw on main canvas if live camera is not running
  if (!appState.active) {
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
    renderReplayTelemetry(frame);
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
    resetRepLog(repTracker.exerciseId);
    nodes.repCount.textContent = '00';
    nodes.feedback.textContent = appState.activeExerciseInstance.getFeedback(appState.exerciseState);
    renderDynamicTelemetry(appState.activeExerciseInstance.getTelemetry(appState.exerciseState));
    announce('Repetition counter reset.');
  }
});

const baselineBtn = document.getElementById('capture-baseline-btn');
if (baselineBtn) {
  baselineBtn.addEventListener('click', () => {
    const a = appState.lastAngles || {};
    const knees = [a.leftKnee, a.rightKnee].filter(Number.isFinite);
    const hips = [a.leftHip, a.rightHip].filter(Number.isFinite);
    const label = document.getElementById('baseline-status');
    if (!knees.length || !hips.length) {
      if (label) label.textContent = 'No valid standing angles yet - start the camera and stand in frame.';
      return;
    }
    const baseline = {
      standingKneeDeg: Math.round((knees.reduce((s, v) => s + v, 0) / knees.length) * 10) / 10,
      standingHipDeg: Math.round((hips.reduce((s, v) => s + v, 0) / hips.length) * 10) / 10,
    };
    settings.update({ exercise: { baseline } });
    if (label) label.textContent = `Baseline saved: knee ${baseline.standingKneeDeg}°, hip ${baseline.standingHipDeg}°.`;
    announce('Standing baseline saved.');
  });
}

const compareBtn = document.getElementById('compare-btn');
if (compareBtn) {
  compareBtn.addEventListener('click', () => {
    const resultEl = document.getElementById('compare-result');
    const a = storage.getSession(document.getElementById('compare-a')?.value);
    const b = storage.getSession(document.getElementById('compare-b')?.value);
    if (!a || !b) { if (resultEl) resultEl.textContent = 'Save at least two sessions to compare.'; return; }
    const r = compareSessions(a, b, nodes.chartJointSelect.value);
    const fmt = v => (v === null || v === undefined ? '—' : `${v}°`);
    if (resultEl) resultEl.textContent = `Δ reps ${r.repDelta > 0 ? '+' : ''}${r.repDelta} · Δ duration ${Math.round(r.durationDeltaMs / 100) / 10}s · mean angle Δ ${fmt(r.meanDeltaDeg)} · max angle Δ ${fmt(r.maxDeltaDeg)} (${r.frameCountA} vs ${r.frameCountB} frames)`;
  });
}

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
  nodes.analyticsDrawer.classList.toggle('open', !nodes.analyticsDrawer.hidden);
  if (!nodes.analyticsDrawer.hidden) {
    updateSessionsList();
    nodes.analyticsDrawer.scrollIntoView({ behavior: 'smooth' });
  }
});

nodes.closeAnalyticsBtn.addEventListener('click', () => {
  nodes.analyticsDrawer.hidden = true;
  nodes.analyticsDrawer.classList.remove('open');
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

nodes.exportRepsBtn?.addEventListener('click', () => {
  const session = appState.currentSessionViewing;
  if (!session) return;
  const reps = analyzeReps(analysisFrames(session), session.exerciseId);
  if (!reps.length) {
    announce('No repetitions with enough tracking data to export for this session.');
    return;
  }
  downloadFile(exportRepsToCSV(reps, profileFor(session.exerciseId)?.phases), `${session.id}-reps.csv`, 'text/csv');
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

if (nodes.sessionSelectTop) {
  nodes.sessionSelectTop.addEventListener('change', (e) => {
    if (e.target.value) {
      loadSessionForPlayback(e.target.value);
    }
  });
}

nodes.deleteSessionBtn.addEventListener('click', async () => {
  if (!appState.currentSessionViewing) return;
  const id = appState.currentSessionViewing.id;
  storage.deleteSession(id);
  await videoStore.deleteVideo(id);
  resetReplayView('Session deleted.');
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
  nodes.cfgTrackLevel.checked = cfg.tracking.levelGuidance;
  nodes.cfgTrackTiltFix.checked = cfg.tracking.tiltCorrection;
  if (nodes.cfgTrackAngleDim) {
    nodes.cfgTrackAngleDim.value = cfg.tracking.angleDimension || 'auto';
  }

  nodes.cfgOverlaySkeleton.checked = cfg.overlays.showSkeleton;
  nodes.cfgOverlayJoints.checked = cfg.overlays.showJointMarkers;
  nodes.cfgOverlayLabels.checked = cfg.overlays.showDegreeLabels;
  nodes.cfgOverlayArcs.checked = cfg.overlays.showAngleArcs;
  nodes.cfgOverlayTrails.checked = cfg.overlays.showMotionTrails;
  nodes.cfgOverlayPredict.checked = cfg.overlays.predictiveOverlay;
  nodes.cfgOverlayConvention.value = cfg.overlays.angleConvention;
  applyAngleConvention(cfg.overlays.angleConvention);
  if (nodes.cfgOverlayDirection) nodes.cfgOverlayDirection.checked = cfg.overlays.showDirection;
  nodes.cfgOverlaySymHud.checked = cfg.overlays.showSymmetryHUD;
  nodes.cfgOverlayGhost.checked = cfg.overlays.showGhost;

  nodes.cfgExerciseSensitivity.value = cfg.exercise.sensitivity;
  nodes.cfgExerciseAudio.checked = cfg.exercise.audioCues;
  if (nodes.cfgExerciseVoice) {
    nodes.cfgExerciseVoice.checked = Boolean(cfg.exercise.voiceCoach);
  }

  const usage = storage.getStorageEstimate();
  const kb = Math.round(usage.bytes / 1024);
  nodes.storageEstimateLabel.textContent = `${usage.count} sessions stored (~${kb} KB locally)`;
}

/**
 * Pushes persisted settings into quick controls and live subsystems so the UI
 * matches storage on load, after edits, and after a reset.
 */
function applySettingsToUI() {
  const cfg = settings.get();
  nodes.analysisMode.value = cfg.exercise.activeExercise;
  nodes.showAngles.checked = cfg.overlays.showDegreeLabels;
  nodes.showArcs.checked = cfg.overlays.showAngleArcs;
  if (!appState.active && !appState.starting) nodes.modelChoice.value = cfg.tracking.model;
  updateAngleDimUI(cfg.tracking.angleDimension, appState.lastView);
  filter.setVisibilityThreshold(cfg.tracking.confidenceThreshold);
  if (filter.presetName !== cfg.tracking.smoothing) filter.setPreset(cfg.tracking.smoothing);
  if (worldFilter.presetName !== cfg.tracking.smoothing) worldFilter.setPreset(cfg.tracking.smoothing);
  adaptive.autoFallbackEnabled = cfg.tracking.autoFallback;
  audio.setEnabled(cfg.exercise.audioCues);
  audio.setVoiceCoachEnabled(cfg.exercise.voiceCoach);
  appState.activeExerciseInstance?.setSensitivity(cfg.exercise.sensitivity);
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
        angleDimension: nodes.cfgTrackAngleDim ? nodes.cfgTrackAngleDim.value : 'auto',
        levelGuidance: nodes.cfgTrackLevel.checked,
        tiltCorrection: nodes.cfgTrackTiltFix.checked,
      },
      overlays: {
        showSkeleton: nodes.cfgOverlaySkeleton.checked,
        showJointMarkers: nodes.cfgOverlayJoints.checked,
        showDegreeLabels: nodes.cfgOverlayLabels.checked,
        showAngleArcs: nodes.cfgOverlayArcs.checked,
        showMotionTrails: nodes.cfgOverlayTrails.checked,
        predictiveOverlay: nodes.cfgOverlayPredict.checked,
        angleConvention: nodes.cfgOverlayConvention.value,
        showDirection: nodes.cfgOverlayDirection ? nodes.cfgOverlayDirection.checked : false,
        showSymmetryHUD: nodes.cfgOverlaySymHud.checked,
        showGhost: nodes.cfgOverlayGhost.checked,
      },
      exercise: {
        sensitivity: nodes.cfgExerciseSensitivity.value,
        audioCues: nodes.cfgExerciseAudio.checked,
        voiceCoach: nodes.cfgExerciseVoice ? nodes.cfgExerciseVoice.checked : false,
      },
    });

    applySettingsToUI();
  };

  const inputs = [
    nodes.cfgCameraFacing, nodes.cfgCameraMirror, nodes.cfgCameraRes,
    nodes.cfgTrackModel, nodes.cfgTrackAutoFallback, nodes.cfgTrackSmoothing,
    nodes.cfgTrackConfidence, nodes.cfgTrackLevel, nodes.cfgTrackTiltFix, nodes.cfgOverlaySkeleton, nodes.cfgOverlayJoints,
    nodes.cfgOverlayLabels, nodes.cfgOverlayArcs, nodes.cfgOverlayTrails, nodes.cfgOverlayPredict, nodes.cfgOverlayConvention, nodes.cfgOverlayDirection,
    nodes.cfgOverlaySymHud, nodes.cfgOverlayGhost,
    nodes.cfgExerciseSensitivity, nodes.cfgExerciseAudio,
  ];
  if (nodes.cfgExerciseVoice) inputs.push(nodes.cfgExerciseVoice);
  if (nodes.cfgTrackAngleDim) inputs.push(nodes.cfgTrackAngleDim);

  for (const input of inputs) {
    input.addEventListener('change', save);
  }
  // Enabling a sensor feature mid-capture starts motion sensing from this user gesture
  // (iOS grants DeviceMotion only from one); start() is a no-op when already listening.
  for (const input of [nodes.cfgTrackLevel, nodes.cfgTrackTiltFix]) {
    input.addEventListener('change', () => {
      if (input.checked && appState.active) deviceLevel.start();
    });
  }

  if (nodes.angleDimToggle) {
    nodes.angleDimToggle.addEventListener('click', () => {
      const cur = settings.get().tracking.angleDimension || 'auto';
      const next = ANGLE_MODES[(ANGLE_MODES.indexOf(cur) + 1) % ANGLE_MODES.length];
      settings.update({ tracking: { angleDimension: next } });
      updateAngleDimUI(next, appState.lastView);
    });
  }
}

nodes.resetDefaultsBtn.addEventListener('click', () => {
  settings.reset();
  populateSettingsForm();
  applySettingsToUI();
  switchExercise(settings.get().exercise.activeExercise);
  announce('Settings reset to defaults.');
});

nodes.clearAllSessionsBtn.addEventListener('click', async () => {
  storage.clearAll();
  await videoStore.clearAll();
  resetReplayView('No session loaded.');
  populateSettingsForm();
  announce('All local sessions and stored videos cleared.');
});

window.addEventListener('pagehide', stopCamera);
window.addEventListener('beforeunload', stopCamera);

// Initialize application on load
bindSettingsInputs();
clearMetricsUI();
setLiveUI(false);
updateSessionsList();
applySettingsToUI();
switchExercise(settings.get().exercise.activeExercise);
