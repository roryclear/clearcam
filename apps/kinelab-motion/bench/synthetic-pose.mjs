/**
 * Synthetic squat generator with known ground-truth joint angles.
 *
 * Produces MediaPipe-shaped output (33 normalized image landmarks + 33 metric
 * world landmarks) for a squatting body seen by a pinhole camera, with a noise
 * model taken from published MediaPipe validation work:
 *  - image-plane landmarks are comparatively precise (~2 px jitter at 720p);
 *  - monocular depth (world z) is much noisier than x/y (Dill et al., Sensors 2024);
 *  - in a lateral view the far (occluded) leg is pulled toward the near leg and
 *    is far less accurate (10.7° vs 25.1° knee RMSE, same study), yet it still
 *    reports visibility above common thresholds.
 */

import { JOINTS } from '../src/biomechanics/landmark-indices.mjs';

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  const u = Math.max(1e-12, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const L = { shank: 0.42, thigh: 0.43, trunk: 0.5, upperArm: 0.3, forearm: 0.27, halfPelvis: 0.1, halfShoulder: 0.18 };

/** Body points in a body frame: x forward, y down, z lateral (+z = right side). */
function bodyPose(kneeFlexDeg) {
  const flex = kneeFlexDeg * Math.PI / 180;
  const shankTilt = 0.4 * flex; // forward shank inclination
  const thighTilt = 0.6 * flex; // backward thigh inclination
  const trunkLean = 0.5 * thighTilt;
  const leg = (z) => {
    const ankle = { x: 0, y: 0, z };
    const knee = { x: ankle.x + L.shank * Math.sin(shankTilt), y: ankle.y - L.shank * Math.cos(shankTilt), z };
    const hip = { x: knee.x - L.thigh * Math.sin(thighTilt), y: knee.y - L.thigh * Math.cos(thighTilt), z };
    return { ankle, knee, hip };
  };
  const left = leg(-L.halfPelvis);
  const right = leg(L.halfPelvis);
  const mid = { x: left.hip.x, y: left.hip.y };
  const shoulderBase = { x: mid.x + L.trunk * Math.sin(trunkLean), y: mid.y - L.trunk * Math.cos(trunkLean) };
  const arm = (z) => {
    const shoulder = { x: shoulderBase.x, y: shoulderBase.y, z };
    const elbow = { x: shoulder.x + L.upperArm * 0.5, y: shoulder.y + L.upperArm * 0.86, z };
    const wrist = { x: elbow.x + L.forearm, y: elbow.y, z };
    return { shoulder, elbow, wrist };
  };
  const lArm = arm(-L.halfShoulder);
  const rArm = arm(L.halfShoulder);
  const pts = new Array(33).fill(null).map(() => ({ x: shoulderBase.x + 0.05, y: shoulderBase.y - 0.25, z: 0 }));
  pts[JOINTS.leftShoulder] = lArm.shoulder; pts[JOINTS.rightShoulder] = rArm.shoulder;
  pts[JOINTS.leftElbow] = lArm.elbow; pts[JOINTS.rightElbow] = rArm.elbow;
  pts[JOINTS.leftWrist] = lArm.wrist; pts[JOINTS.rightWrist] = rArm.wrist;
  pts[JOINTS.leftHip] = left.hip; pts[JOINTS.rightHip] = right.hip;
  pts[JOINTS.leftKnee] = left.knee; pts[JOINTS.rightKnee] = right.knee;
  pts[JOINTS.leftAnkle] = left.ankle; pts[JOINTS.rightAnkle] = right.ankle;
  for (const [heel, toe, ankle] of [[29, 31, JOINTS.leftAnkle], [30, 32, JOINTS.rightAnkle]]) {
    pts[heel] = { x: pts[ankle].x - 0.06, y: 0.05, z: pts[ankle].z };
    pts[toe] = { x: pts[ankle].x + 0.18, y: 0.07, z: pts[ankle].z };
  }
  return pts;
}

/**
 * Rotates the body about the vertical axis. yawDeg = 0 is a pure lateral view
 * (left side toward the camera); 90 is a frontal view.
 */
function toCamera(pts, yawDeg) {
  const a = yawDeg * Math.PI / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  return pts.map(p => ({ x: p.x * c - p.z * s, y: p.y, z: p.x * s + p.z * c }));
}

function truthAngle(a, b, c) {
  const u = [a.x - b.x, a.y - b.y, a.z - b.z];
  const v = [c.x - b.x, c.y - b.y, c.z - b.z];
  const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  return Math.acos(Math.max(-1, Math.min(1, dot / (Math.hypot(...u) * Math.hypot(...v))))) * 180 / Math.PI;
}

/**
 * @param {Object} o
 * @param {number} [o.reps=6]
 * @param {number} [o.cycleMs=2400]
 * @param {number} [o.fps=30]
 * @param {number} [o.yawDeg=0] camera yaw relative to a pure side view
 * @param {number} [o.maxFlexDeg=85] peak knee flexion (included angle 180-maxFlex)
 * @param {number} [o.seed=1]
 * @param {number} [o.holdMs=1500] standing still before and after the set
 */
export function generateSquatSequence({ reps = 6, cycleMs = 2400, fps = 30, yawDeg = 0, maxFlexDeg = 85, seed = 1, holdMs = 1500 } = {}) {
  const rand = mulberry32(seed);
  const dt = 1000 / fps;
  const total = holdMs * 2 + reps * cycleMs;
  const camDist = 3.0;
  const focal = 1.2; // normalized focal length (person ~65% of frame height)
  const aspect = 16 / 9;
  // Slowly varying error makes depth noise realistic (not pure white noise).
  const drift = new Array(33).fill(0).map(() => ({ x: 0, y: 0, z: 0 }));
  const frames = [];
  const nearIsLeft = Math.cos(yawDeg * Math.PI / 180) >= 0;
  const farLeg = nearIsLeft
    ? [JOINTS.rightHip, JOINTS.rightKnee, JOINTS.rightAnkle]
    : [JOINTS.leftHip, JOINTS.leftKnee, JOINTS.leftAnkle];
  const nearLeg = nearIsLeft
    ? [JOINTS.leftHip, JOINTS.leftKnee, JOINTS.leftAnkle]
    : [JOINTS.rightHip, JOINTS.rightKnee, JOINTS.rightAnkle];
  const occlusion = Math.abs(Math.cos(yawDeg * Math.PI / 180)); // 1 in side view, 0 frontal

  for (let t = 0; t <= total; t += dt) {
    let flex = 0;
    if (t > holdMs && t < holdMs + reps * cycleMs) {
      const phase = ((t - holdMs) % cycleMs) / cycleMs;
      flex = maxFlexDeg * 0.5 * (1 - Math.cos(phase * 2 * Math.PI));
    }
    const body = bodyPose(flex);
    const cam = toCamera(body, yawDeg);
    const mid = { x: (cam[23].x + cam[24].x) / 2, y: (cam[23].y + cam[24].y) / 2, z: (cam[23].z + cam[24].z) / 2 };
    const truth = {
      leftKnee: truthAngle(body[23], body[25], body[27]),
      rightKnee: truthAngle(body[24], body[26], body[28]),
      leftHip: truthAngle(body[11], body[23], body[25]),
      rightHip: truthAngle(body[12], body[24], body[26]),
    };

    for (const d of drift) {
      d.x = 0.95 * d.x + 0.004 * gaussian(rand);
      d.y = 0.95 * d.y + 0.004 * gaussian(rand);
      d.z = 0.95 * d.z + 0.015 * gaussian(rand);
    }

    const world = cam.map((p, i) => ({
      x: p.x - mid.x + drift[i].x + 0.006 * gaussian(rand),
      y: p.y - mid.y + drift[i].y + 0.006 * gaussian(rand),
      z: p.z - mid.z + drift[i].z + 0.03 * gaussian(rand),
      visibility: 0.97,
      presence: 0.99,
    }));
    // Camera placed so the body's hip height sits mid-frame.
    const image = cam.map((p, i) => {
      const depth = camDist + p.z;
      const pxNoise = 0.0028;
      return {
        x: 0.5 + (focal / aspect) * (p.x / depth) + pxNoise * gaussian(rand),
        y: 0.62 + focal * ((p.y + 0.45) / depth) + pxNoise * gaussian(rand),
        z: p.z,
        visibility: 0.97,
        presence: 0.99,
      };
    });

    // Occluded far leg: pulled toward the near leg, noisier, still "visible".
    if (occlusion > 0.3) {
      for (let k = 0; k < 3; k++) {
        const f = farLeg[k];
        const n = nearLeg[k];
        const pull = 0.55 * occlusion;
        for (const arr of [image, world]) {
          arr[f].x += pull * (arr[n].x - arr[f].x);
          arr[f].y += pull * (arr[n].y - arr[f].y);
          arr[f].visibility = 0.6 + 0.15 * rand();
        }
        image[f].x += 0.012 * gaussian(rand);
        image[f].y += 0.012 * gaussian(rand);
        world[f].z += 0.06 * gaussian(rand);
      }
    }

    frames.push({ timeMs: t, landmarks: image, worldLandmarks: world, truth, width: 1280, height: 720 });
  }
  return { frames, reps, nearSide: nearIsLeft ? 'left' : 'right' };
}
