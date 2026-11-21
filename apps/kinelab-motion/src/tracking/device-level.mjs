/**
 * Phone tilt from the gravity sensor, for camera-setup guidance and an optional
 * correction of segment-to-vertical angles (trunk, shank).
 *
 * Image roll adds 1:1 to segment-to-vertical angles and cancels out of joint angles.
 * Placement studies: camera at hip height, 2–3 m away gave hip MAE 1.9° / knee 2.4°
 * (PMC10886083); front-oblique views were worst (PMC11819822).
 *
 * Roll is computed from the gravity *axis* (atan of a ratio, folded to ±90°), so it
 * does not depend on the sign convention of accelerationIncludingGravity (which has
 * differed between browsers) or on which way a landscape screen is rotated.
 */

const DEG = 180 / Math.PI;

/**
 * Camera tilt from a gravity vector in device coordinates.
 * @param {{x:number,y:number,z:number}} g accelerationIncludingGravity
 * @param {number} [screenAngle=0] screen.orientation.angle (0, 90, 180, 270)
 * @returns {{ rollDeg: number, pitchDeg: number }|null} rollDeg is clockwise-positive as
 *   seen on the screen, in (−90, 90]; pitchDeg is |forward/back tilt| from upright
 */
export function cameraTiltFromGravity(g, screenAngle = 0) {
  if (!g || ![g.x, g.y, g.z].every(Number.isFinite)) return null;
  const landscape = ((Math.round(screenAngle / 90) % 2) + 2) % 2 === 1;
  // Component along the screen's vertical and horizontal axes.
  const vert = landscape ? g.x : g.y;
  const horiz = landscape ? -g.y : g.x;
  const inPlane = Math.hypot(vert, horiz);
  if (inPlane < 1e-6) return null; // phone lying flat: no meaningful roll
  let rollDeg = Math.atan(-horiz / vert) * DEG;
  if (!Number.isFinite(rollDeg)) rollDeg = 90;
  const pitchDeg = Math.atan(Math.abs(g.z) / inPlane) * DEG;
  return { rollDeg, pitchDeg };
}

/**
 * Rotates image-space landmarks to undo camera roll, about the image centre in pixel
 * space (so angles are preserved for any aspect ratio). Rear and front cameras see the
 * scene rotate in opposite directions when the phone rolls.
 * @param {Array<Object>|null} landmarks normalized landmarks
 * @param {number} rollDeg from cameraTiltFromGravity
 * @param {number} width
 * @param {number} height
 * @param {'user'|'environment'} facing
 */
export function correctImageRoll(landmarks, rollDeg, width, height, facing = 'environment') {
  if (!Array.isArray(landmarks) || !Number.isFinite(rollDeg) || rollDeg === 0) return landmarks;
  const sign = facing === 'user' ? -1 : 1;
  const a = (sign * rollDeg) / DEG;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const cx = width / 2;
  const cy = height / 2;
  return landmarks.map(p => {
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return p;
    const px = p.x * width - cx;
    const py = p.y * height - cy;
    // Visually clockwise rotation in y-down pixel coordinates.
    return { ...p, x: (cx + px * c - py * s) / width, y: (cy + px * s + py * c) / height };
  });
}

/** World landmarks: same correction about the hip-centred origin (x right, y down). */
export function correctWorldRoll(world, rollDeg, facing = 'environment') {
  if (!Array.isArray(world) || !Number.isFinite(rollDeg) || rollDeg === 0) return world;
  const a = ((facing === 'user' ? -1 : 1) * rollDeg) / DEG;
  const c = Math.cos(a);
  const s = Math.sin(a);
  return world.map(p => (p && Number.isFinite(p.x) && Number.isFinite(p.y)
    ? { ...p, x: p.x * c - p.y * s, y: p.x * s + p.y * c }
    : p));
}

/** Tilt below which no levelling prompt is shown. */
export const LEVEL_TOLERANCE_DEG = 3;
/** Forward/back tilt below which the hip-height check is trusted. */
export const PITCH_TOLERANCE_DEG = 8;

/**
 * Listens to devicemotion and keeps a smoothed gravity vector.
 * iOS requires DeviceMotionEvent.requestPermission() from a user gesture.
 */
export class DeviceLevel {
  constructor({ target = globalThis, alpha = 0.2, staleMs = 1500, now = () => performance.now() } = {}) {
    this.target = target;
    this.alpha = alpha;
    this.staleMs = staleMs;
    this.now = now;
    this.g = null;
    this.lastAt = -Infinity;
    this._onMotion = e => this.update(e?.accelerationIncludingGravity);
    this.listening = false;
    this.generation = 0; // bumped by stop(); a start() pending on permission then gives up
  }

  /** Call from a user gesture. Resolves true when motion events may be delivered. */
  async start() {
    const DME = this.target.DeviceMotionEvent;
    if (!DME) return false;
    const gen = this.generation;
    try {
      if (typeof DME.requestPermission === 'function') {
        const res = await DME.requestPermission();
        if (res !== 'granted') return false;
      }
    } catch {
      return false;
    }
    if (gen !== this.generation) return false; // stopped while the permission prompt was open
    if (!this.listening) {
      this.target.addEventListener?.('devicemotion', this._onMotion);
      this.listening = true;
    }
    return true;
  }

  stop() {
    this.generation++;
    if (this.listening) this.target.removeEventListener?.('devicemotion', this._onMotion);
    this.listening = false;
    this.g = null;
  }

  update(raw) {
    if (!raw || ![raw.x, raw.y, raw.z].every(Number.isFinite)) return;
    const a = this.alpha;
    this.g = this.g
      ? { x: this.g.x + a * (raw.x - this.g.x), y: this.g.y + a * (raw.y - this.g.y), z: this.g.z + a * (raw.z - this.g.z) }
      : { x: raw.x, y: raw.y, z: raw.z };
    this.lastAt = this.now();
  }

  /** Current tilt, or null without recent sensor data. */
  tilt(screenAngle = globalThis.screen?.orientation?.angle ?? 0) {
    if (!this.g || this.now() - this.lastAt > this.staleMs) return null;
    return cameraTiltFromGravity(this.g, screenAngle);
  }
}
