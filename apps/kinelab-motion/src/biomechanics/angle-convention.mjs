/**
 * Display convention for joint angles.
 *
 * Internally every angle is the interior 3-point angle (knee/elbow 180° = straight,
 * hip = trunk–thigh, shoulder = elbow–shoulder–hip, 0° = arm at the side). Clinical
 * goniometry reports flexion from the straight position, so in 'clinical' mode knee,
 * hip and elbow are shown as 180° − interior. Hip is trunk-referenced (ISB/goniometry
 * reference the pelvis, which MediaPipe hip points cannot see), so it is labelled
 * "trunk–thigh". Exports keep interior angles.
 */

/** Joints whose clinical value is flexion = 180° − interior angle. */
const FLEXION_JOINTS = ['Knee', 'Hip', 'Elbow'];

let current = 'clinical';

/** @param {'clinical'|'interior'} convention */
export function setAngleConvention(convention) {
  current = convention === 'interior' ? 'interior' : 'clinical';
}

export function getAngleConvention() {
  return current;
}

function jointOf(key) {
  const m = /(Knee|Hip|Elbow|Shoulder)$/i.exec(String(key ?? ''));
  return m ? m[1][0].toUpperCase() + m[1].slice(1).toLowerCase() : null;
}

/**
 * @param {string} key angle key ('leftKnee') or joint name ('Knee')
 * @param {number} interiorDeg
 * @param {'clinical'|'interior'} [convention]
 */
export function displayAngle(key, interiorDeg, convention = current) {
  if (!Number.isFinite(interiorDeg)) return null;
  if (convention === 'clinical' && FLEXION_JOINTS.includes(jointOf(key))) return 180 - interiorDeg;
  return interiorDeg;
}

export function formatAngle(key, interiorDeg, convention = current) {
  const v = displayAngle(key, interiorDeg, convention);
  return v === null ? '—' : `${Math.round(v)}°`;
}

/** Row label for an angle definition, e.g. 'LEFT KNEE FLEX' / 'LEFT HIP FLEX (TRUNK–THIGH)'. */
export function angleLabel(def, convention = current) {
  if (convention !== 'clinical') return def.label;
  const joint = jointOf(def.key);
  if (joint === 'Hip') return `${def.label} FLEX (TRUNK–THIGH)`;
  if (FLEXION_JOINTS.includes(joint)) return `${def.label} FLEX`;
  if (joint === 'Shoulder') return `${def.label} ELEV`;
  return def.label;
}
