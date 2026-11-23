import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { SquatExercise } from '../src/exercises/squat-exercise.mjs';

// See docs/validation.md for the dataset, checksum and joint convention.
const root = process.argv[2];
if (!root) {
  console.error('usage: node bench/uiprmd-squat-probe.mjs <Segmented Movements/Kinect>');
  process.exit(1);
}

function readRows(path) {
  return readFileSync(path, 'utf8').trim().split(/\r?\n/)
    .map(line => line.split(',').map(field => field.trim() ? Number(field) : NaN));
}

const vector = (row, joint) => row.slice(3 * (joint - 1), 3 * joint);
function rotate([x, y, z], angles) {
  const [rx, ry, rz] = angles.map(degrees => degrees * Math.PI / 180);
  const rotatedY = Math.cos(rx) * y - Math.sin(rx) * z;
  const rotatedZ = Math.sin(rx) * y + Math.cos(rx) * z;
  const rotatedX = Math.cos(ry) * x + Math.sin(ry) * rotatedZ;
  return [
    Math.cos(rz) * rotatedX - Math.sin(rz) * rotatedY,
    Math.sin(rz) * rotatedX + Math.cos(rz) * rotatedY,
    -Math.sin(ry) * x + Math.cos(ry) * rotatedZ,
  ];
}

// Compute both vectors in the hip joint's frame. Common ancestor rotations
// cancel, so only the knee's local rotation is needed for this interior angle.
function kneeAngle(positions, rotations, joint) {
  const thigh = vector(positions, joint).map(value => -value);
  const shin = rotate(vector(positions, joint + 1), vector(rotations, joint));
  const norm = Math.hypot(...thigh) * Math.hypot(...shin);
  if (norm === 0) return null;
  const cosine = thigh.reduce((sum, value, i) => sum + value * shin[i], 0) / norm;
  return Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI;
}

function movingAverage(samples, width) {
  return samples.map((row, i) => {
    const first = Math.max(0, i - Math.floor((width - 1) / 2));
    const last = Math.min(samples.length, i + 1 + Math.floor(width / 2));
    return row.map((_, j) => {
      let total = 0;
      for (let k = first; k < last; k++) total += samples[k][j];
      return total / (last - first);
    });
  });
}
const exercise = new SquatExercise();
const clips = [];
for (let subject = 1; subject <= 10; subject++) {
  for (let rep = 1; rep <= 10; rep++) {
    const id = `m01_s${String(subject).padStart(2, '0')}_e${String(rep).padStart(2, '0')}`;
    const positions = readRows(join(root, 'Positions', `${id}_positions.txt`));
    const rotations = readRows(join(root, 'Angles', `${id}_angles.txt`));
    if (positions.length !== rotations.length || [...positions, ...rotations].some(row => row.length !== 66 || !row.every(Number.isFinite))) {
      throw new Error(`Invalid paired input for ${id}`);
    }
    clips.push({ id, positions, rotations });
  }
}
for (const [fps, posWindow, angleWindow] of [[30, 1, 1], [60, 1, 1], [100, 1, 1], [120, 1, 1], [30, 5, 20]]) {
  const counts = {};
  const phases = {};
  const misses = [];
  for (const clip of clips) {
    const positions = posWindow === 1 ? clip.positions : movingAverage(clip.positions, posWindow);
    const rotations = angleWindow === 1 ? clip.rotations : movingAverage(clip.rotations, angleWindow);
    let state = exercise.createState();
    const sequence = [state.phase];
    for (let i = 0; i < positions.length; i++) {
      const angles = {
        leftKnee: kneeAngle(positions[i], rotations[i], 16),
        rightKnee: kneeAngle(positions[i], rotations[i], 20),
      };
      state = exercise.update(state, { angles, landmarks: null, timestampMs: i * 1000 / fps, trackingQuality: 'good' });
      if (sequence.at(-1) !== state.phase) sequence.push(state.phase);
    }
    counts[state.reps] = (counts[state.reps] || 0) + 1;
    const key = sequence.join('>');
    phases[key] = (phases[key] || 0) + 1;
    if (state.reps !== 1) misses.push(clip.id);
  }
  console.log(JSON.stringify({ fps, posWindow, angleWindow, counts, phases, misses }));
}
