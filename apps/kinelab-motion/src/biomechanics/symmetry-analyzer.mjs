/**
 * Bilateral symmetry analysis for paired limbs (knees, hips, elbows, shoulders).
 */

export const PAIRED_JOINTS = Object.freeze([
  { id: 'knee', label: 'Knees', left: 'leftKnee', right: 'rightKnee' },
  { id: 'hip', label: 'Hips', left: 'leftHip', right: 'rightHip' },
  { id: 'elbow', label: 'Elbows', left: 'leftElbow', right: 'rightElbow' },
  { id: 'shoulder', label: 'Shoulders', left: 'leftShoulder', right: 'rightShoulder' },
]);

/**
 * Computes bilateral symmetry metrics from calculated joint angles.
 * @param {Object} angles Object containing left/right joint angle measurements
 * @returns {Object} Symmetry report per paired joint and overall symmetry index
 */
export function analyzeSymmetry(angles) {
  if (!angles) return { pairs: {}, overallAsymmetryDeg: null };

  const pairs = {};
  let totalDiff = 0;
  let count = 0;

  for (const pair of PAIRED_JOINTS) {
    const leftVal = angles[pair.left];
    const rightVal = angles[pair.right];

    if (Number.isFinite(leftVal) && Number.isFinite(rightVal)) {
      const diff = Math.round(Math.abs(leftVal - rightVal) * 10) / 10;
      const avg = (leftVal + rightVal) / 2;
      const percentDiff = avg > 0 ? Math.round((diff / avg) * 100) : 0;

      pairs[pair.id] = {
        label: pair.label,
        left: leftVal,
        right: rightVal,
        diffDeg: diff,
        percentDiff,
        isSymmetric: diff <= 12.0, // within normal biological variance threshold
      };
      totalDiff += diff;
      count++;
    } else {
      pairs[pair.id] = {
        label: pair.label,
        left: leftVal ?? null,
        right: rightVal ?? null,
        diffDeg: null,
        percentDiff: null,
        isSymmetric: null,
      };
    }
  }

  return {
    pairs,
    overallAsymmetryDeg: count > 0 ? Math.round((totalDiff / count) * 10) / 10 : null,
    validPairCount: count,
  };
}
