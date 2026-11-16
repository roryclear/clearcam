import test from 'node:test';
import assert from 'node:assert/strict';
import { angleAt, getJointAngles, createSquatState, advanceSquat, trackingStatus } from '../src/pose-math.mjs';

const point = (x, y, visibility = 0.99) => ({ x, y, visibility, presence: 0.99 });
const blankPose = () => Array.from({ length: 33 }, () => point(0.5, 0.5));

test('angleAt returns 90 degrees at perpendicular vectors', () => {
  assert.equal(angleAt(point(0.3, 0.5), point(0.5, 0.5), point(0.5, 0.7), 1280, 720), 90);
});

test('angleAt honors pixel aspect ratio instead of normalized distortion', () => {
  const measured = angleAt(point(0.4, 0.5), point(0.5, 0.5), point(0.6, 0.4), 1000, 500);
  assert.ok(Math.abs(measured - 153.435) < 0.01);
});

test('angleAt rejects occluded landmarks, invalid dimensions and degenerate joints', () => {
  assert.equal(angleAt(point(.3,.5,.2), point(.5,.5), point(.5,.7), 1280, 720), null);
  assert.equal(angleAt(point(.5,.5), point(.5,.5), point(.6,.6), 1280, 720), null);
  assert.equal(angleAt(point(.3,.5), point(.5,.5), point(.5,.7), 0, 720), null);
});

test('getJointAngles returns independent left and right joint values', () => {
  const pose = blankPose();
  pose[23] = point(.4,.2); pose[25] = point(.4,.4); pose[27] = point(.6,.4);
  pose[24] = point(.6,.2); pose[26] = point(.6,.4); pose[28] = point(.6,.6);
  const angle = getJointAngles(pose, 1000, 1000);
  assert.equal(angle.leftKnee, 90);
  assert.equal(angle.rightKnee, 180);
});

test('squat count increments only after controlled down and return to extension', () => {
  let state = createSquatState();
  for (const [value, timestamp] of [[170,0],[110,100],[107,200],[100,450],[161,800],[164,900]]) {
    state = advanceSquat(state, value, timestamp);
  }
  assert.equal(state.reps, 1);
  assert.equal(state.phase, 'ready');
});

test('squat tracker rejects partial squats and one-frame jitter', () => {
  let state = createSquatState();
  for (const [value, timestamp] of [[171,0],[130,100],[125,200],[162,300],[110,400],[162,500],[169,600]]) {
    state = advanceSquat(state, value, timestamp);
  }
  assert.equal(state.reps, 0);
});

test('missing pose does not create reps or silently mark a posture', () => {
  let state = createSquatState();
  state = advanceSquat(state, null, 100);
  assert.equal(state.reps, 0);
  assert.equal(state.status, 'tracking-lost');
  assert.equal(trackingStatus([]), 'lost');
});


test('losing the body midway through a squat discards an incomplete repetition', () => {
  let state = createSquatState();
  for (const [knee, time] of [[110, 0], [108, 120], [null, 400], [160, 650], [165, 850]]) {
    state = advanceSquat(state, knee, time);
  }
  assert.equal(state.reps, 0);
  assert.equal(state.phase, 'ready');
});
