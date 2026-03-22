"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from .association import *
from ocsort_tracker.STrack import STrack
from collections import defaultdict
from copy import deepcopy
from numpy import zeros
from copy import deepcopy

MAX_STEPS = 300

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))

def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed, speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.
        """

        from .kalmanfilter import KalmanFilterNew as KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.occurrences = defaultdict(float)
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = []
        self.delta_t = delta_t
        self.velocity = np.array((-math.inf, -math.inf))
        self.avg_vel = np.array((0, 0))
        self.speed = 0
        self.obs_ages = []
        self.last_ob = np.array([-1, -1, -1, -1, -1])
        self.last_converted = None

    def update2(self):
        self.kf.history_obs.append(None)
        if self.kf.observed:
            self.kf.attr_saved = deepcopy(self.kf.__dict__)
            self.kf.time_gap = 1
        self.kf.time_gap += 1
        self.kf.observed = False
        self.kf.z = np.array([[None]*self.kf.dim_z]).T
        self.kf.y = zeros((self.kf.dim_z, 1))


    def update(self, bbox, score=None, class_id=None):
        self.occurrences[class_id] += score
        self.class_id = max(self.occurrences, key=self.occurrences.get)
        valid = float(self.last_observation.sum() >= 0)
        candidate_ages = self.age - np.arange(self.delta_t, 0, -1)
        obs_ages = np.fromiter(self.obs_ages, dtype=int) if len(self.observations) > 0 else np.array([], dtype=int)
        obs_boxes = np.stack(list(self.observations)) if len(self.observations) > 0 else np.empty((0, len(bbox)))
        matches = obs_ages[:, None] == candidate_ages[None, :] if obs_ages.size else np.zeros((0, self.delta_t), dtype=bool)
        match_per_col = matches.any(axis=0)
        col_idx = np.argmax(match_per_col)
        row_idx = np.argmax(matches[:, col_idx]) if matches.size else 0
        has_match = float(match_per_col.any())
        all_boxes = np.vstack([self.last_observation[None, :], obs_boxes])
        safe_idx = int(has_match) * (row_idx + 1)
        previous_box = all_boxes[safe_idx]
        dist, velocity = speed_direction(previous_box, bbox)

        self.velocity = valid * velocity + (1.0 - valid) * self.velocity
        self.avg_vel = np.array(self.avg_vel) + valid * (dist / float(self.age))
        self.speed = valid * np.abs(self.avg_vel).sum() + (1.0 - valid) * self.speed
        self.last_observation = bbox
        self.obs_ages.append(self.age)
        self.last_ob = bbox
        self.observations.append(bbox)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        converted = convert_bbox_to_z(bbox)
        self.kf.history_obs.append(converted)
        if not self.kf.observed and self.kf.attr_saved:
            time_gap = self.kf.time_gap
            x1, y1, s1, r1 = self.last_converted
            x2, y2, s2, r2 = converted
            w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
            w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
            t = np.arange(1, MAX_STEPS + 1)
            x = x1 + t * (x2 - x1) / time_gap
            y = y1 + t * (y2 - y1) / time_gap
            w = w1 + t * (w2 - w1) / time_gap
            h = h1 + t * (h2 - h1) / time_gap
            s = w * h
            r = w / h

            boxes = np.stack([x, y, s, r], axis=1).reshape(-1, 4, 1)
            xs = np.zeros((MAX_STEPS, *self.kf.x.shape))
            Ps = np.zeros((MAX_STEPS, *self.kf.P.shape))
            if time_gap <= MAX_STEPS:
                self.kf.__dict__ = self.kf.attr_saved
                self.kf.history_obs.extend(boxes[:MAX_STEPS])
                for i in range(MAX_STEPS):
                    z = boxes[i]
                    self.kf.update(z)
                    xs[i] = self.kf.x
                    Ps[i] = self.kf.P
                    self.kf.x = self.kf.F @ self.kf.x
                    self.kf.P = self.kf._alpha_sq * (self.kf.F @ self.kf.P @ self.kf.F.T) + self.kf.Q

                self.kf.x = xs[time_gap - 1]
                self.kf.P = Ps[time_gap - 1]
                #self.kf.history_obs = self.kf.history_obs[:- (MAX_STEPS - time_gap)]
        self.last_converted = converted
        self.kf.update(converted)

    def predict(self):
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0): self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

class OCSort(object):
    def __init__(self, det_thresh=0.25, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.delta_t = delta_t
        self.asso_func = iou_batch
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

    def update(self, output_results, det_thresh=0.25):
        MAX=300
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]  # x1y1x2y2
        class_ids = output_results[:, 5].astype(int)

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_inds = scores > det_thresh
        dets = dets[remain_inds]
        class_ids = class_ids[remain_inds]
        scores = scores[remain_inds]

        # get predicted locations from existing trackers.

        trackers_pad = [KalmanBoxTracker(bbox=[0,0,0,0])] * MAX
        trackers_pad[:len(self.trackers)] = self.trackers

        vel_pad = [v.velocity for v in trackers_pad]
        vel_pad = np.array(vel_pad)

        trks = np.zeros((MAX, 5))
        ret = []
        for t, trk in enumerate(trks):
            pos = trackers_pad[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

        trks[trks[:, 0] == 0] = 0

        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        
        k_observations = np.array([trk.last_ob for trk in trackers_pad])


        dets_pad = np.zeros((MAX,5), dtype=dets.dtype)
        
        dets_pad[:dets.shape[0]] = dets

        matched, unmatched_dets, unmatched_trks = associate(dets_pad, trks, self.iou_threshold, vel_pad, k_observations, self.inertia)
        for m in matched: self.trackers[m[1]].update(bbox=dets[m[0], :], score=scores[m[0]], class_id=class_ids[m[0]])

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], scores[det_ind], class_ids[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks: self.trackers[m].update2()

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            trk.class_id = class_ids[i]
            trk.score = scores[i]
            trk.occurrences[class_ids[i]] += 1
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1, trk.age, trk.class_id, trk.score, trk.speed])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                if trk.speed > 2 or trk.time_since_update > 600: self.trackers.pop(i)

        out = []
        for x in ret: out.append(STrack(tlwh=[x[0][0], x[0][1], (x[0][2] - x[0][0]), (x[0][3] - x[0][1])], score=x[0][7], class_id=x[0][6], track_id=x[0][4], age=x[0][5], speed=x[0][8]))
        return out

