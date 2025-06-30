from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.tensor import Tensor
from tinygrad.device import is_dtype_supported
from tinygrad import dtypes
import numpy as np
from itertools import chain
from pathlib import Path
import cv2
from collections import defaultdict
import time, sys
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
import json
from tinygrad import TinyJit
import lap
from collections import OrderedDict
import pickle


class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.  
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor = np.linalg.cholesky(projected_cov)
        y = np.linalg.solve(chol_factor, np.dot(covariance, self._update_mat.T).T)
        kalman_gain = np.linalg.solve(chol_factor.T, y).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

class TrackState(object):
    New = 1
    Tracked = 2
    Lost = 3
    Removed = 4
    Replaced = 5

def tlbr_np(values, mean):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`."""
    if mean is None:
        ret = values[:4].copy()
    else:
        ret = mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
    ret[2:] += ret[:2]
    return ret

def tlbr_np_batch(tracked_stracks_values, tracked_stracks_means):
    ret = np.empty((len(tracked_stracks_values), 4))
    use_mean = np.array([m is not None for m in tracked_stracks_means], dtype=bool)
    if np.any(use_mean):
        means = np.array([m for m in tracked_stracks_means if m is not None])
        ret[use_mean, :] = means[:, :4].copy()
        ret[use_mean, 2] *= ret[use_mean, 3]
        ret[use_mean, :2] -= ret[use_mean, 2:] / 2
    if not np.all(use_mean):
        ret[~use_mean, :] = tracked_stracks_values[~use_mean, :4].copy()
    ret[:, 2:] += ret[:, :2]
    return ret

def tlwh_np(values,mean):
    """Get current position in bounding box format `(top left x, top left y,
            width, height)`.
    """
    if mean is None:
        return values[:4].copy()
    ret = mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

def bbox_ious(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    
    boxes_area = (
        (boxes[:, 2] - boxes[:, 0] + 1) *
        (boxes[:, 3] - boxes[:, 1] + 1)
    ).reshape(N, 1)
    
    query_areas = (
        (query_boxes[:, 2] - query_boxes[:, 0] + 1) *
        (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    ).reshape(1, K)
    
    ixmin = np.maximum(boxes[:, 0].reshape(N, 1), query_boxes[:, 0].reshape(1, K))
    iymin = np.maximum(boxes[:, 1].reshape(N, 1), query_boxes[:, 1].reshape(1, K))
    ixmax = np.minimum(boxes[:, 2].reshape(N, 1), query_boxes[:, 2].reshape(1, K))
    iymax = np.minimum(boxes[:, 3].reshape(N, 1), query_boxes[:, 3].reshape(1, K))
    
    iw = np.maximum(ixmax - ixmin + 1, 0)
    ih = np.maximum(iymax - iymin + 1, 0)
    intersection = iw * ih
    
    union = boxes_area + query_areas - intersection
    
    overlaps = np.where(
        (iw > 0) & (ih > 0),
        intersection / union,
        np.zeros_like(intersection)
    ) 
    return overlaps

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self._count = 0

        self.lost_stracks_ids_tg = Tensor.empty()
        self.lost_stracks_fids_tg = Tensor.empty()
        self.lost_stracks_startframes_tg = Tensor.empty()
        self.lost_stracks_states_tg = Tensor.empty()
        self.lost_stracks_bools_tg = Tensor.empty()
        self.lost_stracks_values_tg = Tensor.empty()
        self.lost_stracks_means_tg = Tensor.empty()
        self.lost_stracks_covs_tg = Tensor.empty()

        self.tracked_stracks_values = []
        self.lost_stracks_values = []

        self.tracked_stracks_means = []
        self.tracked_stracks_bools = []
        self.tracked_stracks_covs = []
        self.tracked_stracks_ids = []
        self.tracked_stracks_fids = []
        self.tracked_stracks_startframes = []
        self.tracked_stracks_states = []
        self.lost_stracks_means = []
        self.lost_stracks_bools = []
        self.lost_stracks_covs = []
        self.lost_stracks_ids = []
        self.lost_stracks_fids = []
        self.lost_stracks_startframes = []
        self.lost_stracks_states = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_stracks_values = []
        activated_stracks_means = []
        activated_stracks_bools = []
        activated_stracks_covs = []
        activated_stracks_ids = []
        activated_stracks_fids = []
        activated_stracks_startframes = []
        activated_stracks_states = []
        refind_stracks_values = []
        refind_stracks_means = []
        refind_stracks_bools = []
        refind_stracks_covs = []
        refind_stracks_ids = []
        refind_stracks_fids = []
        refind_stracks_startframes = []
        refind_stracks_states = []
        lost_stracks_means = []
        lost_stracks_bools = []
        lost_stracks_covs = []
        lost_stracks_values = []
        lost_stracks_startframes = []
        lost_stracks_states = []
        lost_stracks_ids = []
        lost_stracks_fids = []
        removed_stracks_ids = []

        classes = output_results[:, 5]
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes = bboxes / scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = inds_low & inds_high

        dets = bboxes * remain_inds.unsqueeze(1)
        dets_second = bboxes * inds_second.unsqueeze(1)
        dets[:, 2:] -= dets[:, :2] #tlbr to tlwh
        dets_second[:, 2:] -= dets_second[:, :2]
        dets_score_classes = dets.cat(scores.reshape(-1,1), dim=1).cat(classes.reshape(-1,1), dim=1)
        dets_score_classes = dets_score_classes.numpy()
        dets_score_classes_second = dets_second.cat(scores.reshape(-1,1), dim=1).cat(classes.reshape(-1,1), dim=1)
        dets_score_classes_second = dets_score_classes_second.numpy()
        detections_means = [None for _ in dets_score_classes]
        detections_bools = [False for _ in dets_score_classes_second]
        detections_cov = [None for _ in dets_score_classes]
        detections_ids = [None for _ in dets_score_classes]
        detections_fids = [None for _ in dets_score_classes]
        detections_startframes = [0 for _ in dets_score_classes]
        detections_states = [TrackState.New for _ in dets_score_classes]
        
        tracked_stracks_means = []
        tracked_stracks_covs = []
        
        mask = np.array(self.tracked_stracks_bools).astype(bool)
        mask_tg = Tensor(mask)

        for i in range(len(self.tracked_stracks_ids)):
            mean = self.tracked_stracks_means[i]
            cov = self.tracked_stracks_covs[i]  
            if self.tracked_stracks_bools[i]:
                tracked_stracks_means.append(mean)
                tracked_stracks_covs.append(cov)

        self.tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids)
        self.tracked_stracks_fids_tg = Tensor(self.tracked_stracks_fids)
        self.tracked_stracks_bools_tg = Tensor(self.tracked_stracks_bools)
        self.tracked_stracks_startframes_tg = Tensor(self.tracked_stracks_startframes)
        self.tracked_stracks_states_tg = Tensor(self.tracked_stracks_states)
        self.tracked_stracks_values_tg = Tensor(self.tracked_stracks_values)
        self.tracked_stracks_covs_tg = Tensor(self.tracked_stracks_covs)
        self.tracked_stracks_means_tg = Tensor(self.tracked_stracks_means)

        tracked_stracks_ids_tg = (self.tracked_stracks_ids_tg * mask_tg)
        tracked_stracks_fids_tg = (self.tracked_stracks_fids_tg * mask_tg)
        tracked_stracks_bools_tg = (self.tracked_stracks_bools_tg * mask_tg)
        tracked_stracks_startframes_tg = (self.tracked_stracks_startframes_tg * mask_tg)
        tracked_stracks_states_tg = (self.tracked_stracks_states_tg * mask_tg)
        tracked_stracks_values_tg = self.tracked_stracks_values_tg * mask_tg.view(-1,1)

        unconfirmed_ids_tg = self.tracked_stracks_ids_tg * ~mask_tg
        unconfirmed_values_tg = self.tracked_stracks_values_tg * ~mask_tg.view(-1,1)
        unconfirmed_bools_tg = self.tracked_stracks_bools_tg * ~mask_tg
        unconfirmed_states_tg = self.tracked_stracks_states_tg * ~mask_tg
        unconfirmed_fids_tg = self.tracked_stracks_fids_tg * ~mask_tg
        unconfirmed_startframes_tg = self.tracked_stracks_startframes_tg * ~mask_tg
        unconfirmed_covs_tg = self.tracked_stracks_covs_tg * ~mask_tg.view(-1,1,1)
        unconfirmed_means_tg = self.tracked_stracks_means_tg * ~mask_tg.view(-1,1)

        if len(self.lost_stracks_values_tg.shape) > 1 and self.lost_stracks_values_tg.shape[0] > 0:
            tracked_stracks_ids_tg = tracked_stracks_ids_tg.cat(self.lost_stracks_ids_tg)
            tracked_stracks_fids_tg = tracked_stracks_fids_tg.cat(self.lost_stracks_fids_tg)
            tracked_stracks_bools_tg = tracked_stracks_bools_tg.cat(self.lost_stracks_bools_tg)
            tracked_stracks_startframes_tg = tracked_stracks_startframes_tg.cat(self.lost_stracks_startframes_tg)
            tracked_stracks_states_tg = tracked_stracks_states_tg.cat(self.lost_stracks_states_tg)
            tracked_stracks_values_tg = tracked_stracks_values_tg.cat(self.lost_stracks_values_tg)

        tracked_stracks_ids = tracked_stracks_ids_tg.numpy() 
        id_mask = tracked_stracks_ids != 0
        tracked_stracks_ids = tracked_stracks_ids[id_mask].tolist()
        tracked_stracks_fids = tracked_stracks_fids_tg.numpy()
        tracked_stracks_fids = tracked_stracks_fids[id_mask].tolist()
        tracked_stracks_bools = tracked_stracks_bools_tg.numpy()
        tracked_stracks_bools = tracked_stracks_bools[id_mask].tolist()
        tracked_stracks_startframes = tracked_stracks_startframes_tg.numpy()
        tracked_stracks_startframes = tracked_stracks_startframes[id_mask].tolist()
        tracked_stracks_states = tracked_stracks_states_tg.numpy()
        tracked_stracks_states = tracked_stracks_states[id_mask].tolist()
        tracked_stracks_values = tracked_stracks_values_tg.numpy()
        tracked_stracks_values = tracked_stracks_values[id_mask].tolist()

        unconfirmed_ids = unconfirmed_ids_tg.numpy()
        id_mask = unconfirmed_ids != 0
        unconfirmed_ids = unconfirmed_ids[id_mask].tolist()
        unconfirmed_values = unconfirmed_values_tg.numpy()
        unconfirmed_values = unconfirmed_values[id_mask].tolist()
        unconfirmed_bools = unconfirmed_bools_tg.numpy()
        unconfirmed_bools = unconfirmed_bools[id_mask].tolist()
        unconfirmed_states = unconfirmed_states_tg.numpy()
        unconfirmed_states = unconfirmed_states[id_mask].tolist()
        unconfirmed_covs = unconfirmed_covs_tg.numpy()
        unconfirmed_covs = unconfirmed_covs[id_mask].tolist()
        unconfirmed_means = unconfirmed_means_tg.numpy()
        unconfirmed_means = unconfirmed_means[id_mask].tolist()
        unconfirmed_fids = unconfirmed_fids_tg.numpy()
        unconfirmed_fids = unconfirmed_fids[id_mask].tolist()
        unconfirmed_startframes = unconfirmed_startframes_tg.numpy()
        unconfirmed_startframes = unconfirmed_startframes[id_mask].tolist()
        tracked_stracks_means = tracked_stracks_means + list(self.lost_stracks_means)
        tracked_stracks_covs = tracked_stracks_covs + list(self.lost_stracks_covs)

        # Predict the current location with KF
        if len(tracked_stracks_ids) > 0:
            multi_mean = np.asarray([st for st in tracked_stracks_means])
            multi_covariance = np.asarray([st for st in tracked_stracks_covs])
            for i in range(len(tracked_stracks_ids)):
                if tracked_stracks_states[i] != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = self.kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i in range(len(tracked_stracks_ids)):
                tracked_stracks_means[i][:] = multi_mean[i].astype(np.float32)
                tracked_stracks_covs[i][:] = multi_covariance[i]
            
            for i, t in enumerate(self.tracked_stracks_covs):
              for j, p in enumerate(tracked_stracks_covs):
                  if t is p:
                      self.tracked_stracks_covs[i] = tracked_stracks_covs[j]

        atlbrs = tlbr_np_batch(tracked_stracks_values, tracked_stracks_means)
        btlbrs = tlbr_np_batch(dets_score_classes,detections_means)
        dists = iou_distance(atlbrs, btlbrs)
        dists = fuse_score(dists, dets_score_classes)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)

        det_values_arr = [dets_score_classes[i] for _, i in matches]

        for idx, (itracked, idet) in enumerate(matches):
            det_xyah = tlwh_to_xyah(tlwh_np(det_values_arr[idx], detections_means[idx]))
            x, y = self.kalman_filter.update(tracked_stracks_means[itracked], tracked_stracks_covs[itracked], det_xyah)
            tracked_stracks_covs[itracked][:] = y
            tracked_stracks_means[itracked][:] = x
            tracked_stracks_fids[itracked] = self.frame_id
            if itracked < len(self.tracked_stracks_fids): self.tracked_stracks_fids[itracked] = self.frame_id

            if tracked_stracks_states[itracked] == TrackState.Tracked:
                activated_stracks_values.append(tracked_stracks_values[itracked])
                activated_stracks_means.append(tracked_stracks_means[itracked])
                activated_stracks_bools.append(tracked_stracks_bools[itracked])
                activated_stracks_covs.append(tracked_stracks_covs[itracked])
                activated_stracks_ids.append(tracked_stracks_ids[itracked])
                activated_stracks_fids.append(tracked_stracks_fids[itracked])
                activated_stracks_startframes.append(tracked_stracks_startframes[itracked])
                activated_stracks_states.append(tracked_stracks_states[itracked])
            else:
                tracked_stracks_states[itracked] = TrackState.Tracked
                tracked_stracks_bools[itracked] = True
                refind_stracks_values.append(tracked_stracks_values[itracked])
                refind_stracks_means.append(tracked_stracks_means[itracked])
                refind_stracks_bools.append(True)
                refind_stracks_covs.append(tracked_stracks_covs[itracked])
                refind_stracks_ids.append(tracked_stracks_ids[itracked])
                refind_stracks_fids.append(tracked_stracks_fids[itracked])
                refind_stracks_startframes.append(tracked_stracks_startframes[itracked])
                refind_stracks_states.append(tracked_stracks_states[itracked])

        r_tracked_stracks_values = []
        r_tracked_stracks_means = []
        r_tracked_stracks_bools = []
        r_tracked_stracks_covs = []
        r_tracked_stracks_ids = []
        r_tracked_stracks_fids = []
        r_tracked_stracks_startframes = []
        r_tracked_stracks_states = []
        
        for i in range(len(u_track)):
            if tracked_stracks_states[u_track[i]] == TrackState.Tracked:
                r_tracked_stracks_values.append(tracked_stracks_values[u_track[i]])
                r_tracked_stracks_means.append(tracked_stracks_means[u_track[i]])
                r_tracked_stracks_bools.append(tracked_stracks_bools[u_track[i]])
                r_tracked_stracks_covs.append(tracked_stracks_covs[u_track[i]])
                r_tracked_stracks_ids.append(tracked_stracks_ids[u_track[i]])
                r_tracked_stracks_fids.append(tracked_stracks_fids[u_track[i]])
                r_tracked_stracks_startframes.append(tracked_stracks_startframes[u_track[i]])
                r_tracked_stracks_states.append(tracked_stracks_states[u_track[i]])
        
        det_values = dets_score_classes_second
        det_means = [None] * len(det_values)  # all means are None initially

        atlbrs = tlbr_np_batch(r_tracked_stracks_values, r_tracked_stracks_means)
        btlbrs = tlbr_np_batch(det_values, det_means)
        dists = iou_distance(atlbrs, btlbrs)

        matches, u_track, _ = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            mean = r_tracked_stracks_means[itracked]
            bool_val = r_tracked_stracks_bools[itracked]
            cov = r_tracked_stracks_covs[itracked]
            startframe = r_tracked_stracks_startframes[itracked]
            state = r_tracked_stracks_states[itracked]
            id_val = r_tracked_stracks_ids[itracked]
            fid = r_tracked_stracks_fids[itracked]
            values = r_tracked_stracks_values[itracked]
            t_val = r_tracked_stracks_values[itracked]
            d_val = det_values[idet]
            d_mean = det_means[idet]

            xyah = tlwh_to_xyah(tlwh_np(d_val, d_mean))
            r_tracked_stracks_means[itracked][:], x = self.kalman_filter.update(mean, cov, xyah)
            r_tracked_stracks_covs[itracked][:] = x
            t_val = list(t_val)
            t_val[4] = d_val[4]  # Update score
            r_tracked_stracks_values[itracked] = tuple(t_val)
            r_tracked_stracks_fids[itracked] = self.frame_id

            # Update tracked_stracks attributes if track exists in tracked_stracks
            for i, t in enumerate(self.tracked_stracks_ids):
                if t is id_val:
                    self.tracked_stracks_fids[i] = self.frame_id
                    self.tracked_stracks_values[i] = tuple(t_val)
                    self.tracked_stracks_means[i] = r_tracked_stracks_means[itracked]
                    self.tracked_stracks_covs[i] = r_tracked_stracks_covs[itracked]
                    self.tracked_stracks_states[i] = TrackState.Tracked
                    break

            if state == TrackState.Tracked:
                activated_stracks_values.append(r_tracked_stracks_values[itracked])
                activated_stracks_means.append(r_tracked_stracks_means[itracked])
                activated_stracks_bools.append(bool_val)
                activated_stracks_covs.append(r_tracked_stracks_covs[itracked])
                activated_stracks_ids.append(id_val)
                activated_stracks_fids.append(self.frame_id)
                activated_stracks_startframes.append(startframe)
                activated_stracks_states.append(state)
            else:
                r_tracked_stracks_bools[itracked] = True
                r_tracked_stracks_states[itracked] = TrackState.Tracked
                refind_stracks_values.append(r_tracked_stracks_values[itracked])
                refind_stracks_means.append(r_tracked_stracks_means[itracked])
                refind_stracks_bools.append(True)
                refind_stracks_covs.append(r_tracked_stracks_covs[itracked])
                refind_stracks_ids.append(id_val)
                refind_stracks_fids.append(self.frame_id)
                refind_stracks_startframes.append(startframe)
                refind_stracks_states.append(state)
        
        for i in range(len(u_track)):
            values = r_tracked_stracks_values[u_track[i]]
            mean = r_tracked_stracks_means[u_track[i]]
            bool_val = r_tracked_stracks_bools[u_track[i]]
            cov = r_tracked_stracks_covs[u_track[i]]
            id_val = r_tracked_stracks_ids[u_track[i]]
            fid = r_tracked_stracks_fids[u_track[i]]
            startframe = r_tracked_stracks_startframes[u_track[i]]
            state = r_tracked_stracks_states[u_track[i]]
            if state != TrackState.Lost:
                for j, t in enumerate(self.tracked_stracks_ids):
                    if t is id_val:
                        self.tracked_stracks_states[j] = TrackState.Lost
                        break
                lost_stracks_values.append(values)
                lost_stracks_means.append(mean)
                lost_stracks_bools.append(bool_val)
                lost_stracks_covs.append(cov)
                lost_stracks_ids.append(id_val)
                lost_stracks_fids.append(fid)
                lost_stracks_startframes.append(startframe)
                lost_stracks_states.append(TrackState.Lost)
                
        u_detection_np = np.array(u_detection)
        detections_ids = np.array(detections_ids)[u_detection_np]
        detections_fids = np.array(detections_fids)[u_detection_np]
        detections_bools = np.array(detections_bools)[u_detection_np]
        detections_means = np.array(detections_means)[u_detection_np]
        detections_cov = np.array(detections_cov)[u_detection_np]
        detections_startframes = np.array(detections_startframes)[u_detection_np]
        detections_states = np.array(detections_states)[u_detection_np]
        dets_score_classes_second = np.array(dets_score_classes)[u_detection_np]
        
        atlbrs = tlbr_np_batch(unconfirmed_values, unconfirmed_means)
        btlbrs = tlbr_np_batch(dets_score_classes_second, detections_means)
        dists = iou_distance(atlbrs, btlbrs)
        dists = fuse_score(dists, dets_score_classes_second)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        updated_means = []
        tracks_values = []
        updated_bools = []
        updated_covs = []
        updated_ids = []
        updated_fids = []
        updated_startframes = []
        updated_states = []

        if len(matches) > 0:
            matches_arr = np.array(matches)
            itracked_arr = matches_arr[:, 0]
            idet_arr = matches_arr[:, 1]
            ids = [unconfirmed_ids[i] for i in itracked_arr]
            fids = [unconfirmed_fids[i] for i in itracked_arr]
            startframes = [unconfirmed_startframes[i] for i in itracked_arr]
            states = [unconfirmed_states[i] for i in itracked_arr]
            updated_bools = [unconfirmed_bools[i] for i in itracked_arr]
            means = [unconfirmed_means[i] for i in itracked_arr]
            tracks_values = [unconfirmed_values[i] for i in itracked_arr]


            covariances = [unconfirmed_covs[i] for i in itracked_arr]
            det_values = dets_score_classes_second[idet_arr]
            tlwhs = det_values[:, :4]
            scores = det_values[:, 4]

            for mean, cov, tlwh, id, sf, fid, state in zip(means, covariances, tlwhs, ids, startframes, fids, states):
                new_mean, new_cov = self.kalman_filter.update(mean, cov, tlwh_to_xyah(tlwh))
                updated_means.append(new_mean)
                updated_covs.append(new_cov)
                updated_ids.append(id)
                updated_fids.append(fid)
                updated_startframes.append(sf)
                updated_states.append(state)

            updated_scores = scores
            frame_id_val = self.frame_id
            for i, (mean, cov, score, values, fids, id) in enumerate(zip(updated_means, updated_covs, updated_scores, tracks_values, updated_fids, updated_ids)):
                covariances[i][:] = cov
                values = list(values)
                values[4] = score
                values = tuple(values)
                updated_fids[i] = frame_id_val
                updated_states[i] = TrackState.Tracked

                if id in self.tracked_stracks_ids:
                  idx = self.tracked_stracks_ids.index(id)
                  self.tracked_stracks_bools[idx] = True
                  self.tracked_stracks_states[idx] = TrackState.Tracked

        activated_stracks_bools.extend(updated_bools)
        activated_stracks_values.extend(tracks_values)
        activated_stracks_means.extend(updated_means)
        activated_stracks_covs.extend(updated_covs)
        activated_stracks_ids.extend(updated_ids)
        activated_stracks_fids.extend(updated_fids)
        activated_stracks_startframes.extend(updated_startframes)
        activated_stracks_states.extend(updated_states)

        u_unconfirmed_np = np.asarray(u_unconfirmed)
        ids = np.fromiter((unconfirmed_ids[key] for key in u_unconfirmed_np), dtype=object)
        if ids.size > 0:
            removed_stracks_ids.extend(ids.tolist())

        u_detection = np.asarray(u_detection)
        track_scores = dets_score_classes_second[u_detection, 4]  # Direct score access
        valid_mask = track_scores >= self.det_thresh
        valid_indices = u_detection[valid_mask].tolist()  # Convert to list of integers

        # Get tracks using proper list indexing
        
        valid_values = dets_score_classes_second[valid_indices]  # Get corresponding values
        valid_means = [detections_means[i] for i in valid_indices]
        valid_bools = [detections_bools[i] for i in valid_indices]
        valid_covs = [detections_cov[i] for i in valid_indices]
        valid_ids = [detections_ids[i] for i in valid_indices]
        valid_fids = [detections_fids[i] for i in valid_indices]
        valid_startframes = [detections_startframes[i] for i in valid_indices]
        valid_states = [detections_states[i] for i in valid_indices]

        for i, (vals, mean) in enumerate(zip(valid_values, valid_means)):
            y = self._count = self._count + 1
            valid_ids[i] = y
            valid_means[i], x = self.kalman_filter.initiate(
                tlwh_to_xyah(vals[:4]))
            valid_covs[i] = x
            if self.frame_id == 1:
                valid_bools[i] = True
            valid_fids[i] = self.frame_id

        activated_stracks_means.extend(valid_means)
        activated_stracks_values.extend(valid_values)
        activated_stracks_bools.extend(valid_bools)
        activated_stracks_covs.extend(valid_covs)
        activated_stracks_ids.extend(valid_ids)
        activated_stracks_fids.extend(valid_fids)
        activated_stracks_startframes.extend(valid_startframes)
        activated_stracks_states.extend(valid_states)
        remove_mask = (self.frame_id - np.array(self.lost_stracks_fids)) > self.max_time_lost
        for t in np.array(self.lost_stracks_states)[remove_mask]: t = TrackState.Removed
        self.lost_stracks_means = np.array(self.lost_stracks_means)[~remove_mask]
        self.lost_stracks_bools = np.array(self.lost_stracks_bools)[~remove_mask]
        self.lost_stracks_values = (np.array(self.lost_stracks_values)[~remove_mask])
        self.lost_stracks_covs = np.array(self.lost_stracks_covs)[~remove_mask]
        self.lost_stracks_ids = np.array(self.lost_stracks_ids)[~remove_mask]
        self.lost_stracks_fids = np.array(self.lost_stracks_fids)[~remove_mask]
        self.lost_stracks_startframes = np.array(self.lost_stracks_startframes)[~remove_mask]
        self.lost_stracks_states = np.array(self.lost_stracks_states)[~remove_mask]

        self.tracked_stracks_states_tg = Tensor(self.tracked_stracks_states)
        mask_tg = self.tracked_stracks_states_tg == TrackState.Tracked
        self.tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids)
        self.tracked_stracks_ids_tg *= mask_tg
        self.tracked_stracks_ids = self.tracked_stracks_ids_tg.numpy()
        mask = self.tracked_stracks_ids != 0

        self.tracked_stracks_values = np.array(self.tracked_stracks_values)[mask]
        self.tracked_stracks_means = np.array(self.tracked_stracks_means)[mask]
        self.tracked_stracks_bools = np.array(self.tracked_stracks_bools)[mask]
        self.tracked_stracks_covs = np.array(self.tracked_stracks_covs)[mask]
        self.tracked_stracks_fids = np.array(self.tracked_stracks_fids)[mask]
        self.tracked_stracks_ids = np.array(self.tracked_stracks_ids)[mask]
        self.tracked_stracks_startframes = np.array(self.tracked_stracks_startframes)[mask]
        self.tracked_stracks_states = np.array(self.tracked_stracks_states)[mask]
            
        keep_activated = np.where(~np.isin(activated_stracks_ids, self.tracked_stracks_ids))[0]
        keep_activated_tg = Tensor(keep_activated)

        self.tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids,dtype=dtypes.int)
        activated_stracks_ids_tg = Tensor(activated_stracks_ids,dtype=dtypes.int)
        self.tracked_stracks_fids_tg = Tensor(self.tracked_stracks_fids,dtype=dtypes.int)
        activated_stracks_fids_tg = Tensor(activated_stracks_fids,dtype=dtypes.int)
        self.tracked_stracks_states_tg = Tensor(self.tracked_stracks_states,dtype=dtypes.int)
        activated_stracks_states_tg = Tensor(activated_stracks_states,dtype=dtypes.int)
        self.tracked_stracks_startframes_tg = Tensor(self.tracked_stracks_startframes,dtype=dtypes.int)
        activated_stracks_startframes_tg = Tensor(activated_stracks_startframes)
        self.tracked_stracks_bools_tg = Tensor(self.tracked_stracks_bools,dtype=dtypes.bool)
        activated_stracks_bools_tg = Tensor(activated_stracks_bools,dtype=dtypes.bool)
        self.tracked_stracks_values_tg = Tensor(self.tracked_stracks_values,dtype=dtypes.float32)
        activated_stracks_values_tg = Tensor(activated_stracks_values,dtype=dtypes.float32)
        self.tracked_stracks_means_tg = Tensor(self.tracked_stracks_means,dtype=dtypes.float32)
        activated_stracks_means_tg = Tensor(activated_stracks_means,dtype=dtypes.float32)
        self.tracked_stracks_covs_tg = Tensor(self.tracked_stracks_covs,dtype=dtypes.float32)
        activated_stracks_covs_tg = Tensor(activated_stracks_covs,dtype=dtypes.float32)


        self.tracked_stracks_ids_tg = self.tracked_stracks_ids_tg.cat(activated_stracks_ids_tg[keep_activated_tg])
        self.tracked_stracks_fids_tg = self.tracked_stracks_fids_tg.cat(activated_stracks_fids_tg[keep_activated_tg])
        self.tracked_stracks_states_tg = self.tracked_stracks_states_tg.cat(activated_stracks_states_tg[keep_activated_tg])
        self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg.cat(activated_stracks_startframes_tg[keep_activated_tg])
        self.tracked_stracks_bools_tg = self.tracked_stracks_bools_tg.cat(activated_stracks_bools_tg[keep_activated_tg])
        if self.tracked_stracks_values_tg.shape[0] > 0:
            self.tracked_stracks_values_tg = self.tracked_stracks_values_tg.cat(activated_stracks_values_tg[keep_activated_tg])
            self.tracked_stracks_means_tg = self.tracked_stracks_means_tg.cat(activated_stracks_means_tg[keep_activated_tg])
            self.tracked_stracks_covs_tg = self.tracked_stracks_covs_tg.cat(activated_stracks_covs_tg[keep_activated_tg])
        else:
           self.tracked_stracks_values_tg = activated_stracks_values_tg[keep_activated_tg]
           self.tracked_stracks_means_tg = activated_stracks_means_tg[keep_activated_tg]
           self.tracked_stracks_covs_tg = activated_stracks_covs_tg[keep_activated_tg]

        self.tracked_stracks_ids = self.tracked_stracks_ids_tg.numpy()
        self.tracked_stracks_covs = self.tracked_stracks_covs_tg.numpy()

        refind_stracks_fids_tg = Tensor(refind_stracks_fids)
        refind_stracks_bools_tg = Tensor(refind_stracks_bools)
        refind_stracks_states_tg = Tensor(refind_stracks_states,dtype=dtypes.int)
        refind_stracks_startframes_tg = Tensor(refind_stracks_startframes)
        refind_stracks_values_tg = Tensor(refind_stracks_values)
        refind_stracks_means_tg = Tensor(refind_stracks_means)
        refind_stracks_covs_tg = Tensor(refind_stracks_covs)

        self.tracked_stracks_fids_tg = self.tracked_stracks_fids_tg.cat(refind_stracks_fids_tg)
        self.tracked_stracks_bools_tg = self.tracked_stracks_bools_tg.cat(refind_stracks_bools_tg)
        self.tracked_stracks_states_tg = self.tracked_stracks_states_tg.cat(refind_stracks_states_tg)
        self.tracked_stracks_startframes_tg = self.tracked_stracks_startframes_tg.cat(refind_stracks_startframes_tg)
        if refind_stracks_means_tg.shape[0] > 0:
            self.tracked_stracks_means_tg = self.tracked_stracks_means_tg.cat(refind_stracks_means_tg)
            self.tracked_stracks_values_tg = self.tracked_stracks_values_tg.cat(refind_stracks_values_tg)
            self.tracked_stracks_covs_tg = self.tracked_stracks_covs_tg.cat(refind_stracks_covs_tg)
      
        self.tracked_stracks_ids = np.array(self.tracked_stracks_ids).tolist() + np.array(refind_stracks_ids).tolist()

        self.tracked_stracks_fids = self.tracked_stracks_fids_tg.numpy()
        self.tracked_stracks_bools = self.tracked_stracks_bools_tg.numpy()
        self.tracked_stracks_states = self.tracked_stracks_states_tg.numpy()
        self.tracked_stracks_startframes = self.tracked_stracks_startframes_tg.numpy()
        self.tracked_stracks_values = self.tracked_stracks_values_tg.numpy()
        self.tracked_stracks_means = self.tracked_stracks_means_tg.numpy()
        self.tracked_stracks_covs = self.tracked_stracks_covs_tg.numpy()

        mask = ~np.isin(self.lost_stracks_ids, self.tracked_stracks_ids)
        mask_tg = Tensor(mask)

        self.lost_stracks_ids_tg = Tensor(self.lost_stracks_ids)
        self.lost_stracks_fids_tg = Tensor(self.lost_stracks_fids)
        self.lost_stracks_startframes_tg = Tensor(self.lost_stracks_startframes)
        self.lost_stracks_states_tg = Tensor(self.lost_stracks_states)
        self.lost_stracks_bools_tg = Tensor(self.lost_stracks_bools)
        self.lost_stracks_values_tg = Tensor(self.lost_stracks_values)
        self.lost_stracks_means_tg = Tensor(self.lost_stracks_means)
        self.lost_stracks_covs_tg = Tensor(self.lost_stracks_covs)


        self.lost_stracks_ids_tg *= mask_tg
        self.lost_stracks_fids_tg *= mask_tg
        self.lost_stracks_startframes_tg *= mask_tg
        self.lost_stracks_states_tg *= mask_tg
        self.lost_stracks_bools_tg *= mask_tg
        self.lost_stracks_values_tg = self.lost_stracks_values_tg * mask_tg.view(-1, 1)
        self.lost_stracks_means_tg = self.lost_stracks_means_tg * mask_tg.view(-1, 1)
        self.lost_stracks_covs_tg = self.lost_stracks_covs_tg * mask_tg.view(-1, 1, 1)

        lost_stracks_values_tg = Tensor(lost_stracks_values)
        lost_stracks_means_tg = Tensor(lost_stracks_means)
        lost_stracks_bools_tg = Tensor(lost_stracks_bools)
        lost_stracks_covs_tg = Tensor(lost_stracks_covs)
        lost_stracks_ids_tg = Tensor(lost_stracks_ids)
        lost_stracks_fids_tg = Tensor(lost_stracks_fids)
        lost_stracks_startframes_tg = Tensor(lost_stracks_startframes)
        lost_stracks_states_tg = Tensor(lost_stracks_states)

        if self.lost_stracks_values_tg.shape[0] == 0:
            self.lost_stracks_values_tg = lost_stracks_values_tg
            self.lost_stracks_means_tg = lost_stracks_means_tg
            self.lost_stracks_bools_tg = lost_stracks_bools_tg
            self.lost_stracks_covs_tg = lost_stracks_covs_tg
            self.lost_stracks_ids_tg = lost_stracks_ids_tg
            self.lost_stracks_fids_tg = lost_stracks_fids_tg
            self.lost_stracks_startframes_tg = lost_stracks_startframes_tg
            self.lost_stracks_states_tg = lost_stracks_states_tg
            self.lost_stracks_states_tg = lost_stracks_states_tg
        elif lost_stracks_values_tg.shape[0] != 0:
            self.lost_stracks_values_tg = self.lost_stracks_values_tg.cat(lost_stracks_values_tg)
            self.lost_stracks_means_tg = self.lost_stracks_means_tg.cat(lost_stracks_means_tg)
            self.lost_stracks_bools_tg = self.lost_stracks_bools_tg.cat(lost_stracks_bools_tg)
            self.lost_stracks_covs_tg = self.lost_stracks_covs_tg.cat(lost_stracks_covs_tg)
            self.lost_stracks_ids_tg = self.lost_stracks_ids_tg.cat(lost_stracks_ids_tg)
            self.lost_stracks_fids_tg = self.lost_stracks_fids_tg.cat(lost_stracks_fids_tg)
            self.lost_stracks_startframes_tg = self.lost_stracks_startframes_tg.cat(lost_stracks_startframes_tg)
            self.lost_stracks_states_tg = self.lost_stracks_states_tg.cat(lost_stracks_states_tg)

        self.lost_stracks_values = self.lost_stracks_values_tg.numpy()
        self.lost_stracks_means = self.lost_stracks_means_tg.numpy()
        self.lost_stracks_bools = self.lost_stracks_bools_tg.numpy()
        self.lost_stracks_covs = self.lost_stracks_covs_tg.numpy()
        self.lost_stracks_ids = self.lost_stracks_ids_tg.numpy()
        self.lost_stracks_fids = self.lost_stracks_fids_tg.numpy()
        self.lost_stracks_startframes = self.lost_stracks_startframes_tg.numpy()
        self.lost_stracks_states = self.lost_stracks_states_tg.numpy()

        self.tracked_stracks_bools_tg = Tensor(self.tracked_stracks_bools).unsqueeze(-1)
        self.tracked_stracks_means_tg = Tensor(self.tracked_stracks_means)
        self.tracked_stracks_ids_tg = Tensor(self.tracked_stracks_ids)
        self.tracked_stracks_values_tg = Tensor(self.tracked_stracks_values)

        
        output_stracks_means_tg = self.tracked_stracks_means_tg * self.tracked_stracks_bools_tg
        output_stracks_values_tg = self.tracked_stracks_values_tg * self.tracked_stracks_bools_tg
        output_stracks_ids_tg = self.tracked_stracks_ids_tg * self.tracked_stracks_bools_tg.squeeze(-1)
        
        zeros = self.lost_stracks_ids != 0
        self.lost_stracks_ids = self.lost_stracks_ids[zeros]
        self.lost_stracks_fids = self.lost_stracks_fids[zeros]
        self.lost_stracks_startframes = self.lost_stracks_startframes[zeros]
        self.lost_stracks_states = self.lost_stracks_states[zeros]
        self.lost_stracks_bools = self.lost_stracks_bools[zeros]
        self.lost_stracks_values = self.lost_stracks_values[zeros]
        self.lost_stracks_means = self.lost_stracks_means[zeros]
        self.lost_stracks_covs = self.lost_stracks_covs[zeros]
        v,m,i = output_stracks_values_tg.numpy(), output_stracks_means_tg.numpy(), output_stracks_ids_tg.numpy()
        return v,m,i


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    
    ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float), np.ascontiguousarray(btlbrs, dtype=np.float))
    return ious

def iou_distance(atlbrs, btlbrs):
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def fuse_score(cost_matrix, det_values):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det[4] for det in det_values])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matched_mask = x >= 0
    matches = np.column_stack((np.arange(len(x))[matched_mask],x[matched_mask]))
    unmatched_a = np.where(~matched_mask)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


#Model architecture from https://github.com/ultralytics/ultralytics/issues/189
#The upsampling class has been taken from this pull request https://github.com/tinygrad/tinygrad/pull/784 by dc-dc-dc. Now 2(?) models use upsampling. (retinet and this)

#Pre processing image functions.
def compute_transform(image, new_shape=(1280, 1280), auto=False, scaleFill=False, scaleup=True, stride=32) -> Tensor:
  shape = image.shape[:2]  # current shape [height, width]
  new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  r = min(r, 1.0) if not scaleup else r
  new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
  dw, dh = (np.mod(dw, stride), np.mod(dh, stride)) if auto else (0.0, 0.0)
  new_unpad = (new_shape[1], new_shape[0]) if scaleFill else new_unpad
  dw /= 2
  dh /= 2
  image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR) if shape[::-1] != new_unpad else image
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
  return Tensor(image)

def preprocess(im, imgsz=1280, model_stride=32, model_pt=True):
  im = compute_transform(im, new_shape=imgsz, auto=True, stride=model_stride)
  im = im.unsqueeze(0)
  im = im[..., ::-1].permute(0, 3, 1, 2)
  im = im / 255.0
  return im

# utility functions for forward pass.
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
  lt, rb = distance.chunk(2, dim)
  x1y1 = anchor_points - lt
  x2y2 = anchor_points + rb
  if xywh:
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return c_xy.cat(wh, dim=1)
  return x1y1.cat(x2y2, dim=1)

def make_anchors(feats, strides, grid_cell_offset=0.5):
  anchor_points, stride_tensor = [], []
  assert feats is not None
  for i, stride in enumerate(strides):
    _, _, h, w = feats[i].shape
    sx = Tensor.arange(w) + grid_cell_offset
    sy = Tensor.arange(h) + grid_cell_offset

    # this is np.meshgrid but in tinygrad
    sx = sx.reshape(1, -1).repeat([h, 1]).reshape(-1)
    sy = sy.reshape(-1, 1).repeat([1, w]).reshape(-1)

    anchor_points.append(Tensor.stack(sx, sy, dim=-1).reshape(-1, 2))
    stride_tensor.append(Tensor.full((h * w), stride))
  anchor_points = anchor_points[0].cat(anchor_points[1], anchor_points[2])
  stride_tensor = stride_tensor[0].cat(stride_tensor[1], stride_tensor[2]).unsqueeze(1)
  return anchor_points, stride_tensor

# this function is from the original implementation
def autopad(k, p=None, d=1):  # kernel, padding, dilation
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

def clip_boxes(boxes, shape):
  boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])  # x1, x2
  boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])  # y1, y2
  return boxes

def scale_boxes(img1_shape, predictions, img0_shape, ratio_pad=None):
  gain = ratio_pad if ratio_pad else min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
  pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
  for pred in predictions:
    boxes_np = pred[:4].numpy() if isinstance(pred[:4], Tensor) else pred[:4]
    boxes_np[..., [0, 2]] -= pad[0]
    boxes_np[..., [1, 3]] -= pad[1]
    boxes_np[..., :4] /= gain
    boxes_np = clip_boxes(boxes_np, img0_shape)
    pred[:4] = boxes_np
  return predictions

def get_variant_multiples(variant):
  return {'n':(0.33, 0.25, 2.0), 's':(0.33, 0.50, 2.0), 'm':(0.67, 0.75, 1.5), 'l':(1.0, 1.0, 1.0), 'x':(1, 1.25, 1.0) }.get(variant, None)

def label_predictions(all_predictions):
  class_index_count = defaultdict(int)
  for pred in all_predictions:
    class_id = int(pred[-1])
    if pred[-2] != 0: class_index_count[class_id] += 1

  return dict(class_index_count)

#this is taken from https://github.com/tinygrad/tinygrad/pull/784/files by dc-dc-dc (Now 2 models use upsampling)
class Upsample:
  def __init__(self, scale_factor:int, mode: str = "nearest") -> None:
    assert mode == "nearest" # only mode supported for now
    self.mode = mode
    self.scale_factor = scale_factor

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) > 2 and len(x.shape) <= 5
    (b, c), _lens = x.shape[:2], len(x.shape[2:])
    tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [self.scale_factor] * _lens)
    return tmp.reshape(list(x.shape) + [self.scale_factor] * _lens).permute([0, 1] + list(chain.from_iterable([[y+2, y+2+_lens] for y in range(_lens)]))).reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])

class Conv_Block:
  def __init__(self, c1, c2, kernel_size=1, stride=1, groups=1, dilation=1, padding=None):
    self.conv = Conv2d(c1,c2, kernel_size, stride, padding=autopad(kernel_size, padding, dilation), bias=False, groups=groups, dilation=dilation)
    self.bn = BatchNorm2d(c2, eps=0.001)

  def __call__(self, x):
    return self.bn(self.conv(x)).silu()

class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, g=1, kernels: list = (3,3), channel_factor=0.5):
    c_ = int(c2 * channel_factor)
    self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=None)
    self.cv2 = Conv_Block(c_, c2, kernel_size=kernels[1], stride=1, padding=None, groups=g)
    self.residual = c1 == c2 and shortcut

  def __call__(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))

class C2f:
  def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
    self.c = int(c2 * e)
    self.cv1 = Conv_Block(c1, 2 * self.c, 1,)
    self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)
    self.bottleneck = [Bottleneck(self.c, self.c, shortcut, g, kernels=[(3, 3), (3, 3)], channel_factor=1.0) for _ in range(n)]

  def __call__(self, x):
    y= list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.bottleneck)
    z = y[0]
    for i in y[1:]: z = z.cat(i, dim=1)
    return self.cv2(z)

class SPPF:
  def __init__(self, c1, c2, k=5):
    c_ = c1 // 2  # hidden channels
    self.cv1 = Conv_Block(c1, c_, 1, 1, padding=None)
    self.cv2 = Conv_Block(c_ * 4, c2, 1, 1, padding=None)

    # TODO: this pads with 0s, whereas torch function pads with -infinity. This results in a < 2% difference in prediction which does not make a difference visually.
    self.maxpool = lambda x : x.pad((k // 2, k // 2, k // 2, k // 2)).max_pool2d(kernel_size=k, stride=1)

  def __call__(self, x):
    x = self.cv1(x)
    x2 = self.maxpool(x)
    x3 = self.maxpool(x2)
    x4 = self.maxpool(x3)
    return self.cv2(x.cat(x2, x3, x4, dim=1))

class DFL:
  def __init__(self, c1=16):
    self.conv = Conv2d(c1, 1, 1, bias=False)
    x = Tensor.arange(c1)
    self.conv.weight.replace(x.reshape(1, c1, 1, 1))
    self.c1 = c1

  def __call__(self, x):
    b, c, a = x.shape # batch, channels, anchors
    return self.conv(x.reshape(b, 4, self.c1, a).transpose(2, 1).softmax(1)).reshape(b, 4, a)

#backbone
class Darknet:
  def __init__(self, w, r, d):
    self.b1 = [Conv_Block(c1=3, c2= int(64*w), kernel_size=3, stride=2, padding=1), Conv_Block(int(64*w), int(128*w), kernel_size=3, stride=2, padding=1)]
    self.b2 = [C2f(c1=int(128*w), c2=int(128*w), n=round(3*d), shortcut=True), Conv_Block(int(128*w), int(256*w), 3, 2, 1), C2f(int(256*w), int(256*w), round(6*d), True)]
    self.b3 = [Conv_Block(int(256*w), int(512*w), kernel_size=3, stride=2, padding=1), C2f(int(512*w), int(512*w), round(6*d), True)]
    self.b4 = [Conv_Block(int(512*w), int(512*w*r), kernel_size=3, stride=2, padding=1), C2f(int(512*w*r), int(512*w*r), round(3*d), True)]
    self.b5 = [SPPF(int(512*w*r), int(512*w*r), 5)]

  def return_modules(self):
    return [*self.b1, *self.b2, *self.b3, *self.b4, *self.b5]

  def __call__(self, x):
    x1 = x.sequential(self.b1)
    x2 = x1.sequential(self.b2)
    x3 = x2.sequential(self.b3)
    x4 = x3.sequential(self.b4)
    x5 = x4.sequential(self.b5)
    return (x2, x3, x5)

#yolo fpn (neck)
class Yolov8NECK:
  def __init__(self, w, r, d):  #width_multiple, ratio_multiple, depth_multiple
    self.up = Upsample(2, mode='nearest')
    self.n1 = C2f(c1=int(512*w*(1+r)), c2=int(512*w), n=round(3*d), shortcut=False)
    self.n2 = C2f(c1=int(768*w), c2=int(256*w), n=round(3*d), shortcut=False)
    self.n3 = Conv_Block(c1=int(256*w), c2=int(256*w), kernel_size=3, stride=2, padding=1)
    self.n4 = C2f(c1=int(768*w), c2=int(512*w), n=round(3*d), shortcut=False)
    self.n5 = Conv_Block(c1=int(512* w), c2=int(512 * w), kernel_size=3, stride=2, padding=1)
    self.n6 = C2f(c1=int(512*w*(1+r)), c2=int(512*w*r), n=round(3*d), shortcut=False)

  def return_modules(self):
    return [self.n1, self.n2, self.n3, self.n4, self.n5, self.n6]

  def __call__(self, p3, p4, p5):
    x = self.n1(self.up(p5).cat(p4, dim=1))
    head_1 = self.n2(self.up(x).cat(p3, dim=1))
    head_2 = self.n4(self.n3(head_1).cat(x, dim=1))
    head_3 = self.n6(self.n5(head_2).cat(p5, dim=1))
    return [head_1, head_2, head_3]

#task specific head.
class DetectionHead:
  def __init__(self, nc=80, filters=()):
    self.ch = 16
    self.nc = nc  # number of classes
    self.nl = len(filters)
    self.no = nc + self.ch * 4  #
    self.stride = [8, 16, 32]
    c1 = max(filters[0], self.nc)
    c2 = max((filters[0] // 4, self.ch * 4))
    self.dfl = DFL(self.ch)
    self.cv3 = [[Conv_Block(x, c1, 3), Conv_Block(c1, c1, 3), Conv2d(c1, self.nc, 1)] for x in filters]
    self.cv2 = [[Conv_Block(x, c2, 3), Conv_Block(c2, c2, 3), Conv2d(c2, 4 * self.ch, 1)] for x in filters]

  def __call__(self, x):
    for i in range(self.nl):
      x[i] = (x[i].sequential(self.cv2[i]).cat(x[i].sequential(self.cv3[i]), dim=1))
    self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    y = [(i.reshape(x[0].shape[0], self.no, -1)) for i in x]
    x_cat = y[0].cat(y[1], y[2], dim=2)
    box, cls = x_cat[:, :self.ch * 4], x_cat[:, self.ch * 4:]
    dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    z = dbox.cat(cls.sigmoid(), dim=1)
    return z

class YOLOv8:
  def __init__(self, w, r,  d, num_classes): #width_multiple, ratio_multiple, depth_multiple
    self.net = Darknet(w, r, d)
    self.fpn = Yolov8NECK(w, r, d)
    self.head = DetectionHead(num_classes, filters=(int(256*w), int(512*w), int(512*w*r)))

  def __call__(self, x):
    x = self.net(x)
    x = self.fpn(*x)
    x = self.head(x)
    # TODO: postprocess needs to be in the model to be compiled to webgpu
    return postprocess(x)

  def return_all_trainable_modules(self):
    backbone_modules = [*range(10)]
    yolov8neck_modules = [12, 15, 16, 18, 19, 21]
    yolov8_head_weights = [(22, self.head)]
    return [*zip(backbone_modules, self.net.return_modules()), *zip(yolov8neck_modules, self.fpn.return_modules()), *yolov8_head_weights]

def convert_f16_safetensor_to_f32(input_file: Path, output_file: Path):
  with open(input_file, 'rb') as f:
    metadata_length = int.from_bytes(f.read(8), 'little')
    metadata = json.loads(f.read(metadata_length).decode())
    float32_values = np.fromfile(f, dtype=np.float16).astype(np.float32)

  for v in metadata.values():
    if v["dtype"] == "F16": v.update({"dtype": "F32", "data_offsets": [offset * 2 for offset in v["data_offsets"]]})

  with open(output_file, 'wb') as f:
    new_metadata_bytes = json.dumps(metadata).encode()
    f.write(len(new_metadata_bytes).to_bytes(8, 'little'))
    f.write(new_metadata_bytes)
    float32_values.tofile(f)

def compute_iou_matrix(boxes):
  x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  areas = (x2 - x1) * (y2 - y1)
  x1 = Tensor.maximum(x1[:, None], x1[None, :])
  y1 = Tensor.maximum(y1[:, None], y1[None, :])
  x2 = Tensor.minimum(x2[:, None], x2[None, :])
  y2 = Tensor.minimum(y2[:, None], y2[None, :])
  w = Tensor.maximum(Tensor(0), x2 - x1)
  h = Tensor.maximum(Tensor(0), y2 - y1)
  intersection = w * h
  union = areas[:, None] + areas[None, :] - intersection
  return intersection / union

def postprocess(output, max_det=300, conf_threshold=0.25, iou_threshold=0.45):
  xc, yc, w, h, class_scores = output[0][0], output[0][1], output[0][2], output[0][3], output[0][4:]
  class_ids = Tensor.argmax(class_scores, axis=0)
  probs = Tensor.max(class_scores, axis=0)
  probs = Tensor.where(probs >= conf_threshold, probs, 0)
  x1 = xc - w / 2
  y1 = yc - h / 2
  x2 = xc + w / 2
  y2 = yc + h / 2
  boxes = Tensor.stack(x1, y1, x2, y2, probs, class_ids, dim=1)
  order = Tensor.topk(probs, max_det)[1]
  boxes = boxes[order]
  iou = compute_iou_matrix(boxes[:, :4])
  iou = Tensor.triu(iou, diagonal=1)
  same_class_mask = boxes[:, -1][:, None] == boxes[:, -1][None, :]
  high_iou_mask = (iou > iou_threshold) & same_class_mask
  no_overlap_mask = high_iou_mask.sum(axis=0) == 0
  boxes = boxes * no_overlap_mask.unsqueeze(-1)
  return boxes

def get_weights_location(yolo_variant: str) -> Path:
  weights_location = Path(__file__).parents[1] / "weights" / f'yolov8{yolo_variant}.safetensors'
  fetch(f'https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8{yolo_variant}.safetensors', weights_location)
  f32_weights = weights_location.with_name(f"{weights_location.stem}_f32.safetensors")
  if not f32_weights.exists(): convert_f16_safetensor_to_f32(weights_location, f32_weights)
  return f32_weights

def draw_predictions_on_frame(frame, predictions, class_labels, color_dict):
  font = cv2.FONT_HERSHEY_SIMPLEX

  def is_bright_color(color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127

  h, w, _ = frame.shape
  box_thickness = int((h + w) / 400)
  font_scale = (h + w) / 2500

  for pred in predictions:
    x1, y1, x2, y2, id, class_id = pred
    x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
    color = color_dict[class_labels[class_id]]
    label = f"{class_labels[class_id]} {int(id)}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    label_y, bg_y = (y1 - 4, y1 - text_size[1] - 4) if y1 - text_size[1] - 4 > 0 else (y1 + text_size[1], y1)
    cv2.rectangle(frame, (x1, bg_y), (x1 + text_size[0], bg_y + text_size[1]), color, -1)
    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    cv2.putText(frame, label, (x1, label_y), font, font_scale, font_color, 1, cv2.LINE_AA)


class Args:
    def __init__(self):
        self.track_thresh = 0.6
        self.track_buffer = 60 #frames, was 30
        self.mot20 = False
        self.match_thresh = 0.9

tracker = BYTETracker(Args())

@TinyJit
def do_inf(image):
  predictions = yolo_infer(image)
  return predictions

from urllib.request import urlopen, urlretrieve

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Error: Video URL or path not provided.")
    sys.exit(1)

  video_path = sys.argv[1]
  yolo_variant = sys.argv[2] if len(sys.argv) >= 3 else (print("No variant given, so choosing 'n' as default.") or 'n')
  print(f'Running inference for YOLOv8 variant: {yolo_variant}')

  output_folder_path = Path('./video_outputs')
  output_folder_path.mkdir(parents=True, exist_ok=True)

  # Download and open the video
  local_video_path = fetch(video_path).as_posix()
  cap = cv2.VideoCapture(local_video_path)

  if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)

  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  out_path = (output_folder_path / f"{Path(local_video_path).stem}_output.mp4").as_posix()
  out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

  # Load YOLOv8 model
  depth, width_mult, ratio = get_variant_multiples(yolo_variant)
  yolo_infer = YOLOv8(w=width_mult, r=ratio, d=depth, num_classes=80)
  state_dict = safe_load(get_weights_location(yolo_variant))
  load_state_dict(yolo_infer, state_dict)

  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  
  
  frame_count = 0
  people = set()
  #outs = []
  expected_values = pickle.load(open('values.pkl', 'rb'))
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frame_count += 1
    
    pre_processed = preprocess(frame)
    predictions = do_inf(pre_processed)
    values, means, track_ids = tracker.update(predictions, [1280,1280], [1280,1280])
    pred_track = np.array([np.append(tlbr_np(v,m), [tid,v[5]]) for v,m,tid in zip(values,means,track_ids)], dtype=np.float32)

    # sanity check print people
    for v,tid in zip(values,track_ids):
      if v[5] == 0 and tid != 0: 
        people.add(tid)
    
    pred_track = scale_boxes(pre_processed.shape[2:], pred_track, frame.shape)
    predictions = predictions.numpy()
    predictions = scale_boxes(pre_processed.shape[2:], predictions, frame.shape)

    # Draw predictions
    #draw_predictions_on_frame(frame, predictions, class_labels, color_dict)
    draw_predictions_on_frame(frame, pred_track, class_labels, color_dict)

    out_writer.write(frame)
    

    if sys.argv[1] == "https://motchallenge.net/sequenceVideos/MOT17-08-DPM-raw.mp4":
       if not np.array_equal(np.array(expected_values[frame_count - 1]), values):
          print("wrong output")
          exit()
       #outs.append(values)


    if frame_count % 10 == 0:
      print(f"Processed frame {frame_count}")
      print(len(people))

  #pickle.dump(outs, open('values.pkl', 'wb'))
  cap.release()
  out_writer.release()
  print(f"Saved processed video to {out_path}")

#https://motchallenge.net/sequenceVideos/MOT17-08-DPM-raw.mp4 73
#https://motchallenge.net/sequenceVideos/MOT17-03-FRCNN-raw.mp4 173

