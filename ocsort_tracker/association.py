import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
import math

def iou_batch(bboxes1, bboxes2):
    bboxes1 = Tensor(bboxes1.astype(np.float32))
    bboxes2 = Tensor(bboxes2.astype(np.float32))

    bboxes2 = bboxes2.unsqueeze(0)
    bboxes1 = bboxes1.unsqueeze(1)

    
    xx1 = (bboxes1[..., 0]).maximum(bboxes2[..., 0])
    yy1 = (bboxes1[..., 1]).maximum(bboxes2[..., 1])
    xx2 = (bboxes1[..., 2]).minimum(bboxes2[..., 2])
    yy2 = (bboxes1[..., 3]).minimum(bboxes2[..., 3])
    
    w = (xx2 - xx1).maximum(0.)
    h = (yy2 - yy1).maximum(0.)
    wh = w * h
    
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) + 
              (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    
    return o.numpy()


def iou_batch2(bboxes1, bboxes2):
    bboxes2 = bboxes2.unsqueeze(0)
    bboxes1 = bboxes1.unsqueeze(1)

    
    xx1 = (bboxes1[..., 0]).maximum(bboxes2[..., 0])
    yy1 = (bboxes1[..., 1]).maximum(bboxes2[..., 1])
    xx2 = (bboxes1[..., 2]).minimum(bboxes2[..., 2])
    yy2 = (bboxes1[..., 3]).minimum(bboxes2[..., 3])
    
    w = (xx2 - xx1).maximum(0.)
    h = (yy2 - yy1).maximum(0.)
    wh = w * h
    
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) + 
              (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    
    return o

@TinyJit
def speed_direction_batch(dets_pad, tracks_pad):
    CX1 = (dets_pad[:,0] + dets_pad[:,2]) / 2.0
    CY1 = (dets_pad[:,1] + dets_pad[:,3]) / 2.0
    CX2 = (tracks_pad[:,0] + tracks_pad[:,2]) / 2.0
    CY2 = (tracks_pad[:,1] + tracks_pad[:,3]) / 2.0

    # broadcast to pairwise (tracks x dets)
    dx = CX1[None, :] - CX2[:, None]
    dy = CY1[None, :] - CY2[:, None]

    norm = Tensor.sqrt(dx**2 + dy**2) + 1e-6
    dx /= norm
    dy /= norm
    return dy, dx


def linear_assignment(cost_matrix):
    cost = cost_matrix.astype(float).copy()
    rows, cols = cost.shape
    num_assignments = min(rows, cols)
    
    # Pre-allocate the result
    assignments = np.empty((num_assignments, 2), dtype=int)
    
    for i in range(num_assignments):
        idx = np.argmin(cost)
        r, c = np.unravel_index(idx, cost.shape)
        assignments[i] = [r, c]
        cost[r, :] = np.inf
        cost[:, c] = np.inf
        
    return assignments


def linear_assignment300(cost):
    assignments = np.zeros((300*300, 2), dtype=int)

    flat = cost.ravel()
    order = np.argsort(flat)
    rows = order // 300
    cols = order % 300
    
    assignments = np.stack((rows, cols), axis=1)
    

    selected = get_selected(rows, cols)

    assignments = assignments[selected]
    valid = cost[assignments[:, 0], assignments[:, 1]] < 1e9
    return assignments[valid]

def get_selected(rows, cols):
    used_rows = np.zeros(300, dtype=bool)
    used_cols = np.zeros(300, dtype=bool)
    selected = np.zeros(300*300, dtype=bool)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if not used_rows[r] and not used_cols[c]:
            used_rows[r] = True
            used_cols[c] = True
            selected[i] = True
    return selected


def associate(dets_pad, trks_pad, iou_threshold, vel_pad, prev_pad, vdc_weight):
    MAX = 300
    trks_pad = Tensor(trks_pad.astype(np.float32))
    vdc_weight = Tensor(vdc_weight)
    dets_pad = Tensor(dets_pad)
    prev_pad = Tensor(prev_pad.astype(np.int32))
    vel_pad = Tensor(vel_pad.astype(np.int32))
    Y, X = speed_direction_batch(dets_pad, prev_pad)
    inertia_Y, inertia_X = vel_pad[:,0], vel_pad[:,1]


    inertia_Y = inertia_Y.reshape(-1, 1).expand(-1, MAX)
    inertia_X = inertia_X.reshape(-1, 1).expand(-1, MAX)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = diff_angle_cos.clip(-1, 1)
    diff_angle = Tensor.acos(diff_angle_cos)
    diff_angle = (math.pi/2.0 - Tensor. abs(diff_angle)) / math.pi
    valid_mask = Tensor.ones(MAX)
    valid_mask *= Tensor.where(prev_pad[:,4] < 0, 1, 0)
    valid_mask = valid_mask.reshape(-1, 1).expand(-1, MAX)
    iou_matrix = iou_batch2(dets_pad, trks_pad)
    scores = dets_pad[:, -1].reshape(-1, 1).expand(-1, MAX)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    full_cost = -(iou_matrix + angle_diff_cost)


    trk_mask = (trks_pad != 0).any(axis=1)

    det_mask = (dets_pad != 0).any(axis=1)
    mask = det_mask.cast(dtypes.float32).reshape(-1, 1)


    full_cost = full_cost * mask + (1 - mask) * 1e9
    trk_mask_col = trk_mask.cast(dtypes.float32).reshape(1, -1)
    full_cost = full_cost * trk_mask_col + (1 - trk_mask_col) * 1e9
    full_cost = full_cost.numpy()
    matched_indices = linear_assignment300(full_cost)
    unmatched_detections = []


    det_mask = det_mask.numpy()
    all_valid_detections = np.where(det_mask)[0]
    trk_mask = trk_mask.numpy()
    all_valid_trackers = np.where(trk_mask)[0]
    matched_detections = matched_indices[:, 0]
    matched_trackers = matched_indices[:, 1]
    unmatched_detections = np.setdiff1d(all_valid_detections, matched_detections)
    unmatched_trackers = np.setdiff1d(all_valid_trackers, matched_trackers)
    iou_matrix = iou_matrix.numpy()
    iou_vals = iou_matrix[matched_detections, matched_trackers]
    low_mask = iou_vals < iou_threshold
    unmatched_detections = np.concatenate([unmatched_detections, matched_detections[low_mask]])
    unmatched_trackers = np.concatenate([unmatched_trackers, matched_trackers[low_mask]])
    matches = matched_indices[~low_mask]
    valid_match_mask = np.all(matches != -1, axis=1)
    matches = matches[valid_match_mask]
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

