import numpy as np
from tinygrad import Tensor, TinyJit

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)  

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
    assignments = np.zeros((300, 2), dtype=int)
    is_valid = np.zeros(300, dtype=bool)
    for i in range(300):
        idx = np.argmin(cost)
        r, c = np.unravel_index(idx, cost.shape)
        is_valid[i] = (cost[r, c] < 1e9)
        assignments[i] = [r, c]
        cost[r, :] = np.inf
        cost[:, c] = np.inf
    return assignments, is_valid


def associate(dets_pad, trks_pad, iou_threshold, vel_pad, prev_pad, vdc_weight):
    MAX = 300
    dets_tensor = Tensor(dets_pad)
    prev_tensor = Tensor(prev_pad.astype(np.int32))
    Y, X = speed_direction_batch(dets_tensor, prev_tensor)
    Y, X = Y.numpy(), X.numpy()
    inertia_Y, inertia_X = vel_pad[:,0], vel_pad[:,1]
    inertia_Y = np.repeat(inertia_Y[:,None], MAX, axis=1)
    inertia_X = np.repeat(inertia_X[:,None], MAX, axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, -1, 1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi/2.0 - np.abs(diff_angle)) / np.pi
    valid_mask = np.ones(MAX)
    valid_mask[np.where(prev_pad[:,4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:,None], MAX, axis=1)
    iou_matrix = iou_batch(dets_pad, trks_pad)
    scores = np.repeat(dets_pad[:,-1][:,None], MAX, axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    det_mask = np.any(dets_pad != 0, axis=1)
    trk_mask = np.any(trks_pad != 0, axis=1)
    full_cost = -(iou_matrix + angle_diff_cost)
    full_cost[~det_mask, :] = np.inf
    full_cost[:, ~trk_mask] = np.inf
    matched_indices, mask = linear_assignment300(full_cost)
    unmatched_detections = []

    matched_indices = matched_indices[mask]

    det_mask = np.any(dets_pad != 0, axis=1)
    trk_mask = np.any(trks_pad != 0, axis=1)
    all_valid_detections = np.where(det_mask)[0]
    all_valid_trackers = np.where(trk_mask)[0]
    matched_detections = matched_indices[:, 0]
    matched_trackers = matched_indices[:, 1]
    unmatched_detections = np.setdiff1d(all_valid_detections, matched_detections)
    unmatched_trackers = np.setdiff1d(all_valid_trackers, matched_trackers)
    iou_vals = iou_matrix[matched_detections, matched_trackers]
    low_mask = iou_vals < iou_threshold
    unmatched_detections = np.concatenate([unmatched_detections, matched_detections[low_mask]])
    unmatched_trackers = np.concatenate([unmatched_trackers, matched_trackers[low_mask]])
    matches = matched_indices[~low_mask]
    valid_match_mask = np.all(matches != -1, axis=1)
    matches = matches[valid_match_mask]
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)