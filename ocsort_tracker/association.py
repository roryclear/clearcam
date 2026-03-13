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
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    cost = cost_matrix.copy()
    rows, cols = cost.shape
    assignments = []
    row_used = np.zeros(rows, dtype=bool)
    col_used = np.zeros(cols, dtype=bool)
    flat_indices = np.argsort(cost, axis=None)
    row_indices = flat_indices // cols
    col_indices = flat_indices % cols
    
    for r, c in zip(row_indices, col_indices):
        if not row_used[r] and not col_used[c]:
            assignments.append([r, c])
            row_used[r] = True
            col_used[c] = True
            if row_used.all() or col_used.all():
                break
    
    return np.array(assignments)

def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    dets_pad = np.zeros((300, 5), dtype=detections.dtype)
    tracks_pad = np.zeros((300, 5), dtype=previous_obs.dtype)
    dets_pad[:detections.shape[0]] = detections
    tracks_pad[:previous_obs.shape[0]] = previous_obs

    tracks_pad = tracks_pad.astype(np.int32)
    dets_pad = Tensor(dets_pad)
    tracks_pad = Tensor(tracks_pad)
    Y, X = speed_direction_batch(dets_pad, tracks_pad)
    Y, X = Y.numpy(), X.numpy()
    dets_pad, tracks_pad = dets_pad.numpy(), tracks_pad.numpy()
    det_mask = np.any(dets_pad != 0, axis=1)
    trk_mask = np.any(tracks_pad != 0, axis=1)
    n_det = det_mask.sum()
    n_trk = trk_mask.sum()
    Y, X = Y[:n_trk, :n_det], X[:n_trk, :n_det]

    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi
    
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)] = 0
    
    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    matched_indices = linear_assignment(-(iou_matrix+angle_diff_cost))
    unmatched_detections = []
    for d, _ in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, _ in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
