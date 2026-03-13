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

    iou_matrix = iou_matrix[det_mask][:, trk_mask]
    angle_diff_cost = angle_diff_cost[det_mask][:, trk_mask]

    matched_indices = linear_assignment(-(iou_matrix+angle_diff_cost))
    unmatched_detections = []

    det_mask = np.any(dets_pad != 0, axis=1)
    trk_mask = np.any(trks_pad != 0, axis=1)
    detections = dets_pad[det_mask]
    trackers = trks_pad[trk_mask]

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
