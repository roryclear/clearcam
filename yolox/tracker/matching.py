import numpy as np
#from scipy.optimize import linear_sum_assignment

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    x, y = lapjv(cost=cost_matrix, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def lapjv(cost: np.ndarray, cost_limit=np.inf):
    n_rows, n_cols = cost.shape
    if cost_limit < np.inf:
      n = n_rows + n_cols
      cost_extended = np.full((n, n), cost_limit / 2.0)
      cost_extended[n_rows:, n_cols:] = 0
      cost_extended[:n_rows, :n_cols] = cost
    else:
      n = max(n_rows, n_cols)
      cost_extended = np.zeros((n, n))
      cost_extended[:n_rows, :n_cols] = cost
    print(cost_extended)
    row_ind, col_ind = linear_sum_assignment(cost_extended)
    x = -np.ones(n_rows, dtype=int)
    y = -np.ones(n_cols, dtype=int)
    for r, c in zip(row_ind, col_ind):
      if r < n_rows and c < n_cols:
        if cost[r, c] < cost_limit:
          x[r] = c
          y[c] = r
    return x, y

def linear_sum_assignment(cost):
    cost = np.asarray(cost, float)
    nr, nc = cost.shape
    u = np.zeros(nr)
    v = np.zeros(nc)
    shortest = np.full(nc, np.inf)
    path = np.full(nc, -1, int)
    col4row = np.full(nr, -1, int)
    row4col = np.full(nc, -1, int)

    for curRow in range(nr):
        minVal = 0.0
        remaining = np.arange(nc - 1, -1, -1).copy()
        num_remaining = nc

        SR = np.zeros(nr, bool)
        SC = np.zeros(nc, bool)
        shortest[:] = np.inf

        sink = -1
        localRow = curRow

        while sink < 0:
            SR[localRow] = True
            jlist = remaining[:num_remaining]
            rcost = minVal + cost[localRow, jlist] - u[localRow] - v[jlist]
            mask_better = rcost < shortest[jlist]
            if mask_better.any():
                idxs = jlist[mask_better]
                shortest[idxs] = rcost[mask_better]
                path[idxs] = localRow
            best_idx = -1
            lowest = np.inf
            for idx_in_remaining, j in enumerate(jlist):
                val = shortest[j]
                if val < lowest or (val == lowest and row4col[j] == -1):
                    lowest = val
                    best_idx = idx_in_remaining
            if lowest == np.inf:
                return np.arange(nr), -np.ones(nr, int)

            minVal = lowest
            j = jlist[best_idx]

            SC[j] = True

            if row4col[j] == -1:
                sink = j
            else:
                localRow = row4col[j]

            remaining[best_idx], remaining[num_remaining - 1] = (
                remaining[num_remaining - 1],
                remaining[best_idx],
            )
            num_remaining -= 1

        u[curRow] += minVal
        SR_indices = np.nonzero(SR)[0]
        for i in SR_indices:
            if i == curRow:
                continue
            assigned_col = col4row[i]
            if assigned_col >= 0:
                u[i] += minVal - shortest[assigned_col]
            else:
                u[i] += minVal
        SC_indices = np.nonzero(SC)[0]
        for jidx in SC_indices:
            v[jidx] -= minVal - shortest[jidx]

        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            prev = col4row[i]
            col4row[i] = j
            j = prev
            if i == curRow:
                break

    a = np.arange(nr, dtype=int)
    b = col4row.copy()
    return a, b

from typing import List, Tuple
import sys
from collections import deque
class Matching:
    left_pairs: List[int] # Maps L vertices to their matched R vertices (-1 if unmatched)
    right_pairs: List[int]  # Maps R vertices to their matched L vertices (-1 if unmatched)
    total_weight: int  # Total weight of the matching

# https://github.com/ShawxingKwok/Kwok-algorithm/blob/main/Python.py
def kwok_linear_sum_assignment(W: np.ndarray) -> Matching:
    n = W.shape[0]
    adj = [
        [(r, int(W[l, r])) for r in range(n)]
        for l in range(n)
    ]

    L_size = n
    R_size = n

    left_pairs = [-1] * L_size
    right_pairs = [-1] * R_size
    right_parents = [-1] * R_size
    right_visited = [False] * R_size

    visited_lefts = []
    visited_rights = []
    on_edge_rights = []
    right_on_edge = [False] * R_size

    left_labels = [max((w for _, w in edges), default=0) for edges in adj]
    right_labels = [0] * R_size
    slacks = [sys.maxsize] * R_size
    q = deque()

    def advance(r: int) -> bool:
        right_on_edge[r] = False
        right_visited[r] = True
        visited_rights.append(r)
        l = right_pairs[r]
        if l != -1:
            q.append(l)
            visited_lefts.append(l)
            return False

        current_r = r
        while current_r != -1:
            l = right_parents[current_r]
            prev_r = left_pairs[l]
            left_pairs[l] = current_r
            right_pairs[current_r] = l
            current_r = prev_r
        return True

    def bfs_until_applies_augment_path(first_unmatched_r: int):
        while True:
            while q:
                l = q.popleft()
                if left_labels[l] == 0:
                    right_parents[first_unmatched_r] = l
                    if advance(first_unmatched_r):
                        return
                if slacks[first_unmatched_r] > left_labels[l]:
                    slacks[first_unmatched_r] = left_labels[l]
                    right_parents[first_unmatched_r] = l
                    if not right_on_edge[first_unmatched_r]:
                        on_edge_rights.append(first_unmatched_r)
                        right_on_edge[first_unmatched_r] = True

                for r, w in adj[l]:
                    if right_visited[r]:
                        continue
                    diff = left_labels[l] + right_labels[r] - w
                    if diff == 0:
                        right_parents[r] = l
                        if advance(r):
                            return
                    elif slacks[r] > diff:
                        right_parents[r] = l
                        slacks[r] = diff
                        if not right_on_edge[r]:
                            on_edge_rights.append(r)
                            right_on_edge[r] = True

            delta = sys.maxsize
            for r in on_edge_rights:
                if right_on_edge[r]:
                    delta = min(delta, slacks[r])

            for l in visited_lefts:
                left_labels[l] -= delta
            for r in visited_rights:
                right_labels[r] += delta

            for r in on_edge_rights:
                if right_on_edge[r]:
                    slacks[r] -= delta
                    if slacks[r] == 0 and advance(r):
                        return

    # initial greedy matching
    for l in range(L_size):
        for r, w in adj[l]:
            if right_pairs[r] == -1 and left_labels[l] + right_labels[r] == w:
                left_pairs[l] = r
                right_pairs[r] = l
                break

    for l in range(L_size):
        if left_pairs[l] != -1:
            continue
        q.clear()

        for r in visited_rights:
            right_visited[r] = False
        for r in on_edge_rights:
            right_on_edge[r] = False
            slacks[r] = sys.maxsize

        visited_lefts.clear()
        visited_rights.clear()
        on_edge_rights.clear()

        visited_lefts.append(l)
        q.append(l)
        first_unmatched_r = next(r for r in range(R_size) if right_pairs[r] == -1)
        bfs_until_applies_augment_path(first_unmatched_r)

    total = 0
    for l in range(L_size):
        matched = False
        for r, w in adj[l]:
            if r == left_pairs[l]:
                total += w
                matched = True
                break
        if not matched:
            r = left_pairs[l]
            if r != -1:
                left_pairs[l] = -1
                right_pairs[r] = -1

    return left_pairs, right_pairs

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious

def bbox_ious(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    
    boxes_area = ((boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)).reshape(N, 1)
    query_areas = ((query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)).reshape(1, K)

    ixmin = np.maximum(boxes[:, 0].reshape(N, 1), query_boxes[:, 0].reshape(1, K))
    iymin = np.maximum(boxes[:, 1].reshape(N, 1), query_boxes[:, 1].reshape(1, K))
    ixmax = np.minimum(boxes[:, 2].reshape(N, 1), query_boxes[:, 2].reshape(1, K))
    iymax = np.minimum(boxes[:, 3].reshape(N, 1), query_boxes[:, 3].reshape(1, K))
    iw = np.maximum(ixmax - ixmin + 1, 0)
    ih = np.maximum(iymax - iymin + 1, 0)
    intersection = iw * ih
    union = boxes_area + query_areas - intersection
    overlaps = np.where((iw > 0) & (ih > 0), intersection / union, np.zeros_like(intersection)) 
    return overlaps

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost