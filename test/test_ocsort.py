import pickle
from ocsort_tracker import ocsort
import numpy as np
ocs_tracker = ocsort.OCSort(max_age=60)
tracks = pickle.load(open('test/tracks.pkl', 'rb'))

for i in range(len(tracks)):
    out = ocs_tracker.update(tracks[i][0], 0.25)
    preds = []
    expected = []
    for x in out: preds.append(np.array([x.tlwh[0], x.tlwh[1], x.tlwh[0] + x.tlwh[2], x.tlwh[1] + x.tlwh[3], x.score, x.class_id]))
    for x in tracks[i][1]: expected.append(np.array([x.tlwh[0], x.tlwh[1], x.tlwh[0] + x.tlwh[2], x.tlwh[1] + x.tlwh[3], x.score, x.class_id]))
    np.testing.assert_allclose(preds, expected, rtol=1e-5)