from detection.yolov9 import YOLOv9, SIZES, safe_load, load_state_dict, Sequential, Silence, Conv, RepNCSPELAN4, AConv,\
ADown, CBLinear, CBFuse, SPPELAN, Upsample, Concat, DDetect, postprocess, fetch, rescale_bounding_boxes, draw_bounding_boxes
import cv2
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes
import numpy as np
from pathlib import Path

if __name__ == "__main__":
  from ocsort_tracker import ocsort
  ocs_tracker = ocsort.OCSort(max_age=60)

  size = "t"
  model = YOLOv9(*SIZES["t"]) if size in SIZES else YOLOv9()
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  Path('./test_outputs').mkdir(parents=True, exist_ok=True)
  cap = cv2.VideoCapture("test/videos/MOT16-03.mp4")
  w, h = int(cap.get(3)), int(cap.get(4))
  out = cv2.VideoWriter(f"test_outputs/out_{size}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

  i = 0
  ppl = set()
  while True:
    ret, im0 = cap.read()
    if not ret: break
    im = im0
    im = Tensor(im).cast(dtypes.float32)
    im = model.preprocess(im, 960)
    pred = model(im).numpy()
    online_targets = ocs_tracker.update(pred, [w, h], [w, h], 0.25)
    preds = []
    for x in online_targets:
      if x.tracklet_len < 1 or x.speed < 2.5: continue
      if x.class_id == 0 and x.track_id not in ppl: ppl.add(x.track_id)
      preds.append(np.array([x.tlwh[0], x.tlwh[1], x.tlwh[0] + x.tlwh[2], x.tlwh[1] + x.tlwh[3], x.score, x.class_id]))
    #tlx tly w h, track_id, age, class_id, score
    print("ppl =",len(ppl))
    _, buffer = cv2.imencode(".jpg", im0)
    out.write(draw_bounding_boxes(buffer, preds, class_labels))
    
    i+=1
    print("frame",i)
  cap.release()
  out.release()
  assert len(ppl) == 154
