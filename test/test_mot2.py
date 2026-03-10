from detection.rfdetr import LWDETR, COCO_CLASSES
from detection.yolov9 import draw_bounding_boxes
import cv2
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes
import numpy as np
from pathlib import Path

def scale_coords(boxes, ratio, dwdh):
  boxes[:, [0, 2]] -= dwdh[0]
  boxes[:, [1, 3]] -= dwdh[1]
  boxes[:, :4] /= ratio[0]
  return boxes

def resize(img, new_size):
  img = img.permute(2,0,1)
  img = Tensor.interpolate(img, size=(new_size[1], new_size[0]), mode='linear', align_corners=False)
  img = img.permute(1, 2, 0)
  return img

if __name__ == "__main__":
  from ocsort_tracker import ocsort
  ocs_tracker = ocsort.OCSort(max_age=60)

  size = "t"
  Path('./test_outputs').mkdir(parents=True, exist_ok=True)
  cap = cv2.VideoCapture("test/videos/MOT16-03.mp4")
  w, h = int(cap.get(3)), int(cap.get(4))
  model = LWDETR("nano", w=w, h=h)
  out = cv2.VideoWriter(f"test_outputs/out_detr.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

  i = 0
  ppl = set()
  threshold = 0.25
  while True:
    ret, im0 = cap.read()
    if not ret: break
    im = im0
    im = Tensor(im).cast(dtype=dtypes.float32)
    im = model.preprocess(im, 384)
    output = model(im).numpy()
    online_targets = ocs_tracker.update(output, [w, h], [w, h], 0.25)
    preds = []
    for x in online_targets:
      if x.tracklet_len < 1 or x.speed < 2.5: continue
      if x.class_id == 1 and x.track_id not in ppl: ppl.add(x.track_id)
      preds.append(np.array([x.tlwh[0], x.tlwh[1], x.tlwh[0] + x.tlwh[2], x.tlwh[1] + x.tlwh[3], x.score, x.class_id]))
    #tlx tly w h, track_id, age, class_id, score
    print("ppl =",len(ppl))
    _, buffer = cv2.imencode(".jpg", im0)

    out.write(draw_bounding_boxes(buffer, preds, COCO_CLASSES))
      
    i+=1
    print("frame",i)
  cap.release()
  out.release()
  assert len(ppl) == 174