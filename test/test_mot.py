from detection.yolov9 import YOLOv9, SIZES, safe_load, load_state_dict, Sequential, Silence, Conv, RepNCSPELAN4, AConv,\
ADown, CBLinear, CBFuse, SPPELAN, Upsample, Concat, DDetect, postprocess, fetch, rescale_bounding_boxes, draw_bounding_boxes
import cv2
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes
import numpy as np
from pathlib import Path

@TinyJit
def do_inf(im, model): return model(im)

if __name__ == "__main__":
  from ocsort_tracker import ocsort
  ocs_tracker = ocsort.OCSort(max_age=60)

  size = "t"
  model = YOLOv9(*SIZES[size]) if size in SIZES else YOLOv9()
  state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/yolov9/resolve/main/yolov9-{size}.safetensors'))
  load_state_dict(model, state_dict)
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  Path('./test_outputs').mkdir(parents=True, exist_ok=True)

  trackers = [ocs_tracker]
  excepted_ppl = [154]
  for j,t in enumerate(trackers):

    cap = cv2.VideoCapture("test/videos/MOT16-03.mp4")
    #cap = cv2.VideoCapture("test/videos/rain_long.mp4")
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(f"test_outputs/out_{size}_{j}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

    i = 0
    ppl = set()
    while True:
      ret, im0 = cap.read()
      if not ret: break
      im = im0
      im = Tensor(im).cast(dtypes.float32)
      im = model.preprocess(im, new_shape=960)
      im = im.unsqueeze(0)
      im = im.permute(0, 3, 1, 2)
      im = im / 255.0
      pred = do_inf(im, model).numpy()
      
      h0, w0 = im0.shape[:2]
      online_targets = t.update(pred, [w0, h0], [w0, h0], 0.25)
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
    assert len(ppl) == excepted_ppl[j]
