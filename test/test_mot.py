from detection.yolov9 import YOLOv9, SIZES, safe_load, load_state_dict, Sequential, Silence, Conv, RepNCSPELAN4, AConv,\
ADown, CBLinear, CBFuse, SPPELAN, Upsample, Concat, DDetect, postprocess, fetch, rescale_bounding_boxes, draw_bounding_boxes
import cv2
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes
import numpy as np
from pathlib import Path

@TinyJit
def do_inf(im, model): return model(im)

def letterbox(im, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad: im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

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
  excepted_ppl = [241] # same as byte-track's was
  for j,t in enumerate(trackers):

    cap = cv2.VideoCapture("test/videos/MOT16-03.mp4")
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(f"test_outputs/out_{size}_{j}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

    i = 0
    ppl = set()
    while True:
      ret, im0 = cap.read()
      if not ret: break
      im = letterbox(im0, new_shape=(960, 960), stride=32, auto=True)[0]
      im = im.transpose((2, 0, 1))[::-1]
      im = np.ascontiguousarray(im)
      im = Tensor(im).cast(dtypes.float32)
      im /= 255
      if len(im.shape) == 3:
          im = im[None]
      pred = do_inf(im, model).numpy()[0]

      # no tracker, todo clean
      #pred = pred[pred[:, 4] >= 0.25]
      #pred = rescale_bounding_boxes(pred, from_size=(im.shape[2:][::-1]), to_size=im0.shape[:2][::-1])
      #_, buffer = cv2.imencode(".jpg", im0)
      #out.write(draw_bounding_boxes(buffer, pred, class_labels))
      
      online_targets = t.update(pred, [960,960], [960,960], 0.25)
      preds = []
      for x in online_targets:
        if x.class_id == 0 and x.track_id not in ppl: ppl.add(x.track_id)
        preds.append(np.array([x.tlwh[0],x.tlwh[1],(x.tlwh[0]+x.tlwh[2]),(x.tlwh[1]+x.tlwh[3]),x.track_id,x.class_id,x.score]))
      #tlx tly w h, track_id, age, class_id, score
      print("ppl =",len(ppl))
      _, buffer = cv2.imencode(".jpg", im0)
      out.write(draw_bounding_boxes(buffer, preds, class_labels))
      
      i+=1
      print("frame",i)
    cap.release()
    out.release()
    assert len(ppl) == excepted_ppl[j]