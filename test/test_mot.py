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
  from yolox.tracker.byte_tracker import BYTETracker
  from yolox.ocsort_tracker import ocsort
  class Args:
    def __init__(self):
        self.track_buffer = 60 # frames, was 30
        self.mot20 = False
        self.match_thresh = 0.9
  tracker = BYTETracker(Args())
  ocs_tracker = ocsort.OCSort(det_thresh=0.25)

  size = "t"
  model = YOLOv9(*SIZES[size]) if size in SIZES else YOLOv9()
  state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/yolov9/resolve/main/yolov9-{size}.safetensors'))
  load_state_dict(model, state_dict)
  cap = cv2.VideoCapture("test/videos/MOT16-03.mp4")
  w, h = int(cap.get(3)), int(cap.get(4))
  out = cv2.VideoWriter(f"outputs/out_{size}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  Path('./outputs').mkdir(parents=True, exist_ok=True)
  i = 0
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
    online_targets = tracker.update(pred, [960,960], [960,960], 0.25)
    oc_online_targets = ocs_tracker.update(pred, [960,960], [960,960])

    #preds = []
    #for x in oc_online_targets: preds.append(np.array([x[0], x[1], x[2], x[3], x[4], x[6]]))

    preds = []
    for x in online_targets: preds.append(np.array([x.tlwh[0],x.tlwh[1],(x.tlwh[0]+x.tlwh[2]),(x.tlwh[1]+x.tlwh[3]),x.score,x.class_id,x.track_id]))

    print(oc_online_targets,"\n\n",preds)
    print("\n\n\n\n\n\n\n\n\n")

    #pred = pred[pred[:, 4] >= 0.25]
    _, buffer = cv2.imencode(".jpg", im0)
    out.write(draw_bounding_boxes(buffer, preds, class_labels))
    i+=1
    print("frame",i)
  cap.release()
  out.release()