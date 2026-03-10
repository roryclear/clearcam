from detection.rfdetr import LWDETR
from detection.yolov9 import draw_bounding_boxes
import cv2
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes
import numpy as np
from pathlib import Path

COCO_CLASSES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella",
31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
89: "hair drier", 90: "toothbrush",
}

COCO_CLASSES = ["","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","","dining table","","","toilet","","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","","book","clock","vase","scissors","teddy bear","hair drier"]

@TinyJit
def do_inf(im, model): return model.predict(im)

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
    output = do_inf(im, model).numpy()
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