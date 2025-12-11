import json
from collections import defaultdict
from clearcam import YOLOv8, get_variant_multiples, get_weights_location, resize, copy_make_border
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import TinyJit
from tinygrad.helpers import fetch
import cv2
import os
from tinygrad import Tensor
import numpy as np
import time

def to_square(x):
  h, w = x.shape[:2]
  scale = 1280 / max(h, w)
  new_w = int(w * scale)
  new_h = int(h * scale)
  resized = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  canvas = np.zeros((1280, 1280, 3), dtype=np.uint8)
  pad_x = (1280 - new_w) // 2
  pad_y = (1280 - new_h) // 2
  canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
  return canvas

@TinyJit
def do_inf(im, yolo_infer):
  im = im.unsqueeze(0)
  im = im[..., ::-1].permute(0, 3, 1, 2)
  im = im / 255.0
  predictions = yolo_infer(im)
  return predictions

def precision_recall_f1(gt, pred):
    classes = set(gt.keys()).union(set(pred.keys()))

    TP = FP = FN = 0

    for cls in classes:
        gt_count = gt[cls]
        pred_count = pred[cls]

        TP += min(gt_count, pred_count)
        FP += max(pred_count - gt_count, 0)
        FN += max(gt_count - pred_count, 0)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return [precision, recall, f1]

if __name__ == "__main__":
  labels = data = json.load(open('dataset/annotations/instances_val2017.json'))
  image_data = {}

  for x in labels["images"]:
    image_data[x["id"]] = (x["file_name"], defaultdict(int))

  for x in labels["annotations"]: image_data[x["image_id"]][1][x["category_id"]] += 1

  #print(image_data)

  yolo_variant = "m"
  depth, width, ratio = get_variant_multiples(yolo_variant)
  yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
  state_dict = safe_load(get_weights_location(yolo_variant))
  load_state_dict(yolo_infer, state_dict)
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")

  i = 0
  total_time = 0
  scores = [0,0,0] # pres, recall, f1
  t = time.time()
  for k in list(image_data.keys()):
    x = cv2.imread(f'dataset/val2017/{image_data[k][0]}')
    x = to_square(x) # just use 1280x1280 square cos the dataset has different shapes
    x = Tensor(x)
    t = time.time()
    preds = do_inf(x, yolo_infer).numpy()
    if i > 3: total_time += (time.time() - t)
    counts = defaultdict(int)
    for p in preds:
      if p[-2] >= 0.25: counts[int(p[-1]) + 1] += 1 # +1 to class needed

    correct = image_data[k][1]
    
    score = precision_recall_f1(correct, counts)
    scores[0] += score[0]
    scores[1] += score[1]
    scores[2] += score[2]

    i+=1
    print(i, len(image_data.keys()))
    
  print("time taken:",total_time)

  print("scores =", scores[0] / i, scores[1] / i, scores[2] / i)
  