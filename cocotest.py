import json
from collections import defaultdict
from clearcam import YOLOv8, get_variant_multiples, get_weights_location, resize, copy_make_border
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import TinyJit
import cv2
import os
from tinygrad import Tensor
import numpy as np
import time

# todo, make jit for different sizes?
def preprocess(image, new_shape=1280, auto=True, scaleFill=False, scaleup=True, stride=32) -> Tensor:
  shape = image.shape[:2]  # current shape [height, width]
  new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  r = min(r, 1.0) if not scaleup else r
  new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
  dw, dh = (np.mod(dw, stride), np.mod(dh, stride)) if auto else (0.0, 0.0)
  new_unpad = (new_shape[1], new_shape[0]) if scaleFill else new_unpad
  dw /= 2
  dh /= 2
  image = resize(image, new_unpad)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  image = copy_make_border(image, top, bottom, left, right, value=(114,114,114))
  return image

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


if __name__ == "__main__":
  labels = data = json.load(open('dataset/annotations/instances_val2017.json'))
  image_data = {}

  for x in labels["images"]:
    image_data[x["id"]] = (x["file_name"], defaultdict(int))

  for x in labels["annotations"]: image_data[x["image_id"]][1][x["category_id"]] += 1

  #print(image_data)

  yolo_variant = "s"
  depth, width, ratio = get_variant_multiples(yolo_variant)
  yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
  state_dict = safe_load(get_weights_location(yolo_variant))
  load_state_dict(yolo_infer, state_dict)

  i = 0
  total_time = 0
  t = time.time()
  for k in list(image_data.keys())[0:100]:
    x = cv2.imread(f'dataset/val2017/{image_data[k][0]}')
    x = to_square(x) # just use 1280x1280 square cos the dataset has different shapes
    x = Tensor(x)
    t = time.time()
    preds = do_inf(x, yolo_infer).numpy()
    if i > 3: total_time += (time.time() - t)
    counts = defaultdict(int)
    for p in preds:
      if p[-2] >= 0.25: counts[int(p[-1])] += 1
    print(counts, image_data[k][1])

    i+=1
    print(i, len(image_data.keys()))
    
  print("time taken:",total_time)