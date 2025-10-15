from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.tensor import Tensor
from tinygrad import TinyJit
from tinygrad.device import is_dtype_supported
from tinygrad import dtypes
import numpy as np
from itertools import chain
from pathlib import Path
import cv2
from collections import defaultdict, deque
import time, sys
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
import json
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import shutil
from datetime import datetime, time as time_obj
import uuid
import urllib
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import struct
import pickle
from urllib.parse import unquote
from urllib.parse import quote
import platform
import ctypes
import zlib

def resize(img, new_size):
    img = img.permute(2,0,1)
    img = Tensor.interpolate(img, size=(new_size[1], new_size[0]), mode='linear', align_corners=False)
    img = img.permute(1, 2, 0)
    return img

@TinyJit
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

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
  lt, rb = distance.chunk(2, dim)
  x1y1 = anchor_points - lt
  x2y2 = anchor_points + rb
  if xywh:
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return c_xy.cat(wh, dim=1)
  return x1y1.cat(x2y2, dim=1)

def make_anchors(feats, strides, grid_cell_offset=0.5):
  anchor_points, stride_tensor = [], []
  assert feats is not None
  for i, stride in enumerate(strides):
    _, _, h, w = feats[i].shape
    sx = Tensor.arange(w) + grid_cell_offset
    sy = Tensor.arange(h) + grid_cell_offset

    # this is np.meshgrid but in tinygrad
    sx = sx.reshape(1, -1).repeat([h, 1]).reshape(-1)
    sy = sy.reshape(-1, 1).repeat([1, w]).reshape(-1)

    anchor_points.append(Tensor.stack(sx, sy, dim=-1).reshape(-1, 2))
    stride_tensor.append(Tensor.full((h * w), stride))
  anchor_points = anchor_points[0].cat(anchor_points[1], anchor_points[2])
  stride_tensor = stride_tensor[0].cat(stride_tensor[1], stride_tensor[2]).unsqueeze(1)
  return anchor_points, stride_tensor

def copy_make_border(img, top, bottom, left, right, value=(0, 0, 0)):
    return img.pad(((top,top),(left,left),(0,0)))

# this function is from the original implementation
def autopad(k, p=None, d=1):  # kernel, padding, dilation
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

def clip_boxes(boxes, shape):
  boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])  # x1, x2
  boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])  # y1, y2
  return boxes

def scale_boxes(img1_shape, predictions, img0_shape, ratio_pad=None):
  gain = ratio_pad if ratio_pad else min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
  pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
  for pred in predictions:
    boxes_np = pred[:4].numpy() if isinstance(pred[:4], Tensor) else pred[:4]
    boxes_np[..., [0, 2]] -= pad[0]
    boxes_np[..., [1, 3]] -= pad[1]
    boxes_np[..., :4] /= gain
    boxes_np = clip_boxes(boxes_np, img0_shape)
    pred[:4] = boxes_np
  return predictions

def get_variant_multiples(variant):
  return {'n':(0.33, 0.25, 2.0), 's':(0.33, 0.50, 2.0), 'm':(0.67, 0.75, 1.5), 'l':(1.0, 1.0, 1.0), 'x':(1, 1.25, 1.0) }.get(variant, None)

def label_predictions(all_predictions):
  class_index_count = defaultdict(int)
  for pred in all_predictions:
    class_id = int(pred[-1])
    if pred[-2] != 0: class_index_count[class_id] += 1

  return dict(class_index_count)

#this is taken from https://github.com/tinygrad/tinygrad/pull/784/files by dc-dc-dc (Now 2 models use upsampling)
class Upsample:
  def __init__(self, scale_factor:int, mode: str = "nearest") -> None:
    assert mode == "nearest" # only mode supported for now
    self.mode = mode
    self.scale_factor = scale_factor

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) > 2 and len(x.shape) <= 5
    (b, c), _lens = x.shape[:2], len(x.shape[2:])
    tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [self.scale_factor] * _lens)
    return tmp.reshape(list(x.shape) + [self.scale_factor] * _lens).permute([0, 1] + list(chain.from_iterable([[y+2, y+2+_lens] for y in range(_lens)]))).reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])

class Conv_Block:
  def __init__(self, c1, c2, kernel_size=1, stride=1, groups=1, dilation=1, padding=None):
    self.conv = Conv2d(c1,c2, kernel_size, stride, padding=autopad(kernel_size, padding, dilation), bias=False, groups=groups, dilation=dilation)
    self.bn = BatchNorm2d(c2, eps=0.001)

  def __call__(self, x):
    return self.bn(self.conv(x)).silu()

class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, g=1, kernels: list = (3,3), channel_factor=0.5):
    c_ = int(c2 * channel_factor)
    self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=None)
    self.cv2 = Conv_Block(c_, c2, kernel_size=kernels[1], stride=1, padding=None, groups=g)
    self.residual = c1 == c2 and shortcut

  def __call__(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))

class C2f:
  def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
    self.c = int(c2 * e)
    self.cv1 = Conv_Block(c1, 2 * self.c, 1,)
    self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)
    self.bottleneck = [Bottleneck(self.c, self.c, shortcut, g, kernels=[(3, 3), (3, 3)], channel_factor=1.0) for _ in range(n)]

  def __call__(self, x):
    y= list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.bottleneck)
    z = y[0]
    for i in y[1:]: z = z.cat(i, dim=1)
    return self.cv2(z)

class SPPF:
  def __init__(self, c1, c2, k=5):
    c_ = c1 // 2  # hidden channels
    self.cv1 = Conv_Block(c1, c_, 1, 1, padding=None)
    self.cv2 = Conv_Block(c_ * 4, c2, 1, 1, padding=None)

    # TODO: this pads with 0s, whereas torch function pads with -infinity. This results in a < 2% difference in prediction which does not make a difference visually.
    self.maxpool = lambda x : x.pad((k // 2, k // 2, k // 2, k // 2)).max_pool2d(kernel_size=k, stride=1)

  def __call__(self, x):
    x = self.cv1(x)
    x2 = self.maxpool(x)
    x3 = self.maxpool(x2)
    x4 = self.maxpool(x3)
    return self.cv2(x.cat(x2, x3, x4, dim=1))

class DFL:
  def __init__(self, c1=16):
    self.conv = Conv2d(c1, 1, 1, bias=False)
    x = Tensor.arange(c1)
    self.conv.weight.replace(x.reshape(1, c1, 1, 1))
    self.c1 = c1

  def __call__(self, x):
    b, c, a = x.shape # batch, channels, anchors
    return self.conv(x.reshape(b, 4, self.c1, a).transpose(2, 1).softmax(1)).reshape(b, 4, a)

#backbone
class Darknet:
  def __init__(self, w, r, d):
    self.b1 = [Conv_Block(c1=3, c2= int(64*w), kernel_size=3, stride=2, padding=1), Conv_Block(int(64*w), int(128*w), kernel_size=3, stride=2, padding=1)]
    self.b2 = [C2f(c1=int(128*w), c2=int(128*w), n=round(3*d), shortcut=True), Conv_Block(int(128*w), int(256*w), 3, 2, 1), C2f(int(256*w), int(256*w), round(6*d), True)]
    self.b3 = [Conv_Block(int(256*w), int(512*w), kernel_size=3, stride=2, padding=1), C2f(int(512*w), int(512*w), round(6*d), True)]
    self.b4 = [Conv_Block(int(512*w), int(512*w*r), kernel_size=3, stride=2, padding=1), C2f(int(512*w*r), int(512*w*r), round(3*d), True)]
    self.b5 = [SPPF(int(512*w*r), int(512*w*r), 5)]

  def return_modules(self):
    return [*self.b1, *self.b2, *self.b3, *self.b4, *self.b5]

  def __call__(self, x):
    x1 = x.sequential(self.b1)
    x2 = x1.sequential(self.b2)
    x3 = x2.sequential(self.b3)
    x4 = x3.sequential(self.b4)
    x5 = x4.sequential(self.b5)
    return (x2, x3, x5)

#yolo fpn (neck)
class Yolov8NECK:
  def __init__(self, w, r, d):  #width_multiple, ratio_multiple, depth_multiple
    self.up = Upsample(2, mode='nearest')
    self.n1 = C2f(c1=int(512*w*(1+r)), c2=int(512*w), n=round(3*d), shortcut=False)
    self.n2 = C2f(c1=int(768*w), c2=int(256*w), n=round(3*d), shortcut=False)
    self.n3 = Conv_Block(c1=int(256*w), c2=int(256*w), kernel_size=3, stride=2, padding=1)
    self.n4 = C2f(c1=int(768*w), c2=int(512*w), n=round(3*d), shortcut=False)
    self.n5 = Conv_Block(c1=int(512* w), c2=int(512 * w), kernel_size=3, stride=2, padding=1)
    self.n6 = C2f(c1=int(512*w*(1+r)), c2=int(512*w*r), n=round(3*d), shortcut=False)

  def return_modules(self):
    return [self.n1, self.n2, self.n3, self.n4, self.n5, self.n6]

  def __call__(self, p3, p4, p5):
    x = self.n1(self.up(p5).cat(p4, dim=1))
    head_1 = self.n2(self.up(x).cat(p3, dim=1))
    head_2 = self.n4(self.n3(head_1).cat(x, dim=1))
    head_3 = self.n6(self.n5(head_2).cat(p5, dim=1))
    return [head_1, head_2, head_3]

#task specific head.
class DetectionHead:
  def __init__(self, nc=80, filters=()):
    self.ch = 16
    self.nc = nc  # number of classes
    self.nl = len(filters)
    self.no = nc + self.ch * 4  #
    self.stride = [8, 16, 32]
    c1 = max(filters[0], self.nc)
    c2 = max((filters[0] // 4, self.ch * 4))
    self.dfl = DFL(self.ch)
    self.cv3 = [[Conv_Block(x, c1, 3), Conv_Block(c1, c1, 3), Conv2d(c1, self.nc, 1)] for x in filters]
    self.cv2 = [[Conv_Block(x, c2, 3), Conv_Block(c2, c2, 3), Conv2d(c2, 4 * self.ch, 1)] for x in filters]

  def __call__(self, x):
    for i in range(self.nl):
      x[i] = (x[i].sequential(self.cv2[i]).cat(x[i].sequential(self.cv3[i]), dim=1))
    self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    y = [(i.reshape(x[0].shape[0], self.no, -1)) for i in x]
    x_cat = y[0].cat(y[1], y[2], dim=2)
    box, cls = x_cat[:, :self.ch * 4], x_cat[:, self.ch * 4:]
    dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    z = dbox.cat(cls.sigmoid(), dim=1)
    return z

class YOLOv8:
  def __init__(self, w, r,  d, num_classes): #width_multiple, ratio_multiple, depth_multiple
    self.net = Darknet(w, r, d)
    self.fpn = Yolov8NECK(w, r, d)
    self.head = DetectionHead(num_classes, filters=(int(256*w), int(512*w), int(512*w*r)))

  def __call__(self, x):
    x = self.net(x)
    x = self.fpn(*x)
    x = self.head(x)
    # TODO: postprocess needs to be in the model to be compiled to webgpu
    return postprocess(x)

def convert_f16_safetensor_to_f32(input_file: Path, output_file: Path):
  with open(input_file, 'rb') as f:
    metadata_length = int.from_bytes(f.read(8), 'little')
    metadata = json.loads(f.read(metadata_length).decode())
    float32_values = np.fromfile(f, dtype=np.float16).astype(np.float32)

  for v in metadata.values():
    if v["dtype"] == "F16": v.update({"dtype": "F32", "data_offsets": [offset * 2 for offset in v["data_offsets"]]})

  with open(output_file, 'wb') as f:
    new_metadata_bytes = json.dumps(metadata).encode()
    f.write(len(new_metadata_bytes).to_bytes(8, 'little'))
    f.write(new_metadata_bytes)
    float32_values.tofile(f)

def compute_iou_matrix(boxes):
  x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  areas = (x2 - x1) * (y2 - y1)
  x1 = Tensor.maximum(x1[:, None], x1[None, :])
  y1 = Tensor.maximum(y1[:, None], y1[None, :])
  x2 = Tensor.minimum(x2[:, None], x2[None, :])
  y2 = Tensor.minimum(y2[:, None], y2[None, :])
  w = Tensor.maximum(Tensor(0), x2 - x1)
  h = Tensor.maximum(Tensor(0), y2 - y1)
  intersection = w * h
  union = areas[:, None] + areas[None, :] - intersection
  return intersection / union

def postprocess(output, max_det=300, conf_threshold=0.25, iou_threshold=0.45):
  xc, yc, w, h, class_scores = output[0][0], output[0][1], output[0][2], output[0][3], output[0][4:]
  class_ids = Tensor.argmax(class_scores, axis=0)
  probs = Tensor.max(class_scores, axis=0)
  probs = Tensor.where(probs >= conf_threshold, probs, 0)
  x1 = xc - w / 2
  y1 = yc - h / 2
  x2 = xc + w / 2
  y2 = yc + h / 2
  boxes = Tensor.stack(x1, y1, x2, y2, probs, class_ids, dim=1)
  order = Tensor.topk(probs, max_det)[1]
  boxes = boxes[order]
  iou = compute_iou_matrix(boxes[:, :4])
  iou = Tensor.triu(iou, diagonal=1)
  same_class_mask = boxes[:, -1][:, None] == boxes[:, -1][None, :]
  high_iou_mask = (iou > iou_threshold) & same_class_mask
  no_overlap_mask = high_iou_mask.sum(axis=0) == 0
  boxes = boxes * no_overlap_mask.unsqueeze(-1)
  return boxes

def get_weights_location(yolo_variant: str) -> Path:
  weights_location = BASE / "weights" / f'yolov8{yolo_variant}.safetensors'
  fetch(f'https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8{yolo_variant}.safetensors', weights_location)
  #f16
  return weights_location.with_name(f"{weights_location.stem}.safetensors")
  f32_weights = weights_location.with_name(f"{weights_location.stem}_f32.safetensors")
  if not f32_weights.exists(): convert_f16_safetensor_to_f32(weights_location, f32_weights)
  return f32_weights

@TinyJit
def do_inf(im):
  im = im.unsqueeze(0)
  im = im[..., ::-1].permute(0, 3, 1, 2)
  im = im / 255.0
  predictions = yolo_infer(im)
  return predictions

# RTSP URL
# Video capture thread
import subprocess
import threading
import time
import numpy as np
import cv2
from datetime import datetime
import os
import threading

BASE = Path.home() / "Library" / "Application Support" / "clearcam"
CAMERA_BASE_DIR = BASE / "cameras"
CAMS_FILE = BASE / "cams.pkl"
NEW_DIR = BASE / "newdir"
CAMERA_BASE_DIR.mkdir(parents=True, exist_ok=True)
NEW_DIR.mkdir(parents=True, exist_ok=True) 

class RollingClassCounter:
  def __init__(self, window_seconds=None, max=None, classes=None, sched=[[0,86399],True,True,True,True,True,True,True],cam_name=None):
    self.window = window_seconds
    self.data = defaultdict(deque)
    self.max = max
    self.classes = classes
    self.last_det = 0
    self.sched = sched
    self.cam_name = cam_name
    self.is_on = True

  def add(self, class_id):
    if self.classes is not None and class_id not in self.classes: return
    now = time.time()
    self.data[class_id].append(now)
    self.cleanup(class_id, now)

  def cleanup(self, class_id, now):
    q = self.data[class_id]
    while self.window and q and now - q[0] > self.window:
        q.popleft()

  def reset_counts(self):
    for class_id, _ in self.data.items():
       self.data[class_id] = deque() # todo, use in reset endpoint?

  def get_counts(self):
    max_reached = False
    now = time.time()
    counts = {}
    for class_id, q in self.data.items():
      while self.window and q and now - q[0] > self.window:
        q.popleft()
      if q:
        counts[class_id] = len(q)
        if self.max and len(q) >= self.max: max_reached = True
    return counts, max_reached
  
  def is_active(self, offset=0):
    if not alerts_on: return False
    if not getattr(self, "is_on", False): return False
    if not self.sched: return True
    now = time.localtime()
    time_of_day = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    if not self.sched[time.localtime().tm_wday + 1]: return False
    return time_of_day < self.sched[0][1] and time_of_day > ((self.sched[0][0] - self.window) + offset)

def write_png(filename, array):
    array = array[..., ::-1]  # BGR to RGB
    height, width, _ = array.shape
    png_signature = b"\x89PNG\r\n\x1a\n"
    def chunk(chunk_type, data):
        return (struct.pack("!I", len(data)) +
                chunk_type +
                data +
                struct.pack("!I", zlib.crc32(chunk_type + data) & 0xffffffff))
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_data = b"".join(b"\x00" + array[y].tobytes() for y in range(height))
    compressed = zlib.compress(raw_data, 9)
    png_bytes = (
        png_signature +
        chunk(b"IHDR", ihdr) +
        chunk(b"IDAT", compressed) +
        chunk(b"IEND", b"")
    )
    with open(filename, "wb") as f:
        f.write(png_bytes)

def find_ffmpeg():
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    common_paths = [
        '/opt/homebrew/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/usr/bin/ffmpeg'
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return 'ffmpeg'

import numpy as np

def draw_rectangle_numpy(img, pt1, pt2, color, thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
    if thickness == -1:  # fill
        img[y1:y2+1, x1:x2+1] = color
    else:
        img[y1:y1+thickness, x1:x2+1] = color
        img[y2-thickness+1:y2+1, x1:x2+1] = color
        img[y1:y2+1, x1:x1+thickness] = color
        img[y1:y2+1, x2-thickness+1:x2+1] = color
    return img


class VideoCapture:
  def __init__(self, src,cam_name="clearcamPy"):
    self.output_dir = CAMERA_BASE_DIR / f'{cam_name}' / "streams"
    self.output_dir_raw = CAMERA_BASE_DIR / f'{cam_name}_raw' / "streams"
    self.current_stream_dir = self._get_new_stream_dir()
    # objects in scene count
    self.counter = RollingClassCounter(cam_name=cam_name)
    self.cam_name = cam_name
    self.object_set = set()

    self.src = src
    self.width = 1920 # todo 1080?
    self.height = 1080
    self.proc = None
    self.hls_proc = None
    self.running = True

    self.raw_frame = None
    self.annotated_frame = None
    self.last_preds = []
    self.dir = None

    self.settings = None
    
    alerts_file = CAMERA_BASE_DIR / cam_name / "alerts.pkl"
    alerts_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file = CAMERA_BASE_DIR / cam_name / "settings.pkl"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(alerts_file, "rb") as f:
            self.alert_counters = pickle.load(f)
            for _,a in self.alert_counters.items():
              for c in a.classes: classes.add(str(c))
    except Exception:
        with open(alerts_file, 'wb') as f:
            self.alert_counters = dict()
            self.alert_counters[str(uuid.uuid4())] = RollingClassCounter(window_seconds=60, max=1, classes={0,1,2,3,5,7},cam_name=cam_name)
            pickle.dump(self.alert_counters, f)
    try:
        with open(settings_file, "rb") as f:
            self.settings = pickle.load(f)
    except Exception:
        print("zone file not found")

    self.lock = threading.Lock()

    self._open_ffmpeg()

    # Start threads
    threading.Thread(target=self.capture_loop, daemon=True).start()
    threading.Thread(target=self.inference_loop, daemon=True).start()

  def _get_new_stream_dir(self):
      timestamp = datetime.now().strftime("%Y-%m-%d")
      stream_dir_raw = self.output_dir_raw / timestamp
      stream_dir = self.output_dir / timestamp
      self.dir = stream_dir_raw
      if stream_dir.exists(): shutil.rmtree(stream_dir)
      if stream_dir_raw.exists(): shutil.rmtree(stream_dir_raw)
      stream_dir.mkdir(parents=True, exist_ok=True)
      stream_dir_raw.mkdir(parents=True, exist_ok=True)
      return stream_dir_raw

  def _open_ffmpeg(self):
    if self.proc:
        self.proc.kill()
    if self.hls_proc:
        self.hls_proc.kill()

    ffmpeg_path = find_ffmpeg()
    
    command = [
        ffmpeg_path,
        "-i", self.src,
        "-loglevel", "quiet",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "2",
        "-an",  # No audio
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={self.width}:{self.height}",
        "-timeout", "5000000",
        "-rw_timeout", "15000000",
        "-"
    ]
    self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    command = [
        ffmpeg_path,
        "-i", self.src,
        "-c", "copy",
        "-f", "hls",
        "-hls_time", "4",
        "-hls_list_size", "0",
        "-hls_flags", "+append_list",
        "-hls_playlist_type", "event",
        "-an",  # No audio
        "-hls_segment_filename", str(self._get_new_stream_dir() / "stream_%06d.ts"),
        str(self._get_new_stream_dir() / "stream.m3u8")
    ]
    self.hls_proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  def capture_loop(self):
    frame_size = self.width * self.height * 3
    fail_count = 0
    last_det = -1
    send_det = False
    last_live_check = time.time()
    last_live_seg = time.time()
    last_preview_time = None
    last_counter_update = time.time()
    count = 0
    while self.running:
        if not (CAMERA_BASE_DIR / self.cam_name).is_dir(): os._exit(1) # deleted cam
        try:
            raw_bytes = self.proc.stdout.read(frame_size)
            if len(raw_bytes) != frame_size:
                fail_count += 1
                print(f"FFmpeg frame read failed (count={fail_count}), restarting stream...")
                if fail_count > 5:
                    self._open_ffmpeg()
                    fail_count = 0
                time.sleep(0.5)
                continue
            fail_count = 0
            frame = np.frombuffer(raw_bytes, np.uint8).reshape((self.height, self.width, 3))
            filtered_preds = [p for p in self.last_preds if (classes is None or str(int(p[5])) in classes)]

            if count > 10:
              if last_preview_time is None or time.time() - last_preview_time >= 3600: # preview every hour
                  last_preview_time = time.time()
                  filename = CAMERA_BASE_DIR / f"{self.cam_name}/preview.png"
                  write_png(filename, self.raw_frame)
              for _,alert in self.alert_counters.items():
                  if not alert.is_active():
                    alert.reset_counts()
                    continue
                  if not alert.is_active(offset=4): alert.last_det = time.time() # don't send alert when just active
                  if alert.get_counts()[1]:
                      if time.time() - alert.last_det >= alert.window:
                          send_det = True
                          timestamp = datetime.now().strftime("%Y-%m-%d")
                          filepath = CAMERA_BASE_DIR / f"{self.cam_name}/event_images/{timestamp}"
                          filepath.mkdir(parents=True, exist_ok=True)
                          filename = filepath / f"{int(time.time() - self.streamer.start_time - 10)}.png"
                          self.annotated_frame = self.draw_predictions(frame.copy(), filtered_preds)
                          write_png(str(filename), self.annotated_frame)
                          text = f"Event Detected ({getattr(alert, 'cam_name')})" if getattr(alert, 'cam_name', None) else None
                          if userID is not None: threading.Thread(target=send_notif, args=(userID,text,), daemon=True).start()
                          last_det = time.time()
                          alert.last_det = time.time()
              if (send_det and userID is not None) and time.time() - last_det >= 15: #send 15ish second clip after
                  os.makedirs(CAMERA_BASE_DIR / self.cam_name / "event_clips", exist_ok=True)
                  mp4_filename = CAMERA_BASE_DIR / f"{self.cam_name}/event_clips/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
                  self.streamer.export_clip(Path(mp4_filename))
                  encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), key)
                  os.unlink(mp4_filename)
                  threading.Thread(target=upload_file, args=(Path(f"""{mp4_filename}.aes"""), userID), daemon=True).start()
                  send_det = False
              if userID and (time.time() - last_live_check) >= 5:
                  last_live_check = time.time()
                  threading.Thread(target=check_upload_link, args=(self.cam_name,), daemon=True).start()
              if (time.time() - last_counter_update) >= 5: #update counter every 5 secs
                counters_file = CAMERA_BASE_DIR / self.cam_name / "counters.pkl"
                if os.path.exists(counters_file):
                    with open(counters_file, 'rb') as f:
                        counter = pickle.load(f)
                    if counter is None: self.counter = RollingClassCounter(cam_name=self.cam_name)
                    counter = self.counter
                    with open(counters_file, 'wb') as f:
                        pickle.dump(counter, f)
                
                added_alerts_file = CAMERA_BASE_DIR / self.cam_name / "added_alerts.pkl"
                if added_alerts_file.exists():
                  with open(added_alerts_file, 'rb') as f:
                    added_alerts = pickle.load(f)
                    for id,a in added_alerts:
                      if a is None:
                        del self.alert_counters[id]
                        continue
                      self.alert_counters[id] = a
                      for c in a.classes: classes.add(str(c))
                    added_alerts_file.unlink()
                
                edited_settings_file = CAMERA_BASE_DIR / self.cam_name / "edited_settings.pkl"
                if edited_settings_file.exists():
                  with open(edited_settings_file, 'rb') as f:
                    zone = pickle.load(f)
                    self.settings = zone
                  edited_settings_file.unlink()
                    
              if userID and live_link[self.cam_name] and (time.time() - last_live_seg) >= 4:
                  last_live_seg = time.time()
                  mp4_filename = f"segment.mp4"
                  self.streamer.export_clip(Path(mp4_filename), live=True)
                  encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), key)
                  Path(mp4_filename).unlink()
                  threading.Thread(target=upload_to_r2, args=(Path(f"""{mp4_filename}.aes"""), live_link[self.cam_name]), daemon=True).start()
            else:
               count+=1
            with self.lock:
                self.raw_frame = frame.copy()
                if self.streamer.feeding_frames: self.annotated_frame = self.draw_predictions(frame.copy(), filtered_preds)
            time.sleep(1 / 30)
        except Exception as e:
            print("Error in capture_loop:", e)
            self._open_ffmpeg()
            time.sleep(1)
  

  def inference_loop(self):
    prev_time = time.time()
    while self.running:
      show_dets = (self.settings.get("show_dets") if self.settings else None) or None
      if show_dets:
        show_dets = int(show_dets)
        if (time.time() - show_dets) < 120 and not self.streamer.feeding_frames:
           self.streamer.feeding_frames = True
           self.streamer._stop_event.clear()
           self.streamer.feeding_frames_thread = threading.Thread(target=self.streamer._feed_frames,daemon=True)
           self.streamer.feeding_frames_thread.start()
        elif self.streamer.feeding_frames:
           self.streamer.feeding_frames = False
           self.streamer._stop_event.set()
      if not any(counter.is_active() for _, counter in self.alert_counters.items()): # don't run inference when no active scheds
        time.sleep(1)
        with self.lock: self.last_preds = [] # to remove annotation when no alerts active
        continue
      with self.lock:
        frame = self.raw_frame.copy() if self.raw_frame is not None else None
      if frame is not None:
        frame = Tensor(frame)
        pre = preprocess(frame)
        preds = do_inf(pre).numpy()
        if track:
          thresh = (self.settings.get("threshold") if self.settings else 0.5) or 0.5 #todo clean!
          online_targets = tracker.update(preds, [1280,1280], [1280,1280], thresh) #todo, zone in js also hardcoded to 1280
          preds = []
          for x in online_targets:
            if x.tracklet_len < 1: continue # dont alert for 1 frame, too many false positives
            if hasattr(self, "settings") and self.settings is not None and self.settings["is_on"]:
              # todo, renmae dims to zone dims or something
              outside = ((x.tlwh[0]+x.tlwh[2])<self.settings["dims"][0] or\
              x.tlwh[0]>=(self.settings["dims"][0]+self.settings["dims"][2]) or\
              (x.tlwh[1]+x.tlwh[3])<self.settings["dims"][1] or\
              x.tlwh[1]>(self.settings["dims"][1]+self.settings["dims"][3]))
              if outside ^ self.settings["outside"]: continue
            preds.append(np.array([x.tlwh[0],x.tlwh[1],(x.tlwh[0]+x.tlwh[2]),(x.tlwh[1]+x.tlwh[3]),x.score,x.class_id]))
            if int(x.track_id) not in self.object_set and (classes is None or str(int(x.class_id)) in classes):
              self.object_set.add(int(x.track_id))
              self.counter.add(int(x.class_id))
              for _, alert in self.alert_counters.items():
                if not alert.get_counts()[1]:
                    alert.add(int(x.class_id)) #only add if empty, don't spam notifs
        preds = np.array(preds)
        preds = scale_boxes(pre.shape[:2], preds, frame.shape)
        with self.lock:
          self.last_preds = preds
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        print(f"\rFPS: {fps:.2f}", end="", flush=True)
  
  def is_bright_color(self,color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127

  def draw_predictions(self, frame, preds):
      for x1, y1, x2, y2, conf, cls in preds:
          x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
          label = f"{class_labels[int(cls)]}:{conf:.2f}"
          color = color_dict[class_labels[int(cls)]]
          frame = draw_rectangle_numpy(frame, (x1, y1), (x2, y2), color, 3)
          (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
          font_color = (0, 0, 0) if self.is_bright_color(color) else (255, 255, 255)
          frame = draw_rectangle_numpy(frame, (x1, y1 - text_height - 10), (x1 + text_width + 2, y1), color, -1)
          cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)
      return frame

  def get_frame(self):
      with self.lock:
          if self.annotated_frame is not None:
              return self.annotated_frame.copy(), self.raw_frame.copy()
      return None, None

  def release(self):
      self.running = False
      if self.proc:
          self.proc.kill()
      if self.hls_proc:
         self.hls_proc.kill()

class HLSStreamer:
    def __init__(self, video_capture, output_dir="streams", segment_time=4, cam_name="clearcampy"):
        self.cam_name = cam_name
        self.cam = video_capture
        self.output_dir = CAMERA_BASE_DIR / self.cam_name / output_dir
        self.output_dir_raw = CAMERA_BASE_DIR / (f"{self.cam_name}_raw") / output_dir
        self.segment_time = segment_time
        self.running = False
        self.ffmpeg_proc = None
        self.ffmpeg_proc_raw = None
        self.start_time = time.time()
        self.feeding_frames = False
        self._stop_event = threading.Event()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_raw.mkdir(parents=True, exist_ok=True)
    
    def _get_new_stream_dir(self):
        timestamp = datetime.now().strftime("%Y-%m-%d")
        stream_dir = self.output_dir / timestamp
        stream_dir_raw = self.output_dir_raw / timestamp
        self.cam.dir = stream_dir
        if stream_dir.exists(): shutil.rmtree(stream_dir)
        if stream_dir_raw.exists(): shutil.rmtree(stream_dir_raw)
        stream_dir.mkdir(exist_ok=True)
        stream_dir_raw.mkdir(exist_ok=True)
        return stream_dir, stream_dir_raw
        
    def start(self):
        self.running = True
        self.current_stream_dir, self.current_stream_dir_raw = self._get_new_stream_dir()
        self.recent_segments = deque(maxlen=4)
        self.recent_segments_raw = deque(maxlen=4) # todo, relies on other stream
        self.start_time = time.time()
        ffmpeg_path = find_ffmpeg()
        ffmpeg_cmd = [
            ffmpeg_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.cam.width}x{self.cam.height}",
            "-use_wallclock_as_timestamps", "1",
            "-fflags", "+genpts",
            "-i", "-",
            "-loglevel", "quiet",
            "-c:v", "libx264",
            "-crf", "21",
            "-preset", "veryfast",
            "-g", str(30 * self.segment_time),
            "-vf", "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='%{localtime}':x=w-tw-10:y=10:fontsize=32:fontcolor=white:box=1:boxcolor=black",
            "-f", "hls",
            "-hls_time", str(self.segment_time),
            "-hls_list_size", "0",
            "-hls_flags", "delete_segments",
            "-hls_allow_cache", "0",
            str(self.current_stream_dir / "stream.m3u8")
        ]

        self.ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        threading.Thread(target=self._track_segments, daemon=True).start()

    def export_clip(self,output_path: Path,live=False):
        if not self.recent_segments_raw:
            print("No segments available to save.")
            return  

        concat_list_path = self.current_stream_dir_raw / "concat_list.txt"
        segments_to_use = [self.recent_segments_raw[-1]] if live else self.recent_segments_raw
        with open(concat_list_path, "w") as f:
            f.writelines(f"file '{segment.resolve()}'\n" for segment in segments_to_use)

        concat_list_path = self.current_stream_dir_raw / "concat_list.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = find_ffmpeg()
        
        if live:
          command = [
              ffmpeg_path,
              "-y",
              "-f", "concat",
              "-safe", "0",
              "-i", str(concat_list_path),
              "-loglevel", "quiet",
              "-vf", "scale=-2:240,fps=24,format=yuv420p", # needed for android playback
              "-c:v", "libx264",
              "-pix_fmt", "yuv420p", # needed for android playback
              "-preset", "veryslow",
              "-crf", "32",
              "-an",
              str(output_path)
          ]
        else:
          command = [
              ffmpeg_path,
              "-y",
              "-f", "concat",
              "-safe", "0",
              "-i", str(concat_list_path),
              "-c:v", "libx264",
              "-pix_fmt", "yuv420p",  # needed for android
              "-an",  # No audio
              str(output_path)
          ]
        try:
            subprocess.run(command, check=True)
            print(f"Saved detection clip to: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to save video: {e}")

    def _track_segments(self):
        while self.running:
            segment_files = sorted(self.current_stream_dir.glob("*.ts"), key=os.path.getmtime)
            self.recent_segments.clear()
            self.recent_segments.extend(segment_files[-4:]) # last 3 for now

            segment_files = sorted(self.current_stream_dir_raw.glob("*.ts"), key=os.path.getmtime)
            self.recent_segments_raw.clear()
            self.recent_segments_raw.extend(segment_files[-4:]) # last 3 for now
            time.sleep(self.segment_time / 2)

    def _feed_frames(self):
        last_frame_time = time.time()
        stall_timeout = 10
        
        while self.running and not self._stop_event.is_set():
            frame, raw_frame = self.cam.get_frame()
            if frame is None:
                if time.time() - last_frame_time > stall_timeout:
                    print("Camera feed stalled, attempting recovery...")
                    self._safe_restart()
                time.sleep(0.1)
                continue

            last_frame_time = time.time()

            try:
                if self.ffmpeg_proc is None or self.ffmpeg_proc.poll() is not None:
                    raise BrokenPipeError("FFmpeg process not running") 
                self.ffmpeg_proc.stdin.write(frame.tobytes())
                self.ffmpeg_proc.stdin.flush()
                
            except (BrokenPipeError, OSError, ValueError) as e:
                print(f"HLS write failed: {e}, restarting...")
                self._safe_restart()
                
            time.sleep(1 / 30)

    def _safe_restart(self):
        try:
            self.stop()
        except Exception as e:
            print(f"Error during stop: {e}")
        time.sleep(2)
        while True:  # 3 second timeout
            if self.cam.get_frame() is not None:
                break
            time.sleep(1)    
        try:
            self.start()
            print("HLS streamer restarted successfully")
        except Exception as e:
            print(f"Failed to restart HLS: {e}")

    
    def stop(self):
        self.running = False
        if self.ffmpeg_proc:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()


def append_to_pickle_list(pkl_path, item): # todo, still needed?
    pkl_path = Path(pkl_path)
    if pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            data = []
    else:
        data = []
    data.append(item)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

class HLSRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.base_dir = CAMERA_BASE_DIR
        self.show_dets = None
        super().__init__(*args, **kwargs)

    def get_camera_path(self, cam_name=None):
        """Get the path for a specific camera or all cameras"""
        if cam_name:
            return self.base_dir / cam_name / "streams"
        return self.base_dir
    
    def do_GET(self):
        parsed_path = urlparse(unquote(self.path))
        query = parse_qs(parsed_path.query)
        cam_name = query.get("cam", [None])[0]

        if parsed_path.path == "/list_cameras":
            available_cams = [d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.name.endswith("_raw")]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(available_cams).encode("utf-8"))
            return

        if parsed_path.path == '/add_camera':
            cam_name = query.get("cam_name", [None])[0]
            rtsp = query.get("rtsp", [None])[0]
            
            if not cam_name or not rtsp:
                self.send_error(400, "Missing cam_name or rtsp")
                return
            
            start_cam(rtsp=rtsp,cam_name=cam_name,yolo_variant=yolo_variant)
            cams[cam_name] = rtsp

            with open(CAMS_FILE, 'wb') as f:
              pickle.dump(cams, f)  

            # Redirect back to home
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            return
        
        if parsed_path.path == "/edit_settings":
            if not cam_name:
                self.send_error(400, "Missing cam or id")
                return
            settings_file = CAMERA_BASE_DIR / cam_name / "settings.pkl"
            edited_settings_file = CAMERA_BASE_DIR / cam_name / "edited_settings.pkl"
            if not settings_file.exists():
                with open(settings_file, "wb") as f:
                    pickle.dump(None, f)
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_file, "rb") as f:
               zone = pickle.load(f)
            if zone is None: zone = {}
            outside = query.get("outside", [None])[0]
            is_on = query.get("is_on", [None])[0]
            show_dets = query.get("show_dets", [self.show_dets])[0]
            if show_dets is not None: self.show_dets = str(int(time.time())) # 2 mins
            threshold = query.get("threshold", ["0.5"])[0] #default 0.5?
            if is_on is not None: is_on = str(is_on).lower() == "true"
            if outside is not None: outside = str(outside).lower() == "true"
            tl_x = query.get("tl_x", [None])[0]
            tl_y = query.get("tl_y", [None])[0]
            w = query.get("w", [None])[0]
            h = query.get("h", [None])[0]
            if tl_x is not None: zone["dims"] = [float(tl_x),float(tl_y),float(w),float(h)]
            if is_on is not None: zone["is_on"] = is_on
            if outside is not None: zone["outside"] = outside
            if threshold is not None: zone["threshold"] = float(threshold)
            if self.show_dets is not None: zone["show_dets"] = self.show_dets
            with open(settings_file, 'wb') as f: pickle.dump(zone, f)
            with open(edited_settings_file, 'wb') as f: pickle.dump(zone, f)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if parsed_path.path == "/edit_alert":
            if not cam_name:
                self.send_error(400, "Missing cam or id")
                return
            alerts_file = CAMERA_BASE_DIR / cam_name / "alerts.pkl"
            with open(alerts_file, "rb") as f:
                raw_alerts = pickle.load(f)
            alert = None
            alert_id = query.get("id", [None])[0]
            is_on = query.get("is_on", [None])[0]
            if alert_id is None: # no id, add alert
                window = query.get("window", [None])[0]
                max_count = query.get("max", [None])[0]
                class_ids = query.get("class_ids", [None])[0]
                sched = json.loads(query.get("sched", ["[[0,86400],[0,86400],[0,86400],[0,86400],[0,86400],[0,86400],[0,86400]]"])[0]) # todo, weekly
                window = int(window)
                max_count = int(max_count)
                classes = [int(c.strip()) for c in class_ids.split(",")]
                alert_id = str(uuid.uuid4())
                alert = RollingClassCounter(
                        window_seconds=window,
                        max=max_count,
                        classes=classes,
                        sched=sched,
                        cam_name=cam_name,
                    )
                raw_alerts[alert_id] = alert
            else:
                if is_on is not None: # todo add other properties to edit
                    is_on = str(is_on).lower() == "true"
                    raw_alerts[alert_id].is_on = is_on
                    alert = raw_alerts[alert_id]
                else:
                    del raw_alerts[alert_id]
            with open(alerts_file, 'wb') as f: pickle.dump(raw_alerts, f)
            added_alerts_file = CAMERA_BASE_DIR / cam_name / "added_alerts.pkl"
            append_to_pickle_list(pkl_path=added_alerts_file,item=[alert_id, alert])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if parsed_path.path == "/get_settings":
            zone = {}
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return
            settings_file = CAMERA_BASE_DIR / cam_name / "settings.pkl"
            if settings_file.exists():
                try:
                    with open(settings_file, "rb") as f:
                        zone = pickle.load(f)
                except Exception as e:
                    self.send_error(500, f"Failed to load zone: {e}")
                    return
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(zone).encode("utf-8"))
            return

        if parsed_path.path == "/get_alerts":
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return

            alerts_file = CAMERA_BASE_DIR / cam_name / "alerts.pkl"
            alert_info = []
            if alerts_file.exists():
                try:
                    with open(alerts_file, "rb") as f:
                        raw_alerts = pickle.load(f) 
                        for key,alert in raw_alerts.items():
                            sched = alert.sched if alert.sched else [[0,86399],True,True,True,True,True,True,True]
                            alert_info.append({
                                "window": alert.window,
                                "max": alert.max,
                                "classes": list(alert.classes),
                                "id": str(key),
                                "sched": sched,
                                "is_on": alert.is_on,
                            })
                except Exception as e:
                    self.send_error(500, f"Failed to load alerts: {e}")
                    return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(alert_info).encode("utf-8"))
            return

        if parsed_path.path == '/delete_camera':
            cam_name = query.get("cam_name", [None])[0]
            if not cam_name:
                self.send_error(400, "Missing cam_name parameter")
                return
            
            cam_name_raw = cam_name + "_raw"
            cam_path = CAMERA_BASE_DIR / cam_name
            cam_path_raw = CAMERA_BASE_DIR / cam_name_raw
            if cam_path.exists() and cam_path.is_dir():
                try:
                    shutil.rmtree(cam_path)
                    shutil.rmtree(cam_path_raw)
                    cams.pop(cam_name, None)
                    cams.pop(cam_name_raw, None) # todo needed?
                    with open(CAMS_FILE, 'wb') as f:
                        pickle.dump(cams, f)
                except Exception as e:
                    self.send_error(500, f"Error deleting camera: {e}")
                    return
            else:
                self.send_error(404, "Camera not found")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"deleted"}')
            return

        if parsed_path.path == "/get_counts":
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return
            counters_file = CAMERA_BASE_DIR / cam_name / "counters.pkl"
            try:
                with open(counters_file, "rb") as f:
                    counter = pickle.load(f)
            except Exception:
                with open(counters_file, 'wb') as f:
                    counter = RollingClassCounter(cam_name=cam_name)
                    pickle.dump(counter, f)

            if not counter:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")
                return

            raw_counts = counter.get_counts()[0]
            labeled_counts = {
                class_labels[int(k)]: v
                for k, v in raw_counts.items()
                if int(k) < len(class_labels)
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(labeled_counts).encode("utf-8"))
            return
        
        if parsed_path.path == "/shutdown": 
          threading.Thread(target=self.server.shutdown).start()
          for proc in active_subprocesses:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
          sys.exit(0)

        if parsed_path.path == "/reset_counts":
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return
            counters_file = CAMERA_BASE_DIR / cam_name / "counters.pkl"

            with open(counters_file, "rb") as f:
                counter = pickle.load(f)
            counter = None
            with open(counters_file, 'wb') as f:
                pickle.dump(counter, f)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
            return


        if parsed_path.path == '/' and "cam" not in query:
          available_cams = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
          html = f"""
          <!DOCTYPE html>
          <html>
          <head>
              <style>
                    #eventImagesContainer {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: center;
                        gap: 14px;
                    }}

                    #eventImagesContainer .image-item {{
                        position: relative;
                        display: inline-block;
                    }}

                    #eventImagesContainer img {{
                        width: 220px;
                        height: auto;
                        border-radius: 10px;
                        border: 1px solid #ccc;
                        transition: transform 0.2s;
                    }}

                    #eventImagesContainer .image-item:hover img {{
                        transform: scale(1.05);
                    }}

                    #eventImagesContainer .image-actions {{
                        position: absolute;
                        bottom: 10px;
                        left: 0;
                        right: 0;
                        display: flex;
                        justify-content: center;
                        gap: 8px;
                        opacity: 0;
                        transition: opacity 0.3s ease;
                    }}

                    #eventImagesContainer .image-item:hover .image-actions {{
                        opacity: 1;
                    }}

                    #eventImagesContainer .image-actions button {{
                        background-color: rgba(0, 0, 0, 0.7);
                        color: white;
                        border: none;
                        padding: 6px 12px;
                        border-radius: 4px;
                        font-size: 0.8rem;
                        cursor: pointer;
                        transition: background-color 0.2s;
                    }}

                    #eventImagesContainer .image-actions button:hover {{
                        background-color: rgba(0, 0, 0, 0.9);
                    }}

                    /* Image preview modal */
                    #imagePreviewModal {{
                        display: none;
                        position: fixed;
                        z-index: 1000;
                        left: 0;
                        top: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0, 0, 0, 0.9);
                        justify-content: center;
                        align-items: center;
                    }}

                    #imagePreviewModal img {{
                        max-width: 90%;
                        max-height: 90%;
                        border-radius: 8px;
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                    }}

                    .close-preview {{
                        position: absolute;
                        top: 20px;
                        right: 30px;
                        color: white;
                        font-size: 40px;
                        font-weight: bold;
                        cursor: pointer;
                        z-index: 1001;
                    }}

                    .close-preview:hover {{
                        color: #ccc;
                    }}

                  .camera-grid {{
                      display: flex;
                      flex-wrap: wrap;
                      gap: 20px;
                  }}
                  .camera-card {{
                      width: 220px;
                      border: 1px solid #ccc;
                      border-radius: 8px;
                      overflow: hidden;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                      text-align: center;
                      font-family: sans-serif;
                  }}
                  .camera-card img {{
                      width: 100%;
                      display: block;
                  }}
                  .camera-card div {{
                      padding: 8px;
                      background: #f9f9f9;
                  }}
                  .form-section {{
                      margin-top: 20px;
                      font-family: sans-serif;
                  }}
                  .form-section input {{
                      padding: 6px;
                      margin: 4px;
                      border: 1px solid #ccc;
                      border-radius: 4px;
                      width: 200px;
                  }}
                  .form-section button {{
                      padding: 6px 12px;
                      background: #007bff;
                      color: white;
                      border: none;
                      border-radius: 4px;
                      cursor: pointer;
                  }}
                  .form-section button:hover {{
                      background: #0056b3;
                  }}
                  .shutdown-wrapper {{
                      margin-top: 20px;
                  }}

                  /* Multi-view overlay grid */
                  #multiView {{
                      display: none;
                      position: fixed;
                      top: 0;
                      left: 0;
                      width: 100%;
                      height: 100vh;
                      background: black;
                      z-index: 9999;
                      padding: 10px;
                      box-sizing: border-box;
                      justify-content: center;
                      align-items: center;
                      grid-gap: 10px;
                  }}
                  #multiView.active {{
                      display: grid;
                      justify-items: center;
                      align-items: center;
                  }}
                  #multiView .video-wrapper {{
                      position: relative;
                      width: 100%;
                      aspect-ratio: 16 / 9;
                      background: #000;
                      display: flex;
                      justify-content: center;
                      align-items: center;
                  }}
                  #multiView video {{
                      width: 100%;
                      height: 100%;
                      object-fit: contain;
                      background: black;
                  }}
                  .multi-view-wrapper {{
                      margin-top: 10px;
                      text-align: left;
                  }}
                  .multi-view-wrapper button {{
                      padding: 6px 12px;
                      background: #007bff;
                      color: white;
                      border: none;
                      border-radius: 4px;
                      cursor: pointer;
                  }}
                  .multi-view-wrapper button:hover {{
                      background: #0056b3;
                  }}
              </style>
          </head>
          <body>
              <div id="cameraList" class="camera-grid"></div>
              <div class="multi-view-wrapper">
                  <button onclick="toggleMultiView()">Multi View</button>
              </div>

              <div class="form-section">
                  <form id="addCameraForm">
                      <input type="text" name="cam_name" placeholder="Camera Name" required>
                      <input type="text" name="rtsp" placeholder="RTSP Link" required>
                      <button type="submit">Add Camera</button>
                  </form>
                  <div class="shutdown-wrapper">
                      <button class="shutdown-button" onclick="shutdownServer()">Shutdown Server</button>
                  </div>
              </div>

              <div id="multiView"></div>

              <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
              <script>
                  async function fetchCameras() {{
                      const res = await fetch('/list_cameras');
                      const cams = await res.json();
                      const container = document.getElementById("cameraList");
                      container.innerHTML = cams.map(cam => `
                          <div class="camera-card">
                              <a href="/?cam=${{cam}}">
                                  <img src="/${{cam}}/preview.png" alt="${{cam}} preview">
                                  <div>${{cam}}</div>
                              </a>
                              <button onclick="deleteCamera('${{cam}}')">Delete</button>
                          </div>
                      `).join('');
                      window.currentCameras = cams;
                  }}

                  async function deleteCamera(cam) {{
                      if (!confirm(`Are you sure you want to delete ${{cam}}?`)) return;
                      const res = await fetch(`/delete_camera?cam_name=${{cam}}`);
                      if (res.ok) {{
                          fetchCameras();
                      }} else {{
                          alert("Failed to delete camera.");
                      }}
                  }}

                  async function shutdownServer() {{
                      if (!confirm("Are you sure you want to shut down the server?")) return;
                      const res = await fetch('/shutdown');
                      if (res.ok) {{
                          alert("Server is shutting down...");
                      }} else {{
                          alert("Shutdown failed.");
                      }}
                  }}

                  document.getElementById("addCameraForm").addEventListener("submit", async (e) => {{
                      e.preventDefault();
                      const form = e.target;
                      const params = new URLSearchParams(new FormData(form)).toString();
                      await fetch(`/add_camera?${{params}}`);
                      form.reset();
                      fetchCameras(); // Immediately refresh
                  }});

                  // --- MULTI VIEW LOGIC ---
                  function closeMultiView() {{
                      const container = document.getElementById('multiView');
                      container.innerHTML = '';
                      container.classList.remove('active');
                      if (document.fullscreenElement) document.exitFullscreen();
                  }}

                  function toggleMultiView() {{
                      const container = document.getElementById('multiView');
                      if (container.classList.contains('active')) {{
                          closeMultiView();
                          return;
                      }}

                      if (!window.currentCameras || window.currentCameras.length === 0) {{
                          alert("No cameras available.");
                          return;
                      }}

                      container.classList.add('active');
                      const cams = window.currentCameras;
                      const today = new Date().toLocaleDateString('en-CA');
                      const base = "http://localhost:8080";

                      const count = cams.length;
                      const cols = Math.ceil(Math.sqrt(count));
                      const rows = Math.ceil(count / cols);
                      container.style.gridTemplateColumns = `repeat(${{cols}}, 1fr)`;
                      container.style.gridTemplateRows = `repeat(${{rows}}, auto)`;

                      cams.forEach(cam => {{
                          const encoded = encodeURIComponent(cam);
                          const videoUrl = `${{base}}/${{encoded}}_raw/streams/${{today}}/stream.m3u8`;

                          const wrapper = document.createElement('div');
                          wrapper.className = 'video-wrapper';
                          const vid = document.createElement('video');
                          vid.autoplay = true;
                          vid.muted = true;
                          vid.playsInline = true;
                          wrapper.appendChild(vid);
                          container.appendChild(wrapper);

                          if (vid.canPlayType('application/vnd.apple.mpegurl')) {{
                              vid.src = videoUrl;
                          }} else if (Hls.isSupported()) {{
                              const hls = new Hls();
                              hls.loadSource(videoUrl);
                              hls.attachMedia(vid);
                          }}
                      }});

                      if (container.requestFullscreen) container.requestFullscreen();
                  }}

                  // Close multi-view on ESC or background click
                  document.addEventListener('keydown', e => {{
                      if (e.key === 'Escape') {{
                          const mv = document.getElementById('multiView');
                          if (mv.classList.contains('active')) closeMultiView();
                      }}
                  }});

                  document.getElementById('multiView').addEventListener('click', e => {{
                      if (e.target.id === 'multiView') closeMultiView();
                  }});

                  // Refresh list every 5 seconds
                  fetchCameras();
                  setInterval(fetchCameras, 5000);
              </script>
          </body>
          </html>
          """
          self.send_response(200)
          self.send_header('Content-type', 'text/html')
          self.end_headers()
          self.wfile.write(html.encode('utf-8'))
          return
                
        if not cam_name:
            available_cams = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
            if available_cams:
                cam_name = available_cams[0]
            else:
                self.send_error(404, "No cameras found")
                return
            
        event_image_dir = self.base_dir / cam_name / "event_images"

        if parsed_path.path == '/event_thumbs' or parsed_path.path.endswith('/event_thumbs'):
            selected_dir = parse_qs(parsed_path.query).get("folder", [datetime.now().strftime("%Y-%m-%d")])[0]
            event_image_path = event_image_dir / selected_dir
            event_images = sorted(
                event_image_path.glob("*.png"),
                key=lambda p: int(p.stem),
                reverse=True
            ) if event_image_path.exists() else []
            
            image_links = ""
            if event_images:
                for img in event_images:
                    ts = int(img.stem)
                    image_url = f"/{img.relative_to(self.base_dir.parent)}"
                    image_links += f"""
                    <div class="image-item">
                        <img src="{image_url}" alt="Event" />
                        <div class="image-actions">
                            <button onclick="viewImage('{image_url}')">View</button>
                            <button onclick="playVideoAtTime({ts})">Play</button>
                        </div>
                    </div>
                    """
            else:
                image_links = '<p style="text-align:center; color:#666;">No alerts detected yet.</p>'
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(image_links.encode('utf-8'))
            return

        if parsed_path.path == '/' or parsed_path.path == f'/{cam_name}':
            selected_dir = parse_qs(parsed_path.query).get("folder", [datetime.now().strftime("%Y-%m-%d")])[0]
            start_param = parse_qs(parsed_path.query).get("start", [None])[0]
            show_detections_param = parse_qs(parsed_path.query).get("show_detections", ["false"])[0]
            show_detections = show_detections_param.lower() in ("true", "1", "yes")
            show_detections_checked = "checked" if show_detections else ""

            event_image_path = event_image_dir / selected_dir
            event_images = sorted(event_image_path.glob("*.png")) if event_image_path.exists() else []
            image_links = ""
            for img in event_images:
                ts = int(img.stem)
                image_links += f"""
                <a href="/?cam={cam_name}&folder={selected_dir}&start={ts}">
                    <img src="/{img.relative_to(self.base_dir.parent)}" width="160" style="margin: 5px; border: 1px solid #ccc;" />
                </a>
                """

            try:
                start_time = float(start_param) if start_param is not None else None
            except ValueError:
                start_time = None

            available_cams = [d.name for d in self.base_dir.iterdir() if d.is_dir()]

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
                <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">
                <style>
                    body {{
                        font-family: 'Inter', sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f9f9f9;
                        color: #333;
                    }}

                    .checkbox-container {{
                        display: grid;
                        grid-template-columns: auto auto;
                        gap: 8px 16px;
                        max-height: 200px;
                        overflow-y: auto;
                        margin: 0 auto;
                        padding: 8px;
                        justify-content: center;
                    }}
    
                    .checkbox-container input[type="checkbox"] {{
                        margin: 0;
                        width: 16px;
                        height: 16px;
                    }}

                    .checkbox-item {{
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        margin-left: 10px;
                    }}

                    .checkbox-item input[type="checkbox"] {{
                        margin: 0;
                        transform: scale(1.2);
                    }}

                    .container {{
                        max-width: 900px;
                        margin: 0 auto;
                        padding: 20px;
                    }}

                    video {{
                        width: 100%;
                        border-radius: 12px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                        margin-bottom: 20px;
                    }}

                    .date-picker-container {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 10px;
                        margin: 20px 0;
                        flex-wrap: wrap;
                    }}

                    .date-picker-container input[type="date"],
                    .date-picker-container button {{
                        height: 40px;
                        font-size: 0.95rem;
                        padding: 6px 12px;
                        border-radius: 6px;
                        border: 1px solid #ccc;
                        box-sizing: border-box;
                    }}

                    .counts-wrapper {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: center;
                        align-items: center;
                        gap: 20px;
                        margin: 20px 0;
                    }}

                    .controls {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 16px;
                        align-items: flex-end;
                        justify-content: center;
                        margin-bottom: 20px;
                    }}

                    .controls label {{
                        font-size: 0.9rem;
                        display: flex;
                        flex-direction: column;
                        flex: 1 1 auto;
                        min-width: 120px;
                    }}

                    .time-inputs {{
                        display: flex;
                        gap: 12px;
                        flex-wrap: nowrap;
                        justify-content: center;
                        flex: 1 1 100%;
                    }}

                    input[type="date"],
                    input[type="time"] {{
                        padding: 6px 10px;
                        border-radius: 8px;
                        border: 1px solid #ccc;
                        font-size: 1rem;
                    }}

                    button {{
                        background-color: #f0f0f0;
                        color: #1a73e8;
                        padding: 10px 24px;
                        border: none;
                        border-radius: 6px;
                        font-size: 1rem;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        min-width: 100px;
                        text-align: center;
                        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                    }}

                    button:hover {{
                        background-color: #e0e0e0;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    }}

                    .form-actions button {{
                        min-width: 80px;
                    }}

                    #alertsContainer table button {{
                        padding: 6px 12px;
                        min-width: 90px;
                        text-align: center;
                        font-size: 0.9rem;
                    }}

                    h3 {{
                        font-size: 1.2rem;
                        margin: 20px 0 10px;
                    }}

                    #eventImagesContainer {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: center;
                        gap: 14px;
                    }}

                    #eventImagesContainer img {{
                        width: 220px;
                        height: auto;
                        border-radius: 10px;
                        border: 1px solid #ccc;
                        transition: transform 0.2s;
                    }}

                    #eventImagesContainer img:hover {{
                        transform: scale(1.05);
                    }}

                    /* Modal styles */
                    .modal {{
                        display: none;
                        position: fixed;
                        z-index: 999;
                        left: 0;
                        top: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0, 0, 0, 0.4);
                        justify-content: center;
                        align-items: center;
                        overflow: auto;
                    }}

                    .modal-content {{
                        max-height: 80vh;
                        overflow-y: auto;
                        display: flex;
                        flex-direction: column;
                    }}
                    .form-group.checkbox-group {{
                        flex: 1;
                        min-height: 0; /* Important for flex children */
                    }}

                    .modal-content {{
                        background-color: #fefefe;
                        padding: 20px;
                        border-radius: 8px;
                        width: 90%;
                        max-width: 500px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    }}

                    .modal-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 15px;
                    }}

                    .modal-header h3 {{
                        margin: 0;
                    }}

                    .close {{
                        color: #aaa;
                        font-size: 28px;
                        font-weight: bold;
                        cursor: pointer;
                    }}

                    .close:hover {{
                        color: black;
                    }}

                    .form-group {{
                        margin-bottom: 15px;
                    }}

                    .form-group label {{
                        display: block;
                        margin-bottom: 5px;
                        font-weight: 500;
                    }}

                    .form-group input {{
                        width: 100%;
                        padding: 8px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        box-sizing: border-box;
                    }}

                    .form-actions {{
                        display: flex;
                        justify-content: flex-end;
                        gap: 10px;
                        margin-top: 20px;
                    }}

                    @media (max-width: 600px) {{
                        button {{
                            padding: 8px 16px;
                            font-size: 0.9rem;
                            min-width: 80px;
                        }}

                        .form-actions button {{
                            min-width: 70px;
                        }}

                        input[type="date"] {{
                            width: 100%;
                            max-width: 200px;
                            padding: 6px 8px;
                            font-size: 0.9rem;
                        }}

                        input[type="time"] {{
                            width: 100px;
                            padding: 6px 8px;
                            font-size: 0.9rem;
                        }}

                        .modal-content {{
                            width: 95%;
                            padding: 15px;
                        }}

                        .form-group input {{
                            padding: 6px 8px;
                            font-size: 0.9rem;
                        }}

                        #alertsContainer table {{
                            font-size: 0.9rem;
                        }}

                        #alertsContainer table button {{
                            min-width: 80px;
                            padding: 5px 10px;
                            font-size: 0.8rem;
                        }}

                        .controls label {{
                            min-width: auto;
                            width: 100%;
                        }}

                        .counts-wrapper {{
                            flex-direction: column;
                            gap: 10px;
                        }}

                        .date-picker-container {{
                            flex-wrap: nowrap;
                        }}

                        .date-picker-container input[type="date"],
                        .date-picker-container button {{
                            height: 36px;
                            font-size: 0.85rem;
                            padding: 6px 10px;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <video id="video" controls></video>
                    <div class="date-picker-container">
                        <input type="date" id="folderPicker" value="{selected_dir}">
                        <label style="display: flex; align-items: center; gap: 6px; margin-left: 10px;">
                            <input type="checkbox" id="showDetections" {show_detections_checked}> Show detections
                        </label>
                    </div>
                    <h3>Active Alerts</h3>
                    <div id="alertsContainer">
                        <p>Loading alerts...</p>
                        <div style="display: flex; gap: 10px; justify-content: flex-start; margin-top: 10px;">
                            <button onclick="openSettingsEditor()">Settings</button>
                            <button onclick="openAlertModal()">Add Alert</button>
                        </div>
                    </div>

                    <!-- The Modal -->
                    <div id="alertModal" class="modal">
                        <div class="modal-content" style="max-width: 400px;">
                            <div class="modal-header">
                                <h3>Add New Alert</h3>
                                <span class="close" onclick="closeAlertModal()">&times;</span>
                            </div>
                            <form id="alertForm" onsubmit="addAlert(event)" style="display: flex; flex-direction: column; align-items: center;">
                                <div class="form-group" style="width: 90%; text-align: center;">
                                    <label for="maxCount">Trigger if there are more than</label>
                                    <input type="number" id="maxCount" name="max" min="1" value="1" required
                                        style="width: 80px; margin: 0 6px; text-align: center; display: inline-block;">
                                    <span>objects detected</span>
                                </div>
                                <div class="form-group checkbox-group" style="width: 90%; text-align: center;">
                                    <label>of class(es)</label>
                                    <div id="checkboxContainer" class="checkbox-container"></div>
                                </div>
                                <div class="form-group" style="width: 90%; text-align: center;">
                                    <label for="windowMinutes">within this time window</label>
                                    <input type="number" id="windowMinutes" name="window" min="1" value="1" required 
                                        style="width: 80px; margin: 0 6px; text-align: center; display: inline-block;">
                                    <span>minutes</span>
                                </div>

                                <div class="form-group" style="width: 90%; text-align: center;">
                                    <label>Schedule (optional)</label>
                                    <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                                        <div>
                                            <label for="scheduleFrom">From</label><br>
                                            <input type="time" id="scheduleFrom" name="schedule_from" step="60" style="text-align: center;">
                                        </div>
                                        <div>
                                            <label for="scheduleTo">To</label><br>
                                            <input type="time" id="scheduleTo" name="schedule_to" step="60" style="text-align: center;">
                                        </div>
                                    </div>
                                </div>
                                <div class="form-group" style="width: 90%; text-align: center;">
                                    <div style="display: flex; flex-direction: column; align-items: center; gap: 5px; margin-top: 5px;">
                                        <label><input type="checkbox" name="days" value="0" checked> Mon</label>
                                        <label><input type="checkbox" name="days" value="1" checked> Tue</label>
                                        <label><input type="checkbox" name="days" value="2" checked> Wed</label>
                                        <label><input type="checkbox" name="days" value="3" checked> Thu</label>
                                        <label><input type="checkbox" name="days" value="4" checked> Fri</label>
                                        <label><input type="checkbox" name="days" value="5" checked> Sat</label>
                                        <label><input type="checkbox" name="days" value="6" checked> Sun</label>
                                    </div>
                                </div>

                                <div class="form-actions" style="display: flex; justify-content: center; gap: 10px; margin-top: 20px; width: 100%;">
                                    <button type="button" onclick="closeAlertModal()">Cancel</button>
                                    <button type="submit">Save Alert</button>
                                </div>
                            </form>
                        </div>
                    </div>


                    <div id="zoneModal" class="modal">
                        <div class="modal-content" style="max-width: 600px;">
                            <div class="modal-header">
                                <h3>Edit Zone</h3>
                                <span class="close" onclick="closeZoneModal()">&times;</span>
                            </div>
                            <div style="position: relative; display: inline-block; max-width: 100%;">
                                <img id="zonePreview" src="/{cam_name}/preview.png"
                                    style="width: 100%; max-height: 400px; object-fit: contain; border: 1px solid #ccc; border-radius: 6px; user-select: none; -webkit-user-drag: none; pointer-events: none;">
                                <div id="zoneRect" 
                                    style="position: absolute; border: 2px dashed red; top: 20px; left: 20px; width: 150px; height: 100px; cursor: move; user-select: none;">
                                    <div class="resize-handle" style="position: absolute; width: 16px; height: 16px; background: red; cursor: nwse-resize; bottom: -8px; right: -8px; z-index: 10; border-radius: 50%;"></div>
                                </div>
                            </div>
                            <div class="form-actions" style="margin-top: 20px; display: flex; justify-content: space-between; align-items: center; width: 100%;">
                                <div style="display: flex; flex-direction: column; gap: 10px;">
                                    <label style="display: flex; align-items: center; gap: 6px;">
                                        <input type="checkbox" id="zoneEnabledCheckbox" checked> Enable Zone
                                    </label>
                                    <label style="display: flex; align-items: center; gap: 6px;">
                                        <input type="checkbox" id="outsideZoneCheckbox"> Detect outside of zone
                                    </label>
                                    <label style="display: flex; align-items: center; gap: 6px;">
                                        Detection Threshold:
                                        <input type="number" id="zoneThreshold" value="50" min="0" max="100" step="1" style="width: 60px;"> %
                                    </label>
                                </div>
                                <div style="display: flex; gap: 10px;">
                                    <button type="button" onclick="closeZoneModal()">Cancel</button>
                                    <button type="button" onclick="saveZone()">Save</button>
                                </div>
                            </div>
                        </div>
                    </div>


                    <div class="counts-wrapper">
                        <table id="objectCounts" style="font-size: 1rem; border-collapse: collapse;">
                            <tbody></tbody>
                        </table>
                        <button onclick="resetCounts()">Reset</button>
                    </div>

                    <h3>Detected Events</h3>
                    <div id="eventImagesContainer">
                        {image_links}
                    </div>
                    <div id="imagePreviewModal" class="modal">
                        <span class="close-preview" onclick="closeImagePreview()">&times;</span>
                        <img id="previewImage" src="" alt="Preview">
                    </div>
                </div>

                <script>
                const classLabels = {class_labels};
                const alertModal = document.getElementById("alertModal");
                const folderPicker = document.getElementById("folderPicker");
                const video = document.getElementById("video");
                const startTime = {start_time if start_time is not None else 'null'};
                const cameraName = "{cam_name}";

                
                window.onload = function () {{
                    const container = document.getElementById('checkboxContainer');
                    container.innerHTML = '';
                    
                    for (let i = 0; i < 80; i++) {{
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.value = i;
                        checkbox.name = 'class_ids';
                        checkbox.id = 'class_' + i;
                        
                        const label = document.createElement('label');
                        label.htmlFor = 'class_' + i;
                        label.textContent = (classLabels[i] !== undefined) ? `${{classLabels[i]}}` : `${{i}}`;
                        
                        container.appendChild(checkbox);
                        container.appendChild(label);
                    }}
                }};

                function viewImage(imageSrc) {{
                    const previewModal = document.getElementById("imagePreviewModal");
                    const previewImage = document.getElementById("previewImage");
                    previewImage.src = imageSrc;
                    previewModal.style.display = "flex";
                }}
                function closeImagePreview() {{
                    document.getElementById("imagePreviewModal").style.display = "none";
                }}
                function playVideoAtTime(timestamp) {{
                    const params = new URLSearchParams(window.location.search);
                    params.set('start', timestamp);
                    window.location.search = params.toString();
                }}
                window.addEventListener("click", function(event) {{
                    const modal = document.getElementById("imagePreviewModal");
                    if (event.target === modal) {{
                        closeImagePreview();
                    }}
                }});
                document.addEventListener('keydown', function(event) {{
                    if (event.key === 'Escape') {{
                        closeImagePreview();
                    }}
                }});
                
                const zoneModal = document.getElementById("zoneModal");
                const zoneRect = document.getElementById("zoneRect");
                let isDragging = false;
                let isResizing = false;
                let resizeHandle = null;
                let startX, startY;
                let startWidth, startHeight;
                let startLeft, startTop;

                function openSettingsEditor() {{
                    zoneModal.style.display = "flex";
                    fetch(`/get_settings?cam=${{encodeURIComponent(cameraName)}}`)
                        .then(res => res.json())
                        .then(data => {{
                            if (data && data.dims) {{
                                const [tl_x, tl_y, w, h] = data.dims;
                                const previewEl = document.getElementById("zonePreview");
                                const videoWidth = 1280;
                                const videoHeight = 720;
                                const left = (tl_x / videoWidth) * previewEl.clientWidth;
                                const top = (tl_y / videoHeight) * previewEl.clientHeight;
                                const width = (w / videoWidth) * previewEl.clientWidth;
                                const height = (h / videoHeight) * previewEl.clientHeight;
                                zoneRect.style.left = left + "px";
                                zoneRect.style.top = top + "px";
                                zoneRect.style.width = width + "px";
                                zoneRect.style.height = height + "px";
                            }}

                            const zoneEnabledCheckbox = document.getElementById("zoneEnabledCheckbox");
                            if (data.is_on !== undefined) {{
                                zoneEnabledCheckbox.checked = data.is_on;
                            }} else if (data.length && data[0] && data[0].is_on !== undefined) {{
                                zoneEnabledCheckbox.checked = data[0].is_on;
                            }} else {{
                                zoneEnabledCheckbox.checked = false;
                            }}
                            const outsideZoneCheckbox = document.getElementById("outsideZoneCheckbox");
                            if (data.outside !== undefined) {{
                                outsideZoneCheckbox.checked = data.outside;
                            }} else if (data.length && data[0] && data[0].outside !== undefined) {{
                                outsideZoneCheckbox.checked = data[0].outside;
                            }} else {{
                                outsideZoneCheckbox.checked = false;
                            }}
                            const thresholdInput = document.getElementById("zoneThreshold");
                            let rawThreshold = undefined;
                            if (data.threshold !== undefined) {{
                                rawThreshold = data.threshold;
                            }} else if (data.length && data[0] && data[0].threshold !== undefined) {{
                                rawThreshold = data[0].threshold;
                            }}
                            if (rawThreshold !== undefined) {{
                                if (rawThreshold <= 1) {{
                                    thresholdInput.value = (rawThreshold * 100).toFixed(0);
                                }} else {{
                                    thresholdInput.value = rawThreshold.toFixed(0);
                                }}
                            }}
                        }})
                        .catch(err => {{
                            console.error("Failed to load zone:", err);
                        }});

                    setTimeout(initZoneEditor, 100);
                }}

                function closeZoneModal() {{
                    zoneModal.style.display = "none";
                }}

                function saveZone() {{
                    const previewEl = document.getElementById("zonePreview");
                    const rect = zoneRect.getBoundingClientRect();
                    const preview = previewEl.getBoundingClientRect();

                    const videoWidth = 1280;
                    const videoHeight = 720;

                    const tl_x = ((rect.left - preview.left) / preview.width) * videoWidth;
                    const tl_y = ((rect.top - preview.top) / preview.height) * videoHeight;
                    const w = (rect.width / preview.width) * videoWidth;
                    const h = (rect.height / preview.height) * videoHeight;

                    const is_on = document.getElementById("zoneEnabledCheckbox").checked;
                    const outside = document.getElementById("outsideZoneCheckbox").checked;

                    const thresholdPercent = parseFloat(document.getElementById("zoneThreshold").value) || 50;
                    const threshold = thresholdPercent / 100;
                    const params = new URLSearchParams({{
                        cam: cameraName,
                        tl_x: tl_x.toFixed(2),
                        tl_y: tl_y.toFixed(2),
                        w: w.toFixed(2),
                        h: h.toFixed(2),
                        is_on: is_on,
                        threshold: threshold.toFixed(2),
                        outside: outside
                    }});

                    fetch(`/edit_settings?${{params.toString()}}`)
                        .then(res => {{
                            if (!res.ok) throw new Error("Failed to save zone");
                            console.log("Zone saved successfully");
                            closeZoneModal();
                        }})
                        .catch(err => {{
                            console.error("Save zone failed:", err);
                            alert("Failed to save zone.");
                        }});
                }}


                function initZoneEditor() {{
                    const handle = zoneRect.querySelector('.resize-handle');
                    
                    // Add event listener to the handle
                    handle.addEventListener('mousedown', function(e) {{
                        e.preventDefault();
                        e.stopPropagation();
                        // Resize mode
                        isResizing = true;
                        resizeHandle = e.target;
                        startX = e.clientX;
                        startY = e.clientY;
                        startWidth = parseInt(document.defaultView.getComputedStyle(zoneRect).width, 10);
                        startHeight = parseInt(document.defaultView.getComputedStyle(zoneRect).height, 10);
                        startLeft = parseInt(document.defaultView.getComputedStyle(zoneRect).left, 10);
                        startTop = parseInt(document.defaultView.getComputedStyle(zoneRect).top, 10);
                    }});
                    
                    // Mouse down event for dragging
                    zoneRect.addEventListener('mousedown', function(e) {{
                        if (!e.target.classList.contains('resize-handle')) {{
                            e.preventDefault();
                            e.stopPropagation();
                            // Drag mode
                            isDragging = true;
                            startX = e.clientX - parseInt(document.defaultView.getComputedStyle(zoneRect).left, 10);
                            startY = e.clientY - parseInt(document.defaultView.getComputedStyle(zoneRect).top, 10);
                        }}
                    }});

                    // Mouse move event
                    document.addEventListener('mousemove', function(e) {{
                        if (!isDragging && !isResizing) return;
                        
                        const preview = document.getElementById("zonePreview").getBoundingClientRect();
                        
                        if (isDragging) {{
                            const previewEl = document.getElementById("zonePreview");
                            const parentRect = previewEl.getBoundingClientRect();

                            let newLeft = e.clientX - startX - parentRect.left;
                            let newTop = e.clientY - startY - parentRect.top;

                            newLeft = Math.max(0, Math.min(newLeft, previewEl.clientWidth - zoneRect.offsetWidth));
                            newTop = Math.max(0, Math.min(newTop, previewEl.clientHeight - zoneRect.offsetHeight));

                            zoneRect.style.left = newLeft + 'px';
                            zoneRect.style.top = newTop + 'px';
                        }} 
                        else if (isResizing) {{
                            const previewEl = document.getElementById("zonePreview");
                            let newWidth = startWidth + (e.clientX - startX);
                            let newHeight = startHeight + (e.clientY - startY);

                            newWidth = Math.max(20, newWidth);
                            newHeight = Math.max(20, newHeight);

                            newWidth = Math.min(newWidth, previewEl.clientWidth - parseInt(zoneRect.style.left));
                            newHeight = Math.min(newHeight, previewEl.clientHeight - parseInt(zoneRect.style.top));

                            zoneRect.style.width = newWidth + 'px';
                            zoneRect.style.height = newHeight + 'px';
                        }}
                    }});

                    // Mouse up event
                    document.addEventListener('mouseup', function() {{
                        isDragging = false;
                        isResizing = false;
                        resizeHandle = null;
                    }});
                }}
                zoneModal.addEventListener('shown', initZoneEditor);

                // Allow dragging of the rectangle
                (function makeDraggable(el) {{
                    let offsetX, offsetY, isDragging = false;

                    el.addEventListener("mousedown", (e) => {{
                        if (e.target === el) {{
                            isDragging = true;
                            offsetX = e.offsetX;
                            offsetY = e.offsetY;
                            document.addEventListener("mousemove", move);
                            document.addEventListener("mouseup", stop);
                        }}
                    }});

                    function move(e) {{
                        if (!isDragging) return;
                        const parent = el.parentElement.getBoundingClientRect();
                        el.style.left = Math.min(Math.max(0, e.clientX - parent.left - offsetX), parent.width - el.offsetWidth) + "px";
                        el.style.top = Math.min(Math.max(0, e.clientY - parent.top - offsetY), parent.height - el.offsetHeight) + "px";
                    }}

                    function stop() {{
                        isDragging = false;
                        document.removeEventListener("mousemove", move);
                        document.removeEventListener("mouseup", stop);
                    }}
                }})(zoneRect);


                function openAlertModal() {{
                    alertModal.style.display = "flex";
                }}

                function closeAlertModal() {{
                    alertModal.style.display = "none";
                }}

                window.addEventListener("click", function(event) {{
                    if (event.target === alertModal) closeAlertModal();
                }});

                document.getElementById("showDetections").addEventListener("change", function() {{
                    const currentTime = video.currentTime;
                    const params = new URLSearchParams(window.location.search);
                    params.set('show_detections', this.checked);
                    params.set('start', currentTime);
                    window.location.search = params.toString();
                }});

                function loadStream(folder) {{
                    const showDetections = document.getElementById("showDetections").checked;
                    const streamSuffix = showDetections ? "" : "_raw";
                    const url = `${{cameraName}}${{streamSuffix}}/streams/${{folder}}/stream.m3u8`;

                    async function waitForManifest(maxRetries = 30, delay = 2000) {{
                        for (let i = 0; i < maxRetries; i++) {{
                            try {{
                                const res = await fetch(url, {{ cache: "no-store" }});
                                if (res.ok) {{
                                    const text = await res.text();
                                    if (text.includes("#EXTINF")) {{
                                        return true; // playlist has segments
                                    }}
                                }}
                            }} catch (err) {{
                                console.warn("Stream not ready yet:", err);
                            }}
                            await new Promise(r => setTimeout(r, delay));
                        }}
                        return false;
                    }}

                    (async () => {{
                        const ready = await waitForManifest();
                        if (!ready) {{
                            console.error("Stream never became available.");
                            return;
                        }}

                        if (Hls.isSupported()) {{
                            const hls = new Hls({{
                                manifestLoadingTimeOut: 20000,
                                manifestLoadingMaxRetry: Infinity,
                                manifestLoadingRetryDelay: 2000,
                            }});

                            hls.loadSource(url);
                            hls.attachMedia(video);

                            hls.on(Hls.Events.MANIFEST_PARSED, function () {{
                                if (startTime !== null) video.currentTime = startTime;
                                video.play();
                            }});

                            hls.on(Hls.Events.ERROR, function (event, data) {{
                                if (data.fatal) {{
                                    switch (data.type) {{
                                        case Hls.ErrorTypes.NETWORK_ERROR:
                                            console.warn("Network error, retrying...");
                                            hls.startLoad(-1);
                                            break;
                                        case Hls.ErrorTypes.MEDIA_ERROR:
                                            console.warn("Media error, recovering...");
                                            hls.recoverMediaError();
                                            break;
                                        default:
                                            console.error("Fatal error, destroying HLS instance.");
                                            hls.destroy();
                                            break;
                                    }}
                                }}
                            }});
                        }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                            video.src = url;
                            video.addEventListener('loadedmetadata', function () {{
                                if (startTime !== null) video.currentTime = startTime;
                                video.play();
                            }});
                        }}
                    }})();
                }}


                function loadEventImages(folder) {{
                    fetch(`/event_thumbs?cam=${{cameraName}}&folder=${{folder}}`)
                        .then(res => res.text())
                        .then(html => {{
                            const container = document.getElementById("eventImagesContainer");

                            // Create a temporary div to parse incoming HTML
                            const temp = document.createElement("div");
                            temp.innerHTML = html.trim();

                            // Compare child count and content
                            const isSameLength = container.children.length === temp.children.length;
                            let isSame = isSameLength;

                            if (isSameLength) {{
                                for (let i = 0; i < container.children.length; i++) {{
                                    if (
                                        container.children[i].outerHTML.trim() !==
                                        temp.children[i].outerHTML.trim()
                                    ) {{
                                        isSame = false;
                                        break;
                                    }}
                                }}
                            }}

                            if (!isSame) {{
                                container.innerHTML = html;
                            }}
                        }})
                        .catch(err => {{
                            console.error("Failed to load images:", err);
                            const container = document.getElementById("eventImagesContainer");
                            container.innerHTML = "<p>No event images found.</p>";
                        }});
                }}

                function fetchCounts() {{
                    fetch(`/get_counts?cam=${{encodeURIComponent(cameraName)}}`)
                        .then(res => res.json())
                        .then(data => {{
                            const tbody = document.querySelector("#objectCounts tbody");
                            tbody.innerHTML = "";

                            const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
                            if (!entries.length) {{
                                tbody.innerHTML = "<tr><td colspan='2'>No detections.</td></tr>";
                                return;
                            }}

                            for (const [label, count] of entries) {{
                                tbody.innerHTML += `
                                    <tr>
                                        <td style="padding:6px 12px; border-bottom:1px solid #eee;">${{label}}</td>
                                        <td style="padding:6px 12px; border-bottom:1px solid #eee;">${{count}}</td>
                                    </tr>`;
                            }}
                        }})
                        .catch(err => {{
                            console.error("Failed to fetch counts:", err);
                            document.querySelector("#objectCounts tbody").innerHTML = "<tr><td colspan='2'>Error fetching counts.</td></tr>";
                        }});
                }}

                function resetCounts() {{
                    if (!confirm("Are you sure you want to reset the counts for this camera?")) return;
                    fetch(`/reset_counts?cam=${{encodeURIComponent(cameraName)}}`)
                        .then(res => res.json())
                        .then(() => {{
                            console.log("Counts reset");
                            fetchCounts();
                        }})
                        .catch(err => {{
                            console.error("Failed to reset counts:", err);
                            alert("Failed to reset counts.");
                        }});
                }}

                function fetchAlerts() {{
                    fetch(`/get_alerts?cam=${{encodeURIComponent(cameraName)}}`)
                        .then(res => res.json())
                        .then(alerts => {{ 
                            const container = document.getElementById("alertsContainer");
                        if (!alerts.length) {{
                            container.innerHTML = `
                                <p>No alerts configured.</p>
                                <div style="display: flex; gap: 10px; justify-content: center; margin-top: 10px;">
                                    <button onclick="openSettingsEditor()">Settings</button>
                                    <button onclick="openAlertModal()">Add Alert</button>
                                </div>`;
                            return;
                        }}

                            let html = `<table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
                                <thead>
                                    <tr>
                                        <th style="text-align:center; padding:6px; border-bottom:1px solid #ccc;"></th>
                                        <th style="text-align:left; padding:6px; border-bottom:1px solid #ccc;">Occurrences of</th>
                                        <th style="text-align:left; padding:6px; border-bottom:1px solid #ccc;">Within</th>
                                        <th style="text-align:left; padding:6px; border-bottom:1px solid #ccc;">Schedule</th>
                                        <th style="text-align:left; padding:6px; border-bottom:1px solid #ccc;"></th>
                                    </tr>
                                </thead><tbody>`;

                            for (const alert of alerts) {{
                                const classNames = alert.classes.map(id => classLabels[id] ?? id).join(", ");
                                const h = Math.floor(alert.window / 3600);
                                const m = Math.floor((alert.window % 3600) / 60);
                                const s = alert.window % 60;
                                const windowStr = `${{String(h).padStart(2,'0')}}:${{String(m).padStart(2,'0')}}:${{String(s).padStart(2,'0')}}`;
                                const fromH = Math.floor(alert.sched[0][0] / 3600);
                                const fromM = Math.floor((alert.sched[0][0] % 3600) / 60);
                                let toSec = alert.sched[0][1];
                                if (toSec % 60 !== 0) {{
                                    toSec += 60 - (toSec % 60);
                                }}
                                if (toSec >= 86400) {{
                                    toSec = 0;
                                }}
                                const toH = Math.floor(toSec / 3600);
                                const toM = Math.floor((toSec % 3600) / 60);
                                const timeRange = `${{String(fromH).padStart(2,'0')}}:${{String(fromM).padStart(2,'0')}} to ${{String(toH).padStart(2,'0')}}:${{String(toM).padStart(2,'0')}}`;

                                const daysOfWeek = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
                                const activeDays = alert.sched.slice(1).map((on, idx) => on ? daysOfWeek[idx] : null).filter(Boolean);
                                let daysStr;
                                if (activeDays.length === 7) {{
                                    daysStr = "Daily";
                                }} else {{
                                    // compress consecutive ranges (Mon–Fri)
                                    const indices = alert.sched.slice(1).map((on, i) => on ? i : -1).filter(i => i >= 0);
                                    let ranges = [];
                                    let start = null;
                                    let prev = null;
                                    for (const idx of indices) {{
                                        if (start === null) {{
                                            start = idx;
                                            prev = idx;
                                        }} else if (idx === prev + 1) {{
                                            prev = idx;
                                        }} else {{
                                            ranges.push([start, prev]);
                                            start = idx;
                                            prev = idx;
                                        }}
                                    }}
                                    if (start !== null) {{
                                        ranges.push([start, prev]);
                                    }}
                                    daysStr = ranges.map(([a, b]) => {{
                                        if (a === b) return daysOfWeek[a];
                                        else return `${{daysOfWeek[a]}}-${{daysOfWeek[b]}}`;
                                    }}).join(", ");
                                }}
                                const schedStr = `${{timeRange}} (${{daysStr}})`;
                                const isOn = alert.hasOwnProperty("is_on") ? alert.is_on : true;
                                const checkedAttr = isOn ? "checked" : "";
                                html += `<tr>
                                    <td style="padding:6px; border-bottom:1px solid #eee;">${{alert.max}}</td>
                                    <td style="padding:6px; border-bottom:1px solid #eee;">${{classNames}}</td>
                                    <td style="padding:6px; border-bottom:1px solid #eee;">${{windowStr}}</td>
                                    <td style="padding:6px; border-bottom:1px solid #eee;">${{schedStr}}</td>
                                    <td style="padding:6px; border-bottom:1px solid #eee; display:flex; justify-content:center; align-items:center;">
                                        <input type="checkbox" data-alert-id="${{alert.id}}" ${{checkedAttr}} 
                                            onchange="toggleAlert('${{alert.id}}', this.checked)" 
                                            style="transform:scale(1.2); cursor:pointer;">
                                    </td>
                                    <td style="padding:6px; border-bottom:1px solid #eee; text-align:center;">
                                        <button onclick="deleteAlert('${{alert.id}}')" style="padding:4px 8px;">Delete</button>
                                    </td>
                                </tr>`;
                            }}


                            html += `</tbody></table>
                                <div style="margin-top:10px; display:flex; gap:10px; justify-content:center;">
                                    <button onclick="openSettingsEditor()">Settings</button>
                                    <button onclick="openAlertModal()">Add Alert</button>
                                </div>`;
                            container.innerHTML = html;
                        }})
                        .catch(err => {{
                            console.error("Failed to fetch alerts:", err);
                            document.getElementById("alertsContainer").innerHTML = "<p>Error loading alerts.</p>";
                        }});
                }}

                function toggleAlert(alertId, isOn) {{
                    fetch(`/edit_alert?cam=${{encodeURIComponent(cameraName)}}&id=${{alertId}}&is_on=${{isOn}}`)
                        .then(res => {{
                            if (!res.ok) throw new Error("Failed to toggle alert");
                            console.log(`Alert ${{alertId}} ${{isOn ? 'enabled' : 'disabled'}}`);
                        }})
                        .catch(err => {{
                            console.error("Toggle alert failed:", err);
                            alert("Failed to toggle alert status.");
                            fetchAlerts();
                        }});
                }}

                function deleteAlert(alertId) {{
                    fetch(`/edit_alert?cam=${{encodeURIComponent(cameraName)}}&id=${{alertId}}`)
                        .then(res => {{
                            if (!res.ok) throw new Error("Failed to delete alert");
                            fetchAlerts();
                        }})
                        .catch(err => {{
                            console.error("Delete failed:", err);
                            alert("Failed to delete alert.");
                        }});
                }}

                function addAlert(event) {{
                    event.preventDefault();
                    const form = event.target;
                    const formData = new FormData(form);

                    const windowMinutes = parseFloat(formData.get('window'));
                    const windowSeconds = Math.round(windowMinutes * 60);

                    const checked = form.querySelectorAll('input[name="class_ids"]:checked');
                    const classIds = Array.from(checked).map(cb => cb.value).join(',');

                    const scheduleFrom = formData.get("schedule_from") || "00:00:00";
                    const scheduleTo = formData.get("schedule_to") || "23:59:59";

                    const toSeconds = (hhmmss) => {{
                        const parts = hhmmss.split(":").map(Number);
                        const [h = 0, m = 0, s = 0] = parts;
                        return h * 3600 + m * 60 + s;
                    }};

                    const fromSec = toSeconds(scheduleFrom);
                    const toSec = toSeconds(scheduleTo);
                    const dayChecks = form.querySelectorAll('input[name="days"]');
                    const dayBools = Array.from(dayChecks).map(cb => cb.checked);
                    const schedArray = [[fromSec, toSec], ...dayBools];

                    const params = new URLSearchParams({{
                        cam: cameraName,
                        window: windowSeconds,
                        max: formData.get('max'),
                        class_ids: classIds,
                        sched: JSON.stringify(schedArray),
                    }});

                    fetch(`/edit_alert?${{params.toString()}}`)
                        .then(res => {{
                            if (!res.ok) throw new Error("Failed to add alert");
                            return res.json();
                        }})
                        .then(() => {{
                            closeAlertModal();
                            form.reset();
                            fetchAlerts();
                        }})
                        .catch(err => {{
                            console.error("Add alert failed:", err);
                            alert("Failed to add alert.");
                        }});
                }}


                folderPicker.addEventListener("change", () => {{
                    const folder = folderPicker.value;
                    const params = new URLSearchParams(window.location.search);
                    params.set('date', folder);
                    window.location.search = params.toString();
                }});

                function getCurrentDate() {{
                    const params = new URLSearchParams(window.location.search);
                    return params.get('date') || folderPicker.value;
                }}

                
                function getQueryParam(name) {{
                    const urlParams = new URLSearchParams(window.location.search);
                    return urlParams.get(name);
                }}

                const initialDate = getQueryParam('date') || "{selected_dir}";
                folderPicker.value = initialDate;


                loadStream(getCurrentDate());
                loadEventImages(getCurrentDate());
                fetchAlerts();
                fetchCounts();
                setInterval(fetchCounts, 5000);
                setInterval(() => loadEventImages(getCurrentDate()), 5000);
            </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
            return
        
        requested_path = parsed_path.path.lstrip('/')
        if requested_path.startswith("cameras/"):
            requested_path = requested_path[len("cameras/"):]
        file_path = self.base_dir / requested_path

        if not file_path.exists():
            self.send_error(404)
            return

        self.send_response(200)
        if file_path.suffix == '.m3u8':
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            self.send_header('Cache-Control', 'no-cache')
        elif file_path.suffix == '.ts':
            self.send_header('Content-Type', 'video/MP2T')
        elif file_path.suffix == '.png':
            self.send_header('Content-Type', 'image/jpeg')
        self.end_headers()

        with open(file_path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile)


def schedule_daily_restart(hls_streamer, restart_time):
    """Schedule daily restarts at a specific time"""
    while True:
        now = datetime.now().time()
        target = time_obj(restart_time[0], restart_time[1])  # (hour, minute)
        
        # Calculate seconds until next restart
        if now >= target:
            # If time already passed today, schedule for tomorrow
            delta = (24 * 3600) - ((now.hour * 3600 + now.minute * 60 + now.second) - (target.hour * 3600 + target.minute * 60))
        else:
            delta = ((target.hour * 3600 + target.minute * 60) - 
                    (now.hour * 3600 + now.minute * 60 + now.second))
        
        print(f"Next stream restart scheduled in {delta//3600}h {(delta%3600)//60}m")
        time.sleep(delta)
        
        print("\nPerforming scheduled stream restart...")
        hls_streamer.stop()
        time.sleep(10) # todo can get away with none or less?
        hls_streamer.start()

def send_notif(session_token: str, text=None):
    host = "www.rors.ai"
    endpoint = "/send" #/test
    boundary = f"Boundary-{uuid.uuid4()}"
    content_type = f"multipart/form-data; boundary={boundary}"
    lines = [
        f"--{boundary}",
        'Content-Disposition: form-data; name="session_token"',
        "",
        session_token,
        f"--{boundary}--",
        ""
    ]
    if text is not None:
      lines.extend([
      f"--{boundary}",
      'Content-Disposition: form-data; name="text"',
      "",
      text,
    ])
    body = "\r\n".join(lines).encode("utf-8")
    conn = http.client.HTTPSConnection(host)
    headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
    try:
        conn.request("POST", endpoint, body, headers)
        response = conn.getresponse()
        print(f"Status: {response.status} {response.reason}")
        print(response.read().decode())
    except Exception as e:
        print(f"Error sending session token: {e}")
    finally:
        conn.close()

import aes
MAGIC_NUMBER = 0x4D41474943
HEADER_SIZE = 8
AES_BLOCK_SIZE = 16
AES_KEY_SIZE = 32

def prepare_key(key: str) -> bytes:
    key_bytes = key.encode('utf-8')[:AES_KEY_SIZE]
    return key_bytes.ljust(AES_KEY_SIZE, b'\0')

def pkcs7_pad(data: bytes, block_size: int) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)

def encrypt_cbc(data: bytes, key: bytes, iv: bytes) -> bytes:
    aes_cipher = aes.AES(key)
    encrypted = bytearray()
    prev_block = iv

    for i in range(0, len(data), AES_BLOCK_SIZE):
        block = data[i:i + AES_BLOCK_SIZE]
        xored = bytes([b ^ p for b, p in zip(block, prev_block)])
        encrypted_block = bytes(aes_cipher.encrypt(xored))
        encrypted += encrypted_block
        prev_block = encrypted_block
    return bytes(encrypted)

def encrypt_file(input_path: Path, output_path: Path, key: str):
    try:
        key_bytes = prepare_key(key)
        iv = os.urandom(AES_BLOCK_SIZE)

        with open(input_path, 'rb') as f:
            plaintext = f.read()
        data = struct.pack('<Q', MAGIC_NUMBER) + plaintext
        padded = pkcs7_pad(data, AES_BLOCK_SIZE)

        ciphertext = encrypt_cbc(padded, key_bytes, iv)

        with open(output_path, 'wb') as f:
            f.write(iv + ciphertext)

        return True

    except Exception as e:
        print(f"ENCRYPTION FAILED: {e}")
        return False

def upload_file(file_path: Path, session_token: str):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        file_name = file_path.name
        file_size = len(file_data)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    try:
        params = {
            "filename": file_name,
            "session_token": session_token,
            "size": str(file_size)
        }
        query_string = urllib.parse.urlencode(params)
        url = f"https://rors.ai/upload?{query_string}"
        
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status != 200:
                print(f"Failed to get upload URL: {response.status}")
                return False
            response_data = json.loads(response.read().decode('utf-8'))
            presigned_url = response_data.get("url")
            if not presigned_url:
                print("Invalid response - missing upload URL")
                return False
    except Exception as e:
        print(f"Error getting upload URL: {e}")
        return False
    success = False
    for attempt in range(10):
        try:
            url_parts = urllib.parse.urlparse(presigned_url)
            if url_parts.scheme == 'https':
                conn = http.client.HTTPSConnection(url_parts.netloc)
            else:
                conn = http.client.HTTPConnection(url_parts.netloc)
            
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(file_size)
            }
            conn.request("PUT", url_parts.path + "?" + url_parts.query, body=file_data, headers=headers)
            upload_response = conn.getresponse()
            
            if 200 <= upload_response.status < 300:
                os.unlink(file_path) # todo remove on failure or success
                print(f"File uploaded successfully on attempt {attempt + 1}")
                success = True
                conn.close()
                break
            else:
                print(f"Upload failed with status {upload_response.status} on attempt {attempt + 1}")
                conn.close()
        except Exception as e:
            print(f"Upload error on attempt {attempt + 1}: {e}")
        if attempt < 3: time.sleep(10 * attempt)

    try:
        file_path.unlink()
        print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Failed to delete file: {e}")
    return success

def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def get_executable_args(): return ([sys.argv[0]], sys.argv[1:]) if getattr(sys, "frozen", False) else ([sys.executable, sys.argv[0]], sys.argv[1:])

def start_cam(rtsp, cam_name, yolo_variant='n'):
    if not rtsp or not cam_name: return
    
    def upsert_arg(args, key, value):
        prefix = f"--{key}="
        for i, arg in enumerate(args):
            if arg.startswith(prefix):
                args[i] = f"{prefix}{value}"
                return args
        return args + [f"{prefix}{value}"]

    executable, base_args = get_executable_args()
    new_args = upsert_arg(base_args, "cam_name", cam_name)
    new_args = upsert_arg(new_args, "rtsp", rtsp)
    new_args = upsert_arg(new_args, "yolo_size", yolo_variant)
    proc = subprocess.Popen(executable + new_args, close_fds=True)
    active_subprocesses.append(proc)


live_link = dict()
alerts_on = True
is_live_lock = threading.Lock()
def check_upload_link(cam_name="clearcampy"):
    global live_link
    global alerts_on
    query_params = urllib.parse.urlencode({
        "name": quote(cam_name),
        "session_token": userID
    })
    url = f"https://rors.ai/get_stream_upload_link?{query_params}"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                response_data = json.loads(response.read().decode('utf-8'))
                upload_link = response_data.get("upload_link")
                alerts_on_res = response_data.get("alerts_on")
                with is_live_lock:
                   live_link[cam_name] = upload_link
                   alerts_on = (alerts_on_res == 1)
            else:
                raise Exception(f"HTTP Error: {response.status}")
    except Exception as e:
        with is_live_lock:
            if cam_name in live_link: live_link[cam_name] = None
        print(f"Error checking upload link: {e}")

def upload_to_r2(file_path: Path, signed_url: str, max_retries: int = 0) -> bool:
    try:
        url_parts = urllib.parse.urlparse(signed_url)
        if url_parts.scheme == 'https':
            conn = http.client.HTTPSConnection(url_parts.netloc)
        else:
            conn = http.client.HTTPConnection(url_parts.netloc)
        
        with file_path.open('rb') as f:
            file_data = f.read()
            headers = {'Content-Type': 'application/octet-stream'}
            conn.request("PUT", url_parts.path + "?" + url_parts.query, body=file_data, headers=headers)
            response = conn.getresponse()
            if 200 <= response.status < 300:
                return True
            return False
    except Exception as e:
        print(f"Error uploading to R2: {e}")
        return False

cams = dict()
active_subprocesses = []
import socket
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server with centralized cleanup management"""
    def __init__(self, server_address, RequestHandlerClass):
        ThreadingMixIn.__init__(self)
        HTTPServer.__init__(self, server_address, RequestHandlerClass)
        self.cleanup_stop_event = threading.Event()
        self.cleanup_thread = None
        self._setup_cleanup_thread()

    def _setup_cleanup_thread(self):
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_stop_event.clear()
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_task,
                daemon=True,
                name="StorageCleanup"
            )
            self.cleanup_thread.start()

    def _cleanup_task(self):
        while not self.cleanup_stop_event.is_set():
            try:
                self._check_and_cleanup_storage()
            except Exception as e:
                print(f"Cleanup error: {e}")
            self.cleanup_stop_event.wait(timeout=600)  # Check every 10 min

    def _check_and_cleanup_storage(self):
        try:
            total_size = sum(f.stat().st_size for f in CAMERA_BASE_DIR.glob('**/*') if f.is_file())
            size_gb = total_size / (1000 ** 3)
            if size_gb > 60:  # 60GB threshold
                self._cleanup_oldest_files()
        except Exception as e:
            print(f"Error checking storage: {e}")

    def _cleanup_oldest_files(self):
        try:
            camera_dirs = []
            for cam_dir in CAMERA_BASE_DIR.iterdir():
                if cam_dir.is_dir():
                    try:
                        dir_size = sum(f.stat().st_size for f in cam_dir.glob('**/*') if f.is_file())
                        camera_dirs.append((cam_dir, dir_size))
                    except Exception as e:
                        print(f"Error calculating size for {cam_dir}: {e}")
            
            if not camera_dirs:
                return
                
            largest_cam = max(camera_dirs, key=lambda x: x[1])[0]
            streams_dir = largest_cam / "streams"
            
            if not streams_dir.exists():
                try:
                    shutil.rmtree(largest_cam)
                    print(f"Deleted camera folder with no streams: {largest_cam}")
                except Exception as e:
                    print(f"Error deleting camera folder {largest_cam}: {e}")
                return
                
            recordings = []
            for rec_dir in streams_dir.iterdir():
                if rec_dir.is_dir():
                    try:
                        recordings.append((rec_dir, rec_dir.stat().st_ctime))
                    except Exception as e:
                        print(f"Error getting ctime for {rec_dir}: {e}")
                    
            if not recordings:
                try:
                    shutil.rmtree(largest_cam)
                    print(f"Deleted camera folder with empty streams: {largest_cam}")
                except Exception as e:
                    print(f"Error deleting empty camera folder {largest_cam}: {e}")
                return
                
            recordings.sort(key=lambda x: x[1])
            oldest_recording = recordings[0][0]
            try:
                shutil.rmtree(oldest_recording)
                event_images_dir = largest_cam / "event_images" / oldest_recording.name
                event_images_dir_raw = f'{largest_cam}_raw' / "event_images" / oldest_recording.name
                if event_images_dir.exists():
                    shutil.rmtree(event_images_dir)
                    shutil.rmtree(event_images_dir_raw)
                print(f"Deleted oldest recording: {oldest_recording}")
            except Exception as e:
                print(f"Error deleting recording {oldest_recording}: {e}")
            
        except Exception as e:
            print(f"Error in cleanup process: {e}")

    def server_close(self):
        """Clean shutdown"""
        if hasattr(self, 'cleanup_stop_event'):
            self.cleanup_stop_event.set()
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        super().server_close()

if __name__ == "__main__":
  if platform.system() == 'Darwin': subprocess.Popen(['caffeinate', '-dimsu'])
  elif platform.system() == 'Windows': ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
  elif platform.system() == 'Linux': subprocess.Popen(['systemd-inhibit', '--why=Running script', '--mode=block', 'sleep', '999999'])

  if os.path.exists(CAMS_FILE):
      with open(CAMS_FILE, 'rb') as f:
        cams = pickle.load(f)
  else:
      with open(CAMS_FILE, 'wb') as f:
         pickle.dump(cams, f)  

  rtsp_url = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--rtsp=")), None)
  classes = {"0","1","2","7"} # person, bike, car, truck, bird (14)

  userID = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--userid=")), None)
  key = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--key=")), None)
  yolo_variant = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--yolo_size=")), None)
  if not yolo_variant: yolo_variant = input("Select YOLOV8 size from [n,s,m,l,x], or press enter to skip (defaults to n):") or "n"

  if rtsp_url is None and userID is None:
    userID = input("enter your Clearcam user id or press Enter to skip: ")
    if len(userID) > 0:
      key = ""
      while len(key) < 1: key = input("enter a password for encryption: ")
      sys.argv.extend([f"--userid={userID}", f"--key={key}", f"--yolo_size={yolo_variant}"])
    else: userID = None

  if userID is not None and key is None:
    print("Error: key is required when userID is provided")
    sys.exit(1)
  
  cam_name = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--cam_name=")), "my_camera")


  track = True #for now
  if track: 
    from yolox.tracker.byte_tracker import BYTETracker
    class Args:
        def __init__(self):
            self.track_buffer = 60 # frames, was 30
            self.mot20 = False
            self.match_thresh = 0.9

    tracker = BYTETracker(Args())
  live_link = dict()
  
  if rtsp_url is None:
    for cam_name in cams.keys():
      start_cam(rtsp=cams[cam_name],cam_name=cam_name,yolo_variant=yolo_variant)

  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  depth, width, ratio = get_variant_multiples(yolo_variant)
  if rtsp_url:
    yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
    state_dict = safe_load(get_weights_location(yolo_variant))
    load_state_dict(yolo_infer, state_dict)
    cam = VideoCapture(rtsp_url,cam_name=cam_name)
    hls_streamer = HLSStreamer(cam,cam_name=cam_name)
    cam.streamer = hls_streamer
  
  try:
    try:
      server = ThreadedHTTPServer(('0.0.0.0', 8080), HLSRequestHandler)
      print(f"Serving at http://{get_lan_ip()}:8080")
    except OSError as e:
      if e.errno == socket.errno.EADDRINUSE:
        print("Port in use, server not started.")
        server = None
      else:
          raise
    
    if rtsp_url:
      hls_streamer.start()
      restart_time = (0, 0)
      threading.Thread(
        target=schedule_daily_restart,
        args=(hls_streamer, restart_time),
        daemon=True
      ).start()

    if server:
      server.serve_forever()
    else:
      while True: time.sleep(3600)

  except KeyboardInterrupt:
    if rtsp_url:
      hls_streamer.stop()
      cam.release()
      server.shutdown()