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
def do_inf(im, yolo_infer):
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

BASE = Path(__file__).parent / "data"
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
    self.is_notif = True
    self.zone = True

  def add(self, class_id):
    if self.classes is not None and class_id not in self.classes: return
    now = time.time()
    self.data[class_id].append(now)
    self.cleanup(class_id, now)

  def cleanup(self, class_id, now):
    q = self.data[class_id]
    window = self.window if self.window else (60 if self.is_notif else 1)
    while window and q and now - q[0] > window:
      q.popleft()

  def reset_counts(self):
    for class_id, _ in self.data.items():
       self.data[class_id] = deque() # todo, use in reset endpoint?

  def get_counts(self):
    window = self.window if self.window else (60 if self.is_notif else 1)
    max_reached = False
    now = time.time()
    counts = {}
    for class_id, q in self.data.items():
      while window and q and now - q[0] > window:
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
    window = self.window if self.window else (60 if self.is_notif else 1)
    return time_of_day < self.sched[0][1] and time_of_day > ((self.sched[0][0] - window) + offset)

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
    self.object_set_zone = set()

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
            self.alert_counters[str(uuid.uuid4())] = RollingClassCounter(window_seconds=None, max=1, classes={0,1,2,3,5,7},cam_name=cam_name)
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
      stream_dir.mkdir(parents=True, exist_ok=True)
      stream_dir_raw.mkdir(parents=True, exist_ok=True)
      return stream_dir_raw

  def _safe_kill_process(self, proc):
    if proc:
      try:
        proc.terminate()
        proc.wait(timeout=5)
      except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
      except Exception:
        pass

  def _open_ffmpeg(self):
    path = self._get_new_stream_dir()
    self._safe_kill_process(self.proc)
    self._safe_kill_process(self.hls_proc)

    ffmpeg_path = find_ffmpeg()
    
    command = [
        ffmpeg_path,
        "-re",
        *(["-rtsp_transport", "tcp"] if self.src.startswith("rtsp") else []),
        "-i", self.src,
        "-c", "copy",
        "-f", "hls",
        "-hls_list_size", "0",
        "-hls_flags", "+append_list",
        "-hls_playlist_type", "event",
        "-an",  # No audio
        "-hls_segment_filename", str(path/ "stream_%06d.ts"),
        str(path / "stream.m3u8")
    ]
    self.hls_proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(8)
    command = [
        ffmpeg_path,
        "-live_start_index", "-1",
        "-i", str(path / "stream.m3u8"),
        "-loglevel", "quiet",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "2",
        "-an",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={self.width}:{self.height}",
        "-timeout", "5000000",
        "-rw_timeout", "15000000",
        "-vsync", "2",
        "-fflags", "+discardcorrupt+fastseek+flush_packets+nobuffer",
        "-avioflags", "direct",
        "-flags", "low_delay",
        "-max_delay", "100000",
        "-threads", "1",
        "-"
    ]
    self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

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
            if fail_count > 5:
              print(f"FFmpeg frame read failed (count={fail_count}), restarting stream...")
              self._open_ffmpeg()
              fail_count = 0
            time.sleep(0.5)
            continue
          else:
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
                window = alert.window if alert.window else (60 if alert.is_notif else 1)
                if not alert.is_active(offset=4): alert.last_det = time.time() # don't send alert when just active
                if alert.get_counts()[1]:
                    if time.time() - alert.last_det >= window:
                        if alert.is_notif: send_det = True
                        timestamp = datetime.now().strftime("%Y-%m-%d")
                        filepath = CAMERA_BASE_DIR / f"{self.cam_name}/event_images/{timestamp}"
                        filepath.mkdir(parents=True, exist_ok=True)
                        self.annotated_frame = self.draw_predictions(frame.copy(), filtered_preds)
                        # todo alerts can be sent with the wrong thumbnail if two happen quickly, use map
                        ts = int(time.time() - self.streamer.start_time - 10)
                        filename = filepath / f"{ts}_notif.jpg" if alert.is_notif else filepath / f"{ts}.jpg"
                        cv2.imwrite(str(filename), self.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) # we've 10MB limit for video file, raw png is 3MB!
                        if (plain := filepath / f"{ts}.jpg").exists() and (filepath / f"{ts}_notif.jpg").exists():
                          plain.unlink() # only one image per event
                          filename = filepath / f"{ts}_notif.jpg"
                        text = f"Event Detected ({getattr(alert, 'cam_name')})" if getattr(alert, 'cam_name', None) else None
                        if userID is not None and alert.is_notif: threading.Thread(target=send_notif, args=(userID,text,), daemon=True).start()
                        last_det = time.time()
                        alert.last_det = time.time()
            if (send_det and userID is not None) and time.time() - last_det >= 6: #send 15ish second clip after
                os.makedirs(CAMERA_BASE_DIR / self.cam_name / "event_clips", exist_ok=True)
                mp4_filename = CAMERA_BASE_DIR / f"{self.cam_name}/event_clips/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
                temp_output = CAMERA_BASE_DIR / f"{self.cam_name}/event_clips/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_temp.mp4"
                self.streamer.export_clip(Path(mp4_filename))

                # img preview?
                subprocess.run(['ffmpeg', '-i', mp4_filename, '-i', str(filename), '-map', '0', '-map', '1', '-c', 'copy', '-disposition:v:1', 'attached_pic', '-y', temp_output])
                os.replace(temp_output, mp4_filename)
                encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), key)
                threading.Thread(target=upload_file, args=(Path(f"""{mp4_filename}.aes"""), userID), daemon=True).start()
                os.unlink(mp4_filename)
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
        elif hasattr(self, 'streamer') and self.streamer.feeding_frames:
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
        preds = do_inf(pre, yolo_infer).numpy()
        if track:
          thresh = (self.settings.get("threshold") if self.settings else 0.5) or 0.5 #todo clean!
          online_targets = tracker.update(preds, [1280,1280], [1280,1280], thresh) #todo, zone in js also hardcoded to 1280
          preds = []
          for x in online_targets:
            if x.tracklet_len < 1: continue # dont alert for 1 frame, too many false positives
            outside = False
            if hasattr(self, "settings") and self.settings is not None and self.settings.get("coords"):
              outside = point_not_in_polygon([[x.tlwh[0], x.tlwh[1]],[(x.tlwh[0]+x.tlwh[2]), x.tlwh[1]],[(x.tlwh[0]), (x.tlwh[1]+x.tlwh[3])],[(x.tlwh[0]+x.tlwh[2]), (x.tlwh[1]+x.tlwh[3])]], self.settings["coords"])
              outside = outside ^ self.settings["outside"]
            non_zone_alert = False
            if outside: # check if any alerts don't use zone
              for _, alert in self.alert_counters.items():
                if not alert.zone:
                  non_zone_alert = True
                  break
              if not non_zone_alert and outside: continue
            preds.append(np.array([x.tlwh[0],x.tlwh[1],(x.tlwh[0]+x.tlwh[2]),(x.tlwh[1]+x.tlwh[3]),x.score,x.class_id]))
            if (classes is None or str(int(x.class_id)) in classes):
              new = int(x.track_id) not in self.object_set
              new_in_zone = int(x.track_id) not in self.object_set_zone and not outside
              if new:
                self.object_set.add(int(x.track_id))
                self.counter.add(int(x.class_id))
              if new_in_zone: self.object_set_zone.add(int(x.track_id))
              for _, alert in self.alert_counters.items():
                if not alert.get_counts()[1] and ((new and not alert.zone) or (new_in_zone and alert.zone)): alert.add(int(x.class_id))
                
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
        stream_dir.mkdir(exist_ok=True)
        stream_dir_raw.mkdir(exist_ok=True)
        return stream_dir, stream_dir_raw
        
    def start(self):
        self.running = True
        self.current_stream_dir, self.current_stream_dir_raw = self._get_new_stream_dir()
        self.recent_segments_raw = deque()
        self.recent_segments_raw_live = deque()
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
            "-f", "hls",
            "-hls_time", "1",
            "-hls_list_size", "0",
            "-hls_flags", "delete_segments",
            "-hls_allow_cache", "0",
            str(self.current_stream_dir / "stream.m3u8")
        ]

        self.ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def export_clip(self,output_path: Path,live=False):
        self._track_segments()

        concat_list_path = self.current_stream_dir_raw / "concat_list.txt"
        segments_to_use = self.recent_segments_raw_live if live else self.recent_segments_raw # last 5 seconds
        with open(concat_list_path, "w") as f:
            f.writelines(f"file '{segment.resolve()}'\n" for segment in segments_to_use)
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
          subprocess.run(command, check=True)
        else:
          with open(self.current_stream_dir_raw / "concat_list.txt", "r") as f: print(" ".join(line.strip() for line in f))
          command = [
            ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",  # needed for android
            "-an",  # No audio
            str(output_path)
          ]
          subprocess.run(command, check=True)
          comp = 5
          file_size = 10*1024*1024
          with open(output_path, "rb") as f:
            file_data = f.read()
            file_size = len(file_data)
          while file_size >= 9*1024*1024: # max size 10MB, # todo, calculate time from ts files
            temp_output = output_path.with_stem(output_path.stem + "_compressed")
            command = [
              ffmpeg_path,
              "-y",
              "-f", "concat",
              "-safe", "0",
              "-i", str(concat_list_path),
              "-c:v", "libx264",
              "-crf", str(18 + comp),
              "-pix_fmt", "yuv420p",  # needed for android
              "-an",  # No audio
              str(temp_output)
            ]
            subprocess.run(command, check=True)
            os.replace(temp_output, output_path)
            with open(output_path, "rb") as f:
              file_data = f.read()
            file_size = len(file_data)
            comp += 5

    def _track_segments(self): # todo shouldn't need a loop here?
      cutoff = time.time() - 20
      live_cutoff = time.time() - 5
      segment_files_raw = sorted(self.current_stream_dir_raw.glob("*.ts"), key=os.path.getmtime)
      recent_raw = [f for f in segment_files_raw if os.path.getmtime(f) >= cutoff]
      recent_raw_live = [f for f in segment_files_raw if os.path.getmtime(f) >= (live_cutoff)]
      self.recent_segments_raw.clear()
      self.recent_segments_raw.extend(recent_raw)
      self.recent_segments_raw_live.clear()
      self.recent_segments_raw_live.extend(recent_raw_live)

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


def point_not_in_polygon(coords, poly):
    n = len(poly)
    for j in range(len(coords)):
      inside = False
      p1x, p1y = poly[0]
      for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if coords[j][1] > min(p1y, p2y):
          if coords[j][1] <= max(p1y, p2y):
            if coords[j][0] <= max(p1x, p2x):
              if p1y != p2y:
                x_intersect = (coords[j][1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
              else:
                x_intersect = p1x
              if p1x == p2x or coords[j][0] <= x_intersect: inside = not inside
        p1x, p1y = p2x, p2y
      if inside:
        return False
    return True

class HLSRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.base_dir = CAMERA_BASE_DIR
        self.show_dets = None
        super().__init__(*args, **kwargs)

    def send_200(self, body=None):
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(json.dumps(body).encode("utf-8"))

    def get_camera_path(self, cam_name=None):
        if cam_name:
            return self.base_dir / cam_name / "streams"
        return self.base_dir
    
    def do_GET(self):
        parsed_path = urlparse(unquote(self.path))
        query = parse_qs(parsed_path.query)
        cam_name = query.get("cam", [None])[0]

        if parsed_path.path == "/set_max_storage":
          max_gb = float(query.get("max", [None])[0])
          self.server.max_gb = max_gb
          self.send_200()
          return
        
        if parsed_path.path == "/get_max_storage":
          self.send_200(body={"max_gb":self.server.max_gb})
          return

        if parsed_path.path == "/list_cameras":
            available_cams = [d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.name.endswith("_raw")]
            self.send_200(available_cams)
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
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            if not settings_file.exists():
                with open(settings_file, "wb") as f:
                    pickle.dump(None, f)
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_file, "rb") as f:
               zone = pickle.load(f)
            if zone is None: zone = {}
            coords_json = query.get("coords", [None])[0]
            if coords_json is not None:
              coords = json.loads(coords_json)
              if isinstance(coords, list) and len(coords) >= 3:
                 zone["coords"] = [[float(x), float(y)] for x, y in coords]
            zone["is_notif"] = (str(is_notif).lower() == "true") if (is_notif := query.get("is_notif", [None])[0]) is not None else zone.get("is_notif")
            zone["outside"] = (str(outside).lower() == "true") if (outside := query.get("outside", [None])[0]) is not None else zone.get("outside")
            query.get("threshold", [None])[0] is not None and zone.update({"threshold": float(query.get("threshold", [None])[0])})
            query.get("show_dets", [self.show_dets])[0] is not None and setattr(self, "show_dets", str(int(time.time()))) and zone.update({"show_dets": self.show_dets})
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
            zone = query.get("zone", [None])[0]
            is_notif = query.get("is_notif", [None])[0]
            if alert_id is None: # no id, add alert
                window = query.get("window", [None])[0]
                max_count = query.get("max", [None])[0]
                class_ids = query.get("class_ids", [None])[0]
                sched = json.loads(query.get("sched", ["[[0,86400],[0,86400],[0,86400],[0,86400],[0,86400],[0,86400],[0,86400]]"])[0]) # todo, weekly
                if window: window = int(window)
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
              if is_on is not None or is_notif is not None or zone is not None:
                if is_on is not None: raw_alerts[alert_id].is_on = str(is_on).lower() == "true"
                if is_notif is not None: raw_alerts[alert_id].is_notif = str(is_notif).lower() == "true"
                if zone is not None: raw_alerts[alert_id].zone = str(zone).lower() == "true"
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
            
            self.send_200(zone)
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
                                "is_notif": alert.is_notif,
                                "zone": alert.zone,
                            })
                except Exception as e:
                    self.send_error(500, f"Failed to load alerts: {e}")
                    return

            self.send_200(alert_info)
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

            self.send_200(labeled_counts)
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
          with open("mainview.html", "r", encoding="utf-8") as f: html = f.read()
          self.send_response(200)
          self.send_header('Content-type', 'text/html')
          self.end_headers()
          self.wfile.write(html.encode('utf-8'))
          return
                            
        if parsed_path.path == '/event_thumbs' or parsed_path.path.endswith('/event_thumbs'): # todo clean
            selected_dir = parse_qs(parsed_path.query).get("folder", [datetime.now().strftime("%Y-%m-%d")])[0]
            name_contains = parse_qs(parsed_path.query).get("name_contains", [None])[0]
            image_data = []
            
            if cam_name is not None:
                event_image_dir = self.base_dir / cam_name / "event_images"
                event_image_path = event_image_dir / selected_dir
                event_images = sorted(
                    event_image_path.glob("*.jpg"),
                    key=lambda p: int(p.stem.split('_')[0]), # n.jpg or n_{s}.jpg?
                    reverse=True
                ) if event_image_path.exists() else []
                
                for img in event_images:
                    if name_contains and name_contains not in img.name: continue
                    ts = int(img.stem.split('_')[0]) # n.jpg or n_{s}.jpg?
                    image_url = f"/{img.relative_to(self.base_dir.parent)}"
                    image_data.append({
                        "url": image_url,
                        "timestamp": ts,
                        "filename": img.name,
                        "cam_name": cam_name,
                        "folder": selected_dir
                    })
            else:
                for camera_dir in self.base_dir.iterdir():
                    if camera_dir.is_dir():
                        event_image_dir = camera_dir / "event_images"
                        event_image_path = event_image_dir / selected_dir
                        if event_image_path.exists():
                            event_images = sorted(
                                event_image_path.glob("*.jpg"),
                                key=lambda p: int(p.stem.split('_')[0]), # n.jpg or n_{s}.jpg?
                                reverse=True
                            )
                            for img in event_images:
                                if name_contains and name_contains not in img.name: continue
                                ts = int(img.stem.split('_')[0]) 
                                image_url = f"/{img.relative_to(self.base_dir.parent)}"
                                image_data.append({
                                    "url": image_url,
                                    "timestamp": ts,
                                    "filename": img.name,
                                    "cam_name": camera_dir.name,
                                    "folder": selected_dir
                                })
            image_data.sort(key=lambda x: x["timestamp"], reverse=True)
            response_data = {
                "images": image_data,
                "count": len(image_data),
                "folder": selected_dir,
                "cam_name": cam_name if cam_name is not None else "all_cameras"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return

        if parsed_path.path == '/' or parsed_path.path == f'/{cam_name}':
            selected_dir = parse_qs(parsed_path.query).get("folder", [datetime.now().strftime("%Y-%m-%d")])[0]
            start_param = parse_qs(parsed_path.query).get("start", [None])[0]
            show_detections_param = parse_qs(parsed_path.query).get("show_detections", ["false"])[0]
            show_detections = show_detections_param.lower() in ("true", "1", "yes")

            try:
                start_time = max(float(start_param),0) if start_param is not None else None
            except ValueError:
                start_time = None

            available_cams = [d.name for d in self.base_dir.iterdir() if d.is_dir()]

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('cameraview.html', 'r', encoding='utf-8') as f: html = f.read()
            replacements = {
                '{selected_dir}': selected_dir,
                '{show_detections_checked}': 'checked' if show_detections else '',
                '{class_labels}': json.dumps(class_labels),
                '{start_time}': str(start_time) if start_time is not None else 'null',
                '{cam_name}': cam_name
            }
            for placeholder, value in replacements.items(): html = html.replace(placeholder, value)
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


def schedule_daily_restart(hls_streamer, videocapture, restart_time):
    while True:
        now = datetime.now().time()
        target = time_obj(restart_time[0], restart_time[1])
        if now >= target:
            delta = (24 * 3600) - ((now.hour * 3600 + now.minute * 60 + now.second) - (target.hour * 3600 + target.minute * 60))
        else:
            delta = ((target.hour * 3600 + target.minute * 60) - 
                    (now.hour * 3600 + now.minute * 60 + now.second))
        time.sleep(delta)
        videocapture._open_ffmpeg() #restart 
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
            with is_live_lock:
                if cam_name in live_link: 
                    live_link[cam_name] = None

def upload_to_r2(file_path: Path, signed_url: str, max_retries: int = 0) -> bool:
    try:
        url_parts = urllib.parse.urlparse(signed_url)
        if url_parts.scheme == 'https':
            conn = http.client.HTTPSConnection(url_parts.netloc)
        else:
            conn = http.client.HTTPConnection(url_parts.netloc)
        
        with file_path.open('rb') as f:
            headers = {'Content-Type': 'application/octet-stream'}
            conn.request("PUT", url_parts.path + "?" + url_parts.query, body=f, headers=headers)
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
    def __init__(self, server_address, RequestHandlerClass):
        ThreadingMixIn.__init__(self)
        HTTPServer.__init__(self, server_address, RequestHandlerClass)
        self.cleanup_stop_event = threading.Event()
        self.cleanup_thread = None
        self.max_gb = 40
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
      total_size = sum(f.stat().st_size for f in CAMERA_BASE_DIR.glob('**/*') if f.is_file())
      size_gb = total_size / (1000 ** 3)
      if size_gb > self.max_gb: self._cleanup_oldest_files()

    def _cleanup_oldest_files(self):
      camera_dirs = []
      for cam_dir in CAMERA_BASE_DIR.iterdir():
        if cam_dir.is_dir():
          dir_size = sum(f.stat().st_size for f in cam_dir.glob('**/*') if f.is_file())
          camera_dirs.append((cam_dir, dir_size))
      
      if not camera_dirs: return
          
      largest_cam = max(camera_dirs, key=lambda x: x[1])[0]
      streams_dir = largest_cam / "streams"
      
      if not streams_dir.exists():
        shutil.rmtree(largest_cam)
        return
          
      recordings = []
      for rec_dir in streams_dir.iterdir():
        if rec_dir.is_dir(): recordings.append((rec_dir, rec_dir.stat().st_ctime))
      if not recordings:
        shutil.rmtree(largest_cam)
        print(f"Deleted camera folder with empty streams: {largest_cam}")
        return
      recordings.sort(key=lambda x: x[1])
      oldest_recording = recordings[0][0]
      shutil.rmtree(oldest_recording)
      event_images_dir = largest_cam.with_name(largest_cam.name.replace("_raw", "")) / Path("event_images") / Path(oldest_recording.name) # todo, remove _raw
      if event_images_dir.exists(): shutil.rmtree(event_images_dir)
      print(f"Deleted oldest recording: {oldest_recording}")

    def server_close(self):
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
        args=(hls_streamer, cam, restart_time),
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