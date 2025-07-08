from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.tensor import Tensor
from tinygrad import TinyJit
from tinygrad.device import is_dtype_supported
from tinygrad import dtypes
import numpy as np
from itertools import chain
from pathlib import Path
import cv2
from collections import defaultdict
import time, sys
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
import json
import cv2
import time
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import shutil
import requests
from datetime import datetime, time as time_obj
import uuid
from collections import deque

from pathlib import Path
import struct

#Model architecture from https://github.com/ultralytics/ultralytics/issues/189
#The upsampling class has been taken from this pull request https://github.com/tinygrad/tinygrad/pull/784 by dc-dc-dc. Now 2(?) models use upsampling. (retinet and this)

#Pre processing image functions.
def compute_transform(image, new_shape=(1280, 1280), auto=False, scaleFill=False, scaleup=True, stride=32) -> Tensor:
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
  image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR) if shape[::-1] != new_unpad else image
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
  return Tensor(image)

def preprocess(im, imgsz=1280, model_stride=32, model_pt=True): return compute_transform(im, new_shape=imgsz, auto=True, stride=model_stride)

# utility functions for forward pass.
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

  def return_all_trainable_modules(self):
    backbone_modules = [*range(10)]
    yolov8neck_modules = [12, 15, 16, 18, 19, 21]
    yolov8_head_weights = [(22, self.head)]
    return [*zip(backbone_modules, self.net.return_modules()), *zip(yolov8neck_modules, self.fpn.return_modules()), *yolov8_head_weights]

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
  weights_location = Path(__file__).parents[1] / "weights" / f'yolov8{yolo_variant}.safetensors'
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

def resolve_youtube_stream_url(youtube_url):
    import yt_dlp
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'noplaylist': True,
        'force_generic_extractor': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        if 'url' in info:
            return info['url']
        elif 'entries' in info and info['entries']:
            return info['entries'][0]['url']
        else:
            raise RuntimeError("Could not resolve YouTube stream URL")

class VideoCapture:
    def __init__(self, src):
        # objects in scene count
        self.object_dict = defaultdict(int)
        self.object_queue = []

        self.src = src
        self.width = 1280  # Reduced resolution for better performance
        self.height = 720
        self.proc = None
        self.running = True

        self.raw_frame = None
        self.annotated_frame = None
        self.last_preds = []
        self.dir = None

        self.lock = threading.Lock()

        self._open_ffmpeg()

        # Start threads
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.inference_loop, daemon=True).start()

    def _open_ffmpeg(self):
        if self.proc:
            self.proc.kill()
        
        if "youtube.com" in self.src or "youtu.be" in self.src: self.src = resolve_youtube_stream_url(self.src)
        
        command = [
            "ffmpeg",
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


    def capture_loop(self):
        frame_size = self.width * self.height * 3
        fail_count = 0
        count = 0
        last_det = -1
        send_det = False
        last_live_check = time.time()
        last_live_seg = time.time()
        while self.running:
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
                filtered_preds = [p for p in self.last_preds if p[4] >= 0.4 and (classes is None or str(int(p[5])) in classes)]
                objects = [int(x[5]) for x in filtered_preds]
                self.object_queue.append(objects)
                if count % 10 == 0: # todo magic 10s
                    last_dict = self.object_dict.copy()
                count = (count + 1) % 10
                for x in objects:
                    self.object_dict[int(x)] += 1
                if len(self.object_queue) > 10:
                    for x in self.object_queue[0]: self.object_dict[int(x)] -= 1
                    del self.object_queue[0]
                    for k in self.object_dict.keys():
                        if abs(self.object_dict[k] - last_dict[k]) > 2:
                            if time.time() - last_det >= 60: # once per min for now
                                send_det = True
                                print("DETECTED") # todo, magic 5, change of 5 over 10 frames = detection
                                os.makedirs("event_images", exist_ok=True)
                                timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                                filename = f"event_images/frame_{timestamp}.jpg"
                                cv2.imwrite(filename, self.annotated_frame)
                                if userID is not None: threading.Thread(target=send_notif(userID), daemon=True).start()
                                last_det = time.time()
                        if (send_det and userID is not None) and time.time() - last_det >= 15: #send 15ish second clip after
                            os.makedirs("event_clips", exist_ok=True)
                            mp4_filename = f"event_clips/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
                            self.streamer.export_last_segments(Path(mp4_filename))
                            encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), key)
                            threading.Thread(target=upload_file, args=(Path(f"""{mp4_filename}.aes"""), userID), daemon=True).start()
                            send_det = False
                    if live and (time.time() - last_live_check) >= 5:
                        last_live_check = time.time()
                        print("CHECKING FOR LIVE",live_link)
                        threading.Thread(target=check_upload_link, daemon=True).start()
                    if live_link and (time.time() - last_live_seg) >= 2:
                        last_live_seg = time.time()
                        mp4_filename = f"segment.mp4"
                        self.streamer.export_last_segments(Path(mp4_filename),last=True)
                        encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), key)
                        threading.Thread(target=upload_to_r2, args=(Path(f"""{mp4_filename}.aes"""), live_link), daemon=True).start()
                with self.lock:
                    self.raw_frame = frame.copy()
                    self.annotated_frame = self.draw_predictions(frame.copy(), filtered_preds)
                time.sleep(1 / 30)
            except Exception as e:
                print("Error in capture_loop:", e)
                self._open_ffmpeg()
                time.sleep(1)
    

    def inference_loop(self):
        prev_time = time.time()
        while self.running:
            with self.lock:
                frame = self.raw_frame.copy() if self.raw_frame is not None else None
            if frame is not None:
                pre = preprocess(frame)
                preds = do_inf(pre).numpy()
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            font_color = (0, 0, 0) if self.is_bright_color(color) else (255, 255, 255)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 2, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)
        return frame

    def get_frame(self):
        with self.lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
        return None

    def release(self):
        self.running = False
        if self.proc:
            self.proc.kill()

class HLSStreamer:
    def __init__(self, video_capture, output_dir="hls_output", segment_time=2):
        self.cam = video_capture
        self.output_dir = Path(output_dir)
        self.segment_time = segment_time
        self.running = False
        self.ffmpeg_proc = None

        if not self.output_dir.exists(): self.output_dir.mkdir()
    
    def _get_new_stream_dir(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        stream_dir = self.output_dir / timestamp
        self.cam.dir = stream_dir
        stream_dir.mkdir(exist_ok=True)
        return stream_dir
        
    def start(self):
        self.running = True
        self.current_stream_dir = self._get_new_stream_dir()
        self.recent_segments = deque(maxlen=8)
        # Start FFmpeg process for HLS streaming to local files
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.cam.width}x{self.cam.height}",
            "-use_wallclock_as_timestamps", "1",
            "-fflags", "+genpts",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "21",
            "-preset", "veryfast",
            "-g", str(30 * self.segment_time),
            "-f", "hls",
            "-hls_time", str(self.segment_time),
            "-hls_list_size", "0",
            "-hls_flags", "delete_segments",
            "-hls_allow_cache", "0",
            str(self.current_stream_dir / "stream.m3u8")
        ]

        self.ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        threading.Thread(target=self._feed_frames, daemon=True).start()
        threading.Thread(target=self._track_segments, daemon=True).start()


    def export_last_segments(self,output_path: Path,last=False):
        if not self.recent_segments:
            print("No segments available to save.")
            return

        concat_list_path = self.current_stream_dir / "concat_list.txt"
        segments_to_use = [self.recent_segments[-1]] if last else self.recent_segments
        with open(concat_list_path, "w") as f:
            f.writelines(f"file '{segment.resolve()}'\n" for segment in segments_to_use)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if last:
            # Re-encode with scaling and compression
            command = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-vf", "scale=-2:240,fps=24",
                "-c:v", "libx264",
                "-preset", "veryslow",
                "-crf", "32",
                "-an",
                str(output_path)
            ]
        else:
            # Just copy original
            command = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-c", "copy",
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
            self.recent_segments.extend(segment_files[-8:]) # last 3 for now
            time.sleep(self.segment_time / 2)

    def _feed_frames(self):
        while self.running:
            frame = self.cam.get_frame()
            if frame is not None:
                try:
                    self.ffmpeg_proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    print("FFmpeg process died, restarting...")
                    self.start()
                    return
            time.sleep(1/30)  # Match the frame rate
    
    def stop(self):
        self.running = False
        if self.ffmpeg_proc:
            self.ffmpeg_proc.terminate()
            try:
                self.ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()

class HLSRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.hls_dir = Path("hls_output")
        super().__init__(*args, **kwargs)
    
    def _get_latest_stream(self):
        """Find the most recent stream directory"""
        stream_dirs = list(self.hls_dir.glob("*"))
        if not stream_dirs:
            return None
        return max(stream_dirs, key=os.path.getmtime)

    def do_GET(self):
      if self.path == '/':
        # Serve a simple HTML page with a video player
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        </head>
        <body>
            <video id="video" controls width="{self.server.cam.width}"></video>
            <script>
                if(Hls.isSupported()) {{
                    var video = document.getElementById('video');
                    var hls = new Hls();
                    hls.loadSource('stream.m3u8');
                    hls.attachMedia(video);
                    hls.on(Hls.Events.MANIFEST_PARSED,function() {{
                        video.play();
                    }});
                }}
                else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                    video.src = 'stream.m3u8';
                    video.addEventListener('loadedmetadata',function() {{
                        video.play();
                    }});
                }}
            </script>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))
        return
      

      latest_dir = self._get_latest_stream()
      if not latest_dir:
          self.send_error(404)
          return
          
      file_path = latest_dir / self.path.lstrip('/')
      if not file_path.exists():
          self.send_error(404)
          return
          
      self.send_response(200)
      if file_path.suffix == '.m3u8':
          self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
          self.send_header('Cache-Control', 'no-cache')
      elif file_path.suffix == '.ts':
          self.send_header('Content-Type', 'video/MP2T')
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

def send_notif(session_token: str):
    host = "www.rors.ai"
    endpoint = "/test/send"
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

        # Add MAGIC header and pad
        data = struct.pack('<Q', MAGIC_NUMBER) + plaintext
        padded = pkcs7_pad(data, AES_BLOCK_SIZE)

        ciphertext = encrypt_cbc(padded, key_bytes, iv)

        with open(output_path, 'wb') as f:
            f.write(iv + ciphertext)  # ObjC expects IV prepended

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
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    file_size = len(file_data)
    try:
        params = {
            "filename": file_name,
            "session_token": session_token,
            "size": str(file_size)
        }
        response = requests.get(
            f"https://rors.ai/test/upload",
            params=params,
            timeout=10
        )
        if response.status_code != 200:
            print(f"Failed to get upload URL: {response.status_code}")
            return False
        response_data = response.json()
        if not response_data.get("url"):
            print("Invalid response - missing upload URL")
            return False 
        presigned_url = response_data["url"]
    except Exception as e:
        print(f"Error getting upload URL: {e}")
        return False

    for attempt in range(4):
        try:
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(file_size)
            }
            
            upload_response = requests.put(
                presigned_url,
                headers=headers,
                data=file_data,
                timeout=30
            )
            
            if 200 <= upload_response.status_code < 300:
                print(f"File uploaded successfully on attempt {attempt + 1}")
                return True
            else:
                print(f"Upload failed with status {upload_response.status_code} on attempt {attempt + 1}")
        except Exception as e:
            print(f"Upload error on attempt {attempt + 1}: {e}")
        
        # Exponential backoff before retrying
        if attempt < 4 - 1:
            delay = 3 * (2 ** attempt)
            print(f"Waiting {delay:.1f} seconds before retrying...")
            time.sleep(delay)
  
    return False

live_link = False
is_live_lock = threading.Lock()
def check_upload_link():
    global live_link
    url = f"https://rors.ai/test/get_stream_upload_link?name=clearcampy&session_token={userID}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        upload_link = json_data.get("upload_link")
        with is_live_lock: live_link = upload_link
    except Exception as e:
        with is_live_lock:
            live_link = None
        print(f"Error checking upload link: {e}")

def upload_to_r2(file_path: Path, signed_url: str, max_retries: int = 0) -> bool:
    with file_path.open('rb') as f:
        _ = requests.put(
            signed_url,
            data=f,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=2
        )
      

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Serve HTTP requests in separate threads."""

if __name__ == "__main__":
  rtsp_url = sys.argv[1] if len(sys.argv) >= 2 else (print("No rtsp url given") or sys.exit(1))
  classes = {"0","1","2","7","14"} # person, bike, car, truck, bird

  userID = next((arg.split("=", 1)[1] for arg in sys.argv[2:] if arg.startswith("--userid=")), None)
  key = next((arg.split("=", 1)[1] for arg in sys.argv[2:] if arg.startswith("--key=")), None)
  if userID is not None and key is None:
    print("Error: key is required when userID is provided")
    sys.exit(1)
  live = next((arg.split("=", 1)[1] for arg in sys.argv[2:] if arg.startswith("--key=")), None)
  live_link = None
  device_name = "clearcam py"
  
  # Model initialization
  yolo_variant = sys.argv[2] if len(sys.argv) >= 3 else (print("No variant given, so choosing 'n' as the default. Yolov8 has different variants, you can choose from ['n', 's', 'm', 'l', 'x']") or 'n')
  depth, width, ratio = get_variant_multiples(yolo_variant)
  yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
  state_dict = safe_load(get_weights_location(yolo_variant))
  load_state_dict(yolo_infer, state_dict)
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  
  cam = VideoCapture(rtsp_url)
  hls_streamer = HLSStreamer(cam)
  cam.streamer = hls_streamer
  
  try:
      server = ThreadedHTTPServer(('0.0.0.0', 8080), HLSRequestHandler)
      server.cam = cam  # Pass camera reference to server
      
      hls_streamer.start()

      restart_time = (0, 0)
      scheduler = threading.Thread(
      target=schedule_daily_restart,
      args=(hls_streamer, restart_time),
      daemon=True
      )
      scheduler.start()
      
      print("\nServing HLS stream on:")
      print(f"  - Web player: http://localhost:8080/")
      print(f"  - HLS URL (for VLC): http://localhost:8080/stream.m3u8")
      print("\nPress Ctrl+C to stop...")
      server.serve_forever()
  except KeyboardInterrupt:
      print("\nShutting down...")
      hls_streamer.stop()
      cam.release()
      server.shutdown()
