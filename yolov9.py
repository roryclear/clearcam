import cv2
from tinygrad import Tensor
from tinygrad.helpers import fetch
from tinygrad.dtype import dtypes
from tinygrad.nn.state import load_state_dict, safe_load
import tinygrad.nn as nn
import numpy as np
from collections import defaultdict
import time
from pathlib import Path

class Sequential():
    def __init__(self, size=0, list=None):
      self.size = size
      if list is not None:
        self.list = list
      else:
       self.list = [None] * size
    def __call__(self, x): return x.sequential(self.list)
    def __len__(self): return len(self.list)
    def __setitem__(self, key, value): self.list[key] = value
    def __getitem__(self, idx): return self.list[idx]

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv():
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride=1, padding:int|tuple[int, ...]|str=0,
               dilation=1, groups=1, bias=True, f=-1):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.f = f
    def __call__(self, x): return self.conv(x).silu()

class ADown():
    def __init__(self, ch0=128, f=-1):
      self.cv1 = Conv(in_channels=ch0, out_channels=ch0, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True)
      self.cv2 = Conv(in_channels=ch0, out_channels=ch0, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
      self.f = f
    def __call__(self, x):
      
      x = Tensor.avg_pool2d(x, 2, 1, 1, 0, False, True)
      x1,x2 = x.chunk(2, 1)
      x1 = self.cv1(x1)
      x2 = Tensor.max_pool2d(x2, kernel_size=3, stride=2, dilation=1, padding=1)
      x2 = self.cv2(x2)
      return Tensor.cat(x1, x2, dim=1)

class AConv():
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride=1, padding:int|tuple[int, ...]|str=0,
               dilation=1, groups=1, bias=True, f=-1):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.f = f

    def __call__(self, x):
        x = Tensor.avg_pool2d(x, kernel_size=2, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        return self.cv1(x)

class ELAN1(): # todo, hardcoded, might work on all though
    def __init__(self, ch0=32, ch1=32, ch2=16, ch3=64, f=-1):  # ch_in, ch_out, number, shortcut, groups, expansion
      self.f = f
      self.cv1 = Conv(in_channels=ch0, out_channels=ch1, kernel_size=1, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
      self.cv2 = Conv(in_channels=ch2, out_channels=ch2, kernel_size=3, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
      self.cv3 = Conv(in_channels=ch2, out_channels=ch2, kernel_size=3, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
      self.cv4 = Conv(in_channels=ch3, out_channels=ch1, kernel_size=1, stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    def __call__(self, x):
      y = self.cv1(x)
      y = y.chunk(2,1)
      y = list(y)
      y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
      y = Tensor.cat(y[0], y[1], y[2], y[3], dim=1)
      y = self.cv4(y)
      return y
    
class RepNBottleneck():
    # Standard bottleneck
    def __init__(self, ch):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.cv1 = Conv(in_channels=ch,out_channels=ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
        self.cv2 = Conv(in_channels=ch,out_channels=ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)

    def __call__(self, x): return x + self.cv2(self.cv1(x))

  
class RepNCSP():
    def __init__(self, a=1, b=1, n=3):
        self.cv1 = Conv(a, b, 1, 1)
        self.cv2 = Conv(a, b, 1, 1)
        self.cv3 = Conv(a, a, 1, 1)
        self.m = Sequential(size=n)
        for i in range(n): self.m[i] = RepNBottleneck(b)

    def __call__(self, x):
      x1 = self.cv1(x)
      x2 = self.m(x1)
      x3 = self.cv2(x)
      x4 = Tensor.cat(x2, x3, dim=1)
      return self.cv3(x4)

class RepNCSPELAN4():
    def __init__(self, a=1, b=1, c=1, n=3, f=-1, size=2):
        self.cv1 = Conv(in_channels=a, out_channels=b*4, kernel_size=1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True)
        self.cv2 = Sequential(size=size)
        self.cv2[0] = RepNCSP(b*2, b, n)
        if size > 1: self.cv2[1] = Conv(in_channels=b*2, out_channels=b*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.cv3 = Sequential(size=2)
        self.cv3[0] = RepNCSP(b*2, b, n)
        if size > 1: self.cv3[1] = Conv(in_channels=b*2, out_channels=b*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.cv4 = Conv(in_channels=b*8, out_channels=c, kernel_size=1, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        self.f = f

    def __call__(self, x):
      x = self.cv1(x)
      y0, y1 = x.chunk(2, 1)
      y2 = self.cv2(y1)
      y3 = self.cv3(y2)
      concat_result = Tensor.cat(y0, y1, y2, y3, dim=1)
      return self.cv4(concat_result)

class SP():
    def __init__(self, k=3, s=1):
        self.k = k
        self.s = s

    def __call__(self, x): return Tensor.max_pool2d(x, self.k, self.s, dilation=1, padding=self.k//2)

class SPPELAN():
    def __init__(self, ch0=128, ch1=64, ch2=256, ch3=128, f=-1):
        self.cv1 = Conv(in_channels=ch0, out_channels=ch1, kernel_size=1, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        self.cv2 = SP(s=1, k=5)
        self.cv3 = SP(s=1, k=5)
        self.cv4 = SP(s=1, k=5)
        self.cv5 = Conv(in_channels=ch2, out_channels=ch3, kernel_size=1, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        self.f = f

    def __call__(self, x):
        y = [self.cv1(x)]
        y.append(self.cv2(y[-1]))
        y.append(self.cv3(y[-1]))
        y.append(self.cv4(y[-1]))
        y = Tensor.cat(*y, dim=1)
        return self.cv5(y)

class Concat():
    def __init__(self, dimension=1, f=-1):
      self.d = dimension
      self.f = f
    def __call__(self, x): return Tensor.cat(x[0],x[1],dim=self.d)

class DDetect():
    def __init__(self, a=64, b=96, c=128, d=80, f=[15, 18, 21]):  # detection layer
        self.anchors = Tensor.empty((2, 22680))
        self.strides = Tensor.empty((1, 22680))
        self.dfl = DFL(c1=16)
        self.cv2 = Sequential(size=3)
        self.cv3 = Sequential(size=3)
        self.cv2[0] = Sequential(size=3)
        self.cv2[1] = Sequential(size=3)
        self.cv2[2] = Sequential(size=3)
        self.cv3[0] = Sequential(size=3)
        self.cv3[1] = Sequential(size=3)
        self.cv3[2] = Sequential(size=3)

        self.cv2[0][0] = Conv(in_channels=a, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True)
        self.cv2[0][1] = Conv(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=True)
        self.cv2[0][2] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=4, bias=True)

        self.cv2[1][0] = Conv(in_channels=b, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True)
        self.cv2[1][1] = Conv(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=True)
        self.cv2[1][2] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=4, bias=True)

        self.cv2[2][0] = Conv(in_channels=c, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True)
        self.cv2[2][1] = Conv(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=True)
        self.cv2[2][2] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=4, bias=True)


        self.cv3[0][0] = Conv(in_channels=a, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True)
        self.cv3[0][1] = Conv(in_channels=d, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=True)
        self.cv3[0][2] = nn.Conv2d(in_channels=d, out_channels=80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, dilation=(1, 1), bias=True)

        self.cv3[1][0] = Conv(in_channels=b, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True)
        self.cv3[1][1] = Conv(in_channels=d, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=True)
        self.cv3[1][2] = nn.Conv2d(in_channels=d, out_channels=80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, dilation=(1, 1), bias=True)

        self.cv3[2][0] = Conv(in_channels=c, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True)
        self.cv3[2][1] = Conv(in_channels=d, out_channels=d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=True)
        self.cv3[2][2] = nn.Conv2d(in_channels=d, out_channels=80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, dilation=(1, 1), bias=True)

        self.nl = 3
        self.no = 144
        self.f = f
        self.reg_max = 16
        self.nc = 80
      
    def __call__(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x0 = self.cv2[i](x[i])
            x1 = self.cv3[i](x[i])
            x[i] = Tensor.cat(x0, x1, dim=1)

        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, [8, 16, 32], 0.5))
        self.shape = shape        
        processed_tensors = []
        for xi in x:
          y = xi.view(shape[0], self.no, -1)
          processed_tensors.append(y)
        concatenated = Tensor.cat(*processed_tensors, dim=2)
        box, cls = concatenated.split((self.reg_max * 4, self.nc), 1)
        dbox = self.dfl(box)
        dbox = dist2bbox(dbox, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = Tensor.cat(dbox, Tensor.sigmoid(cls), dim=1)
        return (y, x)

class CBLinear():
    def __init__(self, ch0=64, ch1=64, c2s=[64], f=1):
        self.conv = nn.Conv2d(in_channels=ch0, out_channels=ch1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True)
        self.c2s = c2s
        self.f = f

    def __call__(self, x): return tuple(self.conv(x).split(self.c2s, dim=1))

class CBFuse():
    def __init__(self, f=1, idx=1):
        self.f = f
        self.idx = idx

    def __call__(self, xs):
        target_size = xs[-1].shape[2:]
        res = []
        for i, x in enumerate(xs[:-1]):
          tensor_to_upsample = x[self.idx[i]]
          upsampled = Tensor.interpolate(tensor_to_upsample, size=target_size, mode='nearest')
          res.append(upsampled)
        
        res += xs[-1:]
        y = Tensor.stack(*res)
        return y.sum(0)

def make_anchors(feats, strides, grid_cell_offset=0.5):
  anchor_points, stride_tensor = [], []
  for i, stride in enumerate(strides):
    _, _, h, w = feats[i].shape
    sx = Tensor.arange(w) + grid_cell_offset
    sy = Tensor.arange(h) + grid_cell_offset

    sx = sx.reshape(1, -1).repeat([h, 1]).reshape(-1)
    sy = sy.reshape(-1, 1).repeat([1, w]).reshape(-1)

    anchor_points.append(Tensor.stack(sx, sy, dim=-1).reshape(-1, 2))
    stride_tensor.append(Tensor.full((h * w), stride))
  anchor_points = anchor_points[0].cat(anchor_points[1], anchor_points[2])
  stride_tensor = stride_tensor[0].cat(stride_tensor[1], stride_tensor[2]).unsqueeze(1)
  return anchor_points, stride_tensor

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
  lt, rb = distance.chunk(2, dim)
  x1y1 = anchor_points - lt
  x2y2 = anchor_points + rb
  if xywh:
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return c_xy.cat(wh, dim=1)
  return x1y1.cat(x2y2, dim=1)

class DFL():
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.c1 = c1
    def __call__(self, x):
        b, _, a = x.shape
        x = x.view(b, 4, self.c1, a)
        return self.conv(x.transpose(2, 1).softmax(1)).view(b, 4, a)


class Upsample():
  def __init__(self, scale_factor=2, f=-1):
      self.scale_factor = scale_factor
      self.f = f
  
  def __call__(self, x):
    s = self.scale_factor
    return x.repeat_interleave(s, dim=2).repeat_interleave(s, dim=3)

class Silence():
    def __init__(self, f=-1): self.f = f
    def __call__(self, x): return x

class DetectionModel():
  def __init__(self, a=16, c=64, e=96, f=24, g=128, h=256, i=224, j=160, k=48, l=144, m=192, n=80, o=32, q=16, s=3, u=96, v=32, y=64, z=128, a1=64, a2=64, a4=128, size=None):
    if size is not None:
      self.model = Sequential(size=23)
      self.model[0] = Conv(in_channels=3, out_channels=a, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True)
      self.model[1] = Conv(in_channels=a, out_channels=a*2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),  groups=1, bias=True)
      self.model[2] = ELAN1(ch0=a*2, ch1=o, ch2=a, ch3=c) if size in ["t", "s"] else RepNCSPELAN4(y, 32, z, n=s)
      self.model[3] = ADown() if size == "c" else AConv(in_channels=o, out_channels=a1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True)
      self.model[4] = RepNCSPELAN4(c, q, a2, n=s)
      self.model[5] = ADown(256) if size == "c" else AConv(in_channels=c, out_channels=u, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True)
      self.model[6] = RepNCSPELAN4(e, f, e, n=s)
      self.model[7] = ADown(256) if size == "c" else AConv(in_channels=u, out_channels=g, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True)
      self.model[8] = RepNCSPELAN4(a4, v, a4, n=s)
      self.model[9] = SPPELAN(ch0=a4, ch1=c, ch2=h, ch3=a4)
      self.model[10] = Upsample()
      self.model[11] = Concat(f=[-1, 6])
      self.model[12] = RepNCSPELAN4(i, f, e, n=s)
      self.model[13] = Upsample()
      self.model[14] = Concat(f=[-1, 4])
      self.model[15] = RepNCSPELAN4(j, q, c, n=s)
      self.model[16] = ADown(128) if size == "c" else AConv(in_channels=a2, out_channels=k, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True)
      self.model[17] = Concat(f=[-1, 12])
      self.model[18] = RepNCSPELAN4(l, f, e, n=s)
      self.model[19] = ADown(256) if size == "c" else AConv(in_channels=u, out_channels=c, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True)
      self.model[20] = Concat(f=[-1, 9])
      self.model[21] = RepNCSPELAN4(m, v, a4, n=s)
      self.model[22] = DDetect(c, e, a4, n, f=[15, 18, 21])
    else:
      self.model = Sequential(size=43)
      self.model[0] = Silence()
      self.model[1] = Conv(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),  groups=1, bias=True)
      self.model[2] = Conv(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),  groups=1, bias=True)
      self.model[3] = RepNCSPELAN4(128, 32, 256, n=2)
      self.model[4] = ADown(ch0=128)
      self.model[5] = RepNCSPELAN4(256, 64, 512, n=2)
      self.model[6] = ADown(ch0=256)
      self.model[7] = RepNCSPELAN4(512, 128, 1024, n=2)
      self.model[8] = ADown(ch0=512)
      self.model[9] = RepNCSPELAN4(1024, 128, 1024, n=2)
      self.model[10] = CBLinear()
      self.model[11] = CBLinear(ch0=256, ch1=192, c2s=[64, 128], f=3)
      self.model[12] = CBLinear(ch0=512, ch1=448, c2s=[64, 128, 256], f=5)
      self.model[13] = CBLinear(ch0=1024, ch1=960, c2s=[64, 128, 256, 512], f=7)
      self.model[14] = CBLinear(ch0=1024, ch1=1984, c2s=[64, 128, 256, 512, 1024], f=9)
      self.model[15] = Conv(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),  groups=1, bias=True, f=0)
      self.model[16] = CBFuse(f=[10, 11, 12, 13, 14, -1], idx=[0, 0, 0, 0, 0])
      self.model[17] = Conv(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1),  groups=1, bias=True)
      self.model[18] = CBFuse(f=[11, 12, 13, 14, -1], idx=[1, 1, 1, 1])
      self.model[19] = RepNCSPELAN4(128, 32, 256, n=2)
      self.model[20] = ADown(ch0=128)
      self.model[21] = CBFuse(f=[12, 13, 14, -1], idx=[2, 2, 2])
      self.model[22] = RepNCSPELAN4(256, 64, 512, n=2)
      self.model[23] = ADown(ch0=256)
      self.model[24] = CBFuse(f=[13, 14, -1], idx=[3, 3])
      self.model[25] = RepNCSPELAN4(512, 128, 1024, n=2)
      self.model[26] = ADown(ch0=512)
      self.model[27] = CBFuse(f=[14, -1], idx=[4])
      self.model[28] = RepNCSPELAN4(1024, 128, 1024, n=2)
      self.model[29] = SPPELAN(ch0=1024, ch1=256, ch2=1024, ch3=512, f=28)
      self.model[30] = Upsample()
      self.model[31] = Concat(f=[-1, 25])
      self.model[32] = RepNCSPELAN4(1536, 128, 512, n=2)
      self.model[33] = Upsample()
      self.model[34] = Concat(f=[-1, 22])
      self.model[35] = RepNCSPELAN4(1024, 64, 256, n=2)
      self.model[36] = ADown(ch0=128)
      self.model[37] = Concat(f=[-1, 32]) 
      self.model[38] = RepNCSPELAN4(768, 128, 512, n=2)
      self.model[39] = ADown(ch0=256)
      self.model[40] = Concat(f=[-1, 29])
      self.model[41] = RepNCSPELAN4(1024, 256, 512, n=2)
      self.model[42] = DDetect(a=256, b=512, c=512, d=256, f=[35, 38, 41]) 
  def __call__(self, x):
    print("rory input =",type(x), x.shape, x.dtype)
    y = []  # outputs
    for i in range(len(self.model)):
      m = self.model[i]
      if m.f != -1: x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
      x = m(x)
      y.append(x)
    
    return postprocess(x[0])

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

def preprocess(im, imgsz=1280, model_stride=32, model_pt=True):
  same_shapes = all(x.shape == im[0].shape for x in im)
  auto = same_shapes and model_pt
  im = [compute_transform(x, new_shape=imgsz, auto=auto, stride=model_stride) for x in im]
  im = Tensor.stack(*im) if len(im) > 1 else im[0].unsqueeze(0)
  im = im[..., ::-1].permute(0, 3, 1, 2)  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
  im = im / 255.0  # 0 - 255 to 0.0 - 1.0
  return im

def rescale_bounding_boxes(predictions, from_size=None, to_size=None):
    from_w, from_h = from_size
    to_w, to_h = to_size
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    
    rescaled_predictions = []
    for pred in predictions:
        x1, y1, x2, y2, conf, class_id = pred
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        rescaled_predictions.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, conf, class_id])
    return rescaled_predictions

def draw_bounding_boxes_and_save(orig_img_path, output_img_path, predictions, class_labels):
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  font = cv2.FONT_HERSHEY_SIMPLEX
  def is_bright_color(color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127

  orig_img = cv2.imread(orig_img_path) if not isinstance(orig_img_path, np.ndarray) else cv2.imdecode(orig_img_path, 1)
  height, width, _ = orig_img.shape
  box_thickness = int((height + width) / 400)
  font_scale = (height + width) / 2500
  object_count = defaultdict(int)

  for pred in predictions:
    x1, y1, x2, y2, conf, class_id = pred
    if conf == 0: continue
    x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
    color = color_dict[class_labels[class_id]]
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, box_thickness)
    label = f"{class_labels[class_id]} {conf:.2f}"
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    label_y, bg_y = (y1 - 4, y1 - text_size[1] - 4) if y1 - text_size[1] - 4 > 0 else (y1 + text_size[1], y1)
    cv2.rectangle(orig_img, (x1, bg_y), (x1 + text_size[0], bg_y + text_size[1]), color, -1)
    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    cv2.putText(orig_img, label, (x1, label_y), font, font_scale, font_color, 1, cv2.LINE_AA)
    object_count[class_labels[class_id]] += 1

  print("Objects detected:")
  for obj, count in object_count.items():
    print(f"- {obj}: {count}")

  cv2.imwrite(output_img_path, orig_img)
  print(f'saved detections at {output_img_path}')

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

SIZES = {"t": [16, 64, 96, 24, 128, 256, 224, 160, 48, 144, 192, 80, 32, 16, 3, 96, 32, 64, 128, 64, 64, 128,"t"],
"s": [32, 128, 192, 48, 256, 512, 448, 320, 96, 288, 384, 128, 64, 32, 3, 192, 64, 64, 128, 128, 128, 256, "s"],
"m": [32, 240, 360, 90, 480, 960, 840, 600, 184, 544, 720, 240, 128, 60, 1, 360, 120, 64, 128, 240, 240, 480, "m"],
"c": [64, 256, 512, 128, 256, 1024, 1024, 1024, 128, 768, 1024, 256, 128, 64, 1, 256, 128, 128, 256, 128, 512, 512, "c"]}

import sys
from pathlib import Path
if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Error: Image URL or path not provided.")
    sys.exit(1)

  img_path = sys.argv[1]
  yolo_variant = sys.argv[2] if len(sys.argv) >= 3 else (print("No variant given, so choosing 't' as the default. Yolov9 has different variants, you can choose from ['t', 's', 'm', 'c', 'e']") or 't')
  print(f'running inference for YOLO version {yolo_variant}')

  output_folder_path = Path('./outputs')
  output_folder_path.mkdir(parents=True, exist_ok=True)
  #absolute image path or URL
  image_location = np.frombuffer(fetch(img_path).read_bytes(), np.uint8)
  image = [cv2.imdecode(image_location, 1)]
  out_path = (output_folder_path / f"{Path(img_path).stem}_output{Path(img_path).suffix or '.png'}").as_posix()
  if not isinstance(image[0], np.ndarray):
    print('Error in image loading. Check your image file.')
    sys.exit(1)
  pre_processed_image = preprocess(image)
  yolo_infer = DetectionModel(*SIZES[yolo_variant])
  state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/yolov9/resolve/main/yolov9-{yolo_variant}.safetensors'))
  load_state_dict(yolo_infer, state_dict)
  st = time.time()
  pred = yolo_infer(pre_processed_image)
  pred = pred.numpy()
  pred = pred[pred[:, 4] >= 0.25]
  print(f'did inference in {int(round(((time.time() - st) * 1000)))}ms')
  #v9 and v3 have same 80 class names for Object Detection
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  pred = scale_boxes(pre_processed_image.shape[2:], pred, image[0].shape)
  draw_bounding_boxes_and_save(orig_img_path=image_location, output_img_path=out_path, predictions=pred, class_labels=class_labels)