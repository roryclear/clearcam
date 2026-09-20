from detection.yolov9 import YOLOv9
import cv2
from tinygrad import Tensor, nn
from utils.helpers import jit_infer
Tensor.manual_seed(42)
import numpy as np

class YOLOv9():
  def __init__(self): self.model =  nn.Conv2d(160, 64, 1, (1,1), (0,0), (1,1), 1, True)

  def __call__(self, frame): return self.model(frame).sum()

if __name__ == "__main__":
  jit_cache = {}
  img = Tensor.rand((1, 160, 68, 120))
  model = YOLOv9()
  for _ in range(3): ret = jit_infer(model, img, jit_cache).numpy()

  expected = 17993.432
  np.testing.assert_allclose(ret, expected, rtol=1e-4)

  # sanity test, BEAM is flakey

