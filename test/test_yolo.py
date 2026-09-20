from detection.yolov9 import YOLOv9
import cv2
from tinygrad import Tensor
from utils.helpers import jit_infer
import numpy as np
if __name__ == "__main__":
  jit_cache = {}
  img = cv2.imread("test/clip_images/front.png")
  img = Tensor(img)
  model = YOLOv9("t", 960)
  for _ in range(3): ret = jit_infer(model, img, jit_cache).numpy()

  expected = 9757807
  np.testing.assert_allclose(expected,ret, rtol=1e-4)

  # sanity test, BEAM is flakey

