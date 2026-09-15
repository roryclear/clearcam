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
  for _ in range(3): preds = jit_infer(model, img, jit_cache).numpy()
  ret = []
  print(preds)
  for p in preds:
    if p[-2] > 0.7: ret.append(list(p))
  print("\n",ret)
  assert len(ret) == 1
  expected = [511.69214, 367.81128, 1294.1671, 899.5974, 0.8302731, 2.0]
  np.testing.assert_allclose(expected,ret[0], rtol=1e-4)

  # sanity test, BEAM is flakey

