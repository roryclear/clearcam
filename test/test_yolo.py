from detection.yolov9 import YOLOv9
import cv2
from tinygrad import Tensor
from utils.helpers import jit_infer
import numpy as np
if __name__ == "__main__":
  jit_cache = {}
  img = cv2.imread("test/clip_images/f40.jpg")
  img = Tensor(img)
  model = YOLOv9("t", 960)
  for _ in range(3): preds = jit_infer(model, img, jit_cache).numpy()
  ret = []
  print(preds)
  for p in preds:
    if p[-2] > 0.5: ret.append(list(p))
  print(ret)
  assert len(ret) == 1
  expected = [70.17624, 212.71173, 895.62067, 517.2943, 0.6730736, 2.0]
  np.testing.assert_allclose(expected,ret[0], rtol=1e-4)

  # sanity test, BEAM is flakey

