import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
from yolox.tracker.matching import linear_sum_assignment2
import time
from clearcam import YOLOv8, get_weights_location, get_variant_multiples, do_inf
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import Tensor
import cv2
from collections import defaultdict

def test_linear_sum_assigment():
  for i in range(1,300):
    input = np.random.rand(i,i)
    a, b = linear_sum_assignment2(input)
    c, d = scipy_linear_sum_assignment(input)
    np.testing.assert_allclose(a,c)
    np.testing.assert_allclose(b,d)
    
def test_linear_sum_assigment_speed():
  total_scipy = 0
  total = 0
  for i in range(1,300):
    input = np.random.rand(i,i)
    st = time.perf_counter()
    linear_sum_assignment2(input)
    t = time.perf_counter() - st
    total += t
    st = time.perf_counter()
    scipy_linear_sum_assignment(input)
    t_scipy = time.perf_counter() - st
    total_scipy += t_scipy
  print("time vs scipy =", f"{(total / total_scipy) * 100:.1f}%")

def test_yolo_infer():
  depth, width, ratio = get_variant_multiples("s")
  yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
  state_dict = safe_load(get_weights_location("s"))
  load_state_dict(yolo_infer, state_dict)
  frame = cv2.VideoCapture("test/MOT16-03.mp4").read()[1]
  frame = Tensor(frame)
  preds = do_inf(frame, yolo_infer).numpy()
  counts = defaultdict(int)
  for x in preds:
    if x[-2] > 0: counts[int(x[-1])] += 1
  assert {int(k): v for k, v in counts.items()} == {2: 1, 0: 35, 1: 2, 3: 1}


test_yolo_infer()
test_linear_sum_assigment()
test_linear_sum_assigment_speed()