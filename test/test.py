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

import pickle
from yolox.tracker.byte_tracker import BYTETracker

def test_bytetracker():
  class Args:
      def __init__(self):
          self.track_buffer = 60
          self.mot20 = False
          self.match_thresh = 0.9
  tracker = BYTETracker(Args())

  inputs = pickle.load(open("test/tracker_inputs.pkl", "rb"))
  outputs = pickle.load(open("test/tracker_outputs.pkl", "rb"))
  total_time = 0
  total_time2 = 0
  for i in range(len(inputs)):
    st = time.perf_counter()
    output = tracker.update(inputs[i], [1280,1280], [1280,1280], threshold=0.5)
    total_time += (time.perf_counter() - st)
    st = time.perf_counter()
    total_time2 += (time.perf_counter() - st)
    assert len(outputs[i]) == len(output)
    for j in range(len(output)):
      np.testing.assert_allclose(output[j]._tlwh, outputs[i][j]._tlwh)
      np.testing.assert_allclose(output[j].score, outputs[i][j].score) 
      np.testing.assert_allclose(output[j].is_activated, outputs[i][j].is_activated)
      np.testing.assert_allclose(output[j].mean, outputs[i][j].mean)
      np.testing.assert_allclose(output[j].covariance, outputs[i][j].covariance)
      np.testing.assert_allclose(output[j].class_id, outputs[i][j].class_id)
    
    assert total_time2 < (total_time * 1.1), "too slow"


def setup_test_bytetracker():
  depth, width, ratio = get_variant_multiples("s")
  yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
  state_dict = safe_load(get_weights_location("s"))
  load_state_dict(yolo_infer, state_dict)

  class Args:
      def __init__(self):
          self.track_buffer = 60
          self.mot20 = False
          self.match_thresh = 0.9

  inputs = []
  outputs = []

  tracker = BYTETracker(Args())
  cap = cv2.VideoCapture("test/MOT16-03.mp4")
  for _ in range(1500):
    _, frame = cap.read()
    frame = Tensor(frame)
    preds = do_inf(frame, yolo_infer).numpy()
    inputs.append(preds)
    preds = tracker.update(preds, [1280,1280], [1280,1280], threshold=0.5)
    outputs.append(preds)
  pickle.dump(inputs, open("test/tracker_inputs.pkl", "wb"))
  pickle.dump(outputs, open("test/tracker_outputs.pkl", "wb"))

setup_test_bytetracker()
test_bytetracker()
#test_yolo_infer()
#test_linear_sum_assigment()
#test_linear_sum_assigment_speed()

