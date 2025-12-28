import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
from yolox.tracker.matching import linear_sum_assignment2
import time
from clearcam import YOLOv9, do_inf, preprocess, SIZES
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import Tensor
import cv2
from collections import defaultdict
from tinygrad.helpers import Context

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


import pickle
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.byte_tracker2 import BYTETracker2

def test_bytetracker():
  class Args:
      def __init__(self):
          self.track_buffer = 60
          self.mot20 = False
          self.match_thresh = 0.9
  tracker = BYTETracker(Args())
  tracker2 = BYTETracker2(Args())

  inputs = pickle.load(open("test/tracker_inputs.pkl", "rb"))
  total_time = 0
  total_time2 = 0
  for i in range(len(inputs)):
    st = time.perf_counter()
    output = tracker.update(inputs[i], [1280,1280], [1280,1280], threshold=0.5)
    total_time += (time.perf_counter() - st)
    st = time.perf_counter()
    output2 = tracker2.update(inputs[i], [1280,1280], [1280,1280], threshold=0.5)
    total_time2 += (time.perf_counter() - st)
    assert len(output2) == len(output)
    for j in range(len(output)):
      np.testing.assert_allclose(output[j]._tlwh, output2[j]._tlwh)
      np.testing.assert_allclose(output[j].score, output2[j].score) 
      np.testing.assert_allclose(output[j].is_activated, output2[j].is_activated)
      np.testing.assert_allclose(output[j].mean, output2[j].mean)
      np.testing.assert_allclose(output[j].covariance, output2[j].covariance)
      np.testing.assert_allclose(output[j].class_id, output2[j].class_id)
    assert total_time2 <= (total_time * 1.0), "slower"


def test_tracker():
  yolo_variant = "t"
  yolo_infer = YOLOv9(*SIZES[yolo_variant]) if yolo_variant in SIZES else YOLOv9()
  class Args:
    def __init__(self):
      self.track_buffer = 60
      self.mot20 = False
      self.match_thresh = 0.9
  tracker = BYTETracker(Args())

  cap = cv2.VideoCapture("test/videos/MOT16-03.mp4")
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = Tensor(frame)
    pre = preprocess(frame)
    preds = do_inf(pre, yolo_infer)[0].numpy()
  cap.release()

test_tracker()
#test_bytetracker()
#test_linear_sum_assigment()
#test_linear_sum_assigment_speed()