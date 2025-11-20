import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
from yolox.tracker.matching import linear_sum_assignment
import time

def test_linear_sum_assigment():
  for i in range(300):
    input = np.random.rand(i,i)
    a, b = linear_sum_assignment(input)
    c, d = scipy_linear_sum_assignment(input)
    np.testing.assert_allclose(a, c)
    np.testing.assert_allclose(b, d)

def test_linear_sum_assigment_speed():
  for i in range(100):
    input = np.random.rand(i,i)
    st = time.perf_counter()
    _, _ = linear_sum_assignment(input)
    t = time.perf_counter() - st
    st = time.perf_counter()
    _, _ = scipy_linear_sum_assignment(input)
    t_scipy = time.perf_counter() - st
    np.testing.assert_(t < t_scipy*50, f"test_linear_sum_assigment_speed slow from {i}")

test_linear_sum_assigment()
test_linear_sum_assigment_speed()