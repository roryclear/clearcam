import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
from yolox.tracker.matching import linear_sum_assignment
from yolox.tracker.matching import kwok_linear_sum_assignment
import time

def test_linear_sum_assigment():
  for i in range(1,300):
    input = np.random.rand(i,i)
    a, b = kwok_linear_sum_assignment(input)
    c, d = scipy_linear_sum_assignment(input)
    x = 0
    for j in range(len(a)): x = x + a[j] + b[j]
    c = c.sum() + c.sum()
    np.testing.assert_equal(x,c)
    assert np.array_equal(np.sort(a), np.arange(i))
    assert np.array_equal(np.sort(b), np.arange(i))

def test_linear_sum_assigment_speed():
  total_scipy = 0
  total = 0
  for i in range(1,300):
    input = np.random.rand(i,i)
    st = time.perf_counter()
    kwok_linear_sum_assignment(input)
    t = time.perf_counter() - st
    total += t
    st = time.perf_counter()
    scipy_linear_sum_assignment(input)
    t_scipy = time.perf_counter() - st
    total_scipy += t_scipy
    np.testing.assert_(t < t_scipy*30, f"test_linear_sum_assigment_speed slow from {i}")
  print("time vs scipy =", f"{(total / total_scipy) * 100:.1f}%")

test_linear_sum_assigment()
test_linear_sum_assigment_speed()