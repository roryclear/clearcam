from tinygrad import Tensor, TinyJit
import numpy as np
import time

@TinyJit
def copy_in_1(x):
  x[0] += 1
  x.realize()

@TinyJit
def copy_in_and_out_1(x):
  x[0] += 1
  return x

@TinyJit
def copy_in_100MB(x):
  x[0] += 1
  x.realize()

@TinyJit
def copy_in_and_out_100MB(x):
  x[0] += 1
  return x

n = 10

if __name__ == "__main__":
  a_np = (np.random.rand(1) * 255).astype(np.uint8)
  for _ in range(2): copy_in_1(Tensor(a_np))
  st = time.perf_counter()
  for _ in range(n): copy_in_1(Tensor(a_np))
  t = time.perf_counter() - st
  copy_in_delay = t / n
  print("copy_in_delay =",copy_in_delay)

  a_np = (np.random.rand(1) * 255).astype(np.uint8)
  for _ in range(2): copy_in_and_out_1(Tensor(a_np)).numpy()
  st = time.perf_counter()
  for _ in range(n): copy_in_and_out_1(Tensor(a_np)).numpy()
  t = time.perf_counter() - st
  copy_out_delay = (t / n) - copy_in_delay
  print("copy_out_delay =",copy_out_delay)

  a_np = (np.random.rand(100_000_000) * 255).astype(np.uint8)
  for _ in range(2): copy_in_100MB(Tensor(a_np))
  st = time.perf_counter()
  for _ in range(n): copy_in_100MB(Tensor(a_np))
  t = time.perf_counter() - st
  copy_in_100MB_time = t / n - copy_in_delay
  print("copy_in_100MB_time =",copy_in_100MB_time)

  a_np = (np.random.rand(100_000_000) * 255).astype(np.uint8)
  for _ in range(2): copy_in_and_out_100MB(Tensor(a_np)).numpy()
  st = time.perf_counter()
  for _ in range(n): copy_in_and_out_100MB(Tensor(a_np)).numpy()
  t = time.perf_counter() - st
  copy_out_100MB_time = (t / n) - (copy_in_100MB_time + copy_out_delay)
  print("copy_out_100MB_time =",copy_out_100MB_time)
  
  print("\n\n")
  print(f"round trip min: {copy_in_delay + copy_out_delay:.2f} seconds")
  print(f"copy in:        {(1 / copy_in_100MB_time) * 100:.2f} MB/s")
  print(f"copy out:       {(1 / copy_out_100MB_time) * 100:.2f} MB/s")

'''  
4C + 9060XT chestnut 100W (PSU isn't faster)
round trip min: 0.05 seconds
copy in:        224.46 MB/s
copy out:       144.53 MB/s

M3 Mac chestnut
round trip min: 0.01 seconds
copy in:        760.33 MB/s
copy out:       395.40 MB/s

M3 Mac chestnut (with dongle + USB-A to USB-C plug)
round trip min: 0.01 seconds
copy in:        373.40 MB/s
copy out:       235.73 MB/s
'''
  