from tinygrad import Tensor, TinyJit
from objects import ObjectFinder
import time
import numpy as np
import sys

def teset_clip_jit(bs=1):
  input1 = np.random.rand(bs, 3, 224, 224).astype(np.float32)
  fun = clip.model.precompute_embedding
  for _ in range(3): _ = fun(Tensor(input1)).numpy()
  ts = time.time()
  _ = fun(Tensor(input1)).numpy()
  print(f"time (bs={bs}) =",round((time.time() - ts) / bs, 2), "seconds")

if __name__ == "__main__":
  clip = ObjectFinder(clip=True)
  bs = int(sys.argv[1])
  teset_clip_jit(bs=bs)