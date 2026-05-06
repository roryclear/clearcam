from tinygrad import Tensor, TinyJit
from objects import ObjectFinder
from utils.helpers import jit_infer
import time
import numpy as np
import sys

def teset_clip_jit():
  for i in range(8):
    bs = 2**i
    input1 = np.random.rand(bs, 3, 224, 224).astype(np.float32)
    for _ in range(3): _ = jit_infer(clip.model.precompute_embedding, Tensor(input1), jit_cache=jit_cache).numpy()
    ts = time.time()
    _ = jit_infer(clip.model.precompute_embedding, Tensor(input1), jit_cache=jit_cache).numpy()
    print(f"time (bs={bs}) =",round((time.time() - ts) / bs, 2), "seconds")

if __name__ == "__main__":
  jit_cache = {}
  clip = ObjectFinder(clip=True)
  teset_clip_jit()