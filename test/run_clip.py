from tinygrad import Tensor, TinyJit
from objects import ObjectFinder, precompute_embeddings_jit, precompute_embedding_jit_bs1
import time
import numpy as np
import sys

def teset_clip_jit(bs=1):
  input1 = np.random.rand(bs, 3, 224, 224).astype(np.float32)
  fun = precompute_embedding_jit_bs1 if bs == 1 else precompute_embeddings_jit
  for _ in range(3): _ = fun(clip.model, Tensor(input1)).numpy()
  ts = time.time()
  _ = fun(clip.model, Tensor(input1)).numpy()
  print(f"time (bs={bs}) =",round((time.time() - ts) / bs, 2), "seconds")

if __name__ == "__main__":
  clip = ObjectFinder(clip=True)
  bs = int(sys.argv[1])
  teset_clip_jit(bs=bs)