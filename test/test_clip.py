from objects import ObjectFinder, precompute_embeddings_jit, precompute_embedding_jit_bs1
import numpy as np
import os
from tinygrad import Tensor
import time

def teset_clip_jit(bs=1):
  clip = ObjectFinder(clip=True)
  input1 = np.random.rand(bs, 3, 224, 224).astype(np.float32)
  fun = precompute_embedding_jit_bs1 if bs == 1 else precompute_embeddings_jit
  for _ in range(3): _ = fun(clip.model, Tensor(input1)).numpy()
  ts = time.time()
  _ = fun(clip.model, Tensor(input1)).numpy()
  print(f"time (bs={bs}) =",round((time.time() - ts), 2), "seconds")

teset_clip_jit()
teset_clip_jit(bs=16)


