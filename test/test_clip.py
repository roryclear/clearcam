from objects import ObjectFinder, precompute_embeddings_jit, precompute_embedding_jit_bs1, preprocess
import numpy as np
from tinygrad import Tensor
import time
import cv2

clip = ObjectFinder(clip=True)

def teset_clip_jit(bs=1):
  input1 = np.random.rand(bs, 3, 224, 224).astype(np.float32)
  fun = precompute_embedding_jit_bs1 if bs == 1 else precompute_embeddings_jit
  for _ in range(3): _ = fun(clip.model, Tensor(input1)).numpy()
  ts = time.time()
  _ = fun(clip.model, Tensor(input1)).numpy()
  print(f"time (bs={bs}) =",round((time.time() - ts), 2), "seconds")

def test_clip_outputs():
  img1 = cv2.imread("test/clip_images/f40.jpg")
  img1 = preprocess(img1)
  emb1 = precompute_embedding_jit_bs1(clip.model, Tensor(img1).unsqueeze(0))
  emb1_text = clip.model._encode_text("ferrari f40")
  sim = (emb1_text @ emb1.T).numpy()[0]
  np.testing.assert_allclose(0.330654, sim, rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
  test_clip_outputs()
  teset_clip_jit()
  teset_clip_jit(bs=16)


