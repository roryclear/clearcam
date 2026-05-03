from objects import ObjectFinder, precompute_embeddings_jit, precompute_embedding_jit_bs1
import numpy as np
from tinygrad import Tensor
import time
import cv2
import subprocess
import sys

def test_clip_outputs():
  img1 = cv2.imread("test/clip_images/f40.jpg")
  img1 = clip.preprocess(img1)
  emb1 = precompute_embedding_jit_bs1(clip.model, Tensor(img1).unsqueeze(0))
  emb1_text = clip.model._encode_text("ferrari f40")
  sim = (emb1_text @ emb1.T).numpy()[0]
  np.testing.assert_allclose(0.330654, sim, rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
  # todo sepatate test for egpu?
  #clip = ObjectFinder(clip=True)
  #test_clip_outputs()
  for i in range(8): subprocess.run([sys.executable, "test/run_clip.py", str(2**i)], check=True)


