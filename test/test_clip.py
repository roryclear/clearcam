from tiny_clip import CachedCLIPSearch as tiny_CachedCLIPSearch
from tiny_clip_search import CLIPSearch as tiny_ClipSearch
import numpy as np
import json
import os

def setup_clip_test():
  if os.path.exists("test/clip_images/embeddings.pkl"): os.remove("test/clip_images/embeddings.pkl")
  scanner = tiny_CachedCLIPSearch()
  scanner.precompute_embeddings("test/clip_images")

def test_clip_search():
  searcher = tiny_ClipSearch()
  searcher._load_single_embeddings_file("test/clip_images/embeddings.pkl")
  res = searcher.search("ferrari f40")
  np.testing.assert_allclose(res[0][1], 0.3566271960735321, rtol=1e-03)
  np.testing.assert_allclose(res[1][1], 0.0718243420124054, rtol=1e-02) # careful now
  assert res[0][0] == "test/clip_images/f40.jpg"
  assert res[1][0] == "test/clip_images/micra.jpg"
  res = searcher.search("nissan micra")
  np.testing.assert_allclose(res[0][1], 0.3218580484390259, rtol=1e-03)
  np.testing.assert_allclose(res[1][1], 0.07153752446174622, rtol=1e-02)
  assert res[1][0] == "test/clip_images/f40.jpg"
  assert res[0][0] == "test/clip_images/micra.jpg"

def test_clip_search_jit():
  searcher = tiny_ClipSearch()
  searcher._load_single_embeddings_file("test/clip_images/embeddings.pkl")
  for _ in range(5): res = searcher.search("ferrari f40")
  np.testing.assert_allclose(res[0][1], 0.3566271960735321, rtol=1e-03)
  np.testing.assert_allclose(res[1][1], 0.0718243420124054, rtol=1e-02)  # careful now
  assert res[0][0] == "test/clip_images/f40.jpg"
  assert res[1][0] == "test/clip_images/micra.jpg"
  for _ in range(5): res = searcher.search("nissan micra")
  np.testing.assert_allclose(res[0][1], 0.3218580484390259, rtol=1e-03)
  np.testing.assert_allclose(res[1][1], 0.07153752446174622, rtol=1e-02)
  assert res[1][0] == "test/clip_images/f40.jpg"
  assert res[0][0] == "test/clip_images/micra.jpg"

setup_clip_test()
test_clip_search()
test_clip_search_jit()

