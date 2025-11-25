from clip import CachedCLIPSearch
from clip_search import CLIPSearch
import numpy as np

def setup_clip_test():
  scanner = CachedCLIPSearch()
  scanner.precompute_embeddings("test/clip_images")

def test_clip_search():
  searcher = CLIPSearch()
  searcher._load_single_embeddings_file("test/clip_images/embeddings.pkl")
  res = searcher.search("ferrari f40")
  np.testing.assert_allclose(res[0][1], 0.33788394927978516)
  np.testing.assert_allclose(res[1][1], 0.07776700705289841)
  assert res[0][0] == "test/clip_images/f40.jpg"
  assert res[1][0] == "test/clip_images/micra.jpg"
  res = searcher.search("nissan micra")
  np.testing.assert_allclose(res[0][1], 0.3227463960647583)
  np.testing.assert_allclose(res[1][1], 0.04420311748981476)
  assert res[1][0] == "test/clip_images/f40.jpg"
  assert res[0][0] == "test/clip_images/micra.jpg"

#setup_clip_test()
test_clip_search()