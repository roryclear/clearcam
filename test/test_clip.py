from tiny_clip import CachedCLIPSearch as tiny_CachedCLIPSearch
from tiny_clip_search import CLIPSearch as tiny_ClipSearch
import numpy as np
import torch
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
  np.testing.assert_allclose(res[0][1], 0.33788394927978516, rtol=1e-03)
  np.testing.assert_allclose(res[1][1], 0.07776700705289841, rtol=1e-02) # careful now
  assert res[0][0] == "test/clip_images/f40.jpg"
  assert res[1][0] == "test/clip_images/micra.jpg"
  res = searcher.search("nissan micra")
  np.testing.assert_allclose(res[0][1], 0.3227463960647583, rtol=1e-03)
  np.testing.assert_allclose(res[1][1], 0.04420311748981476, rtol=1e-02)
  assert res[1][0] == "test/clip_images/f40.jpg"
  assert res[0][0] == "test/clip_images/micra.jpg"

def test_clip_search_jit():
  searcher = tiny_ClipSearch()
  searcher._load_single_embeddings_file("test/clip_images/embeddings.pkl")
  for _ in range(5): res = searcher.search("ferrari f40")
  np.testing.assert_allclose(res[0][1], 0.33788394927978516, rtol=1e-03)
  np.testing.assert_allclose(res[1][1], 0.07776700705289841, rtol=1e-02) # careful now
  assert res[0][0] == "test/clip_images/f40.jpg"
  assert res[1][0] == "test/clip_images/micra.jpg"
  for _ in range(5): res = searcher.search("nissan micra")
  np.testing.assert_allclose(res[0][1], 0.3227463960647583, rtol=1e-03)
  np.testing.assert_allclose(res[1][1], 0.04420311748981476, rtol=1e-02)
  assert res[1][0] == "test/clip_images/f40.jpg"
  assert res[0][0] == "test/clip_images/micra.jpg"

def test_tokenizer(s):
  a = open_clip.tokenize([s]).to(torch.device("cpu")).detach().numpy()
  b = my_tokenizer(s)
  np.testing.assert_allclose(b, a)

def my_tokenizer(s):
  with open("tokenizer.json", "r") as f: tok = json.load(f)
  vocab = tok["model"]["vocab"]
  #ret = [49406, 49407]
  ret = [49406] # start
  
  if len(s) > 0: s += "</w>"
  s = s.replace("  ", " ")
  s = s.replace(" ", "</w>")

  longest = max(len(k) for k in vocab)
  i = 0
  while i < len(s):
    j = longest
    while j >= 0 and s[i:i+j] not in vocab: j -= 1
    token = vocab[s[i:i+j]]
    ret.append(token)
    i+=j

  ret.append(49407) # end
  if len(ret) < 77: ret += [0] * (77 - len(ret))
  return np.asarray([ret])

#for x in ["", "a", "car", "a car", "a bugatti veyron", "deep learning engineer","sxokwasikwqoiwdjwqdioqjdi"]: test_tokenizer(x) # todo failing, not bpe yet
#for x in ["mp4-12c"]: test_tokenizer(x) # failing
setup_clip_test()
test_clip_search()
test_clip_search_jit()

