from clip import CachedCLIPSearch
from clip_search import CLIPSearch

scanner = CachedCLIPSearch()
scanner.precompute_embeddings("test/clip_images")

searcher = CLIPSearch()
searcher._load_single_embeddings_file("test/clip_images/embeddings.pkl")
print(searcher.search("ferrari f40"))
print(searcher.search("nissan micra"))