# search_images.py
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import pickle

class CLIPSearch:
    def __init__(self, cache_file="embeddings.pkl"):
      self.cache_file = cache_file
      self.image_embeddings = {}
      self.image_paths = {}
      self._load_embeddings()
    
    def _load_embeddings(self):
      """Load precomputed embeddings"""
      if not os.path.exists(self.cache_file):
        print("No embeddings found. Run update_embeddings.py first.")
        return
      
      with open(self.cache_file, 'rb') as f:
        cache = pickle.load(f)
        self.image_embeddings = cache['embeddings']
        self.image_paths = cache['paths']
      print(f"Loaded {len(self.image_embeddings)} image embeddings")
    
    def search(self, query, top_k=10):
      """Search using precomputed embeddings"""
      if not self.image_embeddings:
        print("No embeddings available.")
        return []
      
      model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
      processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
      
      with torch.no_grad():
        text_inputs = processor(text=[query], return_tensors="pt", padding=True)
        text_embedding = model.get_text_features(**text_inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
      
      similarities = []
      for path, img_embedding in self.image_embeddings.items():
        similarity = (img_embedding @ text_embedding.T).item()
        similarities.append((path, similarity))
      
      similarities.sort(key=lambda x: x[1], reverse=True)
      return similarities[:top_k]

if __name__ == "__main__":
  searcher = CLIPSearch()
  
  while True:
    query = input("\nEnter search query (or 'quit' to exit): ").strip()
    if query.lower() == 'quit':
      break
    
    if not query:
      continue
        
    results = searcher.search(query, top_k=50)
    print(f"\nTop results for '{query}':")
    for i, (path, score) in enumerate(results, 1):
      print(f"{i}. Score: {score:.3f} - {os.path.basename(path)}")