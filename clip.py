import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import pickle

class CachedCLIPSearch:
    def __init__(self, model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_embeddings = {}
        self.image_paths = {}
        
    def precompute_embeddings(self, folder_path, cache_file="embeddings.pkl"):
        """Precompute and cache all image embeddings"""
        if os.path.exists(cache_file):
            print("Loading cached embeddings...")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.image_embeddings = cache['embeddings']
                self.image_paths = cache['paths']
            return
        
        print("Computing embeddings for all images...")
        images = []
        paths = []
        
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_file)
                images.append(Image.open(img_path))
                paths.append(img_path)
        
        if images:
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
            for path, embedding in zip(paths, embeddings):
                self.image_embeddings[path] = embedding
                self.image_paths[path] = path
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'embeddings': self.image_embeddings, 'paths': self.image_paths}, f)
        print(f"Embeddings computed and cached for {len(images)} images")
    
    def search(self, query, top_k=5):
        """Search using precomputed embeddings"""
        if not self.image_embeddings:
            print("No embeddings found. Call precompute_embeddings() first.")
            return []
        
        with torch.no_grad():
            text_inputs = self.processor(text=[query], return_tensors="pt", padding=True)
            text_embedding = self.model.get_text_features(**text_inputs)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = []
        for path, img_embedding in self.image_embeddings.items():
            similarity = (img_embedding @ text_embedding.T).item()
            similarities.append((path, similarity))
        
        # Sort and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage with caching
searcher = CachedCLIPSearch()
searcher.precompute_embeddings("test_clip_images")  # Slow first time

# Subsequent searches are fast!
results = searcher.search("a red car")
for path, score in results:
    print(f"Score: {score:.3f} - {os.path.basename(path)}")

# Try different queries - they'll be much faster!
print("\nSearching for 'soccer player':")
results2 = searcher.search("soccer player")
for path, score in results2:
    print(f"Score: {score:.3f} - {os.path.basename(path)}")