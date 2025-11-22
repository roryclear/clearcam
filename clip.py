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
        
    def precompute_embeddings(self, folder_path, cache_file="embeddings.pkl", batch_size=32):
        """Precompute and cache all image embeddings with batching"""
        if os.path.exists(cache_file):
            print("Loading cached embeddings...")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.image_embeddings = cache['embeddings']
                self.image_paths = cache['paths']
            return
        
        print("Computing embeddings for all images...")
        image_paths = []
        
        # First, collect all image paths
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_file)
                image_paths.append(img_path)
        
        if not image_paths:
            print("No images found!")
            return
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load and process batch
            for img_path in batch_paths:
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        batch_images.append(img.copy())  # Make a copy to close the original
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if batch_images:
                with torch.no_grad():
                    inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                    embeddings = self.model.get_image_features(**inputs)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    
                for path, embedding in zip(batch_paths, embeddings):
                    self.image_embeddings[path] = embedding
                    self.image_paths[path] = path
            
            print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images...")
            
            # Clean up batch images to free memory
            for img in batch_images:
                img.close()
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'embeddings': self.image_embeddings, 'paths': self.image_paths}, f)
        print(f"Embeddings computed and cached for {len(image_paths)} images")
    
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

# Delete old cache and run
if os.path.exists("embeddings.pkl"): 
    os.remove("embeddings.pkl")

# Usage with caching
searcher = CachedCLIPSearch()
searcher.precompute_embeddings("data/cameras/city/objects/2025-11-22", batch_size=16)  # Adjust batch size as needed

# Subsequent searches are fast!
results = searcher.search("red car")
for path, score in results:
    print(f"Score: {score:.3f} - {os.path.basename(path)}")

q = "toyota land cruiser"
print(f"results for {q}")
results2 = searcher.search(q)
for path, score in results2:
    print(f"Score: {score:.3f} - {os.path.basename(path)}")