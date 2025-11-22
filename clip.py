# update_embeddings.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import pickle
import time
from datetime import datetime

class CachedCLIPSearch:
    def __init__(self, model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"):
      self.model = CLIPModel.from_pretrained(model_name)
      self.processor = CLIPProcessor.from_pretrained(model_name)
      self.image_embeddings = {}
      self.image_paths = {}
        
    def precompute_embeddings(self, folder_path, cache_file="embeddings.pkl", batch_size=16):
      """Precompute embeddings, only for new images"""
      # Load existing cache if it exists
      if os.path.exists(cache_file):
        print(f"{datetime.now()}: Loading cached embeddings...")
        with open(cache_file, 'rb') as f:
          cache = pickle.load(f)
          self.image_embeddings = cache['embeddings']
          self.image_paths = cache['paths']
      
      # Find all current images
      current_images = set()
      for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
          img_path = os.path.join(folder_path, img_file)
          current_images.add(img_path)
      
      # Find new images (not in cache)
      cached_images = set(self.image_embeddings.keys())
      new_images = current_images - cached_images
      deleted_images = cached_images - current_images
      
      # Remove deleted images
      for img_path in deleted_images:
        if img_path in self.image_embeddings:
          del self.image_embeddings[img_path]
          del self.image_paths[img_path]
      
      if not new_images:
        print(f"{datetime.now()}: No new images found.")
        # Still save to update deletions
        with open(cache_file, 'wb') as f:
          pickle.dump({'embeddings': self.image_embeddings, 'paths': self.image_paths}, f)
        return
      
      print(f"{datetime.now()}: Found {len(new_images)} new images, processing...")
      new_image_list = list(new_images)
      
      # Process new images in batches
      for i in range(0, len(new_image_list), batch_size):
        batch_paths = new_image_list[i:i + batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    batch_images.append(img.copy())
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
        
        print(f"Processed {min(i + batch_size, len(new_image_list))}/{len(new_image_list)} new images...")
        
        for img in batch_images:
          img.close()
    
      # Save updated cache
      with open(cache_file, 'wb') as f:
        pickle.dump({'embeddings': self.image_embeddings, 'paths': self.image_paths}, f)
      print(f"{datetime.now()}: Updated cache with {len(new_images)} new images. Total: {len(self.image_embeddings)}")

if __name__ == "__main__":
  searcher = CachedCLIPSearch()
  #searcher = CachedCLIPSearch(model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
  #searcher = CachedCLIPSearch("laion/CLIP-ViT-B-16-laion400m_e32")
  #searcher = CachedCLIPSearch("laion/CLIP-ViT-L-14-laion400m_e32")
  #searcher = CachedCLIPSearch("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
  
  while True:
    searcher.precompute_embeddings("data/cameras/city/objects/2025-11-22")
    print(f"{datetime.now()}: Sleeping for 1 hour...")
    time.sleep(3600)  # 1 hour