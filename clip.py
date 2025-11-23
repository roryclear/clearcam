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
    
    def find_object_folders(self, base_path="data/cameras"):
        """Find all object folders in the structure: data/cameras/{camera_name}/objects/{date}"""
        object_folders = []
        
        if not os.path.exists(base_path):
            print(f"Base path {base_path} does not exist!")
            return object_folders
            
        # Look through each camera folder
        for camera_folder in os.listdir(base_path):
            camera_path = os.path.join(base_path, camera_folder)
            if os.path.isdir(camera_path):
                # Look for objects folder under camera
                objects_path = os.path.join(camera_path, "objects")
                if os.path.exists(objects_path) and os.path.isdir(objects_path):
                    # Look through each date folder under objects
                    for date_folder in os.listdir(objects_path):
                        date_path = os.path.join(objects_path, date_folder)
                        if os.path.isdir(date_path):
                            object_folders.append(date_path)
                            print(f"Found object folder: {date_path}")
        
        return object_folders
        
    def precompute_embeddings(self, folder_path, batch_size=16):
        cache_file = os.path.join(folder_path, "embeddings.pkl")

        # Load cached embeddings for THIS folder only
        folder_embeddings = {}
        folder_paths = {}
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
                folder_embeddings = cache.get("embeddings", {})
                folder_paths = cache.get("paths", {})

        # Detect current images
        current_images = {
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        cached_images = set(folder_embeddings.keys())
        
        new_images = current_images - cached_images
        deleted_images = cached_images - current_images

        # Remove deleted images
        for img in deleted_images:
            folder_embeddings.pop(img, None)
            folder_paths.pop(img, None)

        if not new_images:
            print(f"{datetime.now()}: No new images found in {folder_path}. Saving cache...")
            with open(cache_file, "wb") as f:
                pickle.dump({"embeddings": folder_embeddings, "paths": folder_paths}, f)
            return

        print(f"{datetime.now()}: Found {len(new_images)} new images in {folder_path}, processing...")

        new_image_list = list(new_images)

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
                    folder_embeddings[path] = embedding
                    folder_paths[path] = path

            print(
                f"Processed {min(i + batch_size, len(new_image_list))}/"
                f"{len(new_image_list)} new images..."
            )

            for img in batch_images:
                img.close()

        # Save folder-specific cache
        with open(cache_file, "wb") as f:
            pickle.dump({"embeddings": folder_embeddings, "paths": folder_paths}, f)

        print(
            f"{datetime.now()}: Updated cache for {folder_path}. "
            f"Total images stored: {len(folder_embeddings)}"
        )

if __name__ == "__main__":
    searcher = CachedCLIPSearch()
    #searcher = CachedCLIPSearch(model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K") # smaller model

    while True:
        object_folders = searcher.find_object_folders("data/cameras")
        print(f"{datetime.now()}: Found {len(object_folders)} object folders")

        for folder in object_folders:
            searcher.precompute_embeddings(folder)

        print(f"{datetime.now()}: Sleeping for 15 mins...")
        time.sleep(900)

