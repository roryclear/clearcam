import torch
import open_clip
import os
import pickle
from datetime import datetime
import cv2
import numpy as np

class CachedCLIPSearch:
    def __init__(self, model_name="ViT-L-14", pretrained_name="laion2b_s32b_b82k"):
        print(f"Loading OpenCLIP model: {model_name} ({pretrained_name})")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name
        )

        self.model = self.model.to("cpu").eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.image_embeddings = {}
        self.image_paths = {}

    def find_object_folders(self, base_path="data/cameras"):
        object_folders = []

        if not os.path.exists(base_path):
            print(f"Base path {base_path} does not exist!")
            return object_folders

        for camera_folder in os.listdir(base_path):
            camera_path = os.path.join(base_path, camera_folder)
            if os.path.isdir(camera_path):
                objects_path = os.path.join(camera_path, "objects")
                if os.path.isdir(objects_path):
                    for date_folder in os.listdir(objects_path):
                        date_path = os.path.join(objects_path, date_folder)
                        if os.path.isdir(date_path):
                            object_folders.append(date_path)
                            print(f"Found object folder: {date_path}")

        return object_folders

    def precompute_embeddings(self, folder_path, batch_size=16):
        cache_file = os.path.join(folder_path, "embeddings.pkl")
        folder_embeddings = {}
        folder_paths = {}

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
                folder_embeddings = cache.get("embeddings", {})
                folder_paths = cache.get("paths", {})

        current_images = {
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        cached_images = set(folder_embeddings.keys())
        new_images = current_images - cached_images
        deleted_images = cached_images - current_images

        # Remove deleted files
        for img in deleted_images:
            folder_embeddings.pop(img, None)
            folder_paths.pop(img, None)

        if not new_images:
            print(f"{datetime.now()}: No new images in {folder_path}. Saving cache...")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump({"embeddings": folder_embeddings, "paths": folder_paths}, f)
            return

        print(f"{datetime.now()}: Found {len(new_images)} new images in {folder_path}, processing...")

        new_image_list = list(new_images)

        for i in range(0, len(new_image_list), batch_size):
            batch_paths = new_image_list[i:i + batch_size]
            batch_np_images = []

            for img_path in batch_paths:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load {img_path}")
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    batch_np_images.append(img)

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

            if batch_np_images:
                with torch.no_grad():
                    batch_tensors = torch.stack([preprocess(img) for img in batch_np_images])
                    embeddings = self.model.encode_image(batch_tensors)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                for path, embedding in zip(batch_paths, embeddings):
                    folder_embeddings[path] = embedding
                    folder_paths[path] = path

            print(f"Processed {min(i + batch_size, len(new_image_list))}/{len(new_image_list)} new images...")

        # Save updated cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"embeddings": folder_embeddings, "paths": folder_paths}, f)

        print(f"{datetime.now()}: Updated cache for {folder_path}. Total images stored: {len(folder_embeddings)}")


def preprocess(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = np.transpose(image, (0, 1, 2))
    tensor = torch.from_numpy(image).permute(2, 0, 1)  # CHW
    return tensor

'''
if __name__ == "__main__":
    searcher = CachedCLIPSearch(
        model_name="ViT-L-14",
        pretrained_name="laion2b_s32b_b82k"
    )

    while True:
        object_folders = searcher.find_object_folders("data/cameras")
        print(f"{datetime.now()}: Found {len(object_folders)} object folders")

        for folder in object_folders:
            searcher.precompute_embeddings(folder)

        print(f"{datetime.now()}: Sleeping for 15 mins...")
        time.sleep(900)
'''