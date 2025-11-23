import torch
from transformers import CLIPProcessor, CLIPModel
import os
import pickle

class CLIPSearch:
    def __init__(self, base_path="data/cameras"):
        self.base_path = base_path
        self.image_embeddings = {}
        self.image_paths = {}
        self._load_all_embeddings()

    def _load_all_embeddings(self):
        """Walk all camera/object/date folders and load any embeddings.pkl files found."""
        total_loaded = 0

        if not os.path.exists(self.base_path):
            print(f"Base path {self.base_path} does not exist!")
            return

        for camera_folder in os.listdir(self.base_path):
            camera_path = os.path.join(self.base_path, camera_folder)
            if not os.path.isdir(camera_path):
                continue

            objects_path = os.path.join(camera_path, "objects")
            if not os.path.isdir(objects_path):
                continue

            for date_folder in os.listdir(objects_path):
                date_path = os.path.join(objects_path, date_folder)
                if not os.path.isdir(date_path):
                    continue

                cache_file = os.path.join(date_path, "embeddings.pkl")

                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "rb") as f:
                            cache = pickle.load(f)

                        folder_embeddings = cache.get("embeddings", {})
                        folder_paths = cache.get("paths", {})

                        self.image_embeddings.update(folder_embeddings)
                        self.image_paths.update(folder_paths)

                        total_loaded += len(folder_embeddings)

                        print(f"Loaded {len(folder_embeddings)} embeddings from {cache_file}")

                    except Exception as e:
                        print(f"Error loading {cache_file}: {e}")

        print(f"\nTotal images loaded: {total_loaded}")

    def precompute_embeddings(self, folder_path, batch_size=16):
        cache_file = os.path.join(folder_path, "embeddings.pkl")

        # Load existing folder cache
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

        # Remove deleted
        for img in deleted_images:
            folder_embeddings.pop(img, None)
            folder_paths.pop(img, None)

        if not new_images:
            print(f"No new images in {folder_path}, saving cache...")
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {"embeddings": folder_embeddings, "paths": folder_paths},
                    f,
                )
            return

        print(f"Found {len(new_images)} new images in {folder_path}")

        new_image_list = list(new_images)

        for i in range(0, len(new_image_list), batch_size):
            batch_paths = new_image_list[i : i + batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    with Image.open(img_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        batch_images.append(img.copy())
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

            if batch_images:
                with torch.no_grad():
                    inputs = self.processor(
                        images=batch_images, return_tensors="pt", padding=True
                    )
                    embeddings = self.model.get_image_features(**inputs)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                for path, embedding in zip(batch_paths, embeddings):
                    folder_embeddings[path] = embedding
                    folder_paths[path] = path

            print(
                f"Processed {min(i + batch_size, len(new_image_list))}/{len(new_image_list)}"
            )

            for img in batch_images:
                img.close()

        # Save updated folder cache
        with open(cache_file, "wb") as f:
            pickle.dump(
                {"embeddings": folder_embeddings, "paths": folder_paths},
                f,
            )

        print(f"Updated cache for {folder_path}: {len(folder_embeddings)} images total.")


    def search(self, query, top_k=10):
        """Search using precomputed embeddings, showing only highest score per object ID."""
        if not self.image_embeddings:
            print("No embeddings available.")
            return []

        model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")

        with torch.no_grad():
            text_inputs = processor(text=[query], return_tensors="pt", padding=True)
            text_embedding = model.get_text_features(**text_inputs)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        # Calculate similarities
        all_similarities = []
        for path, img_embedding in self.image_embeddings.items():
            similarity = (img_embedding @ text_embedding.T).item()
            filename = os.path.basename(path)

            # Extract object ID from filename like "camera1_123.jpg"
            if "_" in filename and filename.endswith((".jpg", ".jpeg", ".png")):
                object_id = filename.split("_")[1].split(".")[0]
                all_similarities.append((path, similarity, object_id))

        # Best match per ID
        best_per_id = {}
        for path, score, object_id in all_similarities:
            if object_id not in best_per_id or score > best_per_id[object_id][1]:
                best_per_id[object_id] = (path, score)

        results = list(best_per_id.values())
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


if __name__ == "__main__":
    searcher = CLIPSearch()
    query = input("\nEnter search query (or 'quit' to exit): ").strip()
    results = searcher.search(query, top_k=10)

    print(f"\nTop results for '{query}' (best per object):")
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f} - \"{path}\"")