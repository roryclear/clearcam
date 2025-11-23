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

    def search(self, query, top_k=10, cam_name=None, timestamp=None):
        if not self.image_embeddings:
            print("No embeddings available.")
            return []
        model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        with torch.no_grad():
            text_inputs = processor(text=[query], return_tensors="pt", padding=True)
            text_embedding = model.get_text_features(**text_inputs)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        all_similarities = []
        for path, img_embedding in self.image_embeddings.items():
            if cam_name and f"/cameras/{cam_name}/" not in path.replace("\\", "/"):
                continue
            if timestamp and f"/objects/{timestamp}/" not in path.replace("\\", "/"):
                continue
            similarity = (img_embedding @ text_embedding.T).item()
            filename = os.path.basename(path)
            if "_" in filename and filename.endswith((".jpg", ".jpeg", ".png")):
                object_id = filename.split("_")[1].split(".")[0]
                all_similarities.append((path, similarity, object_id))
        best_per_id = {}
        for path, score, object_id in all_similarities:
            if object_id not in best_per_id or score > best_per_id[object_id][1]:
                best_per_id[object_id] = (path, score)
        results = list(best_per_id.values())
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

'''
if __name__ == "__main__":
    searcher = CLIPSearch()
    query = input("\nEnter search query (or 'quit' to exit): ").strip()
    results = searcher.search(query, top_k=10)

    print(f"\nTop results for '{query}' (best per object):")
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f} - \"{path}\"")
'''