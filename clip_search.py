import os
import pickle
import torch
import open_clip

class CLIPSearch:
    def __init__(self, base_path="data/cameras"):
        self.base_path = base_path
        self.image_embeddings = {}
        self.image_paths = {}
        self.device = torch.device("cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='laion2b_s32b_b82k'
        )
        self.model = self.model.to(self.device).eval()

        self._load_all_embeddings()

    def _load_all_embeddings(self):
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

    def _encode_text(self, query):
        tokens = open_clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb

    def search(self, query, top_k=10, cam_name=None, timestamp=None):
        if not self.image_embeddings:
            print("No embeddings available.")
            return []
        text_embedding = self._encode_text(query)
        all_similarities = []
        for path, img_embedding in self.image_embeddings.items():
            normalized_path = path.replace("\\", "/")
            if cam_name and f"/cameras/{cam_name}/" not in normalized_path:
                continue
            if timestamp and f"/objects/{timestamp}/" not in normalized_path:
                continue
            similarity = (img_embedding @ text_embedding.T).item()

            filename = os.path.basename(path)
            if "_" in filename and filename.lower().endswith((".jpg", ".jpeg", ".png")):
                object_id = filename.split("_")[1].split(".")[0]
                all_similarities.append((path, similarity, object_id))

        best_per_id = {}
        for path, score, object_id in all_similarities:
            if object_id not in best_per_id or score > best_per_id[object_id][1]:
                best_per_id[object_id] = (path, score)

        # Sort by similarity
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