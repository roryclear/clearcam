import os
import pickle
import torch
import open_clip
from typing import Optional
from torch.nn import functional as F
import torch.nn as nn
from tinygrad import nn as tiny_nn, Tensor as tiny_Tensor

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

    def _load_single_embeddings_file(self, cache_file):
        """Load embeddings from a single cache file."""
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)

            folder_embeddings = cache.get("embeddings", {})
            folder_paths = cache.get("paths", {})

            self.image_embeddings.update(folder_embeddings)
            self.image_paths.update(folder_paths)

            loaded_count = len(folder_embeddings)
            print(f"Loaded {loaded_count} embeddings from {cache_file}")
            return loaded_count
            
        except Exception as e:
            print(f"Error loading {cache_file}: {e}")
            return 0

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
                    total_loaded += self._load_single_embeddings_file(cache_file)

        print(f"\nTotal images loaded: {total_loaded}")

    def _encode_text(self, query):
        tokens = open_clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_emb = encode_text(self.model, tokens)
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
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                object_id = filename.split("_")[1].split(".")[0] if "_" in filename else None
                all_similarities.append((path, similarity, object_id))
        
        if any(item[2] for item in all_similarities):
            best_per_id = {}
            for path, score, object_id in all_similarities:
                if object_id is not None:
                    if object_id not in best_per_id or score > best_per_id[object_id][1]:
                        best_per_id[object_id] = (path, score)

            items_without_id = [(path, score) for path, score, object_id in all_similarities if object_id is None]
            results = list(best_per_id.values()) + items_without_id
        else:
            results = [(path, score) for path, score, _ in all_similarities]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class tiny_Embedding(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, weight):
        super().__init__()
        self.embedding = torch.nn.modules.sparse.Embedding(num_embeddings=num_embeddings, embedding_dim=embeddings_dim)
        self.tiny_embedding = tiny_nn.Embedding(num_embeddings, embeddings_dim)
        self.tiny_embedding.weight = tiny_Tensor(weight.detach().numpy())
        
    def forward(self, x):
        return self.tiny_embedding(tiny_Tensor(x.detach().numpy()))
        return torch.Tensor(ret)

def encode_text(model, text, normalize: bool = False):
    if not isinstance(model.token_embedding, tiny_Embedding):
        model.token_embedding = tiny_Embedding(model.token_embedding.num_embeddings, model.token_embedding.embedding_dim, model.token_embedding.weight.data.clone())
    x = model.token_embedding(text)
    if not hasattr(model, 'tiny_positional_embedding'): model.tiny_positional_embedding = tiny_Tensor(model.positional_embedding.detach().numpy())
    x = x + model.tiny_positional_embedding
    x = torch.Tensor(x.numpy())
    
    for resblock in model.transformer.resblocks:
        residual = x
        x = resblock.ln_1(x)
        attn_output, _ = resblock.attn(
            x, x, x, 
            attn_mask=model.attn_mask,
            need_weights=False
        )
        x = residual + resblock.ls_1(attn_output)
        
        residual = x
        x = resblock.ln_2(x)
        x = resblock.mlp(x)
        x = residual + resblock.ls_2(x)


    x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    x = text_global_pool(x, text, model.text_pool_type, eos_token_id=getattr(model, "text_eos_id", None))
    if model.text_projection is not None:
        if isinstance(model.text_projection, nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection

    return F.normalize(x, dim=-1) if normalize else x


def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'argmax',
        eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
    elif pool_type == 'eos':
        # take features from tokenizer specific eos
        assert text is not None
        assert eos_token_id is not None
        idx = (text == eos_token_id).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), idx]
    else:
        pooled = x

    return pooled


'''
if __name__ == "__main__":
    searcher = CLIPSearch()
    query = input("\nEnter search query (or 'quit' to exit): ").strip()
    results = searcher.search(query, top_k=10)

    print(f"\nTop results for '{query}' (best per object):")
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f} - \"{path}\"")
'''