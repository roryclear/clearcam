import os
import pickle
import torch
import open_clip
from typing import Optional
from torch.nn import functional as F
import torch.nn as nn
from tinygrad import nn as tiny_nn, Tensor as tiny_Tensor, TinyJit

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

        # convert here
        # .copy() or trouble
        self.model.tiny_text_projection = tiny_Tensor(self.model.text_projection.detach().numpy().copy())
        self.model.tiny_token_embedding = tiny_nn.Embedding(self.model.token_embedding.num_embeddings, self.model.token_embedding.embedding_dim)
        self.model.tiny_token_embedding.weight = tiny_Tensor(self.model.token_embedding.weight.data.detach().numpy().copy())
        
        self.model.tiny_ln_1 = []
        self.model.tiny_ln_2 = []
        self.model.out_proj_bias_tiny = []
        for resblock in self.model.transformer.resblocks:
            layernorm = tiny_nn.LayerNorm(normalized_shape=resblock.ln_1.normalized_shape, eps=resblock.ln_1.eps, elementwise_affine=True)
            layernorm.weight = tiny_Tensor(resblock.ln_1.weight.detach().numpy().copy())
            layernorm.bias = tiny_Tensor(resblock.ln_1.bias.detach().numpy().copy())
            self.model.tiny_ln_1.append(layernorm)

            layernorm = tiny_nn.LayerNorm(normalized_shape=resblock.ln_2.normalized_shape, eps=resblock.ln_2.eps, elementwise_affine=True)
            layernorm.weight = tiny_Tensor(resblock.ln_2.weight.detach().numpy().copy())
            layernorm.bias = tiny_Tensor(resblock.ln_2.bias.detach().numpy().copy())
            self.model.tiny_ln_2.append(layernorm)

            opb = tiny_Tensor(resblock.attn.out_proj.bias.detach().numpy().copy())
            self.model.out_proj_bias_tiny.append(opb)

            # todo
            resblock.mlp.c_fc = tiny_Linear(resblock.mlp.c_fc.weight, resblock.mlp.c_fc.bias)
            resblock.mlp.c_proj = tiny_Linear(resblock.mlp.c_proj.weight , resblock.mlp.c_proj.bias)

        self.model.tiny_ln_final = tiny_nn.LayerNorm(normalized_shape=self.model.ln_final.normalized_shape, eps=self.model.ln_final.eps, elementwise_affine=True)
        self.model.tiny_ln_final .weight = tiny_Tensor(self.model.ln_final.weight.detach().numpy().copy())
        self.model.tiny_ln_final .bias = tiny_Tensor(self.model.ln_final.bias.detach().numpy().copy())

    def _load_single_embeddings_file(self, cache_file):
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
        tokens = tiny_Tensor(tokens.detach().numpy())
        text_emb = encode_text(self.model, tokens)
        text_emb = torch.Tensor(text_emb.numpy())
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
        return self.tiny_embedding(x)

class tiny_Linear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.linear = tiny_nn.Linear(weight.shape[0], weight.shape[1])
        self.linear.weight = tiny_Tensor(weight.detach().numpy())
        self.linear.bias = tiny_Tensor(bias.detach().numpy())

    def forward(self, x): return self.linear(x)

@TinyJit
def encode_text(model, text):
    x = text
    x = model.tiny_token_embedding(x)
    if not hasattr(model, 'tiny_positional_embedding'): model.tiny_positional_embedding = tiny_Tensor(model.positional_embedding.detach().numpy())
    x = x + model.tiny_positional_embedding
    
    for i, resblock in enumerate(model.transformer.resblocks):
        residual = x
        x = model.tiny_ln_1[i](x)
        
        B, L, D = x.shape
        H = resblock.attn.num_heads
        d_head = D // H

        in_proj_weight_tiny = tiny_Tensor(resblock.attn.in_proj_weight.detach().numpy()) # todo store these
        in_proj_bias_tiny = tiny_Tensor(resblock.attn.in_proj_bias.detach().numpy())
        out_proj_weight_tiny = tiny_Tensor(resblock.attn.out_proj.weight.detach().numpy())
        attn_mask = tiny_Tensor(model.attn_mask.detach().numpy())
        

        qkv = x.matmul(in_proj_weight_tiny.T) + in_proj_bias_tiny
        x = torch.Tensor(x.numpy())
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, L, H, d_head).transpose(1, 2)
        k = k.view(B, L, H, d_head).transpose(1, 2)
        v = v.view(B, L, H, d_head).transpose(1, 2)

        scale = 1.0 / (d_head ** 0.5)
        attn_scores = q.matmul(k.transpose(-2, -1)) * scale  # (B,H,L,L)
        bool_mask = attn_mask < 0
        attn_scores = attn_scores.masked_fill(bool_mask, float("-inf"))
        attn_probs = tiny_Tensor.softmax(attn_scores)
        context = attn_probs.matmul(v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        x = context.matmul(out_proj_weight_tiny.T) + model.out_proj_bias_tiny[i]

        x += residual
        residual = x
        x = model.tiny_ln_2[i](x)
        x = resblock.mlp.c_fc(x)
        x = x.gelu()
        x = resblock.mlp.c_proj(x)
        x += residual

    x = model.tiny_ln_final(x)  # [batch_size, n_ctx, transformer.width]
    argmax = text.argmax()
    x = x[0][argmax]
    return x @ model.tiny_text_projection

'''
if __name__ == "__main__":
    searcher = CLIPSearch()
    query = input("\nEnter search query (or 'quit' to exit): ").strip()
    results = searcher.search(query, top_k=10)

    print(f"\nTop results for '{query}' (best per object):")
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f} - \"{path}\"")
'''

