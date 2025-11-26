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
        return self.tiny_embedding(x)

class tiny_LayerNorm(nn.Module):
    def __init__(self, weight, bias, eps, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = tiny_Tensor(weight.detach().numpy())
        self.bias = tiny_Tensor(bias.detach().numpy())
        self.eps = eps
        self.axis = tuple(-1-i for i in range(len(self.normalized_shape)))

    def forward(self, x):
        x = x.layernorm(eps=self.eps, axis=self.axis)
        return x * self.weight + self.bias

class tiny_Linear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.linear = tiny_nn.Linear(weight.shape[0], weight.shape[1])
        self.linear.weight = tiny_Tensor(weight.detach().numpy())
        self.linear.bias = tiny_Tensor(bias.detach().numpy())

    def forward(self, x): return self.linear(x)

def encode_text(model, text, normalize: bool = False):
    if not isinstance(model.token_embedding, tiny_Embedding):
        model.token_embedding = tiny_Embedding(model.token_embedding.num_embeddings, model.token_embedding.embedding_dim, model.token_embedding.weight.data.clone())
    x = tiny_Tensor(text.detach().numpy())
    x = model.token_embedding(x)
    if not hasattr(model, 'tiny_positional_embedding'): model.tiny_positional_embedding = tiny_Tensor(model.positional_embedding.detach().numpy())
    x = x + model.tiny_positional_embedding
    x = torch.Tensor(x.numpy())
    
    for resblock in model.transformer.resblocks:
        residual = x
        if not isinstance(resblock.ln_1, tiny_LayerNorm):
            resblock.ln_1 = tiny_LayerNorm(resblock.ln_1.weight, resblock.ln_1.bias, resblock.ln_1.eps, resblock.ln_1.normalized_shape)
        if not isinstance(resblock.ln_2, tiny_LayerNorm):
            resblock.ln_2 = tiny_LayerNorm(resblock.ln_2.weight, resblock.ln_2.bias, resblock.ln_2.eps, resblock.ln_2.normalized_shape)
        x = tiny_Tensor(x.detach().numpy())
        x = resblock.ln_1(x)
        x = torch.Tensor(x.numpy())

        if not isinstance(resblock.attn.out_proj, nn.modules.linear.Linear):
            resblock.attn.out_proj = nn.modules.linear.Linear(resblock.attn.out_proj.weight.shape, None)
        
        # https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/activation.py#L1252

        key_padding_mask=None
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(model.attn_mask),
            other_name="attn_mask",
            target_type=x.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=model.attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=x.dtype,
            check_other=False,
        )

        merged_mask, mask_type = resblock.attn.merge_masks(
            attn_mask, key_padding_mask, x
        )

        attn_output = inline_mha(
            x,
            resblock.attn.embed_dim,
            resblock.attn.num_heads,
            resblock.attn.in_proj_weight,
            resblock.attn.in_proj_bias,
            resblock.attn.out_proj.weight,
            resblock.attn.out_proj.bias,
            merged_mask
        )[0]

        
        x = residual + attn_output
        residual = x
        x = tiny_Tensor(x.detach().numpy())
        x = resblock.ln_2(x)
        if not isinstance(resblock.mlp.c_fc, tiny_Linear): resblock.mlp.c_fc = tiny_Linear(resblock.mlp.c_fc.weight, resblock.mlp.c_fc.bias)
        x = resblock.mlp.c_fc(x)
        x = x.gelu()
        if not isinstance(resblock.mlp.c_proj, tiny_Linear): resblock.mlp.c_proj = tiny_Linear(resblock.mlp.c_proj.weight , resblock.mlp.c_proj.bias)
        x = resblock.mlp.c_proj(x)
        x = torch.Tensor(x.numpy())
        x += residual

    if not isinstance(model.ln_final, tiny_LayerNorm):
        model.ln_final = tiny_LayerNorm(model.ln_final.weight, model.ln_final.bias, model.ln_final.eps, model.ln_final.normalized_shape)

    x = tiny_Tensor(x.detach().numpy())
    x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    x = torch.Tensor(x.numpy())
    x = text_global_pool(x, text, model.text_pool_type, eos_token_id=getattr(model, "text_eos_id", None))
    if model.text_projection is not None:
        if isinstance(model.text_projection, nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection

    return F.normalize(x, dim=-1) if normalize else x


import torch
import torch.nn.functional as F

def inline_mha(
    x,
    embed_dim,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    attn_mask=None
):
    B, L, D = x.shape
    H = num_heads
    d_head = D // H
    qkv = F.linear(x, in_proj_weight, in_proj_bias)
    q, k, v = qkv.split(D, dim=-1)
    def shape(x): return x.view(B, L, H, d_head).transpose(1, 2)
    q = shape(q)
    k = shape(k)
    v = shape(v)
    scale = 1.0 / (d_head ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,L,L)
    if attn_mask.dtype != torch.bool:
        if torch.is_floating_point(attn_mask):
            bool_mask = attn_mask < 0 
        else:
            bool_mask = attn_mask != 0
    else:
        bool_mask = attn_mask

    attn_scores = attn_scores.masked_fill(bool_mask, float("-inf"))
    attn_probs = F.softmax(attn_scores, dim=-1)
    context = torch.matmul(attn_probs, v)
    context = context.transpose(1, 2).contiguous().view(B, L, D)
    out = F.linear(context, out_proj_weight, out_proj_bias)
    return out, attn_probs.mean(dim=1)


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
