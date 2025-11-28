import os
import pickle
import open_clip
from tinygrad import nn as tiny_nn, Tensor as tiny_Tensor, TinyJit, Device
from utils.clip_tokenizer import SimpleTokenizer
import torch

class TinyModel:
    pass

class CLIPSearch:
    def __init__(self, base_path="data/cameras"):
        self.base_path = base_path
        self.image_embeddings = {}
        self.image_paths = {}
        self.model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='laion2b_s32b_b82k'
        )
        self.model = self.model.eval()
        self.tokenizer = SimpleTokenizer()

        device = Device.DEFAULT


        self.tiny_model = TinyModel()

        weights = tiny_nn.state.safe_load("open_clip_pytorch_model.safetensors")
        #print(weights)
        #print(list(weights.keys()))
        #print(type(weights["visual.transformer.resblocks.9.mlp.c_proj.bias"]))
        #print(weights["visual.transformer.resblocks.9.mlp.c_proj.bias"].device)


        self.tiny_model.text_projection = weights["text_projection"].to(device)

        self.tiny_model.token_embedding = tiny_nn.Embedding(49408, 768) # todo unhardcode
        self.tiny_model.token_embedding.weight = weights["token_embedding.weight"].to(device)

        self.model.tiny_ln_final = tiny_nn.LayerNorm(768, eps=1e-5, elementwise_affine=True)
        self.model.tiny_ln_final.weight = weights["ln_final.weight"].to(device)
        self.model.tiny_ln_final.bias = weights["ln_final.bias"].to(device)
        
        self.model.tiny_ln_2 = []
        self.model.out_proj_bias_tiny = []
        self.model.mlp_c_fc = []
        self.model.mlp_c_proj = []

        self.tiny_model.resblocks = []
        
        for i in range(12):
            resblock = TinyModel()
            layernorm = tiny_nn.LayerNorm(768, 1e-5, elementwise_affine=True)
            layernorm.weight = weights[f"transformer.resblocks.{i}.ln_1.weight"].to(device)
            layernorm.bias = weights[f"transformer.resblocks.{i}.ln_1.bias"].to(device)
            resblock.ln_1 = layernorm

            layernorm = tiny_nn.LayerNorm(768, 1e-5, elementwise_affine=True)
            layernorm.weight = weights[f"transformer.resblocks.{i}.ln_2.weight"].to(device)
            layernorm.bias = weights[f"transformer.resblocks.{i}.ln_2.bias"].to(device)
            resblock.ln_2 = layernorm

            weight = weights[f"transformer.resblocks.{i}.attn.out_proj.weight"].to(device)
            resblock.attn_out_proj_weight = weight

            bias = weights[f"transformer.resblocks.{i}.attn.out_proj.bias"].to(device)
            resblock.attn_out_proj_bias = bias

            mlpcfc = tiny_nn.Linear(3072, 768)
            mlpcfc.weight = weights[f"transformer.resblocks.{i}.mlp.c_fc.weight"].to(device)
            mlpcfc.bias = weights[f"transformer.resblocks.{i}.mlp.c_fc.bias"].to(device)
            resblock.mlp_c_fc = mlpcfc

            mlpcp = tiny_nn.Linear(768, 3072)
            mlpcp.weight = weights[f"transformer.resblocks.{i}.mlp.c_proj.weight"].to(device)
            mlpcp.bias = weights[f"transformer.resblocks.{i}.mlp.c_proj.bias"].to(device)
            resblock.mlp_c_proj = mlpcp

            self.tiny_model.resblocks.append(resblock)


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
        tokens = [49406]
        tokens += self.tokenizer.encode(query)
        tokens.append(49407)
        if len(tokens) < 77: tokens += [0] * (77 - len(tokens))
        tokens = tiny_Tensor([tokens])
        text_emb = encode_text(self.model, self.tiny_model, tokens)
        return text_emb

    def search(self, query, top_k=10, cam_name=None, timestamp=None):
        if not self.image_embeddings:
            print("No embeddings available.")
            return []
        text_embedding = self._encode_text(query)
        text_embedding = text_embedding.numpy()
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

@TinyJit
def encode_text(model, tiny_model, text):
    x = text
    x = tiny_model.token_embedding(x)
    if not hasattr(model, 'tiny_positional_embedding'): model.tiny_positional_embedding = tiny_Tensor(model.positional_embedding.detach().numpy())
    x = x + model.tiny_positional_embedding
    
    for i, resblock in enumerate(model.transformer.resblocks):
        residual = x
        x = tiny_model.resblocks[i].ln_1(x)
        
        B, L, D = x.shape
        H = resblock.attn.num_heads
        d_head = D // H

        in_proj_weight_tiny = tiny_Tensor(resblock.attn.in_proj_weight.detach().numpy()) # todo store these
        in_proj_bias_tiny = tiny_Tensor(resblock.attn.in_proj_bias.detach().numpy())
        attn_mask = tiny_Tensor(model.attn_mask.detach().numpy())
        

        qkv = x.matmul(in_proj_weight_tiny.T) + in_proj_bias_tiny
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
        x = context.matmul(tiny_model.resblocks[i].attn_out_proj_weight.T) + tiny_model.resblocks[i].attn_out_proj_bias

        x += residual
        residual = x
        x = tiny_model.resblocks[i].ln_2(x)
        x = tiny_model.resblocks[i].mlp_c_fc(x)
        x = x.gelu()
        x = tiny_model.resblocks[i].mlp_c_proj(x)
        x += residual

    x = model.tiny_ln_final(x)  # [batch_size, n_ctx, transformer.width]
    argmax = text.argmax()
    x = x[0][argmax]
    x = x @ tiny_model.text_projection
    return x / (x * x).sum(axis=-1, keepdim=True).sqrt()

'''
if __name__ == "__main__":
    searcher = CLIPSearch()
    query = input("\nEnter search query (or 'quit' to exit): ").strip()
    results = searcher.search(query, top_k=10)

    print(f"\nTop results for '{query}' (best per object):")
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f} - \"{path}\"")
'''
