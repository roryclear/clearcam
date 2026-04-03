import os
import pickle
from tinygrad import nn, Tensor, TinyJit, Device
from tinygrad.helpers import fetch
from clearcam import event_img_info
from utils.clip_tokenizer import SimpleTokenizer
import numpy as np
import math
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

class Blank: pass

class CLIPSearch:
    def __init__(self, base_path="data/cameras"):
        self.base_path = base_path
        self.image_embeddings = {}
        self.face_embeddings = {}
        self.image_paths = {}
        self.face_paths = {}
        self.tokenizer = SimpleTokenizer()
        self.text_projection = Tensor.empty(768, 768)
        self.positional_embedding_text = Tensor.empty(77, 768)
        self.token_embedding = nn.Embedding(49408, 768)

        self.ln_final = nn.LayerNorm(768, eps=1e-5, elementwise_affine=True)
        self.attn_mask = Tensor.ones(77, 77).tril().where(0.0, -math.inf).cast("float32")
        self.resblocks = []
        
        for i in range(12):
            resblock = Blank()
            resblock.ln_1 = nn.LayerNorm(768, 1e-5, elementwise_affine=True)
            resblock.ln_2 = nn.LayerNorm(768, 1e-5, elementwise_affine=True)
            resblock.attn_out_proj_weight = Tensor.empty(768, 768)
            resblock.attn_out_proj_bias = Tensor.empty(768)
            resblock.mlp_c_fc = nn.Linear(768, 3072)
            resblock.mlp_c_proj = nn.Linear(3072, 768)
            resblock.in_proj_weight = Tensor.empty(2304, 768)
            resblock.in_proj_bias = Tensor.empty(2304)        
            self.resblocks.append(resblock)
        
        state_dict = safe_load("model_comb.safetensors")
        load_state_dict(self, state_dict)

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

    def _load_all_embeddings(self, face=False):
        total_loaded = 0
        valid_paths = set()
        cache_filename = "embeddings.pkl"
        target_embeddings = self.face_embeddings if face else self.image_embeddings
        target_paths = self.face_paths if face else self.image_paths

        for camera_folder in os.listdir(self.base_path):
            camera_path = os.path.join(self.base_path, camera_folder)
            if not os.path.isdir(camera_path): continue
            objects_path = os.path.join(camera_path, "faces" if face else "objects")
            if not os.path.isdir(objects_path): continue
            for date_folder in os.listdir(objects_path):
                date_path = os.path.join(objects_path, date_folder)
                if not os.path.isdir(date_path): continue
                cache_file = os.path.join(date_path, cache_filename)
                if not os.path.exists(cache_file): continue
                with open(cache_file, "rb") as f:
                    cache = pickle.load(f)
                folder_embeddings = cache.get("embeddings", {})
                folder_paths = cache.get("paths", {})
                valid_paths.update(folder_embeddings.keys())
                target_embeddings.update(folder_embeddings)
                target_paths.update(folder_paths)
                total_loaded += len(folder_embeddings)
        stale_keys = set(target_embeddings.keys()) - valid_paths
        for k in stale_keys:
            del target_embeddings[k]
            target_paths.pop(k, None)

        if face:
            self.face_embeddings = target_embeddings
        else:
            self.image_embeddings = target_embeddings

        print(f"\nTotal {'face' if face else 'image'} embeddings loaded: {total_loaded}")

    def _encode_text(self, query, realize=False):
        tokens = [49406]
        tokens += self.tokenizer.encode(query)
        tokens.append(49407)
        if len(tokens) < 77: tokens += [0] * (77 - len(tokens))
        tokens = Tensor([tokens])
        text_emb = encode_text(self, tokens)
        if realize: return text_emb.numpy()
        return text_emb

    def search(self, query=None, top_k=10, cam_name=None, timestamp=None, text_embedding=None, is_face=False):
        embeddings = self.face_embeddings if is_face else self.image_embeddings
        if not embeddings:
            print("No embeddings available.")
            return []
        if text_embedding is None:
            text_embedding = self._encode_text(query)
            text_embedding = text_embedding.numpy()
        all_similarities = []
        for path, img_embedding in embeddings.items():
            normalized_path = path.replace("\\", "/")
            if cam_name and f"/cameras/{cam_name}/" not in normalized_path:
                continue
            if timestamp and f"/objects/{timestamp}/" not in normalized_path and "/objects/video/" not in normalized_path:
                continue
            similarity = (img_embedding @ text_embedding.T).item()
            filename = os.path.basename(path)
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                object_id = event_img_info(filename.split(".")[0])["object_id"] if "_" in filename else None
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
def encode_text(model, text):
    x = text
    x = model.token_embedding(x)
    x = x + model.positional_embedding_text
    
    for i in range(len(model.resblocks)):
        residual = x
        x = model.resblocks[i].ln_1(x)
        
        B, L, D = x.shape
        H = 12 #resblock.attn.num_heads
        d_head = D // H

        qkv = x.matmul(model.resblocks[i].in_proj_weight.T) + model.resblocks[i].in_proj_bias
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, L, H, d_head).transpose(1, 2)
        k = k.view(B, L, H, d_head).transpose(1, 2)
        v = v.view(B, L, H, d_head).transpose(1, 2)

        scale = 1.0 / (d_head ** 0.5)
        attn_scores = q.matmul(k.transpose(-2, -1)) * scale  # (B,H,L,L)
        bool_mask = model.attn_mask < 0
        attn_scores = attn_scores.masked_fill(bool_mask, float("-inf"))
        attn_probs = Tensor.softmax(attn_scores)
        context = attn_probs.matmul(v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        x = context.matmul(model.resblocks[i].attn_out_proj_weight.T) + model.resblocks[i].attn_out_proj_bias

        x += residual
        residual = x
        x = model.resblocks[i].ln_2(x)
        x = model.resblocks[i].mlp_c_fc(x)
        x = x.gelu()
        x = model.resblocks[i].mlp_c_proj(x)
        x += residual

    x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    argmax = text.argmax()
    x = x[0][argmax]
    x = x @ model.text_projection
    return x / (x * x).sum(axis=-1, keepdim=True).sqrt()
