from PIL import Image
import os
import pickle
from datetime import datetime
from tinygrad import nn as tiny_nn, Tensor as tiny_Tensor, TinyJit, Device
from tinygrad.helpers import fetch
from torchvision.transforms import functional as F
from tinygrad.dtype import dtypes
import numpy as np
import gc

class TinyModel: pass


class CachedCLIPSearch:
    def __init__(self, model_name="ViT-L-14", pretrained_name="laion2b_s32b_b82k"):
        self.image_embeddings = {}
        self.image_paths = {}
        
        self.tiny_model = TinyModel()
        device = Device.DEFAULT
        # convert
        weights = tiny_nn.state.safe_load(fetch("http://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.safetensors"))

        self.tiny_model.visual_conv1 = tiny_nn.Conv2d(3, 1024, (14, 14), (14, 14), (0, 0), (1, 1), 1, bias=False)
        self.tiny_model.visual_conv1.weight = weights["visual.conv1.weight"].to(device)

        self.tiny_model.class_embedding = weights["visual.class_embedding"].to(device)
        self.tiny_model.positional_embedding = weights["visual.positional_embedding"].to(device)
        
 
        self.tiny_model.ln_pre = tiny_nn.LayerNorm(1024)
        self.tiny_model.ln_pre.weight = weights["visual.ln_pre.weight"].to(device)
        self.tiny_model.ln_pre.bias = weights["visual.ln_pre.bias"].to(device)
        

        self.tiny_model.ln_post = tiny_nn.LayerNorm(1024)
        self.tiny_model.ln_post.weight = weights["visual.ln_post.weight"].to(device)
        self.tiny_model.ln_post.bias = weights["visual.ln_post.bias"].to(device)

        self.tiny_model.proj = weights["visual.proj"].to(device)

        self.tiny_model.resblocks = []

        for i in range(24):
            tiny_resblock = TinyModel()

            tiny_resblock.ln_1 = tiny_nn.LayerNorm(1024, 1e-05, elementwise_affine=True)
            tiny_resblock.ln_1.weight = weights[f"visual.transformer.resblocks.{i}.ln_1.weight"].to(device)
            tiny_resblock.ln_1.bias = weights[f"visual.transformer.resblocks.{i}.ln_1.bias"].to(device)

            tiny_resblock.ln_2 = tiny_nn.LayerNorm(1024, 1e-05, elementwise_affine=True)
            tiny_resblock.ln_2.weight = weights[f"visual.transformer.resblocks.{i}.ln_2.weight"].to(device)
            tiny_resblock.ln_2.bias = weights[f"visual.transformer.resblocks.{i}.ln_2.bias"].to(device)

            tiny_resblock.in_proj_weight = weights[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"].to(device)
            tiny_resblock.in_proj_bias = weights[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"].to(device)

            tiny_resblock.out_proj_weight = weights[f"visual.transformer.resblocks.{i}.attn.out_proj.weight"].to(device)
            tiny_resblock.out_proj_bias = weights[f"visual.transformer.resblocks.{i}.attn.out_proj.bias"].to(device)


            tiny_resblock.mlp_c_fc = tiny_nn.Linear(4096, 1024)
            tiny_resblock.mlp_c_fc.weight = weights[f"visual.transformer.resblocks.{i}.mlp.c_fc.weight"].to(device)
            tiny_resblock.mlp_c_fc.bias = weights[f"visual.transformer.resblocks.{i}.mlp.c_fc.bias"].to(device)

            tiny_resblock.mlp_c_proj = tiny_nn.Linear(1024, 4096)
            tiny_resblock.mlp_c_proj.weight = weights[f"visual.transformer.resblocks.{i}.mlp.c_proj.weight"].to(device)
            tiny_resblock.mlp_c_proj.bias = weights[f"visual.transformer.resblocks.{i}.mlp.c_proj.bias"].to(device)

            self.tiny_model.resblocks.append(tiny_resblock)
        
        weights = None
        gc.collect()
        
        # for BEAM
        tiny_precompute_embeddings(self.tiny_model, tiny_Tensor.rand((1, 3, 224, 224), dtype=dtypes.float32))
        tiny_precompute_embeddings(self.tiny_model, tiny_Tensor.rand((16, 3, 224, 224), dtype=dtypes.float32))

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

            batch_tensors = np.stack([inline_preprocess(img) for img in batch_images])
            if len(batch_images) == batch_size:
                x = tiny_Tensor(batch_tensors)
                embeddings = tiny_precompute_embeddings(self.tiny_model, x)
                embeddings = embeddings.numpy()
            else:
                embeddings = []
                for i in range(len(batch_images)): # one by one if not batch_size
                    single_x = tiny_Tensor(batch_tensors[i:i+1])
                    single_embedding = tiny_precompute_embedding(self.tiny_model, single_x)
                    single_embedding_torch = single_embedding.numpy()
                    embeddings.append(single_embedding_torch)

            for path, embedding in zip(batch_paths, embeddings):
                folder_embeddings[path] = embedding
                folder_paths[path] = path

            print(f"Processed {min(i + batch_size, len(new_image_list))}/{len(new_image_list)} new images...")

            for img in batch_images:
                img.close()

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"embeddings": folder_embeddings, "paths": folder_paths}, f)

        print(f"{datetime.now()}: Updated cache for {folder_path}. Total images stored: {len(folder_embeddings)}")

@TinyJit
def tiny_precompute_embeddings(tiny_model, x): return tiny_precompute_embedding(tiny_model, x)

def tiny_precompute_embedding(tiny_model, x):
    x = tiny_model.visual_conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls_emb = tiny_model.class_embedding
    cls_emb = cls_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
    x = tiny_Tensor.cat(cls_emb, x, dim=1)
    x = x + tiny_model.positional_embedding
    x = tiny_model.ln_pre(x)
    # https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/activation.py#L1252
    for i in range(len(tiny_model.resblocks)):
        x_ln1 = tiny_model.resblocks[i].ln_1(x)
        B, L, D = x_ln1.shape
        H = 16 #block.attn.num_heads
        d_head = D // H
        qkv = x_ln1 @ tiny_model.resblocks[i].in_proj_weight.T + tiny_model.resblocks[i].in_proj_bias           
        q, k, v = qkv.split(D, dim=-1)
        def shape(x): return x.view(B, L, H, d_head).transpose(1, 2)
        q = shape(q)
        k = shape(k)
        v = shape(v)
        scale = 1.0 / (d_head ** 0.5)
        attn_scores = q.matmul(k.transpose(-2, -1)) * scale
        attn_probs = tiny_Tensor.softmax(attn_scores)
        context = attn_probs.matmul(v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = context @ tiny_model.resblocks[i].out_proj_weight.T + tiny_model.resblocks[i].out_proj_bias      
        attn_scaled = attn_out
        x = x + attn_scaled
        x_ln2 = tiny_model.resblocks[i].ln_2(x)
        ff = tiny_model.resblocks[i].mlp_c_fc(x_ln2)
        ff = ff.gelu()
        ff = tiny_model.resblocks[i].mlp_c_proj(ff)
        x = x + ff

    x = tiny_model.ln_post(x)
    image_embeds = x[:, 0, :]
    embeddings = image_embeds @ tiny_model.proj
    embeddings = embeddings / (embeddings.pow(2).sum(axis=-1, keepdim=True).sqrt() + 1e-8)
    return embeddings

def inline_preprocess(image):
    image = F.resize(image, size=224, interpolation=F.InterpolationMode.BICUBIC, antialias=True)
    image = F.center_crop(image, output_size=(224, 224))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = F.to_tensor(image)
    image = F.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
   
    return image

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
