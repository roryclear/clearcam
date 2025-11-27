import torch
from PIL import Image
import open_clip
import os
import pickle
import time
from datetime import datetime
from tinygrad import nn as tiny_nn, Tensor as tiny_Tensor, TinyJit
from torchvision.transforms import functional as F
from tinygrad.dtype import dtypes
# DO NOT USE WITH BEAM
class CachedCLIPSearch:
    def __init__(self, model_name="ViT-L-14", pretrained_name="laion2b_s32b_b82k"):
        print(f"Loading OpenCLIP model: {model_name} ({pretrained_name})")

        if os.path.exists("laion2b_s32b_b82k.pkl"):
            # Load from existing pickle file
            with open("laion2b_s32b_b82k.pkl", 'rb') as f:
                self.model = pickle.load(f)
            print("Loaded data from pickle file")
        else:
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_name
            )

            self.model = self.model.to("cpu").eval()
            
            self.tokenizer = open_clip.get_tokenizer(model_name)

            self.image_embeddings = {}
            self.image_paths = {}

            # convert
            self.model.visual.tiny_vc = tiny_nn.Conv2d(in_channels=self.model.visual.conv1.in_channels, out_channels=self.model.visual.conv1.out_channels, kernel_size=self.model.visual.conv1.kernel_size, stride=self.model.visual.conv1.stride,\
            padding=self.model.visual.conv1.padding, dilation=self.model.visual.conv1.dilation, groups=self.model.visual.conv1.groups, bias=False)
            self.model.visual.tiny_vc.weight = tiny_Tensor(self.model.visual.conv1.weight.detach().numpy().copy())

            self.model.visual.tiny_class_embedding = tiny_Tensor(self.model.visual.class_embedding.detach().numpy().copy())
            self.model.visual.tiny_positional_embedding = tiny_Tensor(self.model.visual.positional_embedding.detach().numpy().copy())

            self.model.visual.tiny_ln_pre = tiny_nn.LayerNorm(self.model.visual.ln_pre.weight.shape[0])
            self.model.visual.tiny_ln_pre.weight = tiny_Tensor(self.model.visual.ln_pre.weight.detach().numpy().copy())
            self.model.visual.tiny_ln_pre.bias = tiny_Tensor(self.model.visual.ln_pre.bias.detach().numpy().copy())

            self.model.visual.tiny_ln_post = tiny_nn.LayerNorm(self.model.visual.ln_post.weight.shape[0])
            self.model.visual.tiny_ln_post.weight = tiny_Tensor(self.model.visual.ln_post.weight.detach().numpy().copy())
            self.model.visual.tiny_ln_post.bias = tiny_Tensor(self.model.visual.ln_post.bias.detach().numpy().copy())
            
            self.model.tiny_ln_1 = []
            self.model.tiny_ln_2 = []
            self.model.tiny_in_proj_weight = []
            self.model.tiny_in_proj_bias = []

            self.model.tiny_out_proj_weight = []
            self.model.tiny_out_proj_bias = []

            self.model.tiny_c_fc = []
            self.model.tiny_c_proj = []

            self.model.visual.tiny_proj = tiny_Tensor(self.model.visual.proj.detach().numpy().copy())

            for resblock in self.model.visual.transformer.resblocks:
                layernorm = tiny_nn.LayerNorm(normalized_shape=resblock.ln_1.normalized_shape, eps=resblock.ln_1.eps, elementwise_affine=True)
                layernorm.weight = tiny_Tensor(resblock.ln_1.weight.detach().numpy().copy())
                layernorm.bias = tiny_Tensor(resblock.ln_1.bias.detach().numpy().copy())
                self.model.tiny_ln_1.append(layernorm)

                layernorm = tiny_nn.LayerNorm(normalized_shape=resblock.ln_2.normalized_shape, eps=resblock.ln_2.eps, elementwise_affine=True)
                layernorm.weight = tiny_Tensor(resblock.ln_2.weight.detach().numpy().copy())
                layernorm.bias = tiny_Tensor(resblock.ln_2.bias.detach().numpy().copy())
                self.model.tiny_ln_2.append(layernorm)

                w = tiny_Tensor(resblock.attn.in_proj_weight.detach().numpy().copy())
                self.model.tiny_in_proj_weight.append(w)

                b = tiny_Tensor(resblock.attn.in_proj_bias.detach().numpy().copy())
                self.model.tiny_in_proj_bias.append(b)

                w = tiny_Tensor(resblock.attn.out_proj.weight.detach().numpy().copy())
                self.model.tiny_out_proj_weight.append(w)

                b = tiny_Tensor(resblock.attn.out_proj.bias.detach().numpy().copy())
                self.model.tiny_out_proj_bias.append(b)

                linear = tiny_nn.Linear(resblock.mlp.c_fc.weight.shape[0], resblock.mlp.c_fc.weight.shape[1])
                linear.weight = tiny_Tensor(resblock.mlp.c_fc.weight.detach().numpy().copy())
                linear.bias = tiny_Tensor(resblock.mlp.c_fc.bias.detach().numpy().copy())
                self.model.tiny_c_fc.append(linear)

                linear = tiny_nn.Linear(resblock.mlp.c_proj.weight.shape[0], resblock.mlp.c_proj.weight.shape[1])
                linear.weight = tiny_Tensor(resblock.mlp.c_proj.weight.detach().numpy().copy())
                linear.bias = tiny_Tensor(resblock.mlp.c_proj.bias.detach().numpy().copy())
                self.model.tiny_c_proj.append(linear)

            with open('laion2b_s32b_b82k.pkl', 'wb') as f: pickle.dump(self.model, f)
        
        # for BEAM
        tiny_precompute_embeddings(self.model, tiny_Tensor.rand((1, 3, 224, 224), dtype=dtypes.float32))
        tiny_precompute_embeddings(self.model, tiny_Tensor.rand((16, 3, 224, 224), dtype=dtypes.float32))

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

            batch_tensors = torch.stack([inline_preprocess(img) for img in batch_images])
            x = tiny_Tensor(batch_tensors.detach().numpy())
            if len(batch_images) == batch_size:
                embeddings = tiny_precompute_embeddings(self.model, x)
                embeddings = torch.Tensor(embeddings.numpy())
            else:
                embeddings = []
                for i in range(len(batch_images)): # one by one if not batch_size
                    single_x = tiny_Tensor(x[i:i+1])
                    single_embedding = tiny_precompute_embedding(self.model, single_x)
                    single_embedding_torch = torch.Tensor(single_embedding.numpy())
                    embeddings.append(single_embedding_torch)
                embeddings = torch.stack(embeddings)

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
def tiny_precompute_embeddings(model, x): return tiny_precompute_embedding(model, x)

def tiny_precompute_embedding(model, x):
    print(x.shape, x.dtype)
    visual = model.visual
    x = visual.tiny_vc(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls_emb = visual.tiny_class_embedding
    cls_emb = cls_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
    x = tiny_Tensor.cat(cls_emb, x, dim=1)
    x = x + visual.tiny_positional_embedding
    x = visual.tiny_ln_pre(x)
    # https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/activation.py#L1252
    for i, block in enumerate(visual.transformer.resblocks):
        x_ln1 = model.tiny_ln_1[i](x)
        B, L, D = x_ln1.shape
        H = block.attn.num_heads
        d_head = D // H
        qkv = x_ln1 @ model.tiny_in_proj_weight[i].T + model.tiny_in_proj_bias[i]           
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
        attn_out = context @ model.tiny_out_proj_weight[i].T + model.tiny_out_proj_bias[i]
        attn_scaled = block.ls_1(attn_out)
        x = x + attn_scaled
        x_ln2 = model.tiny_ln_2[i](x)
        ff = model.tiny_c_fc[i](x_ln2)
        ff = ff.gelu()
        ff = model.tiny_c_proj[i](ff)
        x = x + ff

    x = visual.tiny_ln_post(x)
    image_embeds = x[:, 0, :]
    embeddings = image_embeds @ visual.tiny_proj
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