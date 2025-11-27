import torch
from PIL import Image
import open_clip
import os
import pickle
import time
from datetime import datetime
from tinygrad import nn as tiny_nn, Tensor as tiny_Tensor, TinyJit
import torch.nn.functional as F
# DO NOT USE WITH BEAM
class CachedCLIPSearch:
    def __init__(self, model_name="ViT-L-14", pretrained_name="laion2b_s32b_b82k"):
        print(f"Loading OpenCLIP model: {model_name} ({pretrained_name})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
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
        
        self.model.tiny_ln_1 = []

        for resblock in self.model.visual.transformer.resblocks:
            layernorm = tiny_nn.LayerNorm(normalized_shape=resblock.ln_1.normalized_shape, eps=resblock.ln_1.eps, elementwise_affine=True)
            layernorm.weight = tiny_Tensor(resblock.ln_1.weight.detach().numpy().copy())
            layernorm.bias = tiny_Tensor(resblock.ln_1.bias.detach().numpy().copy())
            self.model.tiny_ln_1.append(layernorm)


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

        # Remove deleted files
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

            if batch_images:
                batch_tensors = torch.stack(
                    [self.preprocess(img) for img in batch_images]
                )

                visual = self.model.visual
                x = batch_tensors
                
                x = tiny_Tensor(x.detach().numpy())
                x = visual.tiny_vc(x)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                cls_emb = visual.tiny_class_embedding
                cls_emb = cls_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
                x = tiny_Tensor.cat(cls_emb, x, dim=1)
                x = x + visual.tiny_positional_embedding
                x = visual.tiny_ln_pre(x)
                print(type(x))

                for i, block in enumerate(visual.transformer.resblocks):
                    x_ln1 = self.model.tiny_ln_1[i](x)
                    x_ln1 = torch.Tensor(x_ln1.numpy())
                    
                    # https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/activation.py#L1252

                    attn_out, _ = torch._native_multi_head_attention(
                        x_ln1,
                        x_ln1,
                        x_ln1,
                        block.attn.embed_dim,
                        block.attn.num_heads,
                        block.attn.in_proj_weight,
                        block.attn.in_proj_bias,
                        block.attn.out_proj.weight,
                        block.attn.out_proj.bias,
                        None,
                        True,
                        True,
                        None,
                    )
                    

                    attn_scaled = block.ls_1(attn_out)
                    x = torch.Tensor(x.numpy())
                    x = x + attn_scaled
                    x_ln2 = block.ln_2(x)
                    ff = block.mlp.c_fc(x_ln2)
                    ff = block.mlp.gelu(ff)
                    ff = block.mlp.c_proj(ff)
                    ff_scaled = block.ls_2(ff)
                    x = x + ff_scaled
                    x = tiny_Tensor(x.detach().numpy())

                x = torch.Tensor(x.numpy())
                x = visual.ln_post(x)
                image_embeds = x[:, 0, :]
                embeddings = image_embeds @ visual.proj
                
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                for path, embedding in zip(batch_paths, embeddings):
                    folder_embeddings[path] = embedding
                    folder_paths[path] = path

            print(f"Processed {min(i + batch_size, len(new_image_list))}/{len(new_image_list)} new images...")

            for img in batch_images:
                img.close()

        # Save updated cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"embeddings": folder_embeddings, "paths": folder_paths}, f)

        print(f"{datetime.now()}: Updated cache for {folder_path}. Total images stored: {len(folder_embeddings)}")



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