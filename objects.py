import os
import pickle
from datetime import datetime
from tinygrad import nn, Tensor, TinyJit, Device
from tinygrad.helpers import fetch
from tinygrad.dtype import dtypes
import numpy as np
import cv2
import time
from utils.helpers import send_notif, export_and_upload, BASE_DIR
from blazeface import BlazeFace
from adaface import ADAFACE
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from utils.clip_tokenizer import SimpleTokenizer
import math
from clearcam import event_img_info
from utils.clip_tokenizer import SimpleTokenizer

class Blank: pass

class OpenCLIP:
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


        self.visual_conv1 = nn.Conv2d(3, 1024, (14, 14), (14, 14), (0, 0), (1, 1), 1, bias=False)
        self.class_embedding = Tensor.empty(1024)
        self.positional_embedding = Tensor.empty(257, 1024)
 
        self.ln_pre = nn.LayerNorm(1024)
        self.ln_post = nn.LayerNorm(1024)
        self.proj = Tensor.empty(1024, 768)

        self.resblocks_img = []
        for i in range(24):
            resblock = Blank()
            resblock.ln_1 = nn.LayerNorm(1024, 1e-05, elementwise_affine=True)
            resblock.ln_2 = nn.LayerNorm(1024, 1e-05, elementwise_affine=True)
            resblock.in_proj_weight = Tensor.empty(3072, 1024)
            resblock.in_proj_bias = Tensor.empty(3072)
            resblock.out_proj_weight = Tensor.empty(1024, 1024)
            resblock.out_proj_bias = Tensor.empty(1024)
            resblock.mlp_c_fc = nn.Linear(1024, 4096)
            resblock.mlp_c_proj = nn.Linear(4096, 1024)
            self.resblocks_img.append(resblock)

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

        state_dict = safe_load(fetch("https://huggingface.co/roryclear/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/CLIP-ViT-L-14-laion2B-s32B-b82K.safetensors"))
        load_state_dict(self, state_dict)

    def _encode_text(self, query, realize=False):
        tokens = [49406]
        tokens += self.tokenizer.encode(query)
        tokens.append(49407)
        if len(tokens) < 77: tokens += [0] * (77 - len(tokens))
        tokens = Tensor([tokens])
        text_emb = encode_text(self, tokens)
        if realize: return text_emb.numpy()
        return text_emb

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

class ObjectFinder:
    def __init__(self, prewarm=False, base_path="data/cameras"):
        self.base_path = base_path
        self.image_embeddings = {}
        self.face_embeddings = {}
        self.image_paths = {}
        self.face_paths = {}
        
        self.model = OpenCLIP()
        
        self.blazeface = BlazeFace()
        self.adaface = ADAFACE()
        
        # prewarm
        if prewarm:
            blazeface_jit(self.blazeface, Tensor.rand((640, 640, 3)).cast(dtype=dtypes.uchar))
            adaface_jit(self.adaface, Tensor.rand((112, 112, 3)).cast(dtype=dtypes.uchar))
            precompute_embeddings_jit(self.model, Tensor.rand((1, 3, 224, 224), dtype=dtypes.float32))
            precompute_embeddings_jit(self.model, Tensor.rand((16, 3, 224, 224), dtype=dtypes.float32))

    def find_object_folders(self, base_path="data/cameras"):
        object_folders = []
        if not os.path.exists(base_path): return object_folders
        for camera_folder in os.listdir(base_path):
            camera_path = os.path.join(base_path, camera_folder)
            if os.path.isdir(camera_path):
                objects_path = os.path.join(camera_path, "objects")
                if os.path.isdir(objects_path):
                    for date_folder in os.listdir(objects_path):
                        date_path = os.path.join(objects_path, date_folder)
                        if os.path.isdir(date_path):
                            object_folders.append(date_path)
        return object_folders
    # db for progress
    def precompute_embeddings(self, folder_path, batch_size=16, vod=False, database=None, cam_name=None, userID=None, key=None):
        folder_embeddings, folder_paths = get_embeddings(folder_path, "embeddings.pkl")
        folder_embeddings_face, folder_paths_face = get_embeddings(folder_path.replace("objects", "faces"), "embeddings.pkl")
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

        if not new_images: return [], []
        new_image_list = list(new_images)
        self.process_faces(new_image_list)
        face_paths, face_embeddings = self.process_faces(new_image_list)
        for path, emb in zip(face_paths, face_embeddings):
          folder_embeddings_face[path] = emb
          folder_paths_face[path] = path
        save_embeddings(folder_path.replace("objects", "faces"), "embeddings.pkl", folder_embeddings_face, folder_paths_face)
        emb_ret = []
        path_ret = []
        for i in range(0, len(new_image_list), batch_size):
            batch_paths = new_image_list[i:i + batch_size]
            batch_np = []

            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = preprocess(img)
                batch_np.append(img)

            if not batch_np: continue
            batch_np = np.stack(batch_np)

            if len(batch_np) == batch_size:
                embeddings = precompute_embeddings_jit(self.model, Tensor(batch_np)).numpy()
            else:
                embeddings = []
                for j in range(len(batch_np)):
                  emb = precompute_embedding_jit_bs1(self.model, Tensor(batch_np[j:j+1])).numpy()
                  embeddings.append(emb)
            emb_ret.extend(embeddings)
            path_ret.extend(batch_paths)
            for path, embedding in zip(batch_paths, embeddings):
                folder_embeddings[path] = embedding
                folder_paths[path] = path
            print(f"Processed {min(i + batch_size, len(new_image_list))}/{len(new_image_list)} new images...")
            if vod: database.run_put("analysis_prog", cam_name, {"Processing":(min(i + batch_size, len(new_image_list))/len(new_image_list))*100})
        save_embeddings(folder_path, "embeddings.pkl", folder_embeddings, folder_paths)
        return emb_ret, path_ret

    def precompute_embedding_bs1_np(self, img): return precompute_embedding_jit_bs1(self.model, Tensor(img)).numpy() # todo remove

    def precompute_face_embedding_bs1_np(self, img): return adaface_jit(self.adaface, Tensor(img)).numpy() # todo remove

    def preprocess_clip(self, img):
      if type(img) == bytes:
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
      else:
        img = f"data/cameras{img}"
        img = cv2.imread(img) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      return [preprocess(img)]
    
    def preprocess_face(self, img):
      if type(img) == bytes: # todo dup of above
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
      else:
        img = f"data/cameras{img}"
        img = cv2.imread(img) 
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      if img.shape != (112, 112, 3): img = self.img_to_face(img)
      return img

    def img_to_face(self, orig):
        h, w = orig.shape[:2]
        scale = 640 / max(h, w)
        resized = cv2.resize(orig, (int(w*scale), int(h*scale)))
        delta_w, delta_h = 640 - resized.shape[1], 640 - resized.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        orig = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        detections = blazeface_jit(self.blazeface, Tensor(orig)).numpy()
        detections = detections[detections[:, 0] != 0]
        # one face per person for now
        if detections.shape[0] > 0:
            y1, x1, y2, x2 = detections[0][:4]
            left_eye = np.array([detections[0][4], detections[0][5]])
            right_eye = np.array([detections[0][6], detections[0][7]])
            
            if (x2 - x1) < 50: return None
            TARGET_LEFT_EYE = np.array([38, 51])
            TARGET_RIGHT_EYE = np.array([73, 51])

            eye_center = (left_eye + right_eye) / 2
            target_eye_distance = np.linalg.norm(TARGET_RIGHT_EYE - TARGET_LEFT_EYE)

            angle_rad = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            angle = np.degrees(angle_rad)

            face_width = x2 - x1
            face_height = y2 - y1

            crop_size = max(face_width, face_height) * 2.0

            x1_crop = int(eye_center[0] - crop_size / 2)
            y1_crop = int(eye_center[1] - crop_size / 2)
            x2_crop = int(eye_center[0] + crop_size / 2)
            y2_crop = int(eye_center[1] + crop_size / 2)

            H, W = orig.shape[:2]
            x1_crop = max(0, x1_crop)
            y1_crop = max(0, y1_crop)
            x2_crop = min(W, x2_crop)
            y2_crop = min(H, y2_crop)

            if x2_crop <= x1_crop or y2_crop <= y1_crop: return None
                
            cropped = orig[y1_crop:y2_crop, x1_crop:x2_crop]
            crop_h, crop_w = cropped.shape[:2]

            if crop_h == 0 or crop_w == 0: return None
            
            left_eye_crop = left_eye - np.array([x1_crop, y1_crop])
            right_eye_crop = right_eye - np.array([x1_crop, y1_crop])

            rot_mat = cv2.getRotationMatrix2D((crop_w/2, crop_h/2), angle, 1.0)

            cos_a = np.abs(rot_mat[0, 0])
            sin_a = np.abs(rot_mat[0, 1])
            new_w = int(crop_h * sin_a + crop_w * cos_a)
            new_h = int(crop_h * cos_a + crop_w * sin_a)
            
            rot_mat[0, 2] += (new_w / 2) - crop_w / 2
            rot_mat[1, 2] += (new_h / 2) - crop_h / 2
            
            rotated = cv2.warpAffine(cropped, rot_mat, (new_w, new_h))
   
            left_eye_rot = rot_mat[:, :2] @ left_eye_crop + rot_mat[:, 2]
            right_eye_rot = rot_mat[:, :2] @ right_eye_crop + rot_mat[:, 2]
 
            rot_eye_distance = np.linalg.norm(right_eye_rot - left_eye_rot)
            final_scale = target_eye_distance / rot_eye_distance
            
            tx = TARGET_LEFT_EYE[0] - left_eye_rot[0] * final_scale
            ty = TARGET_LEFT_EYE[1] - left_eye_rot[1] * final_scale
            
            transform_mat = np.array([[final_scale, 0, tx], [0, final_scale, ty]], dtype=np.float32)
            
            face_img = cv2.warpAffine(rotated, transform_mat, (112, 112))
            
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            # Debug draw dots for the eyes at target positions
            #cv2.circle(face_img, (38, 51), 2, (0, 255, 0), -1)
            #cv2.circle(face_img, (73, 51), 2, (0, 255, 0), -1)
            
            return face_img
        
        return None

    def process_faces(self, paths):
      ret_paths = []
      ret_embeddings = []
      for path in paths:
        if path.endswith("_0.jpg"): # person
          orig = cv2.imread(path)
          orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
          face_img = self.img_to_face(orig)
          if face_img is None: continue
          cv2.imwrite(path.replace("objects","faces"), face_img)
          embeddings = adaface_jit(self.adaface, Tensor(face_img)).numpy()
          ret_embeddings.append(embeddings)
          ret_paths.append(path)

      return ret_paths, ret_embeddings

    def search(self, query=None, top_k=10, cam_name=None, timestamp=None, text_embedding=None, is_face=False):
        embeddings = self.face_embeddings if is_face else self.image_embeddings
        if not embeddings:
            print("No embeddings available.")
            return []
        if text_embedding is None:
            text_embedding = self.model._encode_text(query)
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

def get_embeddings(path, filename):
  cache_file = os.path.join(path, filename)
  folder_embeddings = {}
  folder_paths = {}
  if os.path.exists(cache_file):
      with open(cache_file, "rb") as f:
          cache = pickle.load(f)
          folder_embeddings = cache.get("embeddings", {})
          folder_paths = cache.get("paths", {})
  return folder_embeddings, folder_paths

def save_embeddings(folder_path, file_name, folder_embeddings, folder_paths):
    cache_file = os.path.join(folder_path, file_name)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f: pickle.dump({"embeddings": folder_embeddings, "paths": folder_paths}, f)

@TinyJit
def precompute_embeddings_jit(model, x): return precompute_embedding(model, x)

@TinyJit
def precompute_embedding_jit_bs1(model, x): return precompute_embedding(model, x)

@TinyJit
def blazeface_jit(model, x): return model(x)

@TinyJit
def adaface_jit(model, x): return model(x)

def precompute_embedding(model, x):
    x = model.visual_conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls_emb = model.class_embedding
    cls_emb = cls_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
    x = Tensor.cat(cls_emb, x, dim=1)
    x = x + model.positional_embedding
    x = model.ln_pre(x)
    # https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/activation.py#L1252
    for i in range(len(model.resblocks_img)):
        x_ln1 = model.resblocks_img[i].ln_1(x)
        B, L, D = x_ln1.shape
        H = 16 #block.attn.num_heads
        d_head = D // H
        qkv = x_ln1 @ model.resblocks_img[i].in_proj_weight.T + model.resblocks_img[i].in_proj_bias           
        q, k, v = qkv.split(D, dim=-1)
        def shape(x): return x.view(B, L, H, d_head).transpose(1, 2)
        q = shape(q)
        k = shape(k)
        v = shape(v)
        scale = 1.0 / (d_head ** 0.5)
        attn_scores = q.matmul(k.transpose(-2, -1)) * scale
        attn_probs = Tensor.softmax(attn_scores)
        context = attn_probs.matmul(v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = context @ model.resblocks_img[i].out_proj_weight.T + model.resblocks_img[i].out_proj_bias      
        attn_scaled = attn_out
        x = x + attn_scaled
        x_ln2 = model.resblocks_img[i].ln_2(x)
        ff = model.resblocks_img[i].mlp_c_fc(x_ln2)
        ff = ff.gelu()
        ff = model.resblocks_img[i].mlp_c_proj(ff)
        x = x + ff

    x = model.ln_post(x)
    image_embeds = x[:, 0, :]
    embeddings = image_embeds @ model.proj
    embeddings = embeddings / (embeddings.pow(2).sum(axis=-1, keepdim=True).sqrt() + 1e-8)
    return embeddings

def preprocess(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return img