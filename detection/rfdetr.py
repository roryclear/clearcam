import numpy as np
import math
from tinygrad.dtype import dtypes
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save, safe_load
from tinygrad.helpers import fetch
from tinygrad import Tensor, nn
from tinygrad import TinyJit

COCO_CLASSES = ["","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","","dining table","","","toilet","","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","","book","clock","vase","scissors","teddy bear","hair drier"]
detr_to_yolo = [80, 0, 1, 2, -1, -1, 5, 6, 7, 8, 9, 10, 80, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 80, 24, 25, 80, 80, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 80, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, -1, -1, 59, 80, -1, 80, 80, 61, 80, -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 80, 73, 74, 75, 76, 77, 78]

class WindowedDinov2WithRegistersEmbeddings():
  def __call__(self, pixel_values):
    batch_size, _, height, width = pixel_values.shape
    embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)

    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = Tensor.cat(cls_tokens, embeddings, dim=1)
    embeddings = embeddings + self.position_embeddings
    num_h_patches = height // 16
    num_w_patches = width // 16
    cls_token_with_pos_embed = embeddings[:, :1]
    pixel_tokens_with_pos_embed = embeddings[:, 1:]
    pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(batch_size, num_h_patches, num_w_patches, -1)

    num_w_patches_per_window = num_w_patches // self.num_windows
    num_h_patches_per_window = num_h_patches // self.num_windows
    windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(batch_size, self.num_windows, num_h_patches_per_window, self.num_windows, num_w_patches_per_window, -1)
    windowed_pixel_tokens = windowed_pixel_tokens.permute(0,1,3,2,4,5)
    windowed_pixel_tokens = windowed_pixel_tokens.reshape(batch_size * self.num_windows**2, num_h_patches_per_window * num_w_patches_per_window, -1)

    windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(self.num_windows ** 2, 1, 1)
    embeddings = Tensor.cat(windowed_cls_token_with_pos_embed, windowed_pixel_tokens, dim=1)
    return embeddings

class Dinov2WithRegistersSdpaSelfAttention():
    def transpose_for_scores(self, x):
      new_x_shape = x.size()[:-1] + (6, 64)
      x = x.view(new_x_shape)
      return x.permute(0, 2, 1, 3)

    def __call__(self, hidden_states, head_mask, output_attentions):
      mixed_query_layer = self.query(hidden_states)
      key_layer = self.transpose_for_scores(self.key(hidden_states))
      value_layer = self.transpose_for_scores(self.value(hidden_states))
      query_layer = self.transpose_for_scores(mixed_query_layer)

      d_k = query_layer.size(-1)
      attn_scores = Tensor.matmul(query_layer, key_layer.transpose(-2, -1)) / math.sqrt(d_k)
      attn_probs = Tensor.softmax(attn_scores, axis=-1)
      context_layer = Tensor.matmul(attn_probs, value_layer)
      context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
      new_context_layer_shape = context_layer.size()[:-2] + (384,)
      context_layer = context_layer.view(new_context_layer_shape)
      return context_layer, None

class Dinov2WithRegistersSdpaAttention():
    def __call__(self, hidden_states, head_mask=None, output_attentions= False):
      self_outputs = self.attention(hidden_states, head_mask, output_attentions)
      attention_output = self.dense(self_outputs[0])
      return (attention_output,) + self_outputs[1:]

class Dinov2WithRegistersMLP():
    def __call__(self, hidden_state):
        hidden_state = self.fc1(hidden_state)
        hidden_state = hidden_state * 0.5 * (1.0 + Tensor.erf(hidden_state / math.sqrt(2.0)))
        hidden_state = self.fc2(hidden_state)
        return hidden_state

class WindowedDinov2WithRegistersLayer():
    def __call__(self, hidden_states, head_mask=None, output_attentions= False, run_full_attention= False):
      shortcut = hidden_states
      if run_full_attention:
        B, HW, C = hidden_states.shape
        num_windows_squared = self.num_windows ** 2
        hidden_states = hidden_states.view(B // num_windows_squared, num_windows_squared * HW, C)
      x = self.norm1(hidden_states)

      # todo
      self_attention_outputs = self.attention(x, head_mask, output_attentions=output_attentions,)
      attention_output = self_attention_outputs[0]

      if run_full_attention:
        B, HW, C = hidden_states.shape
        num_windows_squared = self.num_windows ** 2
        attention_output = attention_output.view(B * num_windows_squared, HW // num_windows_squared, C)
      attention_output = (attention_output) * self.lambda1
      outputs = self_attention_outputs[1:]
      hidden_states = attention_output + shortcut

      # in Dinov2WithRegisters, layernorm is also applied after self-attention
      layer_output = self.norm2(hidden_states)
      layer_output = self.mlp(layer_output)
      layer_output = layer_output * self.lambda2
      layer_output = layer_output + hidden_states
      return (layer_output,) + outputs

class WindowedDinov2WithRegistersEncoder():
    def __call__(self, hidden_states, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True,):
      all_hidden_states = () if output_hidden_states else None
      all_self_attentions = () if output_attentions else None

      for i, layer_module in enumerate(self.layer):
        all_hidden_states = all_hidden_states + (hidden_states,)
        run_full_attention = i not in [0, 1, 2, 4, 5, 7, 8, 10, 11]
        layer_head_mask = None
        layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, run_full_attention)
        hidden_states = layer_outputs[0]

      all_hidden_states = all_hidden_states + (hidden_states,)
      return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

class WindowedDinov2WithRegistersBackbone():
    def __init__(self):
      self.config = {}
      self.stage_names = ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7', 'stage8', 'stage9', 'stage10', 'stage11', 'stage12']
      self.out_features = ['stage3', 'stage6', 'stage9', 'stage12']

    def __call__(self, pixel_values, output_hidden_states=None, output_attentions=None, return_dict=None,):
        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict)

        hidden_states = outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
          if stage in self.out_features:
            hidden_state = self.layernorm(hidden_state)
            hidden_state = hidden_state[:, 1 :]
            # this was actually a bug in the original implementation that we copied here,
            # cause normally the order is height, width
            batch_size, _, height, width = pixel_values.shape
            patch_size = 16


            num_h_patches = height // patch_size
            num_w_patches = width // patch_size

            # undo windowing
            num_windows_squared = self.num_windows ** 2
            B, HW, C = hidden_state.shape

            num_h_patches_per_window = num_h_patches // self.num_windows
            num_w_patches_per_window = num_w_patches // self.num_windows

            hidden_state = hidden_state.reshape(B // num_windows_squared, self.num_windows, self.num_windows, num_h_patches_per_window, num_w_patches_per_window, C)
            hidden_state = hidden_state.permute(0,1,3,2,4,5)
            hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, C)
            hidden_state = hidden_state.permute(0,3,1,2).contiguous()
            feature_maps += (hidden_state,)

        output = (feature_maps,) + outputs[2:]
        return output
    
def ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights):
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
    value_l_ = value.view(B * n_heads, head_dim, int(value_spatial_shapes), int(value_spatial_shapes))
    sampling_grid_l_ = sampling_grids[:, :, :, 0].transpose(1, 2).flatten(0, 1)

    N, C, H, W = value_l_.shape
    _, H_out, W_out, _ = sampling_grid_l_.shape
    x = (sampling_grid_l_[..., 0] + 1) * W / 2 - 0.5
    y = (sampling_grid_l_[..., 1] + 1) * H / 2 - 0.5
    x0 = x.floor().cast(dtype=dtypes.int64)
    y0 = y.floor().cast(dtype=dtypes.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0.float()

    wy = y - y0.cast(dtype=dtypes.float)
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy


    v00 = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)

    v01 = (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)

    v10 = (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
    v11 = (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)

    x0c = x0.clamp(0, W - 1)
    x1c = x1.clamp(0, W - 1)
    y0c = y0.clamp(0, H - 1)
    y1c = y1.clamp(0, H - 1)

    value_flat = value_l_.view(N, C, H * W)
    def gather(xi, yi):
        idx = (yi * W + xi).view(N, 1, -1).expand(-1, C, -1)
        return Tensor.gather(value_flat, 2, idx).view(N, C, H_out, W_out)

    g00 = gather(x0c, y0c)
    g01 = gather(x0c, y1c)
    g10 = gather(x1c, y0c)
    g11 = gather(x1c, y1c)
    sampling_value_l_ = (g00 * (w00 * v00).unsqueeze(1) + g01 * (w01 * v01).unsqueeze(1) +
    g10 * (w10 * v10).unsqueeze(1) + g11 * (w11 * v11).unsqueeze(1))
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    output = (sampling_value_l_ * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    ret = output.transpose(1, 2).contiguous()
    return ret    

class TransformerDecoderLayer(): # todo, remove unused
    def __call__(self, tgt, memory, memory_key_padding_mask, query_pos,
      reference_points=None, spatial_shapes=None, level_start_index=None):  
        q = k = tgt + query_pos
        v = tgt

        C = 256
        B, T, C = q.shape
        H = 8
        D = C // H
        w = self.in_proj_weight
        b = self.in_proj_bias
        wo = self.out_proj_weight
        bo = self.out_proj_bias
        wq, wk, wv = w.chunk(3, dim=0)
        bq, bk, bv = b.chunk(3, dim=0)

        q = q @ wq.T + bq
        k = k @ wk.T + bk
        v = v @ wv.T + bv

        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        attn = Tensor.scaled_dot_product_attention(q,k,v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        tgt2 = attn @ wo.T + bo
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt+query_pos, reference_points, memory, spatial_shapes, memory_key_padding_mask)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        x = self.linear1(tgt)
        x = x.relu()
        tgt2 = self.linear2(x)
        tgt += tgt2
        tgt = self.norm3(tgt)
        return tgt
    
def gen_sineembed_for_position(pos_tensor, dim=128):
  scale = 2 * math.pi
  dim_t = Tensor.arange(dim)
  dim_t = 10000 ** (2 * (dim_t // 2) / dim)
  x_embed = pos_tensor[:, :, 0] * scale
  y_embed = pos_tensor[:, :, 1] * scale
  pos_x = x_embed[:, :, None] / dim_t
  pos_y = y_embed[:, :, None] / dim_t
  pos_x = Tensor.stack(pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos(), dim=3).flatten(2)
  pos_y = Tensor.stack(pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos(), dim=3).flatten(2)
  w_embed = pos_tensor[:, :, 2] * scale
  pos_w = w_embed[:, :, None] / dim_t
  pos_w = Tensor.stack(pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos(), dim=3).flatten(2)

  h_embed = pos_tensor[:, :, 3] * scale
  pos_h = h_embed[:, :, None] / dim_t
  pos_h = Tensor.stack(pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos(), dim=3).flatten(2)
  pos = Tensor.cat(pos_y, pos_x, pos_w, pos_h, dim=2)
  return pos

class TransformerDecoder(): # todo remove unused
    def __call__(self, tgt, memory, memory_key_padding_mask=None,
      refpoints_unsigmoid=None, level_start_index=None, spatial_shapes=None, valid_ratios=None):
        intermediate = []
        def get_reference(refpoints_unsigmoid, valid_ratios):
          obj_center = refpoints_unsigmoid[..., :4]
          refpoints_input = obj_center[:, :, None] * Tensor.cat(valid_ratios, valid_ratios, dim=-1)[:, None]
          query_sine_embed = gen_sineembed_for_position(
              refpoints_input[:, :, 0, :], 256 / 2) # bs, nq, 256*2
          query_pos = self.ref_point_head(query_sine_embed)
          return refpoints_input, query_pos

        for _, layer in enumerate(self.layers):
          refpoints_input, query_pos = get_reference(refpoints_unsigmoid, valid_ratios) #todo

          tgt = layer(tgt, memory,
            memory_key_padding_mask=memory_key_padding_mask,
            query_pos=query_pos,
            reference_points=refpoints_input,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

          x = self.norm(tgt)
          intermediate.append(x)
        
        tgt = self.norm(tgt)
        intermediate.pop()
        intermediate.append(tgt)
        return [(Tensor.stack(intermediate)), (refpoints_unsigmoid.unsqueeze(0))]

def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shape, unsigmoid=True):
    memory_padding_mask = memory_padding_mask.cast(dtype=dtypes.bool)

    H_, W_ = spatial_shape, spatial_shape
    mask = memory_padding_mask.reshape(1, H_, W_)

    valid_H = (~mask[:, :, 0]).sum(axis=1).unsqueeze(-1)
    valid_W = (~mask[:, 0, :]).sum(axis=1).unsqueeze(-1)

    x = Tensor.linspace(0, H_ - 1, H_)
    y = Tensor.linspace(0, W_ - 1, W_)
    grid_y, grid_x = Tensor.meshgrid(y, x)

    grid = Tensor.cat(grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), dim=-1)
    scale = Tensor.cat(valid_W, valid_H, dim=1).view(1, 1, 1, 2)
    grid = (grid.unsqueeze(0).expand(1, -1, -1, -1) + 0.5) / scale

    wh = Tensor.ones_like(grid) * 0.05
    output_proposals = Tensor.cat(grid, wh, dim=-1).view(1, -1, 4)

    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))
    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals

class MSDeformAttn():
    def __call__(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.masked_fill(input_padding_mask[..., None], float(0))
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, 16, 1, 2, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, 16, 1 * 2)
        sampling_locations = reference_points[:, :, None, :, None, :2] \
                                + sampling_offsets / 2 * reference_points[:, :, None, :, None, 2:] * 0.5
        attention_weights = attention_weights.softmax(-1)
        value = value.transpose(1, 2).contiguous().view(N, 16, 256 // 16, Len_in)
        output = ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

class Transformer():
    def __call__(self, srcs, masks, refpoint_embed, query_feat):

        self.enc_out_class_embed_w = self.enc_out_class_embed[0].weight
        self.enc_out_class_embed_b = self.enc_out_class_embed[0].bias

        src = srcs[0] if type(srcs) == list else srcs
        bs, _, h, w = src.shape
        src = src.flatten(2).transpose(1, 2)              # bs, hw, c
        mask = masks[0].flatten(1) if type(masks) == list else masks.flatten(1)
        level_start_index = Tensor([0])
        output_memory, output_proposals = gen_encoder_output_proposals(
            src, mask, h, unsigmoid=True)
        
        output_memory_gidx = self.enc_output_norm(self.enc_output(output_memory))
        enc_outputs_class_unselected_gidx = output_memory_gidx @ self.enc_out_class_embed_w.T + self.enc_out_class_embed_b

        
        enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[0](output_memory_gidx)

        enc_outputs_coord_cxcy_gidx = enc_outputs_coord_delta_gidx[...,
            :2] * output_proposals[..., 2:] + output_proposals[..., :2]
        enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
        enc_outputs_coord_unselected_gidx = Tensor.cat(enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx, dim=-1)
        topk = min(300, enc_outputs_class_unselected_gidx.shape[-2])
        x = enc_outputs_class_unselected_gidx.max(-1)
        topk_proposals_gidx = Tensor.topk(x, topk, dim=1)[1]
        boxes_ts = enc_outputs_coord_unselected_gidx.gather(dim=1, index=topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4))
        tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)

        ts_len = boxes_ts.shape[-2]
        refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
        refpoint_embed_subset = refpoint_embed[..., ts_len:, :]


        refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * boxes_ts[..., 2:]
        refpoint_embed_cxcy = refpoint_embed_cxcy + boxes_ts[..., :2]
        refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * boxes_ts[..., 2:]
        refpoint_embed_ts_subset = Tensor.cat(refpoint_embed_cxcy, refpoint_embed_wh, dim=-1)
        refpoint_embed = Tensor.cat(refpoint_embed_ts_subset, refpoint_embed_subset, dim=-2)
        hs, references = self.decoder(tgt, src, memory_key_padding_mask=mask,
                        refpoints_unsigmoid=refpoint_embed,
                        level_start_index=level_start_index,
                        spatial_shapes=h,
                        valid_ratios=Tensor([[[1., 1.]]]))

        return hs, references

    
class ConvX():
    def __call__(self, x):
      x = self.conv(x)
      x = self.bn(x)
      return Tensor.silu(x)

class Bottleneck():
    def __call__(self, x): return self.cv2(self.cv1(x))

class C2f():
    def __init__(self): self.c = 128
    def __call__(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = Tensor.cat(*y, dim=1)
        y = self.cv2(y)
        return y

class LayerNorm():
    def __init__(self): self.eps = 1e-6
    def __call__(self, x):
      x = x.permute(0, 2, 3, 1)
      x -= x.mean(axis=-1, keepdim=True)
      var = (x ** 2).mean(axis=-1, keepdim=True) + self.eps
      var = Tensor.sqrt(var)
      x_norm = x / var
      x_norm = x_norm * self.weight
      x_norm = x_norm + self.bias
      x = x_norm
      x = x.permute(0, 3, 1, 2)
      return x

class MultiScaleProjector():
    def __call__(self, x):
      feat_fuse = Tensor.cat(*x, dim=1)
      stage_output = self.stages[0](feat_fuse)
      return [stage_output]

class PositionEmbeddingSine():
    def __init__(self):
      self.scale = 6.283185307179586
      self.num_pos_feats = 128
      self.temperature = 10000

    def __call__(self, tensors, mask, align_dim_orders = True):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = Tensor.arange(self.num_pos_feats)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = Tensor.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = Tensor.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = Tensor.cat(pos_y, pos_x, dim=3).permute(0, 3, 1, 2)
        return pos

class Backbone():
    def __call__(self, tensors ,mask):
      feats = list(self.encoder(tensors)[0])
      feats = self.projector(feats)
      m = mask
      mask = ~Tensor.interpolate(m.unsqueeze(0), size=feats[0].shape[-2:])[0]
      return feats[0], mask

class MLP():
    def __call__(self, x):            
      for i, layer in enumerate(self.layers): x = Tensor.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
      return x
    
class RFDETR():
  def __init__(self, name, res=None):
    self.num_queries = 300
    config = {"nano":{"n_layers":2, "size": 577, "res": 384}, "small":{"n_layers":3, "size":1025, "res":512},
                "medium":{"n_layers":4, "size":1297, "res":576}, "large":{"n_layers":4, "size":1937, "res":704}}
    self.res = config[name]["res"] if res is None else res

    num_windows = 2
    self.position_embedding = PositionEmbeddingSine()
    self.query_feat = Tensor.empty((3900, 256))
    self.refpoint_embed = Tensor.empty((3900, 4))
    self.class_embed = nn.Linear(256, 91)
    self.bbox_embed = MLP()
    self.bbox_embed.num_layers = 3
    self.bbox_embed.layers = seq(size=3)
    self.bbox_embed.layers[0] = nn.Linear(256, 256)
    self.bbox_embed.layers[1] = nn.Linear(256, 256)
    self.bbox_embed.layers[2] = nn.Linear(256, 4)

    self.bbox_embed.layers = seq(size=3)
    self.bbox_embed.layers[0] = nn.Linear(256, 256)
    self.bbox_embed.layers[1] = nn.Linear(256, 256)
    self.bbox_embed.layers[2] = nn.Linear(256, 4)


    self.transformer = Transformer()
    self.transformer.enc_output = nn.Linear(256, 256)
    self.transformer.enc_output_norm = nn.LayerNorm(256)
    self.transformer.decoder = TransformerDecoder()
    self.transformer.decoder.norm = nn.LayerNorm(256)
    self.transformer.decoder.layers = seq(config[name]["n_layers"])
    for i in range(config[name]["n_layers"]):
      self.transformer.decoder.layers[i] = TransformerDecoderLayer()
      self.transformer.decoder.layers[i].cross_attn = MSDeformAttn()
      self.transformer.decoder.layers[i].linear1 = nn.Linear(256, 2048)
      self.transformer.decoder.layers[i].linear2 = nn.Linear(2048, 256)

      self.transformer.decoder.layers[i].in_proj_weight = Tensor.empty(768, 256)
      self.transformer.decoder.layers[i].in_proj_bias = Tensor.empty(768)
      self.transformer.decoder.layers[i].out_proj_weight = Tensor.empty(256, 256)
      self.transformer.decoder.layers[i].out_proj_bias = Tensor.empty(256)

      self.transformer.decoder.layers[i].norm1 = nn.LayerNorm(256)
      self.transformer.decoder.layers[i].norm2 = nn.LayerNorm(256)
      self.transformer.decoder.layers[i].norm3 = nn.LayerNorm(256)

      self.transformer.decoder.layers[i].cross_attn.value_proj = nn.Linear(256, 256)
      self.transformer.decoder.layers[i].cross_attn.output_proj = nn.Linear(256, 256)
      self.transformer.decoder.layers[i].cross_attn.sampling_offsets = nn.Linear(256, 64)
      self.transformer.decoder.layers[i].cross_attn.attention_weights = nn.Linear(256, 32)

    self.transformer.enc_out_bbox_embed = seq(13)
    self.transformer.enc_out_class_embed = seq(13)
    for i in range(13):
      self.transformer.enc_out_bbox_embed[i] = MLP()
      self.transformer.enc_out_bbox_embed[i].num_layers = 3
      self.transformer.enc_out_bbox_embed[i].layers = seq(3)
      self.transformer.enc_out_bbox_embed[i].layers[0] = nn.Linear(256, 256)
      self.transformer.enc_out_bbox_embed[i].layers[1] = nn.Linear(256, 256)
      self.transformer.enc_out_bbox_embed[i].layers[2] = nn.Linear(256, 4)
      self.transformer.enc_out_bbox_embed[i].layers = seq(3)
      self.transformer.enc_out_bbox_embed[i].layers[0] = nn.Linear(256, 256)
      self.transformer.enc_out_bbox_embed[i].layers[1] = nn.Linear(256, 256)
      self.transformer.enc_out_bbox_embed[i].layers[2] = nn.Linear(256, 4)

      self.transformer.enc_out_class_embed[i] = nn.Linear(256, 91)         

    self.transformer.decoder.ref_point_head = MLP()
    self.transformer.decoder.ref_point_head.layers = seq(size=2)
    self.transformer.decoder.ref_point_head.num_layers = 2
    self.transformer.decoder.ref_point_head.layers = seq(size=2)
    self.transformer.decoder.ref_point_head.layers[0] = nn.Linear(512, 256)
    self.transformer.decoder.ref_point_head.layers[1] = nn.Linear(256, 256)
    self.transformer.decoder.ref_point_head.layers[0] = nn.Linear(512, 256) # todo remove
    self.transformer.decoder.ref_point_head.layers[1] = nn.Linear(256, 256)

    self.backbone = Backbone()
    self.backbone.projector = MultiScaleProjector()
    self.backbone.projector.stages = seq(1) # todo, this is dumb
    self.backbone.projector.stages[0] = seq(2)
    self.backbone.projector.stages[0][0] = C2f()
    self.backbone.projector.stages[0][1] = LayerNorm()
    self.backbone.projector.stages[0][1].weight = Tensor.empty((256))
    self.backbone.projector.stages[0][1].bias = Tensor.empty((256))
    self.backbone.projector.stages[0][0].cv1 = ConvX()
    self.backbone.projector.stages[0][0].cv2 = ConvX()          
    self.backbone.projector.stages[0][0].cv1.conv = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, bias=False, padding=(0, 0))
    self.backbone.projector.stages[0][0].cv2.conv = nn.Conv2d(in_channels=640, out_channels=256, kernel_size=1, bias=False, padding=(0, 0))

    self.backbone.projector.stages[0][0].cv1.bn = LayerNorm()
    self.backbone.projector.stages[0][0].cv1.bn.weight = Tensor.empty((256))
    self.backbone.projector.stages[0][0].cv1.bn.bias = Tensor.empty((256))
    self.backbone.projector.stages[0][0].cv2.bn = LayerNorm()
    self.backbone.projector.stages[0][0].cv2.bn.weight = Tensor.empty((256))
    self.backbone.projector.stages[0][0].cv2.bn.bias = Tensor.empty((256))
  
    self.backbone.projector.stages[0][0].m = seq(size=3)
    for i in range(3):
      self.backbone.projector.stages[0][0].m[i] = Bottleneck()
      self.backbone.projector.stages[0][0].m[i].cv1 = ConvX()
      self.backbone.projector.stages[0][0].m[i].cv1.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=False, padding=(1, 1))
      self.backbone.projector.stages[0][0].m[i].cv2 = ConvX()
      self.backbone.projector.stages[0][0].m[i].cv2.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=False, padding=(1, 1))            
      self.backbone.projector.stages[0][0].m[i].cv1.bn = LayerNorm()
      self.backbone.projector.stages[0][0].m[i].cv1.bn.weight = Tensor.empty((128))
      self.backbone.projector.stages[0][0].m[i].cv1.bn.bias = Tensor.empty((128))
      self.backbone.projector.stages[0][0].m[i].cv2.bn = LayerNorm()
      self.backbone.projector.stages[0][0].m[i].cv2.bn.weight = Tensor.empty((128))
      self.backbone.projector.stages[0][0].m[i].cv2.bn.bias = Tensor.empty((128))
    

    self.backbone.patch_size = 16 # todo, not in state_dict?
    self.backbone.num_windows = num_windows # todo, not in state_dict?
    self.backbone.encoder = WindowedDinov2WithRegistersBackbone()
    self.backbone.encoder.num_windows = num_windows
    self.backbone.encoder.layernorm = nn.LayerNorm(384)
    self.backbone.encoder.encoder = WindowedDinov2WithRegistersEncoder()
    self.backbone.encoder.embeddings = WindowedDinov2WithRegistersEmbeddings()
    self.backbone.encoder.embeddings.num_windows = num_windows
    self.backbone.encoder.embeddings.projection = nn.Conv2d(in_channels=3, out_channels=384, kernel_size=16, stride=(16, 16))
    self.backbone.encoder.embeddings.cls_token = Tensor.empty((1, 1, 384))
    self.backbone.encoder.embeddings.position_embeddings = Tensor.empty((1, config[name]["size"], 384))
    self.backbone.encoder.encoder.layer = seq(size=13)
    for i in range(12):
      self.backbone.encoder.encoder.layer[i] = WindowedDinov2WithRegistersLayer()
      self.backbone.encoder.encoder.layer[i].num_windows = num_windows
      self.backbone.encoder.encoder.layer[i].norm1 = nn.LayerNorm(384)
      self.backbone.encoder.encoder.layer[i].norm2 = nn.LayerNorm(384)
      self.backbone.encoder.encoder.layer[i].attention = Dinov2WithRegistersSdpaAttention()
      self.backbone.encoder.encoder.layer[i].attention.attention = Dinov2WithRegistersSdpaSelfAttention()
      self.backbone.encoder.encoder.layer[i].attention.attention.query = nn.Linear(384, 384)
      self.backbone.encoder.encoder.layer[i].attention.attention.key = nn.Linear(384, 384)
      self.backbone.encoder.encoder.layer[i].attention.attention.value = nn.Linear(384, 384)
      self.backbone.encoder.encoder.layer[i].attention.dense = nn.Linear(384, 384)
      self.backbone.encoder.encoder.layer[i].lambda1 = Tensor.empty((384))
      self.backbone.encoder.encoder.layer[i].lambda2 = Tensor.empty((384))
      self.backbone.encoder.encoder.layer[i].mlp = Dinov2WithRegistersMLP()
      self.backbone.encoder.encoder.layer[i].mlp.fc1 = nn.Linear(384, 1536)
      self.backbone.encoder.encoder.layer[i].mlp.fc2 = nn.Linear(1536, 384)
    state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/rf-detr/resolve/main/{name}.safetensors'))
    load_state_dict(self, state_dict)

  @TinyJit
  def __call__(self, frame):
    pre = self.preprocess(frame)
    predictions = self.predict(pre)
    out_logits, out_bbox = predictions
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = Tensor.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = Tensor.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    ret = Tensor.cat(boxes.squeeze(0), topk_values.squeeze(0).unsqueeze(1), labels.squeeze(0).unsqueeze(1), dim=1)
    ret = self.scale_boxes(pre.shape[:2], ret, frame.shape)
    return ret

  def predict(self, samples, targets=None):
    _, _, h, w = samples.shape
    mask = Tensor.zeros((1, h, w), dtype=dtypes.bool)
    feature, mask = self.backbone(samples, mask)
    refpoint_embed_weight = self.refpoint_embed[:self.num_queries]
    query_feat_weight = self.query_feat[:self.num_queries]
    hs, ref_unsigmoid = self.transformer(feature, mask, refpoint_embed_weight, query_feat_weight)
    outputs_coord_delta = self.bbox_embed(hs)

    outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
    outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
    outputs_coord = Tensor.cat(outputs_coord_cxcy, outputs_coord_wh, dim=-1)
    outputs_class = self.class_embed(hs)[-1]
    return outputs_class, outputs_coord[-1]

  def preprocess(self, img):
    img = img.cast(dtypes.float32)
    img = img[:, :, ::-1] # BGR to RGB
    img /= 255.0
    means = Tensor([[[0.485, 0.456, 0.406]]])
    stds = Tensor([[[0.229, 0.224, 0.225]]])
    img = resize(img, (self.res, self.res))
    img = (img - means) / stds
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img

  def scale_boxes(self, img1_shape, predictions, img0_shape):
    predictions[:,:4] *= Tensor([img0_shape[1], img0_shape[0], img0_shape[1], img0_shape[0]])
    return predictions

def compute_iou_matrix(boxes):
  x1s = boxes[:, :, 0]
  y1s = boxes[:, :, 1]
  x2s = boxes[:, :, 2]
  y2s = boxes[:, :, 3]
  areas = (x2s - x1s) * (y2s - y1s)
  x1 = Tensor.maximum(x1s[:, :, None], x1s[:, None, :])
  y1 = Tensor.maximum(y1s[:, :, None], y1s[:, None, :])
  x2 = Tensor.minimum(x2s[:, :, None], x2s[:, None, :])
  y2 = Tensor.minimum(y2s[:, :, None], y2s[:, None, :])
  w = Tensor.maximum(Tensor(0), x2 - x1)
  h = Tensor.maximum(Tensor(0), y2 - y1)
  intersection = w * h
  union = areas[:, :, None] + areas[:, None, :] - intersection
  return intersection / union

def box_cxcywh_to_xyxy(x):
  x_c, y_c, w, h = [t.squeeze(-1) for t in x.split(1, dim=-1)]
  w_pos = w.clip(0.0, float("inf"))
  h_pos = h.clip(0.0, float("inf"))
  b = [x_c - 0.5 * w_pos, y_c - 0.5 * h_pos, x_c + 0.5 * w_pos, y_c + 0.5 * h_pos]
  return Tensor.stack(b, dim=-1)

class seq:
  def __init__(self, size=0): self.size = size
  def __setitem__(self, key, value): setattr(self, str(key), value)
  def __getitem__(self, idx):
    try:
      return getattr(self, str(idx))
    except AttributeError:
      raise IndexError(idx)
  def __len__(self): return self.size
  def __call__(self, x):
    for i in range(self.size):
      layer = getattr(self, str(i))
      x = layer(x)
    return x

def resize(img, new_size):
  img = img.permute(2,0,1)
  img = Tensor.interpolate(img, size=(new_size[1], new_size[0]), mode='linear', align_corners=False)
  img = img.permute(1, 2, 0)
  return img
