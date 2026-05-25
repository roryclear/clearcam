import unicodedata, re, math, typing, sys, cv2, time
import numpy as np
from tinygrad import Tensor, nn, TinyJit, Variable, dtypes
Tensor.manual_seed(42)
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import partition, fetch
from gguf import gguf_load
from model import Transformer


class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    ret = tokens + self._encode_sentence(text[pos:])
    return ret

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'kimi-k2': return self.encode("<|im_" + role + "|>" + role + "<|im_middle|>")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'glm4': return self.encode("<|" + role + "|>")
    if self.preset == 'tekken':
      if role == 'user': return self.encode("[INST]")
      if role == 'assistant': return []
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.preset}'")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'kimi-k2': return [self.eos_id]
    if self.preset == 'qwen2': return [self.eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []
    if self.preset == 'tekken': return self.encode("[/INST]")
    return [self.eos_id]
  def prefix(self) -> list[int]:
    return ([] if self.bos_id is None else [self.bos_id]) + (self.encode("<sop>") if self.preset == 'glm4' else [])
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)

temp = 0.7
top_k = 20
top_p = 0.8

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    ret = Tensor.cat(-x2, x1, dim=-1)
    return ret


def sample(logits, temp: float, k: int, p: float, af: float, ap: float):
  assert logits.ndim == 1, "only works on 1d tensors"
  assert 0 <= p <= 1, "p must be between 0 and 1"
  assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

  # if temperature is very low just use argmax
  if temp < 1e-6: return logits.argmax()

  # alpha sampling
  if af or ap:
    if not hasattr(sample, "alpha_counter"):
      setattr(sample, "alpha_counter", Tensor.zeros_like(logits, dtype=dtypes.int32).contiguous())
    logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

  # replace NaNs with -inf
  logits = (logits != logits).where(-float("inf"), logits)

  # softmax
  t = (logits / temp).softmax()

  counter, counter2 = Tensor.arange(t.numel(), device=logits.device).contiguous(), Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous()
  # top k
  if k:
    output, output_indices = Tensor.zeros(k, device=logits.device).contiguous(), Tensor.zeros(k, device=logits.device, dtype=dtypes.int32).contiguous()
    for i in range(k):
      t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(dtypes.default_int)
      output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
      output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
      t = (counter == t_argmax).where(0, t)

    # approximate top p
    # because we are already limited to top k elements we can do top p "without sorting"
    output_cumsum = output[::-1].cumsum()[::-1] + t.sum()
    output = (output_cumsum >= (1 - p)) * output
    output_indices = (output_cumsum >= (1 - p)) * output_indices

    # sample
    output_idx = output.multinomial()
    output_token = output_indices[output_idx]
  else:
    output_token = t.multinomial()

  # increase alpha counter
  if af or ap:
    sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

  return output_token

def smart_resize(height, width, factor, min_pixels, max_pixels):
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

@TinyJit
def preprocess_img(image):
  image = image.permute(2, 0, 1)
  patch_size = 16
  merge_size = 2
  temporal_patch_size = 2
  height, width = image.shape[-2:]
  resized_height, resized_width = smart_resize(
      height,
      width,
      factor=patch_size * merge_size,
      min_pixels=65536,
      max_pixels=16777216,
  )
  image = image.unsqueeze(0).float()
  image = image.interpolate(size=(resized_height, resized_width))
  resized_height, resized_width = image.shape[-2:]
  patches = (image - 127.5) / 127.5
  batch_size, channel = patches.shape[:2]
  grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
  patches = patches.reshape(
      batch_size,
      channel,
      grid_h // merge_size,
      merge_size,
      patch_size,
      grid_w // merge_size,
      merge_size,
      patch_size,
  )
  patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
  pixel_values = (
      patches.unsqueeze(6)
      .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
      .reshape(
          batch_size,
          grid_h * grid_w,
          channel * temporal_patch_size * patch_size * patch_size,
      )
  )[0]
  return pixel_values, Tensor([1, grid_h, grid_w])

class Qwen3VL():
  def __init__(self, size="2B"):
    self.vis = qwen3vl_vis(size=size)
    self.lang, kv = Transformer.from_gguf(fetch(f"https://huggingface.co/Qwen/Qwen3-VL-{size}-Instruct-GGUF/resolve/main/Qwen3VL-{size}-Instruct-F16.gguf"), 2000) # max context
    self.tok = SimpleTokenizer.from_gguf_kv(kv)
    self.prewarmed = False

  def preprocess(self, image, prompt):
    pixel_values, image_grid_thw = preprocess_img(image=Tensor(image))
    image_grid_thw = image_grid_thw.numpy().tolist()
    text_inputs = self.tok.encode(prompt)
    image_token_id = 151655
    image_token_positions = [i for i, tid in enumerate(text_inputs) if tid == image_token_id]
    num_image_tokens = ((image_grid_thw[0]*image_grid_thw[1]*image_grid_thw[2]) / 4)
    for pos in reversed(image_token_positions): text_inputs[pos:pos+1] = [image_token_id] * int(num_image_tokens)
    seq_len=len(text_inputs)
    input_ids = Tensor([text_inputs])
    return pixel_values, input_ids, seq_len, image_grid_thw

  def prewarm(self, res, prompt):
    pixel_values, input_ids, seq_len, image_grid_thw = self.preprocess(image=np.random.randint(0, 256, size=res, dtype=np.uint8), prompt=prompt)
    for _ in range(3): preprocess_img(image=Tensor.rand(res).cast(dtypes.uint8))
    for _ in range(3): self.prefill(pixel_values=pixel_values, input_ids=input_ids, image_grid_thw=image_grid_thw)
    for _ in range(3):  self.fwd(token=Tensor([[42]]), seq_len=Variable("pos",1,2000).bind(seq_len))
    self.prewarmed = True

  def forward(self, prompt, image):
    pixel_values, input_ids, seq_len, image_grid_thw = self.preprocess(image=image, prompt=prompt)
    if not self.prewarmed: self.prewarm(image.shape, prompt)

    toks_out = []
    prefill_done = False
    ts = time.time()
    token = self.prefill(pixel_values=pixel_values, input_ids=input_ids, image_grid_thw=image_grid_thw)
    while True:
        if prefill_done:
          ts = time.time()
          token = self.fwd(token=next_token_tensor, seq_len=Variable("pos",1,2000).bind(seq_len))
          seq_len+=1
        else:
          prefill_done = True
        next_token = int(token.numpy()[0])
        next_token_tensor = Tensor([[next_token]])  # shape (1,1)

        if next_token == 151645 or seq_len == 406: break
        toks_out.append(next_token)
        print(self.tok.decode(toks_out))
        print(f"TOK/S = {1 / (time.time() - ts):.2f}")

    return self.tok.decode(toks_out)


  @TinyJit
  def fwd(self, token, seq_len):
    hidden_states = self.lang.token_embd(token)
    for i in range(len(self.lang.blk)): hidden_states = self.lang.blk[i](hidden_states, start_pos=seq_len)  
    hidden_states = self.lang.output_norm(hidden_states)
    outputs = hidden_states[:, -1:, :] @ self.lang.token_embd.weight.T
    next_token_logits = outputs[:, -1, :]
    scores = next_token_logits / temp
    token = sample(scores[0], temp=temp, k=top_k, p=top_p, af=None, ap=None)
    return token

  @TinyJit
  def prefill(self, pixel_values, input_ids, image_grid_thw):
      hidden_states = pixel_values.view(-1, 3, 2, 16, 16)
      hidden_states = hidden_states.cast(dtype=dtypes.bfloat16)

      B, C, D, H, W = hidden_states.shape
      x = hidden_states.reshape(B, C * D, H, W)
      w = Tensor.stack(self.vis.v.patch_embd.weight, self.vis.v.patch_embd.weight2, dim=2)
      out_C, in_C, kD, kH, kW = w.shape
      w2d = w.reshape(out_C, in_C * kD, kH, kW)

      hidden_states = x.conv2d(
          weight=w2d,
          bias=self.vis.v.patch_embd.bias,
          stride=(16, 16),
          padding=(0, 0),
          dilation=(1, 1),
          groups=1
      )

      hidden_states = hidden_states.view(-1, 1024)
          
      grid_ts = image_grid_thw[0]
      grid_hs = image_grid_thw[1]
      grid_ws = image_grid_thw[2]

      h_idxs = Tensor.linspace(0, self.vis.v.num_grid_per_side - 1, grid_hs)
      w_idxs = Tensor.linspace(0, self.vis.v.num_grid_per_side - 1, grid_ws)

      h_idxs_floor = h_idxs.cast(dtypes.int32)
      w_idxs_floor = w_idxs.cast(dtypes.int32)
      h_idxs_ceil = (h_idxs_floor.int() + 1).clip(self.vis.v.num_grid_per_side - 1)
      w_idxs_ceil = (w_idxs_floor.int() + 1).clip(self.vis.v.num_grid_per_side - 1)
      dh = h_idxs - h_idxs_floor
      dw = w_idxs - w_idxs_floor

      base_h = h_idxs_floor * self.vis.v.num_grid_per_side
      base_h_ceil = h_idxs_ceil * self.vis.v.num_grid_per_side


      idx_tensor = Tensor.stack(
          (base_h[None].T + w_idxs_floor[None]).flatten(),
          (base_h[None].T + w_idxs_ceil[None]).flatten(),
          (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
          (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
      ).cast(dtypes.int32)

      weight_tensor = Tensor.stack(
          ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
          ((1 - dh)[None].T * dw[None]).flatten(),
          (dh[None].T * (1 - dw)[None]).flatten(),
          (dh[None].T * dw[None]).flatten(),
      ).cast(dtypes.bfloat16)

      pos_embeds = self.vis.v.position_embd(idx_tensor)
      pos_embeds *= weight_tensor[:, :, None]
      patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

      patch_pos_embeds = patch_pos_embeds[:grid_hs * grid_ws]

      merge_size = 2
      pos_embeds = patch_pos_embeds.repeat(grid_ts, 1)
      pos_embeds = (pos_embeds.view(grid_ts, grid_hs // merge_size, merge_size, grid_ws // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
      hidden_states = hidden_states + pos_embeds
      
      hpos_ids = Tensor.arange(image_grid_thw[1]).unsqueeze(1).expand(-1, image_grid_thw[2])
      hpos_ids = hpos_ids.reshape(image_grid_thw[1] // merge_size, merge_size, image_grid_thw[2] // merge_size, merge_size).transpose(1, 2).flatten()

      wpos_ids = Tensor.arange(image_grid_thw[2]).unsqueeze(0).expand(image_grid_thw[1], -1)
      wpos_ids = wpos_ids.reshape(image_grid_thw[1] // merge_size, merge_size, image_grid_thw[2] // merge_size, merge_size).transpose(1, 2).flatten()

      pos_ids = Tensor.stack(hpos_ids, wpos_ids, dim=-1).repeat(image_grid_thw[0], 1)

      rotary_pos_emb = (pos_ids.unsqueeze(-1) * self.vis.inv_freq).flatten(1)

      sqlen, _ = hidden_states.size()
      hidden_states = hidden_states.reshape(sqlen, -1)
      rotary_pos_emb = rotary_pos_emb.reshape(sqlen, -1)
      emb = Tensor.cat(rotary_pos_emb, rotary_pos_emb, dim=-1)
      cos, sin = emb.cos(), emb.sin()
      cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
      
      for i in range(len(self.vis.v.blk)):
        hidden_states_input = self.vis.v.blk[i].ln1(hidden_states)
        seq_length = hidden_states_input.shape[0]
        qkv = self.vis.v.blk[i].attn_qkv(hidden_states_input)
        qkv_reshaped = qkv.reshape(seq_length, 3, 16, -1)
        qkv_permuted = qkv_reshaped.permute(1, 0, 2, 3)
        query, key, value = qkv_permuted.chunk(3, dim=0)
        query = query.squeeze(0)
        key   = key.squeeze(0)
        value = value.squeeze(0)
        query, key = query.cast(dtypes.float32), key.cast(dtypes.float32)
        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)

        attn_weight = query @ key.transpose(-2, -1) * 0.125
        attn_weight = Tensor.softmax(attn_weight)
        attn_output = attn_weight @ value
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = attn_output.cast(dtypes.bfloat16)
        attn_output = self.vis.v.blk[i].attn_out(attn_output)
        attn_output = attn_output.cast(dtypes.bfloat16) # todo
        hidden_states += attn_output
        norm = self.vis.v.blk[i].ln2(hidden_states)
        x = self.vis.v.blk[i].ffn_up(norm)
        x = Tensor.gelu(x)
        norm = self.vis.v.blk[i].ffn_down(x)
        hidden_states = hidden_states + norm
      
      image_embeds = self.vis.v.post_ln(hidden_states)
      image_embeds = image_embeds.view(-1, 4096)
      image_embeds = self.vis.mm[0](image_embeds)
      image_embeds = Tensor.gelu(image_embeds)
      image_embeds = self.vis.mm[2](image_embeds)
      
      image_mask = input_ids == 151655

      inputs_embeds = self.lang.token_embd(input_ids)
      image_mask = image_mask.unsqueeze(-1).expand(inputs_embeds.shape)
      image_embeds = image_embeds.view(-1)
      flat_mask = image_mask.view(-1)
      idx = (flat_mask.cumsum(0) - 1).clamp(0)
      expanded = image_embeds[idx] * flat_mask
      flat_inputs = inputs_embeds.view(-1)
      flat_inputs = flat_inputs * (~flat_mask) + expanded
      hidden_states = flat_inputs.view(inputs_embeds.shape)
      
      for i in range(len(self.lang.blk)): # todo same block above
        self.lang.blk[i]._init_state(Tensor.zeros(1, 1))
        hidden_states = self.lang.blk[i](hidden_states, start_pos=0)

      hidden_states = self.lang.output_norm(hidden_states)
      outputs = hidden_states[:, -1:, :] @ self.lang.token_embd.weight.T

      next_token_logits = outputs[:, -1, :]
      scores = next_token_logits / temp
      token = sample(scores[0], temp=temp, k=top_k, p=top_p, af=None, ap=None)
      return token

class qwen3vl_vis():
  def __init__(self, size="2B"):
    _, state_dict_visual = gguf_load(fetch(f"https://huggingface.co/Qwen/Qwen3-VL-{size}-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-{size}-Instruct-F16.gguf"))
    self.v = qwen3_vis_v()
    sizes = {"2B": 2048, "4B":2560}
    self.mm = [nn.Linear(4096, 4096, bias=True), None, nn.Linear(4096, sizes[size], bias=True)]
    state_dict_visual["v.patch_embd.weight2"] = state_dict_visual["v.patch_embd.weight.1"] # todo
    load_state_dict(self, state_dict_visual)
    self.inv_freq = 1.0 / (10000.0 ** (Tensor.arange(0, 32, 2, dtype=dtypes.float) / 32))

class qwen3_patch_embd():
  def __init__(self):
    self.weight = Tensor.zeros(1024, 3, 16, 16)
    self.weight2 = Tensor.zeros(1024, 3, 16, 16)
    self.bias = Tensor.zeros(1024)
    
class qwen3_vis_v():
  def __init__(self):
    self.blk = []
    for _ in range(24): self.blk.append(qwen3_vis_block())
    self.patch_embd = qwen3_patch_embd()
    self.num_grid_per_side = 48
    self.position_embd = nn.Embedding(2304, 1024)
    self.post_ln = nn.LayerNorm(1024, eps=1e-6, elementwise_affine=True)

class qwen3_vis_block():
  def __init__(self):
    self.ffn_up = nn.Linear(1024, 4096)
    self.ffn_down = nn.Linear(4096, 1024)
    self.ln1 = nn.LayerNorm(1024, eps=1e-6, elementwise_affine=True)
    self.ln2 = nn.LayerNorm(1024, eps=1e-6, elementwise_affine=True)
    self.attn_out = nn.Linear(1024, 1024)
    self.attn_qkv = nn.Linear(1024, 3072)