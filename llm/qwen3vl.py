import unicodedata, re, math, typing, sys, cv2, time
import numpy as np
from tinygrad import Tensor, nn, TinyJit, Variable, dtypes
Tensor.manual_seed(42)
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import partition, fetch
from utils.gguf import gguf_load
from utils.model import Transformer

TEMP = 0.7
TOP_K = 20
TOP_P = 0.8

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

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    ret = Tensor.cat(-x2, x1, dim=-1)
    return ret

class Qwen3VL():
  def __init__(self, size="2B", res=(640, 640)):
    self.res = res
    self.max_context = 2000
    self.vis = Qwen3VLVis(size=size)
    self.lang, kv = Transformer.from_gguf(fetch(f"https://huggingface.co/Qwen/Qwen3-VL-{size}-Instruct-GGUF/resolve/main/Qwen3VL-{size}-Instruct-F16.gguf"), self.max_context) # max context
    self.tok = SimpleTokenizer.from_gguf_kv(kv)
    self.start_pos = 0

  def prewarm(self):
    for _ in range(2):
      prefill(vis=self.vis, lang=self.lang, image=Tensor.rand(*self.res, 3).cast(dtypes.uint8), start_pos=Variable("pos",0,self.max_context).bind(42))
      self.lang(tokens=Tensor([[42]]).clone(), start_pos=Variable("pos",1,self.max_context).bind(42), temperature=Tensor(0.7).clone())
      self.lang.prefill_jit(tokens=Tensor([[42]*self.max_context]).clone()[:, :Variable("len",1,self.max_context).bind(42)], \
      start_pos=Variable("pos",1,self.max_context).bind(42), temperature=Tensor(0.7).clone())

  def generate(self, prompt=None, image=None, reset=False):
    if reset: self.start_pos = 0
    if image is not None:
      prefill_img(vis=self.vis, lang=self.lang, image=image, start_pos=Variable("pos",0,self.max_context).bind(self.start_pos), res=self.res)
      self.start_pos += ((self.res[0] * self.res[1]) // (32*32)) + 8
    if prompt is None: return
    prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    prompt = self.tok.encode(prompt)
    prompt_len = len(prompt)
    prompt = prompt + [0] * (self.max_context - prompt_len)
    tokens = Tensor(prompt).unsqueeze(0)
    token = self.lang.prefill_jit(tokens=tokens[:, :Variable("len",1,self.max_context).bind(prompt_len)], start_pos=Variable("pos",1,self.max_context).bind(self.start_pos), temperature=Tensor(0.7).clone())[0]
    self.start_pos += prompt_len
    toks_out = []
    decoded = ""

    while True:
      ts = time.time()
      if toks_out:
        token = self.lang(tokens=next_token_tensor.clone(), start_pos=Variable("pos",1,self.max_context).bind(self.start_pos), temperature=Tensor(0.7).clone())[0]
        self.start_pos += 1
      next_token = int(token.numpy()[0])
      next_token_tensor = Tensor([[next_token]])
      if next_token == self.tok.eos_id: break
      toks_out.append(next_token)
      new_text = self.tok.decode([next_token])
      decoded += new_text
      tok_s = f" ({1/(time.time()-ts):.1f} tok/s)"
      print(new_text + tok_s, end="", flush=True)
      print("\b" * len(tok_s), end="", flush=True)
    print("\n")
    return self.tok.decode(toks_out)
  
def prefill_img(vis, lang, image, start_pos, res=(640, 640)):
  if image.shape[:2] != res:
    target_h, target_w = res[:2]
    s = min(target_w / image.shape[1], target_h / image.shape[0])
    r = cv2.resize(image, (int(image.shape[1] * s), int(image.shape[0] * s)))
    image = cv2.copyMakeBorder(r, (target_h - r.shape[0]) // 2, target_h - r.shape[0] - (target_h - r.shape[0]) // 2, (target_w - r.shape[1]) // 2, target_w - r.shape[1] - (target_w - r.shape[1]) // 2, cv2.BORDER_CONSTANT, value=0)
  prefill(vis=vis, lang=lang, image=Tensor(image), start_pos=start_pos)

@TinyJit
def prefill(vis, lang, image, start_pos):
  image = image.permute(2, 0, 1)
  height, width = image.shape[-2:]
  image = image.unsqueeze(0).float()
  image = image.interpolate(size=(height, width))
  resized_height, resized_width = image.shape[-2:]
  patches = (image - 127.5) / 127.5
  batch_size, channel = 1, 3
  # https://github.com/huggingface/transformers/blob/4ae05b0fba41860adaaeb708774fc1f48c92c049/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L195
  grid_h, grid_w = resized_height // vis.patch_size, resized_width // vis.patch_size
  patches = patches.reshape(
      batch_size,
      channel,
      grid_h // vis.merge_size,
      vis.merge_size,
      vis.patch_size,
      grid_w // vis.merge_size,
      vis.merge_size,
      vis.patch_size,
  )
  patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
  pixel_values = (
      patches.unsqueeze(6)
      .expand(-1, -1, -1, -1, -1, -1, vis.temporal_patch_size, -1, -1)
      .reshape(
          batch_size,
          grid_h * grid_w,
          channel * vis.temporal_patch_size * vis.patch_size * vis.patch_size, # 1536
      )
  )[0]
  pixel_values = pixel_values.cast(dtypes.bfloat16)

  # f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n<|im_end|>\n" fill size of img with image token
  # <|im_end|>\n<|im_start|>assistant\n
  num_image_tokens = int((grid_h*grid_w) / 4)
  input_ids = Tensor.cat(Tensor([151644, 872, 198, 151652]), Tensor.zeros(num_image_tokens), Tensor([151653, 198, 151645, 198])).unsqueeze(0).cast(dtypes.int)

  image_embeds, hidden_states, deepstack_feature_lists = vis(pixel_values, [grid_h, grid_w])
  hidden_states = lang.token_embd(input_ids).cast(dtypes.float)
  # 4 to -4 because of <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n<|im_end|>\n tokens before and after image_pad
  hidden_states[:, 4:-4, :] = image_embeds.unsqueeze(0)
  
  # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modular_qwen3_vl.py#L626
  # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modular_qwen3_vl.py#L543
  for i in range(len(lang.blk)):
    hidden_states = lang.blk[i](hidden_states, start_pos=start_pos)
    # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L692
    if i in vis.v.deepstack_idx:
      # 4 to -4 because of <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n<|im_end|>\n tokens before and after image_pad
      hidden_states[:, 4:-4, :] += deepstack_feature_lists[vis.v.deepstack_idx.index(i)]
  hidden_states.realize()

class Qwen3VLVis():
  def __init__(self, size="2B"):
    kv, state_dict = gguf_load(fetch(f"https://huggingface.co/Qwen/Qwen3-VL-{size}-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-{size}-Instruct-F16.gguf"))
    self.merge_size = kv["clip.vision.spatial_merge_size"]
    self.patch_size = kv["clip.vision.patch_size"]
    self.temporal_patch_size = 2
    self.v = Qwen3VisBlocks(kv=kv, weights=state_dict)
    self.mm = [nn.Linear(*state_dict["mm.0.weight"].shape[::-1], bias=True), None, nn.Linear(*state_dict["mm.2.weight"].shape[::-1], bias=True)]
    state_dict["v.patch_embd.weight1"] = state_dict["v.patch_embd.weight.1"]
    load_state_dict(self, state_dict)
    self.inv_freq = 1.0 / (10000.0 ** (Tensor.arange(0, 32, 2, dtype=dtypes.float) / 32))

  # https://github.com/huggingface/transformers/blob/15bb519bd4277f4ab5309154aedf3c231e8b4ca8/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L679
  def __call__(self, pixel_values, image_grid_size):        
    grid_hs = image_grid_size[0]
    grid_ws = image_grid_size[1]

    h_idxs = Tensor.linspace(0, self.v.num_grid_per_side - 1, grid_hs)
    w_idxs = Tensor.linspace(0, self.v.num_grid_per_side - 1, grid_ws)

    h_idxs_floor = h_idxs.cast(dtypes.int32)
    w_idxs_floor = w_idxs.cast(dtypes.int32)
    h_idxs_ceil = (h_idxs_floor.int() + 1).clip(self.v.num_grid_per_side - 1)
    w_idxs_ceil = (w_idxs_floor.int() + 1).clip(self.v.num_grid_per_side - 1)
    dh = h_idxs - h_idxs_floor
    dw = w_idxs - w_idxs_floor

    base_h = h_idxs_floor * self.v.num_grid_per_side
    base_h_ceil = h_idxs_ceil * self.v.num_grid_per_side

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

    pos_embeds = self.v.position_embd(idx_tensor)
    pos_embeds *= weight_tensor[:, :, None]
    pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    merge_size = 2
    pos_embeds = (pos_embeds.view(1, grid_hs // merge_size, merge_size, grid_ws // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
    
    hpos_ids = Tensor.arange(grid_hs).unsqueeze(1).expand(-1, grid_ws)
    hpos_ids = hpos_ids.reshape(grid_hs // merge_size, merge_size, grid_ws // merge_size, merge_size).transpose(1, 2).flatten()

    wpos_ids = Tensor.arange(grid_ws).unsqueeze(0).expand(grid_hs, -1)
    wpos_ids = wpos_ids.reshape(grid_hs // merge_size, merge_size, grid_ws // merge_size, merge_size).transpose(1, 2).flatten()

    pos_ids = Tensor.stack(hpos_ids, wpos_ids, dim=-1).repeat(1, 1)

    rotary_pos_emb = (pos_ids.unsqueeze(-1) * self.inv_freq).flatten(1)

    hidden_states = pixel_values.view(-1, 3, 2, 16, 16)
    hidden_states = hidden_states.flatten(1, 2)
    w = Tensor.stack(self.v.patch_embd.weight, self.v.patch_embd.weight1, dim=2)
    out_C, in_C, kD, kH, kW = w.shape
    w2d = w.reshape(out_C, in_C * kD, kH, kW)

    hidden_states = hidden_states.conv2d(
        weight=w2d,
        bias=self.v.patch_embd.bias,
        stride=(16, 16),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1
    )

    hidden_states = hidden_states.view(-1, 1024)
    hidden_states = hidden_states + pos_embeds

    sqlen, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(sqlen, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(sqlen, -1)
    emb = Tensor.cat(rotary_pos_emb, rotary_pos_emb, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    
    deepstack_feature_lists = []
    for i in range(len(self.v.blk)):
      hidden_states = self.v.blk[i](hidden_states, cos, sin)
      if i in self.v.deepstack_idx: deepstack_feature_lists.append(self.v.deepstack[i](hidden_states))

    image_embeds = self.v.post_ln(hidden_states)
    image_embeds = image_embeds.view(-1, 4096)
    image_embeds = self.mm[0](image_embeds)
    image_embeds = Tensor.gelu(image_embeds)
    image_embeds = self.mm[2](image_embeds)
    return image_embeds, hidden_states, deepstack_feature_lists

class Qwen3PatchEmbed():
  def __init__(self, kv=None):
    self.weight = Tensor.zeros(kv["clip.vision.embedding_length"], 3, 16, 16)
    self.weight1 = Tensor.zeros(kv["clip.vision.embedding_length"], 3, 16, 16)
    self.bias = Tensor.zeros(kv["clip.vision.embedding_length"])
    
class Qwen3VisBlocks():
  def __init__(self, kv=None, weights=None):
    self.blk = []
    for _ in range(kv["clip.vision.block_count"]): self.blk.append(Qwen3VisBlock(kv, weights=weights))
    self.patch_embd = Qwen3PatchEmbed(kv=kv)
    self.num_grid_per_side = 48 # todo unhardcode
    self.deepstack_layers = kv["clip.vision.is_deepstack_layers"]
    self.deepstack_idx = [i for i, val in enumerate(self.deepstack_layers) if val]
    self.deepstack = []
    for i in range(len(self.deepstack_layers)):
      if i in self.deepstack_idx:
        self.deepstack.append(DeepstackLayer(i, weights))
      else:
        self.deepstack.append(None)
    self.position_embd = nn.Embedding(*weights["v.position_embd.weight"].shape)
    self.post_ln = nn.LayerNorm(weights["v.post_ln.weight"].shape[0], eps=1e-6, elementwise_affine=True)

class DeepstackLayer:
  def __init__(self, index, weights):
    self.fc1 = nn.Linear(*weights[f"v.deepstack.{index}.fc1.weight"].shape[::-1])
    self.fc2 = nn.Linear(*weights[f"v.deepstack.{index}.fc2.weight"].shape[::-1])
    self.norm = nn.LayerNorm(weights[f"v.deepstack.{index}.norm.weight"].shape[0], eps=1e-6, elementwise_affine=True)
    self.hidden_size = weights[f"v.deepstack.{index}.norm.weight"].shape[0]

  #https://github.com/huggingface/transformers/blob/027d1a97025295a1346c2eb5c361259e69eedfe7/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L112
  def __call__(self, hidden_states):
      deepstack_feature = (hidden_states.view(-1, self.hidden_size)).view(-1, self.hidden_size)
      return self.fc2(Tensor.gelu(self.fc1(deepstack_feature)))

class Qwen3VisBlock():
  def __init__(self, kv=None, weights=None):
    self.ffn_up = nn.Linear(kv["clip.vision.embedding_length"], kv["clip.vision.feed_forward_length"])
    self.ffn_down = nn.Linear(kv["clip.vision.feed_forward_length"], kv["clip.vision.embedding_length"])
    self.ln1 = nn.LayerNorm(kv["clip.vision.embedding_length"], eps=1e-6, elementwise_affine=True)
    self.ln2 = nn.LayerNorm(kv["clip.vision.embedding_length"], eps=1e-6, elementwise_affine=True)
    self.attn_out = nn.Linear(kv["clip.vision.embedding_length"], kv["clip.vision.embedding_length"])
    self.attn_qkv = nn.Linear(*weights["v.blk.0.attn_qkv.weight"].shape[::-1])
  
  def __call__(self, hidden_states, cos, sin):
    hidden_states_input = self.ln1(hidden_states)
    seq_length = hidden_states_input.shape[0]
    qkv = self.attn_qkv(hidden_states_input)
    qkv = qkv.reshape(seq_length, 3, 16, -1).permute(1, 0, 2, 3)
    query, key, value = qkv.chunk(3, dim=0)
    query = query.squeeze(0)
    key   = key.squeeze(0)
    value = value.squeeze(0)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)

    query = query.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)

    attn_weight = query @ key.transpose(0, 1).unsqueeze(0).transpose(-2, -1) * 0.125
    attn_weight = Tensor.softmax(attn_weight)
    attn_output = attn_weight @ value
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.attn_out(attn_output)
    hidden_states += attn_output
    norm = self.ln2(hidden_states)
    norm = self.ffn_up(norm).gelu()
    norm = self.ffn_down(norm)
    return hidden_states + norm