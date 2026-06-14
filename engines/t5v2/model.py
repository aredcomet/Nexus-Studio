import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Dict, Any, List

class DecoderCache:
    def __init__(self, num_layers: int):
        self.self_keys = [None] * num_layers
        self.self_values = [None] * num_layers
        self.cross_keys = [None] * num_layers
        self.cross_values = [None] * num_layers
        self.offset = 0

# Activation function for T5Gemma2/Gemma3
def gelu_pytorch_tanh(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3.0))))

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Gemma RMSNorm is scaled by (1 + weight) where weight is initialized to 0
        self.weight = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        output = x * mx.rsqrt(variance + self.eps)
        output = output * (1.0 + self.weight.astype(mx.float32))
        return output.astype(orig_dtype)

class GemmaMLP(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Act is gelu_pytorch_tanh
        return self.down_proj(gelu_pytorch_tanh(self.gate_proj(x)) * self.up_proj(x))

class T5Gemma2RotaryEmbedding:
    def __init__(self, config: Dict[str, Any], layer_type: str):
        self.head_dim = config["head_dim"]
        rope_params = config["rope_parameters"][layer_type]
        self.base = rope_params["rope_theta"]
        self.factor = rope_params.get("factor", 1.0)
        
        # Compute inverse frequencies
        dim = self.head_dim
        arange = mx.arange(0, dim, 2, dtype=mx.float32)
        self.inv_freq = 1.0 / (self.base ** (arange / dim))
        self.inv_freq = self.inv_freq / self.factor

    def __call__(self, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        # position_ids: [B, S]
        # inv_freq: [head_dim // 2]
        # freqs: [B, S, head_dim // 2]
        freqs = position_ids[:, :, None] * self.inv_freq[None, None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos, sin

def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    # q: [B, num_heads, S, head_dim]
    # k: [B, num_kv_heads, S, head_dim]
    # cos, sin: [B, S, head_dim]
    cos = mx.expand_dims(cos, axis=1) # [B, 1, S, head_dim]
    sin = mx.expand_dims(sin, axis=1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    if n_rep == 1:
        return x
    B, n_kv_heads, S, D = x.shape
    x = mx.broadcast_to(x[:, :, None, :, :], [B, n_kv_heads, n_rep, S, D])
    return x.reshape(B, n_kv_heads * n_rep, S, D)

def clip_residual(x: mx.array, y: mx.array) -> mx.array:
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(mx.float16)

class T5Gemma2SelfAttention(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.layer_type = config["layer_types"][layer_idx]
        self.layer_idx = layer_idx
        self.head_dim = config["head_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = config["query_pre_attn_scalar"] ** -0.5
        
        self.q_proj = nn.Linear(config["hidden_size"], self.num_attention_heads * self.head_dim, bias=config["attention_bias"])
        self.k_proj = nn.Linear(config["hidden_size"], self.num_key_value_heads * self.head_dim, bias=config["attention_bias"])
        self.v_proj = nn.Linear(config["hidden_size"], self.num_key_value_heads * self.head_dim, bias=config["attention_bias"])
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config["hidden_size"], bias=config["attention_bias"])
        
        self.q_norm = GemmaRMSNorm(dim=self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = GemmaRMSNorm(dim=self.head_dim, eps=config["rms_norm_eps"])

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, S, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states).reshape(B, S, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = self.k_proj(hidden_states).reshape(B, S, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = self.v_proj(hidden_states).reshape(B, S, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        scores = (query_states @ key_states.transpose(0, 1, 3, 2)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(hidden_states.dtype)
        attn_output = attn_weights @ value_states
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(attn_output)

class T5Gemma2MergedAttention(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.layer_type = config["layer_types"][layer_idx]
        self.layer_idx = layer_idx
        self.head_dim = config["head_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = config["query_pre_attn_scalar"] ** -0.5
        
        self.q_proj = nn.Linear(config["hidden_size"], self.num_attention_heads * self.head_dim, bias=config["attention_bias"])
        self.k_proj = nn.Linear(config["hidden_size"], self.num_key_value_heads * self.head_dim, bias=config["attention_bias"])
        self.v_proj = nn.Linear(config["hidden_size"], self.num_key_value_heads * self.head_dim, bias=config["attention_bias"])
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config["hidden_size"], bias=config["attention_bias"])
        
        self.q_norm = GemmaRMSNorm(dim=self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = GemmaRMSNorm(dim=self.head_dim, eps=config["rms_norm_eps"])

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        merged_attention_mask: Optional[mx.array],
        encoder_hidden_states: mx.array,
        past_key_values: Optional[Any] = None,
        is_sliding: bool = False,
        sliding_window: int = 512,
    ) -> mx.array:
        B, S, _ = hidden_states.shape
        
        # Self-attention projections
        query_states = self.q_proj(hidden_states).reshape(B, S, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = self.k_proj(hidden_states).reshape(B, S, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = self.v_proj(hidden_states).reshape(B, S, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Update self-attention cache
        if past_key_values is not None:
            prev_keys = past_key_values.self_keys[self.layer_idx]
            prev_values = past_key_values.self_values[self.layer_idx]
            
            if prev_keys is None:
                keys_out = key_states
                values_out = value_states
            else:
                keys_out = mx.concatenate([prev_keys, key_states], axis=2)
                values_out = mx.concatenate([prev_values, value_states], axis=2)
                
            if is_sliding and keys_out.shape[2] > sliding_window:
                keys_out = keys_out[:, :, -sliding_window:, :]
                values_out = values_out[:, :, -sliding_window:, :]
                
            past_key_values.self_keys[self.layer_idx] = keys_out
            past_key_values.self_values[self.layer_idx] = values_out
            key_states, value_states = keys_out, values_out
            
        # Get cross-attention projections (computed once and cached)
        if past_key_values is not None and past_key_values.cross_keys[self.layer_idx] is not None:
            cross_key_states = past_key_values.cross_keys[self.layer_idx]
            cross_value_states = past_key_values.cross_values[self.layer_idx]
        else:
            B_enc, S_enc, _ = encoder_hidden_states.shape
            cross_key_states = self.k_proj(encoder_hidden_states).reshape(B_enc, S_enc, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
            cross_value_states = self.v_proj(encoder_hidden_states).reshape(B_enc, S_enc, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
            cross_key_states = self.k_norm(cross_key_states)
            
            if past_key_values is not None:
                past_key_values.cross_keys[self.layer_idx] = cross_key_states
                past_key_values.cross_values[self.layer_idx] = cross_value_states
                
        # Merge self and cross key/values
        # key_states: [B, num_kv_heads, S_dec_all, head_dim]
        # cross_key_states: [B, num_kv_heads, S_enc, head_dim]
        merged_keys = mx.concatenate([key_states, cross_key_states], axis=2)
        merged_values = mx.concatenate([value_states, cross_value_states], axis=2)
        
        merged_keys = repeat_kv(merged_keys, self.num_key_value_groups)
        merged_values = repeat_kv(merged_values, self.num_key_value_groups)
        
        scores = (query_states @ merged_keys.transpose(0, 1, 3, 2)) * self.scaling
        if merged_attention_mask is not None:
            scores = scores + merged_attention_mask
            
        attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(hidden_states.dtype)
        attn_output = attn_weights @ merged_values
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(attn_output)

class T5Gemma2EncoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.self_attn = T5Gemma2SelfAttention(config, layer_idx)
        self.pre_self_attn_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_self_attn_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        
        self.mlp = GemmaMLP(config)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_feedforward_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = clip_residual(residual, hidden_states)
        
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = clip_residual(residual, hidden_states)
        return hidden_states

class T5Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.layer_type = config["layer_types"][layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.get("sliding_window", 512)
        
        self.self_attn = T5Gemma2MergedAttention(config, layer_idx)
        self.pre_self_attn_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_self_attn_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        
        self.mlp = GemmaMLP(config)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_feedforward_layernorm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        merged_attention_mask: Optional[mx.array],
        encoder_hidden_states: mx.array,
        past_key_values: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            merged_attention_mask=merged_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            is_sliding=self.is_sliding,
            sliding_window=self.sliding_window,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = clip_residual(residual, hidden_states)
        
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = clip_residual(residual, hidden_states)
        return hidden_states

class T5Gemma2TextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
        eoi_token_index: int = 256000,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale
        self.eoi_token_index = eoi_token_index
        # eoi embedding is an extra learned parameter
        self.eoi_embedding = mx.zeros((embedding_dim,))

    def __call__(self, input_ids: mx.array) -> mx.array:
        input_embeddings = super().__call__(input_ids) * self.embed_scale
        
        # Replace EOI token embedding
        mask = (input_ids == self.eoi_token_index)
        if mx.any(mask):
            # We can use mx.where or scatter/assignment
            # In MLX, mx.where is easy:
            # self.eoi_embedding[None, None, :] broadcasted
            input_embeddings = mx.where(mask[:, :, None], self.eoi_embedding[None, None, :], input_embeddings)
        return input_embeddings

class T5Gemma2TextEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any], eoi_token_index: int = 256000):
        super().__init__()
        self.config = config
        self.padding_idx = config.get("pad_token_id", 0)
        
        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            config["vocab_size"],
            config["hidden_size"],
            self.padding_idx,
            embed_scale=config["hidden_size"] ** 0.5,
            eoi_token_index=eoi_token_index,
        )
        self.norm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.layers = [T5Gemma2EncoderLayer(config, i) for i in range(config["num_hidden_layers"])]
        
        # Initialize rotary embeddings
        self.rotary_embs = {}
        for layer_type in set(config["layer_types"]):
            self.rotary_embs[layer_type] = T5Gemma2RotaryEmbedding(config, layer_type)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        B, S, _ = inputs_embeds.shape
        position_ids = mx.arange(S, dtype=mx.float32)[None, :]
        
        # Compute masks
        # attention_mask has shape [B, S] where 1 is keep, 0 is pad
        if attention_mask is None:
            attention_mask = mx.ones((B, S), dtype=mx.float32)
            
        # Global position embeddings for each layer type
        position_embeddings = {}
        for layer_type, rotary in self.rotary_embs.items():
            position_embeddings[layer_type] = rotary(position_ids)
            
        hidden_states = inputs_embeds
        
        for i, layer in enumerate(self.layers):
            layer_type = self.config["layer_types"][i]
            is_sliding = (layer_type == "sliding_attention")
            sliding_window = self.config.get("sliding_window", 512)
            
            # Encoder mask for this layer
            # enc_mask shape: [B, 1, 1, S] or [B, 1, S, S] if sliding
            enc_mask = (1.0 - attention_mask[:, None, None, :]) * -1e9
            if is_sliding:
                q_idx = mx.arange(S)[:, None]
                kv_idx = mx.arange(S)[None, :]
                dist = q_idx - kv_idx
                left_window_size = (sliding_window + 1) // 2
                right_window_size = sliding_window // 2 + 1
                valid = ((dist >= 0) & (dist < left_window_size)) | ((dist < 0) & (-dist < right_window_size))
                sliding_mask = (1.0 - valid.astype(mx.float32)) * -1e9
                enc_mask = enc_mask + sliding_mask[None, None, :, :]
                
            hidden_states = layer(
                hidden_states,
                position_embeddings[layer_type],
                enc_mask
            )
            
        return self.norm(hidden_states)

# SigLIP Vision Tower Implementation in MLX
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embed_dim = config["hidden_size"]
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config["num_channels"],
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        
    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values input: [B, C, H, W]
        # MLX Conv2d expects [B, H, W, C]
        x = pixel_values.transpose(0, 2, 3, 1)
        patch_embeds = self.patch_embedding(x) # [B, H_p, W_p, D]
        B, H_p, W_p, D = patch_embeds.shape
        patch_embeds = patch_embeds.reshape(B, H_p * W_p, D)
        
        pos_ids = mx.arange(self.num_patches)[None, :]
        pos_embeds = self.position_embedding(pos_ids)
        return patch_embeds + pos_embeds

class SiglipAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embed_dim = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = attn_weights @ values
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)

class SiglipMLP(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=True)
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=True)
        
    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(gelu_pytorch_tanh(self.fc1(x)))

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        
    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.layer_norm1(x))
        out = h + self.mlp(self.layer_norm2(h))
        return out

class SiglipVisionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = nn.Sequential(
            *[SiglipEncoderLayer(config) for _ in range(config["num_hidden_layers"])]
        )
        self.post_layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        
    def __call__(self, pixel_values: mx.array) -> mx.array:
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        return self.post_layernorm(x)

class T5Gemma2MultiModalProjector(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Project vision hidden size -> text hidden size
        # mm_input_projection_weight is initialized to zeros
        self.mm_input_projection_weight = mx.zeros((config["vision_config"]["hidden_size"], config["text_config"]["hidden_size"]))
        self.mm_soft_emb_norm = GemmaRMSNorm(config["vision_config"]["hidden_size"], eps=config["vision_config"]["layer_norm_eps"])
        
        # average pooling params
        self.patches_per_image = int(config["vision_config"]["image_size"] // config["vision_config"]["patch_size"])
        self.tokens_per_side = int(config["mm_tokens_per_image"] ** 0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side

    def __call__(self, vision_outputs: mx.array) -> mx.array:
        # vision_outputs: [B, num_patches, D_vision]
        B, N, D_vision = vision_outputs.shape
        
        # Avg Pooling over grid space [patches_per_image, patches_per_image]
        # We can implement average pool manually by reshaping
        # reshaped: [B, D_vision, patches, patches]
        # in MLX, grid pooling is simple:
        # Transpose to [B, patches, patches, D_vision]
        grid = vision_outputs.reshape(B, self.patches_per_image, self.patches_per_image, D_vision)
        # Pool: kernel_size x kernel_size
        k = self.kernel_size
        # Reshape to group patches into pooling windows:
        # [B, tokens_per_side, k, tokens_per_side, k, D_vision]
        pooled = grid.reshape(B, self.tokens_per_side, k, self.tokens_per_side, k, D_vision)
        # Average over the k x k window axes (axis=2 and axis=4)
        pooled = mx.mean(pooled, axis=(2, 4)) # [B, tokens_per_side, tokens_per_side, D_vision]
        pooled = pooled.reshape(B, self.tokens_per_side * self.tokens_per_side, D_vision)
        
        normed = self.mm_soft_emb_norm(pooled)
        projected = normed @ self.mm_input_projection_weight
        return projected

class T5Gemma2Encoder(nn.Module):
    def __init__(self, config: Dict[str, Any], eoi_token_index: int = 256000):
        super().__init__()
        self.config = config
        self.image_token_id = config.get("image_token_id", 256001)
        
        self.text_model = T5Gemma2TextEncoder(config["text_config"], eoi_token_index)
        self.vision_tower = SiglipVisionModel(config["vision_config"])
        self.multi_modal_projector = T5Gemma2MultiModalProjector(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ) -> mx.array:
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        
        if pixel_values is not None:
            # Run vision tower
            vision_features = self.vision_tower(pixel_values) # [B, num_patches, D_vision]
            # Project to text space
            image_features = self.multi_modal_projector(vision_features) # [B, tokens_per_image, D_text]
            image_features = image_features.astype(inputs_embeds.dtype)
            
            # Find the indices of image tokens in inputs_ids
            # Since MLX doesn't have masked_scatter, we can use where
            # Wait, image_features has shape [B, tokens_per_image, D_text]
            # For each item in batch, we want to replace the image_token_id tokens in inputs_embeds
            # with the corresponding tokens from image_features.
            # We can do this with scatter or a boolean mask if we flatten
            image_mask = (input_ids == self.image_token_id) # [B, S]
            
            if mx.any(image_mask):
                # Flatten embeds and features
                B, S, D = inputs_embeds.shape
                # Flatten masks
                flat_mask = image_mask.reshape(-1)
                flat_embeds = inputs_embeds.reshape(-1, D)
                flat_features = image_features.reshape(-1, D)
                
                # Scatter flat_features into flat_embeds at indices where flat_mask is True
                indices = mx.arange(B * S)[flat_mask]
                flat_embeds[indices] = flat_features
                inputs_embeds = flat_embeds.reshape(B, S, D)
                
        return self.text_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

class T5Gemma2Decoder(nn.Module):
    def __init__(self, config: Dict[str, Any], eoi_token_index: int = 256000):
        super().__init__()
        self.config = config
        self.padding_idx = config.get("pad_token_id", 0)
        
        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            config["vocab_size"],
            config["hidden_size"],
            self.padding_idx,
            embed_scale=config["hidden_size"] ** 0.5,
            eoi_token_index=eoi_token_index,
        )
        self.norm = GemmaRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.layers = [T5Gemma2DecoderLayer(config, i) for i in range(config["num_hidden_layers"])]
        
        # Initialize rotary embeddings
        self.rotary_embs = {}
        for layer_type in set(config["layer_types"]):
            self.rotary_embs[layer_type] = T5Gemma2RotaryEmbedding(config, layer_type)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_hidden_states: mx.array = None,
        encoder_attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Any] = None,
    ) -> mx.array:
        inputs_embeds = self.embed_tokens(input_ids)
        
        B, S, _ = inputs_embeds.shape
        past_seen_tokens = past_key_values.offset if past_key_values is not None else 0
        position_ids = mx.arange(S, dtype=mx.float32)[None, :] + past_seen_tokens
        
        if past_key_values is not None:
            past_key_values.offset += S
            
        # Global position embeddings for each layer type
        position_embeddings = {}
        for layer_type, rotary in self.rotary_embs.items():
            position_embeddings[layer_type] = rotary(position_ids)
            
        hidden_states = inputs_embeds
        
        # Total tokens in cache including current ones
        cache_len = past_seen_tokens + S
        
        for i, layer in enumerate(self.layers):
            layer_type = self.config["layer_types"][i]
            is_sliding = (layer_type == "sliding_attention")
            sliding_window = self.config.get("sliding_window", 512)
            
            # Merged mask for this layer
            merged_mask = None
            if encoder_attention_mask is not None:
                # 1. self mask
                if S == 1:
                    self_mask = mx.zeros((B, 1, 1, cache_len), dtype=mx.float32)
                else:
                    q_idx = mx.arange(S)[:, None]
                    kv_idx = mx.arange(cache_len)[None, :]
                    causal_valid = (kv_idx <= (q_idx + past_seen_tokens))
                    if is_sliding:
                        causal_valid = causal_valid & ((q_idx + past_seen_tokens) - kv_idx < sliding_window)
                    self_mask = (1.0 - causal_valid.astype(mx.float32)) * -1e9
                    self_mask = mx.broadcast_to(self_mask[None, None, :, :], [B, 1, S, cache_len])
                    
                # 2. cross mask
                cross_mask = (1.0 - encoder_attention_mask[:, None, None, :]) * -1e9
                cross_mask = mx.broadcast_to(cross_mask, [B, 1, S, encoder_attention_mask.shape[1]])
                
                # 3. merge
                merged_mask = mx.concatenate([self_mask, cross_mask], axis=-1)
                
            hidden_states = layer(
                hidden_states,
                position_embeddings[layer_type],
                merged_mask,
                encoder_hidden_states,
                past_key_values,
            )
            
        return self.norm(hidden_states)

class T5Gemma2Model(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        eoi_token_index = config.get("eoi_token_index", 256000)
        self.encoder = T5Gemma2Encoder(config["encoder"], eoi_token_index)
        self.decoder = T5Gemma2Decoder(config["decoder"], eoi_token_index)

    def __call__(
        self,
        input_ids: mx.array,
        decoder_input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        decoder_attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        past_key_values: Optional[Any] = None,
        encoder_outputs: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        return decoder_outputs, encoder_outputs

class T5Gemma2ForConditionalGeneration(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = T5Gemma2Model(config)
        self.lm_head = nn.Linear(config["decoder"]["hidden_size"], config["decoder"]["vocab_size"], bias=False)
        self.vocab_size = config["decoder"]["vocab_size"]

    def __call__(
        self,
        input_ids: mx.array,
        decoder_input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        decoder_attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        past_key_values: Optional[Any] = None,
        encoder_outputs: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        decoder_outputs, encoder_outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
        )
        
        logits = self.lm_head(decoder_outputs)
        
        # Softcapping if defined in config
        final_logit_softcapping = self.config["decoder"].get("final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = mx.tanh(logits)
            logits = logits * final_logit_softcapping
            
        return logits, encoder_outputs
