import math
import mlx.core as mx
import mlx.nn as nn

class ModelConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 256000)
        self.hidden_size = kwargs.get("hidden_size", 2560)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 26)
        self.intermediate_size = kwargs.get("intermediate_size", 15360)
        self.num_attention_heads = kwargs.get("num_attention_heads", 10)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 1)
        self.head_dim = kwargs.get("head_dim", 256)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 0.5)
        self.attention_window_size = kwargs.get("attention_window_size", 2048)
        self.logits_soft_cap = kwargs.get("logits_soft_cap", 30.0)
        self.lru_width = kwargs.get("lru_width", 2560)
        self.conv1d_width = kwargs.get("conv1d_width", 4)
        
        block_pattern = kwargs.get("block_types", kwargs.get("_block_types", ["recurrent", "recurrent", "attention"]))
        self.layers_block_type = (
            block_pattern * (self.num_hidden_layers // len(block_pattern))
            + block_pattern[:(self.num_hidden_layers % len(block_pattern))]
        )

class DecoderCache:
    def __init__(self, num_layers, lru_width, hidden_size):
        self.conv1d_states = [None] * num_layers
        self.recurrent_states = [None] * num_layers
        self.kv_caches = [None] * num_layers
        self.lru_width = lru_width
        self.hidden_size = hidden_size

    def get_conv1d_state(self, layer_idx, batch_size, dtype):
        if self.conv1d_states[layer_idx] is None or self.conv1d_states[layer_idx].shape[0] != batch_size:
            # conv1d_state shape: [B, 3, lru_width]
            self.conv1d_states[layer_idx] = mx.zeros((batch_size, 3, self.lru_width), dtype=dtype)
        return self.conv1d_states[layer_idx]

    def update_conv1d_state(self, layer_idx, new_state):
        self.conv1d_states[layer_idx] = new_state

    def get_recurrent_state(self, layer_idx, batch_size):
        if self.recurrent_states[layer_idx] is None or self.recurrent_states[layer_idx].shape[0] != batch_size:
            # recurrent_states always computed in full precision (float32)
            self.recurrent_states[layer_idx] = mx.zeros((batch_size, self.lru_width), dtype=mx.float32)
        return self.recurrent_states[layer_idx]

    def update_recurrent_state(self, layer_idx, new_state):
        self.recurrent_states[layer_idx] = new_state

class RecurrentGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dim,))

    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x ** 2, axis=-1, keepdims=True)
        # Apply normalization: (x * rsqrt(variance + eps))
        output = x * mx.rsqrt(variance + self.eps)
        # Apply scaling: output * (1.0 + weight)
        output = output * (1.0 + self.weight)
        return output.astype(orig_dtype)

class RecurrentGemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x, offset=0):
        # x is expected to have shape [B, H, L, D]
        seq_len = x.shape[2]
        t = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    # x shape: [B, H, L, D]
    # cos, sin shape: [L, D]
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rx = mx.concatenate([-x2, x1], axis=-1)
    
    # Reshape cos/sin for broadcasting
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (rx * sin)

class RecurrentGemmaSdpaAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.partial_rotary_factor = config.partial_rotary_factor
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=True)
        self.rotary_emb = RecurrentGemmaRotaryEmbedding(dim=self.rotary_dim, base=config.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        offset: int = 0,
        cache: DecoderCache | None = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Split for partial rotary embedding
        q_rot = query_states[..., :self.rotary_dim]
        q_pass = query_states[..., self.rotary_dim:]
        k_rot = key_states[..., :self.rotary_dim]
        k_pass = key_states[..., self.rotary_dim:]

        cos, sin = self.rotary_emb(q_rot, offset=offset)
        q_rot = apply_rotary_pos_emb(q_rot, cos, sin)
        k_rot = apply_rotary_pos_emb(k_rot, cos, sin)

        query_states = mx.concatenate([q_rot, q_pass], axis=-1)
        key_states = mx.concatenate([k_rot, k_pass], axis=-1)

        if cache is not None:
            kv_cache = cache.kv_caches[self.layer_idx]
            if kv_cache is not None:
                prev_k, prev_v = kv_cache
                key_states = mx.concatenate([prev_k, key_states], axis=2)
                value_states = mx.concatenate([prev_v, value_states], axis=2)
            
            # Sliding window attention caching
            if key_states.shape[2] > self.config.attention_window_size:
                key_states = key_states[:, :, -self.config.attention_window_size:, :]
                value_states = value_states[:, :, -self.config.attention_window_size:, :]
            
            cache.kv_caches[self.layer_idx] = (key_states, value_states)

        # Repeat KV to match query heads
        if self.num_key_value_groups > 1:
            key_states = mx.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = mx.repeat(value_states, self.num_key_value_groups, axis=1)

        # Scaled dot-product attention
        scores = mx.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_dim))

        # Attention mask
        if L > 1:
            mask = mx.triu(mx.full((L, L), -float("inf"), dtype=query_states.dtype), k=1)
            
            # Sliding window masking for sequence prefill
            row_idx = mx.arange(L)[:, None]
            col_idx = mx.arange(L)[None, :]
            window_mask = (row_idx - col_idx) >= self.config.attention_window_size
            mask = mx.where(window_mask, -float("inf"), mask)
            
            scores = scores + mask

        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(query_states.dtype)
        attn_output = mx.matmul(probs, value_states)
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_output)

class RecurrentGemmaRglru(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.block_width = config.lru_width // self.num_attention_heads
        self.lru_width = config.lru_width

        self.recurrent_param = mx.zeros((config.lru_width,))
        self.input_gate_weight = mx.zeros((self.num_attention_heads, self.block_width, self.block_width))
        self.input_gate_bias = mx.zeros((self.num_attention_heads, self.block_width))

        self.recurrent_gate_weight = mx.zeros((self.num_attention_heads, self.block_width, self.block_width))
        self.recurrent_gate_bias = mx.zeros((self.num_attention_heads, self.block_width))

    def __call__(
        self,
        activations: mx.array,
        position_ids: mx.array,
        cache: DecoderCache | None = None,
        layer_idx: int = 0,
    ) -> mx.array:
        batch_size, seq_len, lru_width = activations.shape
        reset = position_ids[:, :, None] == 0

        # Reshape activations for batched matmul: [num_heads, batch * seq_len, block_width]
        reshape_act = activations.reshape(batch_size * seq_len, self.num_attention_heads, self.block_width)
        reshape_act = reshape_act.transpose(1, 0, 2)

        # Compute input gate
        in_res = mx.matmul(reshape_act, self.input_gate_weight) + self.input_gate_bias[:, None, :]
        in_res = in_res.transpose(1, 0, 2).reshape(batch_size, seq_len, lru_width)
        input_gate = mx.sigmoid(in_res)

        # Compute recurrent gate
        rec_res = mx.matmul(reshape_act, self.recurrent_gate_weight) + self.recurrent_gate_bias[:, None, :]
        rec_res = rec_res.transpose(1, 0, 2).reshape(batch_size, seq_len, lru_width)
        recurrent_gate = mx.sigmoid(rec_res)

        # Recurrent parameter `A` computation
        softplus_param = mx.maximum(self.recurrent_param, 0.0) + mx.log1p(mx.exp(-mx.abs(self.recurrent_param)))
        log_recurrent_gate = -8.0 * recurrent_gate * softplus_param
        recurrent_gate = mx.exp(log_recurrent_gate)
        a_square = mx.exp(2.0 * log_recurrent_gate)

        gated_inputs = activations * input_gate

        # Apply normalization multiplier: sqrt(1 - a^2)
        multiplier = mx.sqrt(mx.maximum(1.0 - a_square, 1e-8))
        multiplier = mx.where(reset, 1.0, multiplier)
        normalized_x = gated_inputs * multiplier.astype(activations.dtype)

        # RNN Scan recurrence
        recurrent_gate = recurrent_gate * mx.logical_not(reset).astype(recurrent_gate.dtype)

        # Fetch cache state
        recurrent_states = None
        if cache is not None:
            recurrent_states = cache.get_recurrent_state(layer_idx, batch_size)

        if seq_len == 1:
            if recurrent_states is None:
                recurrent_states = mx.zeros((batch_size, lru_width), dtype=mx.float32)
            
            # Recurrence: h_t = a_t * h_{t-1} + x_t
            contextualized = recurrent_gate * recurrent_states[:, None, :] + normalized_x
            new_recurrent_states = contextualized[:, 0].astype(mx.float32)
            
            if cache is not None:
                cache.update_recurrent_state(layer_idx, new_recurrent_states)
                
            return contextualized
        else:
            if recurrent_states is None:
                recurrent_states = mx.zeros((batch_size, lru_width), dtype=mx.float32)
                
            contextualized_states = []
            for t in range(seq_len):
                recurrent_states = recurrent_gate[:, t] * recurrent_states + normalized_x[:, t]
                contextualized_states.append(recurrent_states[:, None, :])
                
            contextualized_states = mx.concatenate(contextualized_states, axis=1)
            
            if cache is not None:
                cache.update_recurrent_state(layer_idx, recurrent_states.astype(mx.float32))
                
            return contextualized_states

class RecurrentGemmaRecurrentBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.lru_width = config.lru_width
        self.hidden_size = config.hidden_size
        self.conv1d_width = config.conv1d_width

        self.linear_y = nn.Linear(config.hidden_size, config.lru_width, bias=True)
        self.linear_x = nn.Linear(config.hidden_size, config.lru_width, bias=True)
        self.linear_out = nn.Linear(config.lru_width, config.hidden_size, bias=True)

        self.conv_1d = nn.Conv1d(
            config.lru_width,
            config.lru_width,
            kernel_size=config.conv1d_width,
            groups=config.lru_width,
            padding=config.conv1d_width - 1,
        )
        self.rg_lru = RecurrentGemmaRglru(config)

    def __call__(
        self,
        input_states: mx.array,
        position_ids: mx.array,
        cache: DecoderCache | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = input_states.shape

        y_branch = self.linear_y(input_states)
        y_branch = self.gelu_tanh(y_branch)

        x_branch = self.linear_x(input_states)

        # Depthwise 1D Convolution
        weight = self.conv_1d.weight.reshape(self.lru_width, self.conv1d_width)
        bias = self.conv_1d.bias

        if cache is not None:
            conv1d_state = cache.get_conv1d_state(self.layer_idx, batch_size, input_states.dtype)

            if seq_len > 1:
                # Prefill mode convolution
                padded = mx.pad(x_branch, ((0, 0), (self.conv1d_width - 1, 0), (0, 0)))
                
                x0 = padded[:, 0:seq_len, :]
                x1 = padded[:, 1:seq_len+1, :]
                x2 = padded[:, 2:seq_len+2, :]
                x3 = padded[:, 3:seq_len+3, :]
                
                x_branch = (
                    x0 * weight[:, 0] +
                    x1 * weight[:, 1] +
                    x2 * weight[:, 2] +
                    x3 * weight[:, 3] +
                    bias
                )
                # Save state (last 3 steps)
                cache.update_conv1d_state(self.layer_idx, padded[:, seq_len:seq_len+3, :])
            else:
                # Decoding mode convolution
                # conv1d_state is shape [B, 3, lru_width], x_branch is [B, 1, lru_width]
                conv_state = mx.concatenate([conv1d_state, x_branch], axis=1) # [B, 4, lru_width]
                
                x_branch = (
                    conv_state[:, 0] * weight[:, 0] +
                    conv_state[:, 1] * weight[:, 1] +
                    conv_state[:, 2] * weight[:, 2] +
                    conv_state[:, 3] * weight[:, 3] +
                    bias
                )
                x_branch = x_branch[:, None, :] # [B, 1, lru_width]
                cache.update_conv1d_state(self.layer_idx, conv_state[:, 1:4, :])
        else:
            # Without cache
            padded = mx.pad(x_branch, ((0, 0), (self.conv1d_width - 1, 0), (0, 0)))
            x0 = padded[:, 0:seq_len, :]
            x1 = padded[:, 1:seq_len+1, :]
            x2 = padded[:, 2:seq_len+2, :]
            x3 = padded[:, 3:seq_len+3, :]
            x_branch = (
                x0 * weight[:, 0] +
                x1 * weight[:, 1] +
                x2 * weight[:, 2] +
                x3 * weight[:, 3] +
                bias
            )

        x_branch = self.rg_lru(x_branch, position_ids, cache, self.layer_idx)

        hidden_states = x_branch * y_branch
        return self.linear_out(hidden_states)

    def gelu_tanh(self, x):
        return 0.5 * x * (1.0 + mx.tanh(0.7978845608 * (x + 0.044715 * (x ** 3))))

class RecurrentGemmaMlp(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=True)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=True)

    def __call__(self, x):
        gate = self.gelu_tanh(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))

    def gelu_tanh(self, x):
        return 0.5 * x * (1.0 + mx.tanh(0.7978845608 * (x + 0.044715 * (x ** 3))))

class RecurrentGemmaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        block_type = config.layers_block_type[layer_idx]
        self.temporal_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if block_type == "recurrent":
            self.temporal_block = RecurrentGemmaRecurrentBlock(config, layer_idx)
        else:
            self.temporal_block = RecurrentGemmaSdpaAttention(config, layer_idx)
            
        self.channel_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_block = RecurrentGemmaMlp(config)

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
        offset: int = 0,
        cache: DecoderCache | None = None,
    ) -> mx.array:
        raw_activations = hidden_states
        inputs_normalized = self.temporal_pre_norm(raw_activations)

        # Call temporal block (attention or recurrent linear LRU)
        if isinstance(self.temporal_block, RecurrentGemmaSdpaAttention):
            hidden_states = self.temporal_block(inputs_normalized, offset=offset, cache=cache)
        else:
            hidden_states = self.temporal_block(inputs_normalized, position_ids, cache=cache)

        residual = hidden_states + raw_activations

        # Channel (MLP) block
        hidden_states = self.channel_pre_norm(residual)
        hidden_states = self.mlp_block(hidden_states)

        return hidden_states + residual

class RecurrentGemmaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            RecurrentGemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.final_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.normalizer = config.hidden_size ** 0.5

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array = None,
        offset: int = 0,
        cache: DecoderCache | None = None,
    ) -> mx.array:
        B, L = input_ids.shape
        
        if position_ids is None:
            position_ids = mx.arange(offset, offset + L, dtype=mx.int32)[None, :]

        hidden_states = self.embed_tokens(input_ids)
        # RecurrentGemma scales token embeddings by sqrt(dim)
        hidden_states = hidden_states * self.normalizer

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                offset=offset,
                cache=cache
            )

        return self.final_norm(hidden_states)

class RecurrentGemmaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = RecurrentGemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array = None,
        offset: int = 0,
        cache: DecoderCache | None = None,
    ) -> mx.array:
        # 1. Forward through backbone model
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            offset=offset,
            cache=cache
        )
        
        # 2. Project to logits
        logits = self.lm_head(hidden_states)
        
        # 3. Apply logits soft-capping (divide by cap, tanh, multiply by cap)
        if self.config.logits_soft_cap is not None:
            cap = self.config.logits_soft_cap
            logits = mx.tanh(logits / cap) * cap
            
        return logits
