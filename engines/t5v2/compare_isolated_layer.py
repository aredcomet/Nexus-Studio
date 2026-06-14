import torch
import mlx.core as mx
import numpy as np
import json
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

# Load in float32 for comparison
hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
hf_model.eval()

with open(f"{model_id}/config.json", "r") as f:
    config = json.load(f)
mlx_model = MLXModel(config)

# Load weights in float32
weights = mx.load(weights_path)
float32_weights = {k: v.astype(mx.float32) for k, v in weights.items()}
mlx_model.load_weights(float32_weights)

prompt = "Today is a beautiful day"
processor = AutoProcessor.from_pretrained(model_id)
inputs = processor(text=prompt, return_tensors="pt")
input_ids_pt = inputs["input_ids"]
attention_mask_pt = inputs["attention_mask"]

# Run embeddings to get dynamic hidden states
with torch.no_grad():
    h_pt = hf_model.model.encoder.text_model.embed_tokens(input_ids_pt)
    
# We will use h_pt as the common input
h_common = mx.array(h_pt.float().numpy())

layer_pt = hf_model.model.encoder.text_model.layers[0]
layer_mx = mlx_model.model.encoder.text_model.layers[0]

print("=== Isolation Testing Layer 0 sub-modules ===")

# 1. Test RMSNorm in isolation
with torch.no_grad():
    norm_h_pt = layer_pt.pre_self_attn_layernorm(h_pt)
norm_h_mx = layer_mx.pre_self_attn_layernorm(h_common)
diff_norm = np.abs(norm_h_pt.float().numpy() - np.array(norm_h_mx)).max()
print(f"pre_self_attn_layernorm max diff: {diff_norm:.2e}")

# 2. Test Self Attention in isolation
norm_h_common = mx.array(norm_h_pt.float().numpy())

position_ids_pt = torch.arange(0, h_pt.shape[1], device=h_pt.device).unsqueeze(0)
position_ids_mx = mx.arange(h_pt.shape[1], dtype=mx.float32)[None, :]

with torch.no_grad():
    pos_emb_pt = hf_model.model.encoder.text_model.rotary_emb(h_pt, position_ids_pt, "sliding_attention")
pos_emb_mx = mlx_model.model.encoder.text_model.rotary_embs["sliding_attention"](position_ids_mx)

# Compare rotary embeddings
cos_pt, sin_pt = pos_emb_pt
cos_mx, sin_mx = pos_emb_mx
print(f"Rotary cos max diff: {np.abs(cos_pt.float().numpy() - np.array(cos_mx)).max():.2e}")
print(f"Rotary sin max diff: {np.abs(sin_pt.float().numpy() - np.array(sin_mx)).max():.2e}")

mask_pt = torch.zeros((1, 1, h_pt.shape[1], h_pt.shape[1]))
mask_mx = mx.zeros((1, 1, h_pt.shape[1], h_pt.shape[1]))

with torch.no_grad():
    attn_pt, _ = layer_pt.self_attn(
        hidden_states=norm_h_pt,
        position_embeddings=pos_emb_pt,
        attention_mask=mask_pt,
    )
    
# Convert pos_emb_pt to MLX arrays for absolute alignment
pos_emb_mx_common = (
    mx.array(cos_pt.float().numpy()),
    mx.array(sin_pt.float().numpy())
)

attn_mx = layer_mx.self_attn(
    hidden_states=norm_h_common,
    position_embeddings=pos_emb_mx_common,
    attention_mask=mask_mx,
)

diff_attn = np.abs(attn_pt.float().numpy() - np.array(attn_mx)).max()
print(f"self_attn max diff: {diff_attn:.2e}")

# Let's inspect query_states, key_states, value_states projections and normalizations
with torch.no_grad():
    q_pt = layer_pt.self_attn.q_proj(norm_h_pt)
    k_pt = layer_pt.self_attn.k_proj(norm_h_pt)
    v_pt = layer_pt.self_attn.v_proj(norm_h_pt)
    
q_mx = layer_mx.self_attn.q_proj(norm_h_common)
k_mx = layer_mx.self_attn.k_proj(norm_h_common)
v_mx = layer_mx.self_attn.v_proj(norm_h_common)

print(f"  q_proj max diff: {np.abs(q_pt.float().numpy() - np.array(q_mx)).max():.2e}")
print(f"  k_proj max diff: {np.abs(k_pt.float().numpy() - np.array(k_mx)).max():.2e}")
print(f"  v_proj max diff: {np.abs(v_pt.float().numpy() - np.array(v_mx)).max():.2e}")

# Check q_norm and k_norm
with torch.no_grad():
    q_norm_pt = layer_pt.self_attn.q_norm(q_pt.view(1, h_pt.shape[1], 4, 256).transpose(1, 2))
    k_norm_pt = layer_pt.self_attn.k_norm(k_pt.view(1, h_pt.shape[1], 1, 256).transpose(1, 2))
    
q_mx_reshaped = q_mx.reshape(1, h_pt.shape[1], 4, 256).transpose(0, 2, 1, 3)
k_mx_reshaped = k_mx.reshape(1, h_pt.shape[1], 1, 256).transpose(0, 2, 1, 3)

q_norm_mx = layer_mx.self_attn.q_norm(q_mx_reshaped)
k_norm_mx = layer_mx.self_attn.k_norm(k_mx_reshaped)

print(f"  q_norm max diff: {np.abs(q_norm_pt.float().numpy() - np.array(q_norm_mx)).max():.2e}")
print(f"  k_norm max diff: {np.abs(k_norm_pt.float().numpy() - np.array(k_norm_mx)).max():.2e}")

# 3. Test MLP in isolation
with torch.no_grad():
    mlp_pt = layer_pt.mlp(h_pt)
mlp_mx = layer_mx.mlp(h_common)
diff_mlp = np.abs(mlp_pt.float().numpy() - np.array(mlp_mx)).max()
print(f"mlp max diff: {diff_mlp:.2e}")
