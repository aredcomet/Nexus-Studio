import torch
import mlx.core as mx
import numpy as np
import json
from transformers import T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu")
hf_model.eval()

with open(f"{model_id}/config.json", "r") as f:
    config = json.load(f)
mlx_model = MLXModel(config)
weights = mx.load(weights_path)
mlx_model.load_weights(weights)

layer_pt = hf_model.model.encoder.text_model.layers[0]
layer_mx = mlx_model.model.encoder.text_model.layers[0]

# Inputs
input_pt = torch.tensor([[2, 17076, 563, 496, 4148, 1719]])
input_mx = mx.array([[2, 17076, 563, 496, 4148, 1719]])

with torch.no_grad():
    h_pt = hf_model.model.encoder.text_model.embed_tokens(input_pt)
h_mx = mlx_model.model.encoder.text_model.embed_tokens(input_mx)

# 1. pre_self_attn_layernorm
with torch.no_grad():
    norm1_pt = layer_pt.pre_self_attn_layernorm(h_pt)
norm1_mx = layer_mx.pre_self_attn_layernorm(h_mx)
print("--- After pre_self_attn_layernorm ---")
print("  PT:  ", norm1_pt[0, 0, :5].detach().float().cpu().numpy())
print("  MLX: ", np.array(norm1_mx[0, 0, :5].astype(mx.float32)))
print("  Diff:", np.abs(norm1_pt.detach().float().cpu().numpy() - np.array(norm1_mx.astype(mx.float32))).max())

# 2. self_attn
position_ids_pt = torch.arange(0, h_pt.shape[1], device=h_pt.device).unsqueeze(0)
position_ids_mx = mx.arange(h_mx.shape[1], dtype=mx.float32)[None, :]

with torch.no_grad():
    pos_emb_pt = hf_model.model.encoder.text_model.rotary_emb(h_pt, position_ids_pt, "sliding_attention")
pos_emb_mx = mlx_model.model.encoder.text_model.rotary_embs["sliding_attention"](position_ids_mx)

mask_pt = torch.zeros((1, 1, h_pt.shape[1], h_pt.shape[1]), dtype=torch.bfloat16)
mask_mx = mx.zeros((1, 1, h_mx.shape[1], h_mx.shape[1]), dtype=mx.bfloat16)

with torch.no_grad():
    attn_pt, _ = layer_pt.self_attn(
        hidden_states=norm1_pt,
        position_embeddings=pos_emb_pt,
        attention_mask=mask_pt,
    )
attn_mx = layer_mx.self_attn(
    hidden_states=norm1_mx,
    position_embeddings=pos_emb_mx,
    attention_mask=mask_mx,
)
print("\n--- After self_attn ---")
print("  PT:  ", attn_pt[0, 0, :5].detach().float().cpu().numpy())
print("  MLX: ", np.array(attn_mx[0, 0, :5].astype(mx.float32)))
print("  Diff:", np.abs(attn_pt.detach().float().cpu().numpy() - np.array(attn_mx.astype(mx.float32))).max())

# 3. post_self_attn_layernorm
with torch.no_grad():
    norm2_pt = layer_pt.post_self_attn_layernorm(attn_pt)
norm2_mx = layer_mx.post_self_attn_layernorm(attn_mx)
print("\n--- After post_self_attn_layernorm ---")
print("  PT:  ", norm2_pt[0, 0, :5].detach().float().cpu().numpy())
print("  MLX: ", np.array(norm2_mx[0, 0, :5].astype(mx.float32)))
print("  Diff:", np.abs(norm2_pt.detach().float().cpu().numpy() - np.array(norm2_mx.astype(mx.float32))).max())

# 4. First residual addition
res1_pt_actual = h_pt + norm2_pt
res1_mx = h_mx + norm2_mx

print("\n--- After first residual ---")
print("  PT actual: ", res1_pt_actual[0, 0, :5].detach().float().cpu().numpy())
print("  MLX:       ", np.array(res1_mx[0, 0, :5].astype(mx.float32)))
print("  Diff:      ", np.abs(res1_pt_actual.detach().float().cpu().numpy() - np.array(res1_mx.astype(mx.float32))).max())
