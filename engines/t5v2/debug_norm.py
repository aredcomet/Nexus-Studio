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

norm_pt = layer_pt.pre_self_attn_layernorm
norm_mx = layer_mx.pre_self_attn_layernorm

# Get weight values
w_pt = norm_pt.weight.detach().float().cpu().numpy()
w_mx = np.array(norm_mx.weight.astype(mx.float32))

print("=== Weights ===")
print("PT Weight First 5: ", w_pt[:5])
print("MLX Weight First 5:", w_mx[:5])
print("Weight Max Diff:   ", np.abs(w_pt - w_mx).max())

# Run embed token of a dummy input
input_pt = torch.tensor([[2, 17076, 563, 496, 4148, 1719]])
input_mx = mx.array([[2, 17076, 563, 496, 4148, 1719]])

with torch.no_grad():
    h_pt = hf_model.model.encoder.text_model.embed_tokens(input_pt)
h_mx = mlx_model.model.encoder.text_model.embed_tokens(input_mx)

print("\n=== Embeddings ===")
print("h_pt shape:", h_pt.shape)
print("h_mx shape:", h_mx.shape)
print("h_pt first 5 elements:", h_pt[0, 0, :5].float().numpy())
print("h_mx first 5 elements:", np.array(h_mx[0, 0, :5].astype(mx.float32)))
print("Embeddings Max Diff:", np.abs(h_pt.float().numpy() - np.array(h_mx.astype(mx.float32))).max())

# Now run RMSNorm using h_pt as input
h_common_mx = mx.array(h_pt.float().numpy()).astype(mx.bfloat16)

with torch.no_grad():
    y_pt = norm_pt(h_pt)
y_mx = norm_mx(h_common_mx)

print("\n=== RMSNorm Outputs ===")
print("y_pt shape:", y_pt.shape)
print("y_mx shape:", y_mx.shape)
print("y_pt first 5 elements:", y_pt[0, 0, :5].float().numpy())
print("y_mx first 5 elements:", np.array(y_mx[0, 0, :5].astype(mx.float32)))
print("RMSNorm Output Max Diff:", np.abs(y_pt.float().numpy() - np.array(y_mx.astype(mx.float32))).max())
