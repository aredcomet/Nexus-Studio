import torch
import mlx.core as mx
import numpy as np
import json
from transformers import T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")

with open(f"{model_id}/config.json", "r") as f:
    config = json.load(f)
mlx_model = MLXModel(config)
weights = mx.load(weights_path)
float32_weights = {k: v.astype(mx.float32) for k, v in weights.items()}
mlx_model.load_weights(float32_weights)

# Let's inspect layer 0 pre_self_attn_layernorm
norm_pt = hf_model.model.encoder.text_model.layers[0].pre_self_attn_layernorm
norm_mx = mlx_model.model.encoder.text_model.layers[0].pre_self_attn_layernorm

# Create a simple input of shape [1, 1, 640]
x_np = np.arange(640, dtype=np.float32).reshape(1, 1, 640) / 100.0
x_pt = torch.tensor(x_np)
x_mx = mx.array(x_np)

with torch.no_grad():
    y_pt = norm_pt(x_pt)
y_mx = norm_mx(x_mx)

y_pt_np = y_pt.numpy()
y_mx_np = np.array(y_mx)

print("Input shape:", x_np.shape)
print("Output shape:", y_pt_np.shape, y_mx_np.shape)
print("Max diff:", np.abs(y_pt_np - y_mx_np).max())
print("Mean diff:", np.abs(y_pt_np - y_mx_np).mean())

print("\nFirst 5 elements of PyTorch output:")
print(y_pt_np[0, 0, :5])
print("First 5 elements of MLX output:")
print(y_mx_np[0, 0, :5])

# Let's check weight values
w_pt = norm_pt.weight.detach().numpy()
w_mx = np.array(norm_mx.weight)
print("\nFirst 5 weight elements:")
print("PT weight: ", w_pt[:5])
print("MLX weight:", w_mx[:5])
