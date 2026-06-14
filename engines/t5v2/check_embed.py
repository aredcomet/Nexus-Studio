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

# Let's get the embedding weight tensor
w_pt = hf_model.model.encoder.text_model.embed_tokens.weight.detach()
w_mx = mlx_model.model.encoder.text_model.embed_tokens.weight

print("PyTorch weight shape:", w_pt.shape)
print("MLX weight shape:", w_mx.shape)

# Compare values for token 100
v_pt = w_pt[100].float().cpu().numpy()
v_mx = np.array(w_mx[100])

diff = np.abs(v_pt - v_mx)
print("Max difference for token 100:", diff.max())
print("Mean difference for token 100:", diff.mean())

print("First 10 elements PyTorch:", v_pt[:10])
print("First 10 elements MLX:    ", v_mx[:10])
print("Difference:               ", (v_pt - v_mx)[:10])
