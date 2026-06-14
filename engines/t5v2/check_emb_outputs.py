import torch
import mlx.core as mx
import numpy as np
import json
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

processor = AutoProcessor.from_pretrained(model_id)
hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")

with open(f"{model_id}/config.json", "r") as f:
    config = json.load(f)
mlx_model = MLXModel(config)
weights = mx.load(weights_path)
float32_weights = {k: v.astype(mx.float32) for k, v in weights.items()}
mlx_model.load_weights(float32_weights)

prompt = "Today is a beautiful day"
inputs = processor(text=prompt, return_tensors="pt")
input_ids_pt = inputs["input_ids"]
input_ids_mx = mx.array(input_ids_pt.numpy())

print("Tokens:", input_ids_pt[0].tolist())

with torch.no_grad():
    h_pt = hf_model.model.encoder.text_model.embed_tokens(input_ids_pt)
h_mx = mlx_model.model.encoder.text_model.embed_tokens(input_ids_mx)

# Print first token embedding
t_pt = h_pt[0, 0].float().cpu().numpy()
t_mx = np.array(h_mx[0, 0])

print("\nFirst 10 elements h_pt:", t_pt[:10])
print("First 10 elements h_mx:", t_mx[:10])
print("Max diff:", np.abs(t_pt - t_mx).max())

# Compare to scaled raw weights
scale = 640 ** 0.5
raw_w = hf_model.model.encoder.text_model.embed_tokens.weight[input_ids_pt[0, 0]].float().numpy()
print("Raw weight scaled:     ", (raw_w * scale)[:10])
