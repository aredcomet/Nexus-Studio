import torch
import mlx.core as mx
import numpy as np
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

processor = AutoProcessor.from_pretrained(model_id)
hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
hf_model.eval()

with open(f"{model_id}/config.json", "r") as f:
    import json
    config = json.load(f)
mlx_model = MLXModel(config)
mlx_model.load_weights(weights_path)

prompt = "Today is a beautiful day"
inputs = processor(text=prompt, return_tensors="pt")
input_ids_pt = inputs["input_ids"]
attention_mask_pt = inputs["attention_mask"]

input_ids_mx = mx.array(input_ids_pt.numpy())
attention_mask_mx = mx.array(attention_mask_pt.numpy())

# Run embeds
with torch.no_grad():
    h_pt = hf_model.model.encoder.text_model.embed_tokens(input_ids_pt)
    
h_mx = mlx_model.model.encoder.text_model.embed_tokens(input_ids_mx)

# Setup rotary embeddings
position_ids_pt = torch.arange(0, h_pt.shape[1], device=h_pt.device).unsqueeze(0)
position_ids_mx = mx.arange(h_mx.shape[1], dtype=mx.float32)[None, :]

position_embeddings_pt = {}
for layer_type in set(hf_model.model.encoder.text_model.config.layer_types):
    position_embeddings_pt[layer_type] = hf_model.model.encoder.text_model.rotary_emb(h_pt, position_ids_pt, layer_type)

position_embeddings_mx = {}
for layer_type, rotary in mlx_model.model.encoder.text_model.rotary_embs.items():
    position_embeddings_mx[layer_type] = rotary(position_ids_mx)

# Check rotary embeddings
for layer_type in position_embeddings_pt.keys():
    cos_pt, sin_pt = position_embeddings_pt[layer_type]
    cos_mx, sin_mx = position_embeddings_mx[layer_type]
    diff_cos = np.abs(cos_pt.float().numpy() - np.array(cos_mx.astype(mx.float32)))
    diff_sin = np.abs(sin_pt.float().numpy() - np.array(sin_mx.astype(mx.float32)))
    print(f"Rotary {layer_type} - cos diff: {diff_cos.max():.2e}, sin diff: {diff_sin.max():.2e}")

# Compare layer by layer
print("\n--- Comparing Encoder Layers ---")
for idx in range(len(mlx_model.model.encoder.text_model.layers)):
    layer_type = config["encoder"]["text_config"]["layer_types"][idx]
    is_sliding = (layer_type == "sliding_attention")
    sliding_window = config["encoder"]["text_config"].get("sliding_window", 512)
    
    # 1. Prepare masks
    # PyTorch
    mask_kwargs = {
        "config": hf_model.model.encoder.text_model.config,
        "inputs_embeds": h_pt,
        "attention_mask": attention_mask_pt,
    }
    from transformers.masking_utils import create_bidirectional_mask
    from transformers.models.t5gemma2.modeling_t5gemma2 import sliding_window_mask_function
    if is_sliding:
        mask_pt = create_bidirectional_mask(
            **mask_kwargs,
            and_mask_function=sliding_window_mask_function(sliding_window, is_causal=False),
        )
    else:
        mask_pt = create_bidirectional_mask(**mask_kwargs)
        
    # MLX
    mask_mx = (1.0 - attention_mask_mx[:, None, None, :]) * -1e9
    if is_sliding:
        S = h_mx.shape[1]
        q_idx = mx.arange(S)[:, None]
        kv_idx = mx.arange(S)[None, :]
        dist = q_idx - kv_idx
        left_window_size = (sliding_window + 1) // 2
        right_window_size = sliding_window // 2 + 1
        valid = ((dist >= 0) & (dist < left_window_size)) | ((dist < 0) & (-dist < right_window_size))
        sliding_mask = (1.0 - valid.astype(mx.float32)) * -1e9
        mask_mx = mask_mx + sliding_mask[None, None, :, :]
        
    # Run layer on the current hidden states
    with torch.no_grad():
        h_pt_next = hf_model.model.encoder.text_model.layers[idx](
            h_pt,
            position_embeddings=position_embeddings_pt[layer_type],
            attention_mask=mask_pt,
        )
        
    h_mx_next = mlx_model.model.encoder.text_model.layers[idx](
        h_mx,
        position_embeddings=position_embeddings_mx[layer_type],
        attention_mask=mask_mx,
    )
    
    # Measure diff on accumulated hidden states
    diff_acc = np.abs(h_pt_next.float().numpy() - np.array(h_mx_next.astype(mx.float32)))
    print(f"Layer {idx} ({layer_type}) accumulated diff: {diff_acc.max():.2e}")
    
    # Run layer on clean PyTorch inputs to isolate layer bugs
    h_mx_clean = mlx_model.model.encoder.text_model.layers[idx](
        mx.array(h_pt.numpy()),
        position_embeddings=position_embeddings_mx[layer_type],
        attention_mask=mask_mx,
    )
    diff_iso = np.abs(h_pt_next.float().numpy() - np.array(h_mx_clean.astype(mx.float32)))
    print(f"Layer {idx} ({layer_type}) isolated diff: {diff_iso.max():.2e}")
    
    # Update state for next layer
    h_pt = h_pt_next
    h_mx = h_mx_next
