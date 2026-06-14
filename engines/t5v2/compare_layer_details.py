import torch
import mlx.core as mx
import numpy as np
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

# Load in float32 for comparison
hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
hf_model.eval()

with open(f"{model_id}/config.json", "r") as f:
    import json
    config = json.load(f)
mlx_model = MLXModel(config)
mlx_model.load_weights(weights_path)

# Cast MLX model parameters to float32 to eliminate dtype differences!
for name, p in mlx_model.parameters().items():
    # We can assign float32 casted array
    # In MLX, assigning to parameter requires modifying the weight dict or using update
    pass

# Wait, let's just cast the weights dictionary to float32 before loading!
weights = mx.load(weights_path)
float32_weights = {k: v.astype(mx.float32) for k, v in weights.items()}
mlx_model.load_weights(float32_weights)

prompt = "Today is a beautiful day"
processor = AutoProcessor.from_pretrained(model_id)
inputs = processor(text=prompt, return_tensors="pt")
input_ids_pt = inputs["input_ids"]
attention_mask_pt = inputs["attention_mask"]

input_ids_mx = mx.array(input_ids_pt.numpy())
attention_mask_mx = mx.array(attention_mask_pt.numpy())

# Run embedding
with torch.no_grad():
    h_pt = hf_model.model.encoder.text_model.embed_tokens(input_ids_pt)
    h_mx = mlx_model.model.encoder.text_model.embed_tokens(input_ids_mx)

    print(f"Embeddings max diff: {np.abs(h_pt.float().numpy() - np.array(h_mx.astype(mx.float32))).max():.2e}")

    # Layer 0
    layer_pt = hf_model.model.encoder.text_model.layers[0]
    layer_mx = mlx_model.model.encoder.text_model.layers[0]

    position_ids_pt = torch.arange(0, h_pt.shape[1], device=h_pt.device).unsqueeze(0)
    position_ids_mx = mx.arange(h_mx.shape[1], dtype=mx.float32)[None, :]

    pos_emb_pt = hf_model.model.encoder.text_model.rotary_emb(h_pt, position_ids_pt, "sliding_attention")
    pos_emb_mx = mlx_model.model.encoder.text_model.rotary_embs["sliding_attention"](position_ids_mx)

    # 1. pre_self_attn_layernorm
    norm_h_pt = layer_pt.pre_self_attn_layernorm(h_pt)
    norm_h_mx = layer_mx.pre_self_attn_layernorm(h_mx)
    print(f"pre_self_attn_layernorm diff: {np.abs(norm_h_pt.float().numpy() - np.array(norm_h_mx)).max():.2e}")

    # 2. self_attn
    mask_pt = torch.zeros((1, 1, h_pt.shape[1], h_pt.shape[1]))
    mask_mx = mx.zeros((1, 1, h_mx.shape[1], h_mx.shape[1]))

    attn_pt, _ = layer_pt.self_attn(
        hidden_states=norm_h_pt,
        position_embeddings=pos_emb_pt,
        attention_mask=mask_pt,
    )
    
    attn_mx = layer_mx.self_attn(
        hidden_states=norm_h_mx,
        position_embeddings=pos_emb_mx,
        attention_mask=mask_mx,
    )
    print(f"self_attn diff: {np.abs(attn_pt.float().numpy() - np.array(attn_mx)).max():.2e}")

    # Let's inspect query_states, key_states, value_states
    q_pt = layer_pt.self_attn.q_proj(norm_h_pt)
    k_pt = layer_pt.self_attn.k_proj(norm_h_pt)
    v_pt = layer_pt.self_attn.v_proj(norm_h_pt)
    
    q_mx = layer_mx.self_attn.q_proj(norm_h_mx)
    k_mx = layer_mx.self_attn.k_proj(norm_h_mx)
    v_mx = layer_mx.self_attn.v_proj(norm_h_mx)

    print(f"q_proj diff: {np.abs(q_pt.float().numpy() - np.array(q_mx)).max():.2e}")
    print(f"k_proj diff: {np.abs(k_pt.float().numpy() - np.array(k_mx)).max():.2e}")
    print(f"v_proj diff: {np.abs(v_pt.float().numpy() - np.array(v_mx)).max():.2e}")

    # Let's check after q_norm and k_norm
    q_norm_pt = layer_pt.self_attn.q_norm(q_pt.view(1, h_pt.shape[1], 4, 256).transpose(1, 2))
    k_norm_pt = layer_pt.self_attn.k_norm(k_pt.view(1, h_pt.shape[1], 1, 256).transpose(1, 2))
    
    q_norm_mx = layer_mx.self_attn.q_norm(q_mx.reshape(1, h_mx.shape[1], 4, 256).transpose(0, 2, 1, 3))
    k_norm_mx = layer_mx.self_attn.k_norm(k_mx.reshape(1, h_mx.shape[1], 1, 256).transpose(0, 2, 1, 3))

    print(f"q_norm diff: {np.abs(q_norm_pt.float().numpy() - np.array(q_norm_mx)).max():.2e}")
    print(f"k_norm diff: {np.abs(k_norm_pt.float().numpy() - np.array(k_norm_mx)).max():.2e}")
