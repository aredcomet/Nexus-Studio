import torch
import mlx.core as mx
import numpy as np
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel
from model import DecoderCache

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

print("1. Loading HF and MLX models...")
processor = AutoProcessor.from_pretrained(model_id)

hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu")
hf_model.eval()

with open(f"{model_id}/config.json", "r") as f:
    import json
    config = json.load(f)
mlx_model = MLXModel(config)
mlx_model.load_weights(weights_path)

# Prepare inputs
prompt = "Translate from English to French: Today is a beautiful day to learn programming."
inputs = processor(text=prompt, return_tensors="pt")
input_ids_pt = inputs["input_ids"]
attention_mask_pt = inputs["attention_mask"]

input_ids_mx = mx.array(input_ids_pt.numpy())
attention_mask_mx = mx.array(attention_mask_pt.numpy())

# STEP 0
print("\n--- STEP 0 (Prefill) ---")
with torch.no_grad():
    enc_outputs_pt = hf_model.model.encoder(
        input_ids=input_ids_pt,
        attention_mask=attention_mask_pt,
    )
    
    # Prefill step with BOS
    decoder_input_ids_pt = torch.tensor([[2]]) # BOS
    dec_outputs_pt_0 = hf_model(
        input_ids=input_ids_pt,
        decoder_input_ids=decoder_input_ids_pt,
        attention_mask=attention_mask_pt,
        encoder_outputs=enc_outputs_pt,
        use_cache=True,
    )
    logits_pt_0 = dec_outputs_pt_0.logits
    past_key_values_pt = dec_outputs_pt_0.past_key_values

# MLX Prefill
encoder_outputs_mx = mlx_model.model.encoder(
    input_ids=input_ids_mx,
    attention_mask=attention_mask_mx,
)
decoder_input_ids_mx = mx.array([[2]])
cache_mx = DecoderCache(mlx_model.config["decoder"]["num_hidden_layers"])

logits_mx_0, _ = mlx_model(
    input_ids=input_ids_mx,
    decoder_input_ids=decoder_input_ids_mx,
    attention_mask=attention_mask_mx,
    past_key_values=cache_mx,
    encoder_outputs=encoder_outputs_mx,
)

print(f"Decoder MLX logits top 5 (Step 0): {np.argsort(-np.array(logits_mx_0[0, -1].astype(mx.float32)))[:5].tolist()}")
print(f"Decoder HF logits top 5 (Step 0): {logits_pt_0[0, -1].float().topk(5).indices.tolist()}")

# STEP 1
print("\n--- STEP 1 (Generation) ---")
# Let's say both select token 262139
next_token = 262139

# PyTorch Decode Step 1
with torch.no_grad():
    decoder_input_ids_pt_1 = torch.tensor([[next_token]])
    dec_outputs_pt_1 = hf_model(
        input_ids=input_ids_pt,
        decoder_input_ids=decoder_input_ids_pt_1,
        attention_mask=attention_mask_pt,
        encoder_outputs=enc_outputs_pt,
        past_key_values=past_key_values_pt,
        use_cache=True,
    )
    logits_pt_1 = dec_outputs_pt_1.logits

# MLX Decode Step 1
decoder_input_ids_mx_1 = mx.array([[next_token]])
logits_mx_1, _ = mlx_model(
    input_ids=input_ids_mx,
    decoder_input_ids=decoder_input_ids_mx_1,
    attention_mask=attention_mask_mx,
    past_key_values=cache_mx,
    encoder_outputs=encoder_outputs_mx,
)

print(f"Decoder MLX logits top 5 (Step 1): {np.argsort(-np.array(logits_mx_1[0, -1].astype(mx.float32)))[:5].tolist()}")
print(f"Decoder HF logits top 5 (Step 1): {logits_pt_1[0, -1].float().topk(5).indices.tolist()}")
