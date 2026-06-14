import torch
import mlx.core as mx
import numpy as np
import json
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel
from model import DecoderCache

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

print("Loading models...")
processor = AutoProcessor.from_pretrained(model_id)
hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu")
hf_model.eval()

with open(f"{model_id}/config.json", "r") as f:
    config = json.load(f)
mlx_model = MLXModel(config)
mlx_model.load_weights(weights_path)

prompt = "Translate from English to French: Today is a beautiful day to learn programming."
inputs = processor(text=prompt, return_tensors="pt")
input_ids_pt = inputs["input_ids"]
attention_mask_pt = inputs["attention_mask"]

input_ids_mx = mx.array(input_ids_pt.numpy())
attention_mask_mx = mx.array(attention_mask_pt.numpy())

# Run Encoder
with torch.no_grad():
    enc_outputs_pt = hf_model.model.encoder(input_ids=input_ids_pt, attention_mask=attention_mask_pt)
encoder_outputs_mx = mlx_model.model.encoder(input_ids=input_ids_mx, attention_mask=attention_mask_mx)

# Autoregressive decoding to step 4
past_key_values_pt = None
decoder_input_ids_pt = torch.tensor([[2]]) # BOS
cache_mx = DecoderCache(mlx_model.config["decoder"]["num_hidden_layers"])
decoder_input_ids_mx = mx.array([[2]])

# Step 0
with torch.no_grad():
    dec_outputs_pt = hf_model(
        input_ids=input_ids_pt,
        decoder_input_ids=decoder_input_ids_pt,
        attention_mask=attention_mask_pt,
        encoder_outputs=enc_outputs_pt,
        use_cache=True,
    )
    logits_pt = dec_outputs_pt.logits[:, -1, :]
    past_key_values_pt = dec_outputs_pt.past_key_values
    token_pt = torch.argmax(logits_pt, dim=-1).item()

logits_mx, _ = mlx_model(
    input_ids=input_ids_mx,
    decoder_input_ids=decoder_input_ids_mx,
    attention_mask=attention_mask_mx,
    past_key_values=cache_mx,
    encoder_outputs=encoder_outputs_mx,
)
logits_mx = logits_mx[:, -1, :]
token_mx = mx.argmax(logits_mx, axis=-1).item()

# Steps 1 to 4
for step in range(1, 5):
    # PyTorch
    with torch.no_grad():
        decoder_input_ids_pt = torch.tensor([[token_pt]])
        dec_outputs_pt = hf_model(
            input_ids=input_ids_pt,
            decoder_input_ids=decoder_input_ids_pt,
            attention_mask=attention_mask_pt,
            encoder_outputs=enc_outputs_pt,
            past_key_values=past_key_values_pt,
            use_cache=True,
        )
        logits_pt = dec_outputs_pt.logits[:, -1, :]
        past_key_values_pt = dec_outputs_pt.past_key_values
        token_pt = torch.argmax(logits_pt, dim=-1).item()
        
    # MLX
    decoder_input_ids_mx = mx.array([[token_mx]])
    logits_mx, _ = mlx_model(
        input_ids=input_ids_mx,
        decoder_input_ids=decoder_input_ids_mx,
        attention_mask=attention_mask_mx,
        past_key_values=cache_mx,
        encoder_outputs=encoder_outputs_mx,
    )
    logits_mx = logits_mx[:, -1, :]
    token_mx = mx.argmax(logits_mx, axis=-1).item()
    
    if step == 4:
        print(f"\n--- STEP 4 DETAIL ---")
        # Logits for 7830 and 5422
        pt_7830 = logits_pt[0, 7830].item()
        pt_5422 = logits_pt[0, 5422].item()
        mx_7830 = logits_mx[0, 7830].item()
        mx_5422 = logits_mx[0, 5422].item()
        
        print(f"PT logit French (7830): {pt_7830}")
        print(f"PT logit English (5422): {pt_5422}")
        print(f"MLX logit French (7830): {mx_7830}")
        print(f"MLX logit English (5422): {mx_5422}")
        
        print(f"PT argmax token: {token_pt}")
        print(f"MLX argmax token: {token_mx}")
        
        # NumPy argsort top 5
        logits_mx_np = np.array(logits_mx.astype(mx.float32)[0])
        top5_mx_np = np.argsort(-logits_mx_np)[:5].tolist()
        print(f"MLX NumPy sorted top 5: {top5_mx_np}")
        for tok in top5_mx_np:
            print(f"  Token {tok}: logit = {logits_mx_np[tok]} ({repr(processor.decode([tok]))})")
            
        logits_pt_np = logits_pt[0].float().numpy()
        top5_pt_np = np.argsort(-logits_pt_np)[:5].tolist()
        print(f"PT NumPy sorted top 5: {top5_pt_np}")
        for tok in top5_pt_np:
            print(f"  Token {tok}: logit = {logits_pt_np[tok]} ({repr(processor.decode([tok]))})")
