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

past_key_values_pt = None
decoder_input_ids_pt = torch.tensor([[2]]) # BOS
cache_mx = DecoderCache(mlx_model.config["decoder"]["num_hidden_layers"])
decoder_input_ids_mx = mx.array([[2]])

token_pt = 2
token_mx = 2

for step in range(3):
    # PyTorch
    with torch.no_grad():
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
        decoder_input_ids_pt = torch.tensor([[token_pt]])
        
    # MLX
    logits_mx, _ = mlx_model(
        input_ids=input_ids_mx,
        decoder_input_ids=decoder_input_ids_mx,
        attention_mask=attention_mask_mx,
        past_key_values=cache_mx,
        encoder_outputs=encoder_outputs_mx,
    )
    logits_mx = logits_mx[:, -1, :]
    token_mx = mx.argmax(logits_mx, axis=-1).item()
    decoder_input_ids_mx = mx.array([[token_mx]])
    
    if step == 2:
        print("\n=== STEP 2 DETAIL ===")
        print(f"PT argmax token: {token_pt} ({repr(processor.decode([token_pt]))})")
        print(f"MLX argmax token: {token_mx} ({repr(processor.decode([token_mx]))})")
        
        # Logits of 40414, 236777, 262138
        for tok in [40414, 236777, 262138]:
            print(f"Token {tok} ({repr(processor.decode([tok]))}) logit - PT: {logits_pt[0, tok].item()}, MLX: {logits_mx[0, tok].item()}")
            
        # Top 5 NumPy
        logits_mx_np = np.array(logits_mx.astype(mx.float32)[0])
        top5_mx = np.argsort(-logits_mx_np)[:5].tolist()
        print("\nMLX Top 5 NumPy:")
        for tok in top5_mx:
            print(f"  Token {tok}: logit = {logits_mx_np[tok]} ({repr(processor.decode([tok]))})")
