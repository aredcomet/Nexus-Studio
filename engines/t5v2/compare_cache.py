import torch
import mlx.core as mx
import numpy as np
import json
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel
from model import DecoderCache

model_id = "models/t5gemma-2-270m-270m"
weights_path = "weights/t5gemma-2-270m-270m/weights.safetensors"

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

for step in range(5):
    print(f"\n=== STEP {step} ===")
    
    # Run PyTorch
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
        
    # Run MLX
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
    
    # Compare top 5
    logits_mx_np = np.array(logits_mx.astype(mx.float32)[0])
    top5_mx = np.argsort(-logits_mx_np)[:5].tolist()
    
    logits_pt_np = logits_pt[0].float().numpy()
    top5_pt = np.argsort(-logits_pt_np)[:5].tolist()
    
    print("PyTorch Top 5:")
    for tok in top5_pt:
        print(f"  Token {tok}: logit = {logits_pt_np[tok]} ({repr(processor.decode([tok]))})")
        
    print("MLX Top 5:")
    for tok in top5_mx:
        print(f"  Token {tok}: logit = {logits_mx_np[tok]} ({repr(processor.decode([tok]))})")
        
    # Check if the token logits for top PyTorch token match
    top_tok = top5_pt[0]
    print(f"Top PyTorch token ({top_tok}) logit - PT: {logits_pt_np[top_tok]}, MLX: {logits_mx_np[top_tok]}, diff: {abs(logits_pt_np[top_tok] - logits_mx_np[top_tok])}")
