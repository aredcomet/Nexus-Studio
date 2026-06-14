import torch
import mlx.core as mx
import numpy as np
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
    import json
    config = json.load(f)
mlx_model = MLXModel(config)
mlx_model.load_weights(weights_path)

prompt = "Translate from English to French: Today is a beautiful day to learn programming."
inputs = processor(text=prompt, return_tensors="pt")
input_ids_pt = inputs["input_ids"]
attention_mask_pt = inputs["attention_mask"]

input_ids_mx = mx.array(input_ids_pt.numpy())
attention_mask_mx = mx.array(attention_mask_pt.numpy())

# 1. Prefill Encoder
print("\nRunning Encoder...")
with torch.no_grad():
    enc_outputs_pt = hf_model.model.encoder(input_ids=input_ids_pt, attention_mask=attention_mask_pt)
    
encoder_outputs_mx = mlx_model.model.encoder(input_ids=input_ids_mx, attention_mask=attention_mask_mx)

# 2. Prefill Decoder (Step 0)
print("\n--- STEP 0 (Prefill) ---")
with torch.no_grad():
    decoder_input_ids_pt = torch.tensor([[2]]) # BOS
    dec_outputs_pt = hf_model(
        input_ids=input_ids_pt,
        decoder_input_ids=decoder_input_ids_pt,
        attention_mask=attention_mask_pt,
        encoder_outputs=enc_outputs_pt,
        use_cache=True,
    )
    logits_pt = dec_outputs_pt.logits[:, -1, :]
    past_key_values_pt = dec_outputs_pt.past_key_values

decoder_input_ids_mx = mx.array([[2]])
cache_mx = DecoderCache(mlx_model.config["decoder"]["num_hidden_layers"])
logits_mx, _ = mlx_model(
    input_ids=input_ids_mx,
    decoder_input_ids=decoder_input_ids_mx,
    attention_mask=attention_mask_mx,
    past_key_values=cache_mx,
    encoder_outputs=encoder_outputs_mx,
)
logits_mx = logits_mx[:, -1, :]

token_pt = torch.argmax(logits_pt, dim=-1).item()
token_mx = mx.argmax(logits_mx, axis=-1).item()

print(f"HF step 0 token: {token_pt} ({repr(processor.decode([token_pt]))})")
print(f"MLX step 0 token: {token_mx} ({repr(processor.decode([token_mx]))})")

# 3. Generation Loop
for step in range(1, 10):
    print(f"\n--- STEP {step} ---")
    
    # Run PyTorch
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
        
    # Run MLX
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
    
    print(f"HF step {step} token: {token_pt} ({repr(processor.decode([token_pt]))})")
    print(f"MLX step {step} token: {token_mx} ({repr(processor.decode([token_mx]))})")
    
    # Print top 3 logits comparison
    top3_pt = logits_pt.float().topk(3).indices[0].tolist()
    top3_mx = np.argsort(-np.array(logits_mx.astype(mx.float32))[0])[:3].tolist()
    print(f"HF top 3: {top3_pt}")
    print(f"MLX top 3: {top3_mx}")
    
    if token_pt != token_mx:
        print("DIVERGENCE FOUND!")
        break
