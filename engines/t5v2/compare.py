import torch
import mlx.core as mx
import numpy as np
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from model import T5Gemma2ForConditionalGeneration as MLXModel

model_id = "models/t5gemma-2-270m-270m"
weights_path = "t5v2/weights.safetensors"

print("1. Loading HF and MLX models...")
processor = AutoProcessor.from_pretrained(model_id)

hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"
)
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

print(f"Input IDs: {input_ids_pt.tolist()}")

# Compare Embeddings
print("\n2. Comparing encoder embeddings...")
with torch.no_grad():
    embeds_pt = hf_model.model.encoder.text_model.embed_tokens(input_ids_pt)
embeds_mx = mlx_model.model.encoder.text_model.embed_tokens(input_ids_mx)

diff_embeds = np.abs(embeds_pt.float().numpy() - np.array(embeds_mx.astype(mx.float32)))
print(f"Embeddings max diff: {diff_embeds.max():.2e}")

# Compare Encoder Outputs
print("\n3. Comparing encoder outputs (after all layers)...")
with torch.no_grad():
    enc_outputs_pt = hf_model.model.encoder(
        input_ids=input_ids_pt,
        attention_mask=attention_mask_pt,
    ).last_hidden_state
enc_outputs_mx = mlx_model.model.encoder(
    input_ids=input_ids_mx,
    attention_mask=attention_mask_mx,
)

diff_enc = np.abs(enc_outputs_pt.float().numpy() - np.array(enc_outputs_mx.astype(mx.float32)))
print(f"Encoder outputs max diff: {diff_enc.max():.2e}")

# If encoder matches, compare Decoder Outputs
print("\n4. Comparing decoder outputs (first step)...")
decoder_input_ids_pt = torch.tensor([[2]]) # BOS
decoder_input_ids_mx = mx.array([[2]])

with torch.no_grad():
    dec_outputs_pt = hf_model(
        input_ids=input_ids_pt,
        decoder_input_ids=decoder_input_ids_pt,
        attention_mask=attention_mask_pt,
        encoder_outputs=None,
    ).logits
    
dec_outputs_mx, _ = mlx_model(
    input_ids=input_ids_mx,
    decoder_input_ids=decoder_input_ids_mx,
    attention_mask=attention_mask_mx,
    encoder_outputs=enc_outputs_mx,
)

diff_dec = np.abs(dec_outputs_pt.float().numpy() - np.array(dec_outputs_mx.astype(mx.float32)))
print(f"Decoder logits max diff: {diff_dec.max():.2e}")
print(f"Decoder HF logits top 5: {dec_outputs_pt[0, -1].float().topk(5).indices.tolist()}")
print(f"Decoder MLX logits top 5: {np.argsort(-np.array(dec_outputs_mx[0, -1].astype(mx.float32)))[:5].tolist()}")
