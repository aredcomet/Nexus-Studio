import torch
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration

model_id = "models/t5gemma-2-270m-270m"
processor = AutoProcessor.from_pretrained(model_id)
hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu")

prompt = "Translate from English to French: Today is a beautiful day to learn programming."
inputs = processor(text=prompt, return_tensors="pt")
input_ids_pt = inputs["input_ids"]
attention_mask_pt = inputs["attention_mask"]

with torch.no_grad():
    enc_outputs = hf_model.model.encoder(input_ids=input_ids_pt, attention_mask=attention_mask_pt)
    
    # Run a few steps to populate the cache
    decoder_input_ids = torch.tensor([[2]]) # BOS
    outputs = hf_model(
        input_ids=input_ids_pt,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask_pt,
        encoder_outputs=enc_outputs,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    sac = past_key_values.self_attention_cache
    
    print("sac.layers len:", len(sac.layers))
    layer0 = sac.layers[0]
    print("Type of sac.layers[0]:", type(layer0))
    print("Attributes of sac.layers[0]:")
    for attr in dir(layer0):
        if not attr.startswith("_"):
            try:
                val = getattr(layer0, attr)
                print(f"  {attr}: type={type(val)}")
                if isinstance(val, (torch.Tensor, list, tuple)):
                    print(f"    shape/len: {val.shape if hasattr(val, 'shape') else len(val)}")
            except Exception as e:
                print(f"  {attr}: error={e}")
