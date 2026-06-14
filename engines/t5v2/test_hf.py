import torch
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration

model_id = "models/t5gemma-2-270m-270m"
print("Loading HF model...")
processor = AutoProcessor.from_pretrained(model_id)
model = T5Gemma2ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"
)

prompt = "Translate from English to French: Today is a beautiful day to learn programming."
print("Preprocessing...")
inputs = processor(text=prompt, return_tensors="pt")

print("Generating...")
outputs = model.generate(**inputs, max_new_tokens=30)

print("\nHF Output:")
print(processor.decode(outputs[0]))
