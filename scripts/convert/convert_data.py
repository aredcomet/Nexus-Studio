from datasets import load_dataset
import json
import os
import random

def convert_to_messages(item):
    """Normalizes GretelAI into the messages format."""
    instruction = item["sql_prompt"].strip()
    schema = item["sql_context"].strip()
    output = item["sql"].strip()

    unified_prompt = f"Instruction: {instruction}\nSchema/Context: {schema}\nSQL:"
    
    return {
        "messages": [
            {"role": "user", "content": unified_prompt},
            {"role": "assistant", "content": output}
        ]
    }

def process_gretel(dataset_id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading dataset {dataset_id}...")
    ds = load_dataset(dataset_id)
    
    train_data = ds["train"]
    valid_data = ds["test"]
    
    print(f"Converting {len(train_data)} training samples...")
    train_messages = [convert_to_messages(item) for item in train_data]
    
    print(f"Converting {len(valid_data)} validation samples...")
    valid_messages = [convert_to_messages(item) for item in valid_data]

    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        for entry in train_messages:
            f.write(json.dumps(entry) + "\n")
            
    with open(os.path.join(output_dir, "valid.jsonl"), "w") as f:
        for entry in valid_messages:
            f.write(json.dumps(entry) + "\n")

    print(f"Finished processing GretelAI dataset!")
    print(f"Total Train: {len(train_messages)} rows")
    print(f"Total Valid: {len(valid_messages)} rows")

if __name__ == "__main__":
    process_gretel("gretelai/synthetic_text_to_sql", "data/mlx_data")
