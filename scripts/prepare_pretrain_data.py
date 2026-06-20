import os
import argparse
import random
import json
from datasets import load_dataset

def split_text(text, ratio=0.5):
    """Splits a document text into prompt (context) and response (target)."""
    words = text.split()
    if len(words) < 10:
        return None, None
        
    split_idx = int(len(words) * ratio)
    prompt = " ".join(words[:split_idx])
    response = " ".join(words[split_idx:])
    return prompt, response

def main():
    parser = argparse.ArgumentParser(description="Prepare pre-training data from HF pseudo-mini-pile for T5Gemma-2")
    parser.add_argument("--dataset", type=str, default="iNeil77/pseudo-mini-pile", help="Hugging Face dataset name")
    parser.add_argument("--subset", type=str, default="openwebtext", help="Subset/config of the dataset")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of documents to load")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Ratio of text to allocate to prompt/context")
    parser.add_argument("--output-dir", type=str, default="data/mlx_pretrain_data", help="Directory to save JSONL files")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set split ratio")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading '{args.subset}' subset from Hugging Face dataset '{args.dataset}' (limit: {args.limit})...")
    
    try:
        # Load a streaming or small slice of the dataset to avoid downloading the entire 91B token corpus
        ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    samples = []
    count = 0
    
    for row in ds:
        text = row.get("content", row.get("text", ""))
        if not text:
            continue
            
        prompt, response = split_text(text, args.split_ratio)
        if prompt and response:
            samples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            })
            count += 1
            if count >= args.limit:
                break
                
    if not samples:
        print("No valid samples extracted.")
        return
        
    print(f"Extracted {len(samples)} valid documents. Splitting into train and validation sets...")
    
    # Shuffle and split
    random.shuffle(samples)
    val_count = int(len(samples) * args.val_ratio)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    
    train_file = os.path.join(args.output_dir, "train.jsonl")
    val_file = os.path.join(args.output_dir, "valid.jsonl")
    
    print(f"Saving {len(train_samples)} training samples to {train_file}...")
    with open(train_file, "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saving {len(val_samples)} validation samples to {val_file}...")
    with open(val_file, "w") as f:
        for item in val_samples:
            f.write(json.dumps(item) + "\n")
            
    print("Data preparation complete! You can now run training using:")
    print(f"python train_t5v2.py --config configs/t5v2_270m.yaml --data {args.output_dir}")

if __name__ == "__main__":
    main()
