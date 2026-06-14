import argparse
import json
import os
import time
import math
import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
from PIL import Image
from transformers import AutoProcessor
from model import T5Gemma2ForConditionalGeneration, DecoderCache

# LoRA Linear Wrapper in MLX
class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: float = 16, lora_dropout: float = 0.0):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / r
        
        in_features = linear_layer.weight.shape[1]
        out_features = linear_layer.weight.shape[0]
        
        # LoRA weights
        self.lora_a = mx.random.normal((in_features, r)) * (1.0 / math.sqrt(in_features))
        self.lora_b = mx.zeros((r, out_features))
        
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else None

    def __call__(self, x: mx.array) -> mx.array:
        # Base projection
        out = self.linear(x)
        
        # LoRA path
        if self.dropout is not None:
            x = self.dropout(x)
        lora_out = (x @ self.lora_a) @ self.lora_b
        
        return out + (self.scale * lora_out)

def apply_lora(model, keys=["q_proj", "v_proj"], rank=8, alpha=16, dropout=0.0):
    """Recursively replaces matching nn.Linear modules with LoRALinear wrappers, handling modules, lists, and dicts."""
    def replace_layers(module, prefix=""):
        if isinstance(module, nn.Module):
            for name, child in list(module.items()):
                child_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Module):
                    if isinstance(child, nn.Linear) and any(k in child_name for k in keys):
                        setattr(module, name, LoRALinear(child, rank, alpha, dropout))
                        print(f"Applied LoRA to: {child_name}")
                    else:
                        replace_layers(child, child_name)
                elif isinstance(child, list):
                    for idx, item in enumerate(child):
                        if isinstance(item, nn.Module):
                            item_name = f"{child_name}.{idx}"
                            if isinstance(item, nn.Linear) and any(k in item_name for k in keys):
                                child[idx] = LoRALinear(item, rank, alpha, dropout)
                                print(f"Applied LoRA to: {item_name}")
                            else:
                                replace_layers(item, item_name)
                elif isinstance(child, dict):
                    for k, item in list(child.items()):
                        if isinstance(item, nn.Module):
                            item_name = f"{child_name}.{k}"
                            if isinstance(item, nn.Linear) and any(k_in in item_name for k_in in keys):
                                child[k] = LoRALinear(item, rank, alpha, dropout)
                                print(f"Applied LoRA to: {item_name}")
                            else:
                                replace_layers(item, item_name)
    replace_layers(model)

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def collate_fn(batch, processor, max_seq_length=512):
    encoder_inputs = []
    decoder_inputs = []
    labels = []
    
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    eos_id = processor.tokenizer.eos_token_id if processor.tokenizer.eos_token_id is not None else 1
    bos_id = processor.tokenizer.bos_token_id if processor.tokenizer.bos_token_id is not None else 2
    
    for item in batch:
        user_msg = item["messages"][0]["content"]
        assistant_msg = item["messages"][1]["content"]
        
        # Tokenize user prompt
        enc_toks = processor.tokenizer.encode(user_msg)
        # Tokenize assistant response
        dec_toks = processor.tokenizer.encode(assistant_msg)
        
        # Encoder input: prompt + EOS
        enc_input = enc_toks + [eos_id]
        
        # Decoder input: BOS + response
        dec_input = [bos_id] + dec_toks
        
        # Target labels: response + EOS
        label = dec_toks + [eos_id]
        
        # Clip to maximum sequence length
        if len(enc_input) > max_seq_length:
            enc_input = enc_input[:max_seq_length]
        if len(dec_input) > max_seq_length:
            dec_input = dec_input[:max_seq_length]
            label = label[:max_seq_length]
            
        encoder_inputs.append(enc_input)
        decoder_inputs.append(dec_input)
        labels.append(label)
        
    # Find max lengths in this batch
    max_enc_len = max(len(x) for x in encoder_inputs)
    max_dec_len = max(len(x) for x in decoder_inputs)
    
    padded_enc = []
    padded_mask = []
    padded_dec = []
    padded_labels = []
    
    for enc, dec, lab in zip(encoder_inputs, decoder_inputs, labels):
        # Pad encoder
        enc_pad_len = max_enc_len - len(enc)
        padded_enc.append(enc + [pad_id] * enc_pad_len)
        padded_mask.append([1] * len(enc) + [0] * enc_pad_len)
        
        # Pad decoder and labels
        dec_pad_len = max_dec_len - len(dec)
        padded_dec.append(dec + [pad_id] * dec_pad_len)
        padded_labels.append(lab + [-100] * dec_pad_len)
        
    return {
        "input_ids": mx.array(padded_enc),
        "attention_mask": mx.array(padded_mask),
        "decoder_input_ids": mx.array(padded_dec),
        "labels": mx.array(padded_labels),
    }

def loss_fn(model, batch):
    logits, _ = model(
        input_ids=batch["input_ids"],
        decoder_input_ids=batch["decoder_input_ids"],
        attention_mask=batch["attention_mask"],
    )
    labels = batch["labels"]
    mask = labels >= 0
    
    V = logits.shape[-1]
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)
    
    loss = nn.losses.cross_entropy(flat_logits, flat_labels, reduction="none")
    flat_mask = mask.reshape(-1)
    loss = mx.where(flat_mask, loss, 0.0)
    
    return mx.sum(loss) / mx.maximum(mx.sum(flat_mask.astype(mx.float32)), 1.0)

def evaluate(model, val_data, processor, batch_size, max_seq_length):
    model.eval()
    total_loss = 0.0
    steps = 0
    
    # Shuffle validation
    random.shuffle(val_data)
    
    for i in range(0, len(val_data), batch_size):
        batch_samples = val_data[i : i + batch_size]
        batch = collate_fn(batch_samples, processor, max_seq_length)
        
        loss = loss_fn(model, batch)
        total_loss += loss.item()
        steps += 1
        
    model.train()
    return total_loss / steps if steps > 0 else 0.0

def train(args):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.processor)
    
    print(f"Loading base model config from {args.config}...")
    with open(args.config, "r") as f:
        config = json.load(f)
        
    print("Initializing model...")
    model = T5Gemma2ForConditionalGeneration(config)
    
    print(f"Loading baseline weights from {args.weights}...")
    model.load_weights(args.weights)
    
    # Apply LoRA layers
    print("\nApplying LoRA to target modules...")
    apply_lora(
        model, 
        keys=args.lora_keys, 
        rank=args.lora_rank, 
        alpha=args.lora_alpha, 
        dropout=args.lora_dropout
    )
    
    # Freeze baseline parameters and unfreeze LoRA layers
    model.freeze()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze()
            module.linear.freeze()
            
    # Count trainable parameters
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"\nTrainable Parameters (LoRA): {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)")
    
    # Load datasets
    print(f"\nLoading data from {args.data}...")
    train_data = load_jsonl(os.path.join(args.data, "train.jsonl"))
    val_data = load_jsonl(os.path.join(args.data, "valid.jsonl"))
    print(f"Loaded {len(train_data)} train rows, {len(val_data)} validation rows.")
    
    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)
    
    # Loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    print("\nStarting training loop...")
    model.train()
    
    train_idx = 0
    accumulated_grads = None
    accumulated_loss = 0.0
    
    # Shuffle training set
    random.shuffle(train_data)
    
    step = 0
    report_loss = 0.0
    start_time = time.perf_counter()
    
    for it in range(args.iters):
        # Gradient Accumulation Loop
        for micro_step in range(args.grad_accumulation_steps):
            # Fetch samples
            if train_idx + args.batch_size > len(train_data):
                # Epoch boundary
                random.shuffle(train_data)
                train_idx = 0
                
            samples = train_data[train_idx : train_idx + args.batch_size]
            train_idx += args.batch_size
            
            batch = collate_fn(samples, processor, args.max_seq_length)
            
            loss, grads = loss_and_grad_fn(model, batch)
            
            # Scale gradients and loss for accumulation
            loss = loss / args.grad_accumulation_steps
            grads = tree_map(lambda g: g / args.grad_accumulation_steps, grads)
            
            accumulated_loss += loss.item()
            
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda x, y: x + y, accumulated_grads, grads)
                
        # Optimize step
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)
        
        step += 1
        report_loss += accumulated_loss
        
        # Reset accumulation states
        accumulated_grads = None
        accumulated_loss = 0.0
        
        # Report progress
        if step % args.steps_per_report == 0:
            elapsed = time.perf_counter() - start_time
            avg_loss = report_loss / args.steps_per_report
            print(f"Iteration {step:4d} | Train Loss: {avg_loss:.4f} | {args.steps_per_report / elapsed:.3f} steps/sec")
            report_loss = 0.0
            start_time = time.perf_counter()
            
        # Evaluate
        if step % args.steps_per_eval == 0 and len(val_data) > 0:
            val_loss = evaluate(model, val_data, processor, args.batch_size, args.max_seq_length)
            print(f"[*] Iteration {step:4d} | Validation Loss: {val_loss:.4f}")
            
        # Save adapter checkpoint
        if step % args.save_every == 0:
            os.makedirs(args.adapter_path, exist_ok=True)
            adapter_file = os.path.join(args.adapter_path, "adapter_model.safetensors")
            
            # Flatten parameter dictionary to save
            lora_dict = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(adapter_file, lora_dict)
            
            # Save configuration
            config_file = os.path.join(args.adapter_path, "adapter_config.json")
            with open(config_file, "w") as f:
                json.dump({
                    "lora_keys": args.lora_keys,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout
                }, f, indent=2)
            print(f"[+] Saved LoRA adapters and config to {args.adapter_path}")
            
    # Save final adapter checkpoint
    os.makedirs(args.adapter_path, exist_ok=True)
    adapter_file = os.path.join(args.adapter_path, "adapter_model.safetensors")
    lora_dict = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(adapter_file, lora_dict)
    
    # Save configuration
    config_file = os.path.join(args.adapter_path, "adapter_config.json")
    with open(config_file, "w") as f:
        json.dump({
            "lora_keys": args.lora_keys,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout
        }, f, indent=2)
    print(f"[+] Finished training! Saved final adapters and config to {args.adapter_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for T5Gemma-2 models in MLX")
    
    # Model and Data Paths
    parser.add_argument("--config", type=str, default="models/t5gemma-2-270m-270m/config.json", help="Path to baseline config.json")
    parser.add_argument("--weights", type=str, default="weights/t5gemma-2-270m-270m/weights.safetensors", help="Path to baseline weights.safetensors")
    parser.add_argument("--processor", type=str, default="models/t5gemma-2-270m-270m", help="Path to baseline processor/tokenizer")
    parser.add_argument("--data", type=str, default="data/mlx_data", help="Directory containing train.jsonl and valid.jsonl")
    
    # LoRA Hyperparameters
    parser.add_argument("--lora-keys", type=str, nargs="+", default=["self_attn.q_proj", "self_attn.v_proj"], help="Module names to apply LoRA to")
    parser.add_argument("--lora-rank", type=int, default=16, help="Rank of LoRA projections")
    parser.add_argument("--lora-alpha", type=int, default=32, help="Alpha scale of LoRA projections")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout rate inside LoRA layers")
    
    # Optimization Hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Mini-batch size")
    parser.add_argument("--iters", type=int, default=1000, help="Number of steps (weight updates)")
    parser.add_argument("--grad-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    
    # Reporting and Saving
    parser.add_argument("--steps-per-report", type=int, default=10, help="Print training stats every N steps")
    parser.add_argument("--steps-per-eval", type=int, default=100, help="Calculate validation loss every N steps")
    parser.add_argument("--save-every", type=int, default=200, help="Save adapters every N steps")
    parser.add_argument("--adapter-path", type=str, default="adapters/t5gemma2-270m", help="Directory to save adapters")
    
    args = parser.parse_args()
    
    train(args)
