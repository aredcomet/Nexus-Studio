import os
import sys
import yaml
import argparse

# Add the t5v2 directory to the path so that imports like 'from model import ...' work
t5v2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "engines", "t5v2"))
sys.path.append(t5v2_dir)

from train import train as run_lora_train

class TrainingArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    parser = argparse.ArgumentParser(description="T5Gemma-2 LoRA training runner using YAML config")
    
    # Config file argument
    parser.add_argument("--config", type=str, default="configs/t5v2_270m.yaml", help="Path to YAML training configuration file")
    
    # Overrides for any YAML parameter
    parser.add_argument("--data", type=str, help="Override training data directory")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--iters", type=int, help="Override training iterations")
    parser.add_argument("--grad-accumulation-steps", type=int, help="Override gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length")
    parser.add_argument("--steps-per-report", type=int, help="Override stats printing interval (iterations)")
    parser.add_argument("--steps-per-eval", type=int, help="Override validation interval (iterations)")
    parser.add_argument("--save-every", type=int, help="Override save interval (iterations)")
    parser.add_argument("--adapter-path", type=str, help="Override path to save adapters")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
        
    print(f"Loading config from {args.config}...")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Map YAML structure to train.py namespace arguments
    lora_params = cfg.get("lora_parameters", {})
    
    mapped_args = {
        "config": cfg.get("model_config", "models/t5gemma-2-270m-270m/config.json"),
        "weights": cfg.get("model_weights", "weights/t5gemma-2-270m-270m/weights.safetensors"),
        "processor": cfg.get("processor", "models/t5gemma-2-270m-270m"),
        "data": args.data if args.data is not None else cfg.get("data", "data/mlx_data"),
        "lora_keys": lora_params.get("keys", ["self_attn.q_proj", "self_attn.v_proj"]),
        "lora_rank": int(lora_params.get("rank", 16)),
        "lora_alpha": float(lora_params.get("alpha", 32)),
        "lora_dropout": float(lora_params.get("dropout", 0.05)),
        "lr": float(args.lr if args.lr is not None else cfg.get("learning_rate", 2e-5)),
        "batch_size": int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1)),
        "iters": int(args.iters if args.iters is not None else cfg.get("iters", 1000)),
        "grad_accumulation_steps": int(args.grad_accumulation_steps if args.grad_accumulation_steps is not None else cfg.get("grad_accumulation_steps", 4)),
        "max_seq_length": int(args.max_seq_length if args.max_seq_length is not None else cfg.get("max_seq_length", 512)),
        "steps_per_report": int(args.steps_per_report if args.steps_per_report is not None else cfg.get("steps_per_report", 10)),
        "steps_per_eval": int(args.steps_per_eval if args.steps_per_eval is not None else cfg.get("steps_per_eval", 100)),
        "save_every": int(args.save_every if args.save_every is not None else cfg.get("save_every", 200)),
        "adapter_path": args.adapter_path if args.adapter_path is not None else cfg.get("adapter_path", "adapters/t5gemma2-270m")
    }
    
    # Create the training namespace object
    train_namespace = TrainingArgs(**mapped_args)
    
    # Run the LoRA training
    run_lora_train(train_namespace)

if __name__ == "__main__":
    main()
