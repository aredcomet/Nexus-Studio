import argparse
import json
import os
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten

from model import ModelConfig, RecurrentGemmaForCausalLM

def quantize_model(model, group_size=64, bits=4, mode="affine"):
    """Recursively replaces nn.Linear layers in the model with nn.QuantizedLinear layers."""
    def replace_layers(module, prefix=""):
        for name, child in list(module.items()):
            child_name = f"{prefix}.{name}" if prefix else name
            
            # Skip embed_tokens and lm_head to preserve quality
            if isinstance(child, nn.Linear):
                if name in ["embed_tokens", "lm_head"] or "embed_tokens" in child_name or "lm_head" in child_name:
                    continue
                # Replace with quantized linear
                quantized_layer = nn.QuantizedLinear.from_linear(child, group_size=group_size, bits=bits, mode=mode)
                setattr(module, name, quantized_layer)
                print(f"Quantized layer: {child_name} ({bits}-bit, group_size={group_size}, mode={mode})")
            elif isinstance(child, nn.Module):
                replace_layers(child, child_name)
            elif isinstance(child, list):
                for idx, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        item_name = f"{child_name}.{idx}"
                        replace_layers(item, item_name)
    replace_layers(model)

def main():
    parser = argparse.ArgumentParser(description="Quantize RecurrentGemma model weights to 4-bit or 8-bit in MLX")
    parser.add_argument("--config", type=str, default="models/recurrentgemma-2b-it/config.json", help="Path to config.json")
    parser.add_argument("--weights", type=str, default="weights/recurrentgemma-2b-it/weights.safetensors", help="Path to baseline float16/bfloat16 weights.safetensors")
    parser.add_argument("--output", type=str, default="weights/recurrentgemma-2b-it-4bit/weights.safetensors", help="Path to save quantized weights")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8], help="Number of quantization bits")
    parser.add_argument("--group-size", type=int, default=64, choices=[16, 32, 64, 128], help="Quantization group size")
    parser.add_argument("--mode", type=str, default="affine", choices=["affine", "mxfp4", "mxfp8", "nvfp4"], help="Quantization mode")
    
    args = parser.parse_args()
    
    # Enforce spec requirements for different modes
    if args.mode == "mxfp4":
        args.group_size = 32
        args.bits = 4
        if args.output == "weights/recurrentgemma-2b-it-4bit/weights.safetensors":
            args.output = "weights/recurrentgemma-2b-it-mxfp4/weights.safetensors"
    elif args.mode == "mxfp8":
        args.group_size = 32
        args.bits = 8
        if args.output == "weights/recurrentgemma-2b-it-4bit/weights.safetensors":
            args.output = "weights/recurrentgemma-2b-it-mxfp8/weights.safetensors"
    elif args.mode == "nvfp4":
        args.group_size = 16
        args.bits = 4
        if args.output == "weights/recurrentgemma-2b-it-4bit/weights.safetensors":
            args.output = "weights/recurrentgemma-2b-it-nvfp4/weights.safetensors"
            
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        return
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found. Run weight conversion first.")
        return
        
    print(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    config = ModelConfig(**config_dict)
    
    print("Initializing model in float16/bfloat16...")
    model = RecurrentGemmaForCausalLM(config)
    
    print(f"Loading baseline weights from {args.weights}...")
    if os.path.isdir(args.weights):
        import glob
        weights = {}
        for f in glob.glob(os.path.join(args.weights, "*.safetensors")):
            weights.update(mx.load(f))
    else:
        weights = mx.load(args.weights)
    dtype = mx.bfloat16 if config_dict.get("torch_dtype") == "bfloat16" else mx.float16
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    
    print(f"\nQuantizing model layers in '{args.mode}' mode...")
    quantize_model(model, group_size=args.group_size, bits=args.bits, mode=args.mode)
    mx.eval(model.parameters())
    
    print(f"\nSaving quantized weights to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Flatten parameters to save
    quantized_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(args.output, quantized_weights)
    print("Quantization complete!")

if __name__ == "__main__":
    main()
