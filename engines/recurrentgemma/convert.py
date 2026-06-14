import os
import glob
import argparse
import mlx.core as mx

def convert_weights(model_path, output_path):
    # Check if model_path is a directory (sharded weights) or a single file
    if os.path.isdir(model_path):
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensors_files:
            raise ValueError(f"No .safetensors files found in directory {model_path}")
        print(f"Loading sharded weights from directory: {model_path} ({len(safetensors_files)} shards)...")
        weights = {}
        for f in sorted(safetensors_files):
            print(f"Loading shard: {f}")
            weights.update(mx.load(f))
    else:
        print(f"Loading weights from single file: {model_path}...")
        weights = mx.load(model_path)
            
    print("Mapping weights to MLX format...")
    mlx_weights = {}
    
    for k, v in weights.items():
        # Transpose Conv1D weights from PyTorch shape [lru_width, 1, 4] to MLX shape [lru_width, 4, 1]
        if "conv_1d.weight" in k:
            v_orig_shape = v.shape
            v = mx.transpose(v, (0, 2, 1))
            print(f"Transposed {k} from PyTorch shape {v_orig_shape} to MLX shape {v.shape}")
            
        mlx_weights[k] = v

    # Tie embeddings explicitly: lm_head.weight = model.embed_tokens.weight
    if "model.embed_tokens.weight" in mlx_weights:
        print("Tying word embeddings explicitly (lm_head.weight)...")
        mlx_weights["lm_head.weight"] = mlx_weights["model.embed_tokens.weight"]

    print(f"Saving MLX weights to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mx.save_safetensors(output_path, mlx_weights)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RecurrentGemma weights to MLX format")
    parser.add_argument("--model-path", type=str, default="models/recurrentgemma-2-2b-it", help="Path to PyTorch model dir or file")
    parser.add_argument("--output-path", type=str, default="weights/recurrentgemma-2b-it/weights.safetensors", help="Path to save MLX weights")
    args = parser.parse_args()
    
    convert_weights(args.model_path, args.output_path)
