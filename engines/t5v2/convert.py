import argparse
import json
import mlx.core as mx

import os
import glob

def convert_weights(model_path, output_path):
    # Check if model_path is a directory (sharded weights) or a single file
    if os.path.isdir(model_path):
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensors_files:
            raise ValueError(f"No .safetensors files found in directory {model_path}")
        print(f"Loading sharded weights from directory: {model_path} ({len(safetensors_files)} shards)")
        weights = {}
        for f in sorted(safetensors_files):
            print(f"Loading shard: {f}")
            weights.update(mx.load(f))
    else:
        print(f"Loading weights from {model_path} using MLX...")
        weights = mx.load(model_path)
            
    print("Mapping weights to MLX format...")
    mlx_weights = {}
    
    # 1. Map encoder/decoder/projector keys
    for k, v in weights.items():
        new_key = k
        # Map encoder layers prefix to match text_model inside T5Gemma2Encoder
        # PyTorch: model.encoder.layers.X -> MLX: model.encoder.text_model.layers.X
        if k.startswith("model.encoder.layers."):
            parts = k.split(".")
            new_key = "model.encoder.text_model.layers." + ".".join(parts[3:])
            
        elif k == "model.encoder.embed_tokens.weight":
            new_key = "model.encoder.text_model.embed_tokens.weight"
            
        elif k == "model.encoder.embed_tokens.eoi_embedding":
            new_key = "model.encoder.text_model.embed_tokens.eoi_embedding"
            
        elif k == "model.encoder.norm.weight":
            new_key = "model.encoder.text_model.norm.weight"
            
        # Map vision_tower keys
        elif k.startswith("model.encoder.vision_tower.vision_model."):
            parts = k.split(".")
            new_key = "model.encoder.vision_tower." + ".".join(parts[4:])
            
            # Conv2d weights transpose in MLX: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
            if "patch_embedding.weight" in new_key:
                v = mx.transpose(v, (0, 2, 3, 1))
                print(f"Transposed vision patch_embedding weight from PyTorch shape {weights[k].shape} to MLX shape {v.shape}")
                
        # Other keys remain the same (e.g. decoder and projector keys)
        mlx_weights[new_key] = v

    # 2. Tie weights explicitly in the saved file to make loading direct and easy
    if "model.encoder.text_model.embed_tokens.weight" in mlx_weights:
        embed_weight = mlx_weights["model.encoder.text_model.embed_tokens.weight"]
        mlx_weights["model.decoder.embed_tokens.weight"] = embed_weight
        mlx_weights["lm_head.weight"] = embed_weight
        
    if "model.encoder.text_model.embed_tokens.eoi_embedding" in mlx_weights:
        eoi_emb = mlx_weights["model.encoder.text_model.embed_tokens.eoi_embedding"]
        mlx_weights["model.decoder.embed_tokens.eoi_embedding"] = eoi_emb

    print(f"Saving MLX weights to {output_path}...")
    mx.save_safetensors(output_path, mlx_weights)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert T5Gemma2 weights to MLX format")
    parser.add_argument("--model-path", type=str, default="models/t5gemma-2-270m-270m/model.safetensors", help="Path to PyTorch safetensors file")
    parser.add_argument("--output-path", type=str, default="weights/t5gemma-2-270m-270m/weights.safetensors", help="Path to save MLX weights")
    args = parser.parse_args()
    
    convert_weights(args.model_path, args.output_path)
