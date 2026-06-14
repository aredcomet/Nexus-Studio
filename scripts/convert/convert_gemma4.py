import os
import sys
import argparse

# Add workspace root to path
workspace_root = os.path.dirname(os.path.abspath(__file__))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Register the model remapping BEFORE importing mlx_lm modules
import mlx_lm.utils
mlx_lm.utils.MODEL_REMAPPING["gemma4_unified"] = "gemma4"

# Monkeypatch mlx_lm.models.gemma4.Model.sanitize to filter out vision_embedder weights
import mlx_lm.models.gemma4
original_sanitize = mlx_lm.models.gemma4.Model.sanitize

def custom_sanitize(self, weights):
    sanitized_weights = {}
    for k, v in weights.items():
        k_clean = k.removeprefix("model.")
        if k_clean.startswith(
            (
                "vision_tower",
                "multi_modal_projector",
                "audio_tower",
                "embed_audio",
                "embed_vision",
                "vision_embedder",
            )
        ):
            continue
        sanitized_weights[k] = v
    return original_sanitize(self, sanitized_weights)

mlx_lm.models.gemma4.Model.sanitize = custom_sanitize

from mlx_lm import convert

def main():
    parser = argparse.ArgumentParser(description="Quantize Gemma-4-12B-it to mxfp4 for MLX")
    parser.add_argument("--hf-path", type=str, default="models/gemma-4-12B-it", help="Path to Hugging Face model directory")
    parser.add_argument("--mlx-path", type=str, default="weights/gemma-4-12B-it-mxfp4", help="Path to save MLX model weights")
    parser.add_argument("--q-mode", type=str, default="mxfp4", choices=["mxfp4", "mxfp8", "affine"], help="Quantization mode")
    
    args = parser.parse_args()
    
    print(f"Starting quantization for model from {args.hf_path} to {args.mlx_path}...")
    print(f"Using quantization mode: {args.q_mode}")
    
    # Check if target directory already exists, delete if so to overwrite cleanly
    import shutil
    if os.path.exists(args.mlx_path):
        print(f"Target path {args.mlx_path} already exists. Removing it first for clean save...")
        if os.path.isdir(args.mlx_path):
            shutil.rmtree(args.mlx_path)
        else:
            os.remove(args.mlx_path)
            
    try:
        convert(
            hf_path=args.hf_path,
            mlx_path=args.mlx_path,
            quantize=True,
            q_mode=args.q_mode,
            trust_remote_code=True
        )
        print("Quantization successfully completed!")
    except Exception as e:
        print(f"Error during quantization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
