import os
import subprocess
import sys

def run_training(model_size="1b"):
    config_file = f"configs/{model_size}.yaml"
    
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found.")
        return

    print(f"Starting MLX LoRA training using {config_file}...")
    
    # Launch training using the YAML config file
    command = [
        "python", "-m", "mlx_lm", "lora",
        "--config", config_file
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
    except FileNotFoundError:
        print("Error: 'python' or 'mlx_lm' module not found. Ensure your .venv is active.")

if __name__ == "__main__":
    # Allow specifying model size as an argument, default to 1b
    size = sys.argv[1] if len(sys.argv) > 1 else "1b"
    
    # Ensure data is converted first
    if not os.path.exists(os.path.join("data/mlx_data", "train.jsonl")):
        print("Data not found. Please run 'python convert_data.py' first.")
    else:
        run_training(size)
