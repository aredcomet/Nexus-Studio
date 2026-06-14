# T5Gemma-2 MLX Explorer

This directory contains custom scripts to locally load, run, and verify Google's `T5Gemma-2` encoder-decoder models using the MLX framework.

## File Structure

* **`model.py`**: Implementation of the complete `T5Gemma2` model, including multi-modal layers, rotary embeddings, merged self/cross-attention blocks, and precision-promoted RMSNorm layers.
* **`convert.py`**: Weight converter that maps Hugging Face PyTorch weights into MLX safetensors, transposing Conv2D layers and tying word embeddings explicitly for easy loading.
* **`generate.py`**: Autoregressive decoding stream generator using Hugging Face's `AutoProcessor` and our custom MLX model.
* **`compare_loop.py`**: Step-by-step verification script comparing MLX token probabilities and logit output against the PyTorch Hugging Face baseline.

---

## Getting Started

### 1. Requirements
Ensure your local Python environment is activated and has the required packages installed:
```bash
# Verify using the local interpreter
.venv/bin/python -c "import mlx.core; import transformers"
```

### 2. Converting weights
Before generating text, map the downloaded PyTorch safetensors into the MLX format:
```bash
.venv/bin/python t5v2/convert.py \
  --model-path models/t5gemma-2-270m-270m/model.safetensors \
  --output-path weights/t5gemma-2-270m-270m/weights.safetensors
```

### 3. Running generation
Generate tokens autoregressively from a text prompt:
```bash
.venv/bin/python t5v2/generate.py \
  --config models/t5gemma-2-270m-270m/config.json \
  --weights weights/t5gemma-2-270m-270m/weights.safetensors \
  --processor models/t5gemma-2-270m-270m \
  --prompt "Translate from English to French: Today is a beautiful day to learn programming." \
  --max-tokens 50
```

### 4. Running Verification
To confirm that your local MLX model's outputs align perfectly with the Hugging Face PyTorch baseline:
```bash
PYTHONPATH=t5v2 .venv/bin/python t5v2/compare_loop.py
```

---

## LoRA Fine-Tuning

You can perform parameter-efficient fine-tuning (LoRA) on your custom dataset using the root-level `train_t5v2.py` script. The script is configured via a YAML file, similar to how standard `mlx_lm` models are trained.

### 1. Training configuration
Create or edit your training config (e.g., [configs/t5v2_270m.yaml](file:///Users/bran/src/play/llm/configs/t5v2_270m.yaml)):
```yaml
# Model and Data Paths
model_config: "models/t5gemma-2-270m-270m/config.json"
model_weights: "weights/t5gemma-2-270m-270m/weights.safetensors"
processor: "models/t5gemma-2-270m-270m"
data: "data/mlx_data"

# LoRA Parameters
lora_parameters:
  keys: ["self_attn.q_proj", "self_attn.v_proj"]
  rank: 16
  alpha: 32
  dropout: 0.05

# Training Hyperparameters
batch_size: 1
iters: 1000
grad_accumulation_steps: 4
learning_rate: 2e-5
max_seq_length: 512
```

### 2. Launching LoRA training
To start the training loop using the configuration file:
```bash
.venv/bin/python train_t5v2.py --config configs/t5v2_270m.yaml
```

You can also override any configuration parameter directly from the command line:
```bash
.venv/bin/python train_t5v2.py --config configs/t5v2_270m.yaml --iters 500 --lr 1e-5
```

### 3. Evaluating trained adapters
To run generation using the base model and your newly trained adapters:
```bash
.venv/bin/python t5v2/generate.py \
  --config models/t5gemma-2-270m-270m/config.json \
  --weights weights/t5gemma-2-270m-270m/weights.safetensors \
  --processor models/t5gemma-2-270m-270m \
  --adapter-path adapters/t5gemma2-270m \
  --prompt "Translate from English to French: Today is a beautiful day to learn programming."
```

