# RecurrentGemma MLX implementation

This directory contains a custom native implementation of Google's `RecurrentGemma-2b-it` model (based on the Griffin architecture combining linear recurrences with local attention) using the Apple MLX framework.

## File Structure

* **[model.py](file:///Users/bran/src/play/llm/engines/recurrentgemma/model.py)**: Native MLX implementation of the Griffin architecture, including:
  - RG-LRU (Real-Gated Linear Recurrent Unit) layer with inference caching and sequential/parallel scanning.
  - Depthwise causal 1D convolution layer with inference states.
  - Multi-query sliding window attention block with partial rotary embeddings (RoPE) and cache management.
  - RMSNorm (with RecurrentGemma's custom `1.0 + weight` scaling).
  - Gated MLP layer (using tanh-approximated GELU).
  - Causal LM wrapper with logit soft-capping.
* **[convert.py](file:///Users/bran/src/play/llm/engines/recurrentgemma/convert.py)**: Weight converter that maps Hugging Face PyTorch safetensors into MLX format, automatically transposing Conv1D weights and explicitly tying word embedding weights.
* **[generate.py](file:///Users/bran/src/play/llm/engines/recurrentgemma/generate.py)**: Autoregressive causal text generation runner, using the tokenizer's chat template and custom caching to run step-by-step decoding.

---

## Getting Started

### 1. Convert PyTorch weights to MLX
Run the conversion script pointing to your Hugging Face PyTorch model directory:
```bash
.venv/bin/python engines/recurrentgemma/convert.py \
  --model-path models/recurrentgemma-2b-it \
  --output-path weights/recurrentgemma-2b-it/weights.safetensors
```

### 2. Run Causal Generation
Run the generation script with a custom prompt:
```bash
PYTHONPATH=engines/recurrentgemma .venv/bin/python engines/recurrentgemma/generate.py \
  --config models/recurrentgemma-2b-it/config.json \
  --tokenizer models/recurrentgemma-2b-it \
  --max-tokens 100 \
  --prompt "Explain the difference between linear recurrence and self-attention in a few sentences."
```

### 3. Quantize the Model
To quantize the model (e.g., to 4-bit) for faster generation speeds and reduced memory footprint:
```bash
.venv/bin/python engines/recurrentgemma/quantize.py \
  --config models/recurrentgemma-2b-it/config.json \
  --weights weights/recurrentgemma-2b-it/weights.safetensors \
  --output weights/recurrentgemma-2b-it-4bit/weights.safetensors \
  --bits 4
```

### 4. Run Quantized Generation
Simply run the generation script pointing to your quantized weights:
```bash
PYTHONPATH=engines/recurrentgemma .venv/bin/python engines/recurrentgemma/generate.py \
  --config models/recurrentgemma-2b-it/config.json \
  --weights weights/recurrentgemma-2b-it-mxfp4/weights.safetensors \
  --tokenizer models/recurrentgemma-2b-it \
  --max-tokens 100 \
  --prompt "Explain the difference between linear recurrence and self-attention in a few sentences."
```
The generator script automatically detects that the loaded weights are quantized, configures the linear layers accordingly, and executes highly optimized quantized kernels.
