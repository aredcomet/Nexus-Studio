import argparse
import json
import os
import mlx.core as mx
import mlx.nn as nn
from PIL import Image
from transformers import AutoProcessor
from model import T5Gemma2ForConditionalGeneration, DecoderCache
from train import apply_lora, LoRALinear

import time

def load_model(config_path, weights_path, adapter_path=None):
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print("Initializing model...")
    model = T5Gemma2ForConditionalGeneration(config)
    
    print(f"Loading weights from {weights_path}...")
    if os.path.isdir(weights_path):
        import glob
        weights = {}
        for f in glob.glob(os.path.join(weights_path, "*.safetensors")):
            weights.update(mx.load(f))
    else:
        weights = mx.load(weights_path)
        
    # Check if weights are quantized
    is_quantized = any(k.endswith(".scales") for k in weights.keys())
    if is_quantized:
        # Find a scales key to deduce bits and group_size
        scales_key = next(k for k in weights.keys() if k.endswith(".scales"))
        weight_key = scales_key.replace(".scales", ".weight")
        bias_key = scales_key.replace(".scales", ".biases")
        
        # Traverse the model to find the original in_features
        def get_in_features(model, key_path):
            parts = key_path.split(".")[:-1] # strip '.scales'
            obj = model
            for part in parts:
                if isinstance(obj, list):
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            return obj.weight.shape[1] # original in_features
            
        in_features = get_in_features(model, scales_key)
        
        scales_shape = weights[scales_key].shape
        weight_shape = weights[weight_key].shape
        
        bits = 32 // (in_features // weight_shape[1])
        group_size = in_features // scales_shape[1]
        
        # Determine mode
        if bias_key in weights:
            mode = "affine"
        elif weights[scales_key].dtype == mx.uint8:
            if bits == 4:
                mode = "mxfp4"
            elif bits == 8:
                mode = "mxfp8"
            else:
                mode = "affine"
        elif group_size == 16 and bits == 4:
            mode = "nvfp4"
        else:
            mode = "affine"
            
        print(f"Detected quantized weights ({bits}-bit, group_size={group_size}, mode='{mode}'). Quantizing model...")
        from quantize import quantize_model
        quantize_model(model, group_size=group_size, bits=bits, mode=mode)
        
    from mlx.utils import tree_unflatten
    dtype = mx.bfloat16 if config.get("torch_dtype") == "bfloat16" else mx.float16
    weights = {k: (v.astype(dtype) if v.dtype not in [mx.uint32, mx.uint8] else v) for k, v in weights.items()}
    model.update(tree_unflatten(list(weights.items())))
    
    if adapter_path is not None:
        config_file = os.path.join(adapter_path, "adapter_config.json")
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        
        print(f"Loading adapter config from {config_file}...")
        with open(config_file, "r") as f:
            adapter_config = json.load(f)
            
        print("Applying LoRA layers to model...")
        apply_lora(
            model,
            keys=adapter_config["lora_keys"],
            rank=adapter_config["lora_rank"],
            alpha=adapter_config["lora_alpha"],
            dropout=adapter_config["lora_dropout"]
        )
        
        print(f"Loading adapter weights from {adapter_file}...")
        model.load_weights(adapter_file, strict=False)
        
    # Put model in eval mode
    mx.eval(model.parameters())
    return model, config

def generate(model, processor, prompt, image_path=None, max_tokens=100, temp=0.0):
    # Prepare inputs using Hugging Face AutoProcessor
    print("Preprocessing inputs...")
    if image_path:
        print(f"Loading image from {image_path}...")
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="np")
        pixel_values = mx.array(inputs["pixel_values"])
    else:
        inputs = processor(text=prompt, return_tensors="np")
        pixel_values = None
        
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])
    
    prompt_tokens = input_ids.shape[1]
    
    # Reset peak memory tracking
    mx.reset_peak_memory()
    
    # 1. Run encoder once + first decoder step (Prefill)
    print("Running encoder and prefill...")
    start_prompt = time.perf_counter()
    
    # Get encoder hidden states
    encoder_outputs = model.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )
    
    # Start token is BOS (usually 2)
    decoder_input_ids = mx.array([[model.config.get("bos_token_id", 2)]])
    
    # Setup cache
    num_decoder_layers = model.config["decoder"]["num_hidden_layers"]
    cache = DecoderCache(num_decoder_layers)
    
    # Generate the first token
    logits, _ = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        past_key_values=cache,
        encoder_outputs=encoder_outputs,
    )
    logits = logits[:, -1, :]
    
    if temp > 0.0:
        token = mx.random.categorical(logits / temp).item()
    else:
        token = mx.argmax(logits, axis=-1).item()
        
    # Force MLX evaluation to record accurate prefill time
    mx.eval(token)
    prompt_time = time.perf_counter() - start_prompt
    
    print("\nGenerated output:")
    print("-----------------")
    
    generated_tokens = [token]
    text = processor.decode(generated_tokens)
    print(text, end="", flush=True)
    prev_text = text
    
    # Update decoder input ids for next step
    decoder_input_ids = mx.array([[token]])
    
    # 2. Decode rest of the tokens
    gen_tokens_count = 1
    start_gen = time.perf_counter()
    
    for token_idx in range(1, max_tokens):
        # Stop on EOS token (usually 1)
        if token == model.config.get("eos_token_id", 1):
            break
            
        # Forward pass of decoder
        logits, _ = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            past_key_values=cache,
            encoder_outputs=encoder_outputs,
        )
        
        logits = logits[:, -1, :]
        
        if temp > 0.0:
            token = mx.random.categorical(logits / temp).item()
        else:
            token = mx.argmax(logits, axis=-1).item()
            
        if token == model.config.get("eos_token_id", 1):
            break
            
        generated_tokens.append(token)
        gen_tokens_count += 1
        
        # Force evaluation per token step
        mx.eval(token)
        
        # Decode and print next token/text
        text = processor.decode(generated_tokens)
        new_text = text[len(prev_text):]
        print(new_text, end="", flush=True)
        prev_text = text
        
        # Update decoder input ids for next step
        decoder_input_ids = mx.array([[token]])
        
    gen_time = time.perf_counter() - start_gen
    print("\n-----------------")
    
    # Compute performance stats
    prompt_tps = prompt_tokens / prompt_time if prompt_time > 0 else 0.0
    gen_tps = gen_tokens_count / gen_time if gen_time > 0 else 0.0
    peak_mem = mx.get_peak_memory() / (1024 * 1024 * 1024)
    
    print("\n==========")
    print(f"Prompt: {prompt_tokens} tokens, {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {gen_tokens_count} tokens, {gen_tps:.3f} tokens-per-sec")
    print(f"Peak memory: {peak_mem:.3f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on T5Gemma2 models using MLX")
    parser.add_argument("--config", type=str, default="models/t5gemma-2-270m-270m/config.json", help="Path to config.json")
    parser.add_argument("--weights", type=str, default="weights/t5gemma-2-270m-270m/weights.safetensors", help="Path to weights.safetensors")
    parser.add_argument("--processor", type=str, default="models/t5gemma-2-270m-270m", help="Path to processor/tokenizer directory")
    parser.add_argument("--prompt", type=str, default="Translate from English to French: Today is a beautiful day to learn programming.", help="Prompt text")
    parser.add_argument("--image", type=str, default=None, help="Optional path to image file")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to trained LoRA adapter directory")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.weights):
        print(f"Error: Weights file {args.weights} not found. Please run convert.py first.")
        exit(1)
        
    model, config = load_model(args.config, args.weights, args.adapter_path)
    
    print(f"Loading processor from {args.processor}...")
    processor = AutoProcessor.from_pretrained(args.processor)
    
    generate(
        model=model,
        processor=processor,
        prompt=args.prompt,
        image_path=args.image,
        max_tokens=args.max_tokens,
        temp=args.temp,
    )
