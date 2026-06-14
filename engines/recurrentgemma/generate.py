import argparse
import time
import json
import os
import mlx.core as mx
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer

from model import ModelConfig, RecurrentGemmaForCausalLM, DecoderCache
from mlx_lm.sample_utils import (
    apply_top_k,
    apply_top_p,
    apply_min_p,
    make_repetition_penalty,
)

def load_model(config_path, weights_path, attention_window=None):
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = ModelConfig(**config_dict)
    
    if attention_window is not None:
        print(f"Overriding sliding window attention size to {attention_window}")
        config.attention_window_size = attention_window
    
    print("Initializing model...")
    model = RecurrentGemmaForCausalLM(config)
    
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
    
    # Cast weights to float16 or bfloat16 for fast GPU inference
    # RecurrentGemma works extremely well in float16/bfloat16
    dtype = mx.bfloat16 if config_dict.get("torch_dtype") == "bfloat16" else mx.float16
    weights = {k: (v.astype(dtype) if v.dtype not in [mx.uint32, mx.uint8] else v) for k, v in weights.items()}
    
    model.update(tree_unflatten(list(weights.items())))
    # Set to eval mode
    mx.eval(model.parameters())
    
    # Warmup model graphs (compiles GPU shaders)
    print("Warming up model graphs...")
    warmup_cache = DecoderCache(config.num_hidden_layers, config.lru_width, config.hidden_size)
    warmup_in = mx.zeros((1, 8), dtype=mx.int32)
    # 1. Warmup prefill path
    model(warmup_in, offset=0, cache=warmup_cache)
    # 2. Warmup decode path
    model(mx.array([[1]]), position_ids=mx.array([[8]]), offset=8, cache=warmup_cache)
    mx.eval(model.parameters())
    
    return model, config

def generate(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    temp=0.0,
    top_k=0,
    top_p=0.0,
    min_p=0.0,
    repeat_penalty=1.0,
    repeat_context=20,
    system_prompt=None,
    context_window=None,
):
    # Apply system prompt by combining with user prompt if present
    if system_prompt:
        combined_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        combined_prompt = prompt
        
    chat = [{"role": "user", "content": combined_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    print("\nFormatted Prompt:")
    print("-" * 40)
    print(formatted_prompt)
    print("-" * 40)
    
    input_ids = mx.array(tokenizer.encode(formatted_prompt))[None, :]
    
    cache = DecoderCache(
        num_layers=model.config.num_hidden_layers,
        lru_width=model.config.lru_width,
        hidden_size=model.config.hidden_size
    )
    
    logits_processors = []
    if repeat_penalty != 1.0:
        logits_processors.append(make_repetition_penalty(repeat_penalty, repeat_context))
        
    def sample_token(logits, generated_tokens):
        if logits_processors and len(generated_tokens) > 0:
            tokens_arr = mx.array(generated_tokens)
            logits = logits[None, :]
            for processor in logits_processors:
                logits = processor(tokens_arr, logits)
            logits = logits[0]
            
        if temp == 0.0:
            return mx.argmax(logits, axis=-1).item()
            
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if top_k > 0:
            logprobs = apply_top_k(logprobs, top_k)
        if min_p > 0.0:
            logprobs = apply_min_p(logprobs, min_p)
        if top_p > 0.0:
            logprobs = apply_top_p(logprobs, top_p)
            
        return mx.random.categorical(logprobs / temp).item()
        
    print("\nRunning encoder/prefill...")
    start_time = time.perf_counter()
    
    # 1. Prefill step
    logits = model(input_ids, offset=0, cache=cache)
    # Get the logits of the last token in the prompt
    token_logits = logits[:, -1, :]
    
    token = sample_token(token_logits[0], [])
        
    prefill_time = time.perf_counter() - start_time
    prefill_tokens = input_ids.shape[1]
    
    # Initialize list of generated tokens
    generated_tokens = [token]
    print(tokenizer.decode([token]), end="", flush=True)
    
    # 2. Generation step
    gen_start_time = time.perf_counter()
    offset = prefill_tokens
    
    for i in range(1, max_tokens):
        # Stop early if we exceed the total context window size
        if context_window is not None and offset >= context_window:
            print("\n[Reached context window limit]")
            break
            
        # We only pass the single last token ID
        curr_input = mx.array([[token]])
        
        # position_ids for decoding is just offset
        position_ids = mx.array([[offset]])
        
        logits = model(curr_input, position_ids=position_ids, offset=offset, cache=cache)
        token_logits = logits[:, -1, :]
        
        token = sample_token(token_logits[0], generated_tokens)
            
        generated_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
        
        if token == tokenizer.eos_token_id:
            break
            
        offset += 1
        
    print() # New line after generation finishes
    
    gen_time = time.perf_counter() - gen_start_time
    gen_tokens_count = len(generated_tokens)
    
    # Get peak memory usage
    peak_mem_gb = mx.get_peak_memory() / 1e9
    
    print("\n" + "=" * 40)
    print(f"Prompt: {prefill_tokens} tokens, {prefill_tokens / prefill_time:.3f} tokens-per-sec")
    print(f"Generation: {gen_tokens_count} tokens, {gen_tokens_count / gen_time:.3f} tokens-per-sec")
    print(f"Peak memory: {peak_mem_gb:.3f} GB")
    print("=" * 40)
    
    return tokenizer.decode(generated_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal text generation with RecurrentGemma in MLX")
    parser.add_argument("--config", type=str, default="models/recurrentgemma-2-2b-it/config.json", help="Path to config.json")
    parser.add_argument("--weights", type=str, default="weights/recurrentgemma-2b-it/weights.safetensors", help="Path to converted weights.safetensors")
    parser.add_argument("--tokenizer", type=str, default="models/recurrentgemma-2b-it", help="Path to tokenizer folder")
    parser.add_argument("--prompt", type=str, default="Explain the difference between linear recurrence and self-attention in a few sentences.", help="Input text prompt")
    parser.add_argument("--max-tokens", type=int, default=150, help="Maximum number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0.0 for greedy decoding)")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt to guide the model's behavior")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling filter")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p (nucleus) sampling filter")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling filter")
    parser.add_argument("--repeat-penalty", type=float, default=1.0, help="Repetition penalty factor")
    parser.add_argument("--repeat-context", type=int, default=20, help="Number of previous tokens to consider for repetition penalty")
    parser.add_argument("--attention-window", type=int, default=None, help="Override sliding window attention size")
    parser.add_argument("--context-window", type=int, default=None, help="Max sequence length (limit context window)")
    
    # Resolve config paths if default doesn't exist
    args = parser.parse_args()
    
    if not os.path.exists(args.config) and "recurrentgemma-2b-it" in args.config:
        # Check models/recurrentgemma-2b-it/config.json
        alt_config = "models/recurrentgemma-2b-it/config.json"
        if os.path.exists(alt_config):
            args.config = alt_config
            args.tokenizer = "models/recurrentgemma-2b-it"
            
    model, config = load_model(args.config, args.weights, attention_window=args.attention_window)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        repeat_context=args.repeat_context,
        system_prompt=args.system_prompt,
        context_window=args.context_window,
    )
