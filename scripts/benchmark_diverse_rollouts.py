import sys
import time
import mlx.core as mx
from mlx_lm import load, generate

# Add path
sys.path.append("/Users/bran/src/play/llm")
from scripts.utils.mlx_batch_generator import MLXBatchGenerator

def run_method_1_sequential(model, tokenizer, problems, prompt_format, group_size, max_tokens):
    print(f"\nRunning Approach 1: Sequential For-loop (Total {len(problems) * group_size} rollouts)...")
    mx.reset_peak_memory()
    mx.clear_cache()
    
    start_time = time.time()
    responses = []
    for p in problems:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_format.format(p)}],
            add_generation_prompt=True
        )
        for _ in range(group_size):
            text = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=max_tokens)
            responses.append(text)
            
    duration = time.time() - start_time
    peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
    return duration, peak_mem_gb

def run_method_2_standard_batch(model, tokenizer, problems, prompt_format, group_size, max_tokens):
    print(f"\nRunning Approach 2: Batched (Duplicated prefill, Batch Size {len(problems) * group_size})...")
    generator = MLXBatchGenerator(model, tokenizer)
    mx.reset_peak_memory()
    mx.clear_cache()
    
    start_time = time.time()
    # Duplicate prompt templates beforehand
    prompts = []
    for p in problems:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_format.format(p)}],
            add_generation_prompt=True
        )
        prompts.extend([formatted] * group_size)
        
    responses = generator.generate(prompts, max_tokens=max_tokens, verbose=False)
    
    duration = time.time() - start_time
    peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
    return duration, peak_mem_gb

def run_method_3_prefix_cache_replicated(model, tokenizer, problems, prompt_format, group_size, max_tokens):
    print(f"\nRunning Approach 3: Prefix-Cache Replication (Prefill Size {len(problems)}, Decode Size {len(problems) * group_size})...")
    generator = MLXBatchGenerator(model, tokenizer)
    mx.reset_peak_memory()
    mx.clear_cache()
    
    start_time = time.time()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_format.format(p)}],
            add_generation_prompt=True
        )
        for p in problems
    ]
    responses = generator.generate_with_diverse_rollouts(prompts, group_size=group_size, max_tokens=max_tokens, verbose=False)
    
    duration = time.time() - start_time
    peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
    return duration, peak_mem_gb

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load(
        "/Users/bran/.lmstudio/models/local/ministral-3-3b-reasoning-2512-mxfp4",
        tokenizer_config={"fix_mistral_regex": True}
    )
    
    problems = [
        "What is 15 + 27?",
        "If a circle has radius 7, what is its approximate area? Use pi = 3.14.",
        "Solve for x: 2x - 5 = 9.",
        "What is the square root of 144?"
    ]
    prompt_format = "Solve the following math problem step-by-step.\nProblem: What is {}?"
    group_size = 4
    max_tokens = 150
    
    # Warmup
    print("Running warmup...")
    _ = run_method_3_prefix_cache_replicated(model, tokenizer, problems[:1], prompt_format, group_size, 20)
    
    # Run tests
    dur1, mem1 = run_method_1_sequential(model, tokenizer, problems, prompt_format, group_size, max_tokens)
    dur2, mem2 = run_method_2_standard_batch(model, tokenizer, problems, prompt_format, group_size, max_tokens)
    dur3, mem3 = run_method_3_prefix_cache_replicated(model, tokenizer, problems, prompt_format, group_size, max_tokens)
    
    # Print results
    print("\n" + "="*80)
    print("                 DIVERSE ROLLOUT BENCHMARK COMPARISON")
    print("="*80)
    print(f"{'Approach':<45} | {'Time (s)':<10} | {'Peak Memory (GB)':<16}")
    print("-" * 77)
    print(f"{'1. Sequential For-loop':<45} | {dur1:<10.2f} | {mem1:<16.3f}")
    print(f"{'2. Standard Batch (Duplicated Prefill)':<45} | {dur2:<10.2f} | {mem2:<16.3f}")
    print(f"{'3. Prefix-Cache Replication (New)':<45} | {dur3:<10.2f} | {mem3:<16.3f}")
    print("="*80)

if __name__ == "__main__":
    main()
