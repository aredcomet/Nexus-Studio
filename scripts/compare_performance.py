import sys
import time
import mlx.core as mx
from mlx_lm import load, generate, batch_generate
from grpo_dataset import generate_math_problem

# Add path
sys.path.append("/Users/bran/src/play/llm")
from scripts.utils.mlx_batch_generator import MLXBatchGenerator

MODEL_PATH = "/Users/bran/.lmstudio/models/local/ministral-3-3b-reasoning-2512-mxfp4"

def run_method_1_for_loop(model, tokenizer, problems, prompt_format, max_tokens):
    print(f"\nRunning Method 1: For-loop (generate) on {len(problems)} problems...")
    mx.reset_peak_memory()
    mx.clear_cache()
    
    start_time = time.time()
    responses = []
    for p in problems:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_format.format(p)}],
            add_generation_prompt=True
        )
        text = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=max_tokens)
        responses.append(text)
        
    duration = time.time() - start_time
    peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
    return duration, peak_mem_gb, responses

def run_method_2_batch_generate(model, tokenizer, problems, prompt_format, max_tokens):
    print(f"\nRunning Method 2: mlx_lm.batch_generate on {len(problems)} problems...")
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
    result = batch_generate(model, tokenizer, prompts, verbose=False, max_tokens=max_tokens)
    responses = result.texts
    
    duration = time.time() - start_time
    peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
    return duration, peak_mem_gb, responses

def run_method_3_custom_batch(model, tokenizer, problems, prompt_format, max_tokens):
    print(f"\nRunning Method 3: Custom MLXBatchGenerator on {len(problems)} problems...")
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
    responses = generator.generate(prompts, max_tokens=max_tokens, verbose=False)
    
    duration = time.time() - start_time
    peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
    return duration, peak_mem_gb, responses

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load(MODEL_PATH,tokenizer_config={"fix_mistral_regex": True})
    
    # Generate 16 random math problems
    print("Generating 16 problems...")
    problems = [generate_math_problem()[0] for _ in range(16)]
    prompt_format = "Solve the following math problem step-by-step.\nProblem: What is {}?"
    max_tokens = 150
    
    # Warmup first to compile metal kernels
    print("Running warmup...")
    _ = run_method_3_custom_batch(model, tokenizer, problems[:1], prompt_format, 20)
    
    # Run tests
    dur1, mem1, _ = run_method_1_for_loop(model, tokenizer, problems, prompt_format, max_tokens)
    dur2, mem2, _ = run_method_2_batch_generate(model, tokenizer, problems, prompt_format, max_tokens)
    dur3, mem3, _ = run_method_3_custom_batch(model, tokenizer, problems, prompt_format, max_tokens)
    
    # Print results
    print("\n" + "="*60)
    print("                    PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Method':<35} | {'Time (s)':<10} | {'Peak Memory (GB)':<16}")
    print("-" * 69)
    print(f"{'1. For-loop (mlx_lm.generate)':<35} | {dur1:<10.2f} | {mem1:<16.3f}")
    print(f"{'2. mlx_lm.batch_generate':<35} | {dur2:<10.2f} | {mem2:<16.3f}")
    print(f"{'3. Custom MLXBatchGenerator':<35} | {dur3:<10.2f} | {mem3:<16.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
