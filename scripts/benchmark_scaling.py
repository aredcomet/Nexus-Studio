import sys
import time
import mlx.core as mx
from mlx_lm import load
from grpo_dataset import generate_math_problem

# Add path
sys.path.append("/Users/bran/src/play/llm")
from scripts.utils.mlx_batch_generator import MLXBatchGenerator

MODEL_PATH = "/Users/bran/.lmstudio/models/local/ministral-3-3b-reasoning-2512-mxfp4"

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load(MODEL_PATH, tokenizer_config={"fix_mistral_regex": True})
    generator = MLXBatchGenerator(model, tokenizer)
    prompt_format = "Solve the following math problem step-by-step.\nProblem: What is {}?"
    max_tokens = 150
    
    # Batch sizes to test
    batch_sizes = [4, 8, 16, 32, 64]
    
    # Generate maximum number of problems needed
    print(f"Generating {max(batch_sizes)} math problems for testing...")
    all_problems = [generate_math_problem()[0] for _ in range(max(batch_sizes))]
    
    # Warmup
    print("Running warmup...")
    _ = generator.generate(
        [tokenizer.apply_chat_template([{"role": "user", "content": prompt_format.format(all_problems[0])}], add_generation_prompt=True)],
        max_tokens=20
    )
    
    print("\n" + "="*70)
    print("                SCALING BENCHMARK (Custom MLXBatchGenerator)")
    print("="*70)
    print(f"{'Batch Size':<12} | {'Total Time (s)':<16} | {'Time/Seq (s)':<14} | {'Peak Memory (GB)':<16}")
    print("-" * 70)
    
    for bs in batch_sizes:
        batch_problems = all_problems[:bs]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_format.format(p)}],
                add_generation_prompt=True
            )
            for p in batch_problems
        ]
        
        # Reset and clear cache
        mx.reset_peak_memory()
        mx.clear_cache()
        
        start_time = time.time()
        _ = generator.generate(prompts, max_tokens=max_tokens, verbose=False)
        duration = time.time() - start_time
        
        time_per_seq = duration / bs
        peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
        
        print(f"{bs:<12} | {duration:<16.2f} | {time_per_seq:<14.2f} | {peak_mem_gb:<16.3f}")
        
    print("="*70)

if __name__ == "__main__":
    main()
