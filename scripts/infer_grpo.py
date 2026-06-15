import mlx.core as mx
from mlx_lm import load, generate
from grpo_dataset import generate_mixed_math_problem
import re

def extract_answer(text):
    numbers = re.findall(r'-?\d+', text)
    return numbers[-1] if numbers else "N/A"

def main():
    print("Loading Base Model...")
    model_path = "storage/models/gemma-3-270m-it"
    base_model, tokenizer = load(model_path)
    
    print("Loading GRPO Fine-Tuned Model...")
    grpo_model, _ = load(model_path)
    # Make sure to load the step 100 checkpoint!
    grpo_model.load_weights("storage/models/grpo_checkpoint_step_100.safetensors")
    
    results = []
    
    # Generate 10 problems to compare
    print("Running Inference on 10 problems...")
    for i in range(10):
        question, ground_truth = generate_mixed_math_problem()
        
        prompt_text = (
            "Solve the following math problem step-by-step. "
            "Ensure that your final numerical answer is the very last number in your response.\n\n"
            f"Problem: What is {question}?"
        )
        
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        def greedy_sampler(logits):
            return mx.argmax(logits, axis=-1)
            
        base_out = generate(base_model, tokenizer, prompt=formatted_prompt, max_tokens=150, sampler=greedy_sampler, verbose=False).strip()
        grpo_out = generate(grpo_model, tokenizer, prompt=formatted_prompt, max_tokens=150, sampler=greedy_sampler, verbose=False).strip()
        
        base_ans = extract_answer(base_out)
        grpo_ans = extract_answer(grpo_out)
        
        results.append({
            "problem": question,
            "truth": ground_truth,
            "base": base_ans,
            "grpo": grpo_ans
        })
        print(f"[{i+1}/10] {question} -> Truth: {ground_truth} | Base: {base_ans} | GRPO: {grpo_ans}")
        
    print("\n\n" + "="*50)
    print("FINAL RESULTS TABLE")
    print("="*50 + "\n")
    
    print("| Problem | Ground Truth | Base Model | GRPO Model |")
    print("| :--- | :---: | :---: | :---: |")
    for r in results:
        base_emoji = "✅" if str(r['base']) == str(r['truth']) else "❌"
        grpo_emoji = "✅" if str(r['grpo']) == str(r['truth']) else "❌"
        print(f"| `{r['problem']}` | **{r['truth']}** | {r['base']} {base_emoji} | {r['grpo']} {grpo_emoji} |")

if __name__ == "__main__":
    main()
