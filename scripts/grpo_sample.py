import mlx.core as mx
from mlx_lm import load, generate
from grpo_dataset import generate_goldilocks_problem, reward_function

def main():
    print("Loading model...")
    model_path = "storage/models/gemma-3-270m-it"
    model, tokenizer = load(model_path)
    
    # Generate a single problem to test the rollout
    question, ground_truth = generate_goldilocks_problem()
    
    prompt_text = (
        "Solve the following math problem. Think step-by-step inside <think>...</think> tags, "
        "and then output your final answer.\n\n"
        "Example:\n"
        "Problem: What is 2 + 3 * 4?\n"
        "<think>\nFirst multiply 3 and 4 to get 12.\nThen add 2 to get 14.\n</think>\n"
        "14\n\n"
        f"Problem: What is {question}?"
    )
    
    print("--- PROMPT ---")
    print(prompt_text)
    print("Ground Truth:", ground_truth)
    print("-" * 40)
    
    # Format the prompt using the model's chat template
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate a group of G=4 responses (The "Rollout" phase)
    G = 4
    temperature = 0.7  # High temperature for diverse reasoning paths
    
    print(f"\nGenerating {G} sample responses (Temperature = {temperature})...")
    
    for i in range(G):
        print(f"\n--- Response {i+1} ---")
        
        def sampler(logits):
            # Scale logits by temperature and sample
            return mx.random.categorical(logits / temperature)

        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=256,
            sampler=sampler,
            verbose=False
        ).strip()
        
        score = reward_function(response, ground_truth)
        print(response)
        print(f"\n[Reward Score: {score}]")

if __name__ == "__main__":
    main()
