import mlx.core as mx
from mlx_lm import load
import time

def generate_batched(model, tokenizer, prompt, G=8, max_tokens=150, temperature=0.7):
    """
    Efficient batched generation with Prefix-Caching.
    Computes the prompt KV cache exactly once, then branches into G different generation paths.
    """
    # 1. Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = mx.array([prompt_tokens])  # Shape: (1, seq_len)
    
    # 2. Compute Prefix Cache
    # We run the prompt through the model with batch_size=1
    cache = model.make_cache()
    logits = model(prompt_array, cache=cache)
    
    # 3. Duplicate the KV Cache for G rollouts
    # The cache arrays have shape (batch_size, n_kv_heads, seq_len, head_dim)
    # We repeat them along the batch_size dimension (axis=0)
    for c in cache:
        c.keys = mx.repeat(c.keys, G, axis=0)
        c.values = mx.repeat(c.values, G, axis=0)
        
    # 4. Duplicate the last token for the start of the generation phase
    # The next input is just the very last token, repeated G times
    # Shape: (G, 1)
    y = mx.repeat(prompt_array[:, -1:], G, axis=0)
    
    # We need to compute the logits for the first step from the last token of the prompt.
    # Wait, the logits we already computed `logits` has shape (1, seq_len, vocab_size).
    # The logits for the NEXT token are at `logits[0, -1, :]`.
    next_logits = logits[:, -1, :]
    next_logits = mx.repeat(next_logits, G, axis=0)
    
    # Sample the first generated token
    def sample(logits):
        return mx.random.categorical(logits / temperature)
        
    y = sample(next_logits)[:, None]  # Shape (G, 1)
    
    # Store the generated tokens for each of the G paths
    generated_tokens = [[] for _ in range(G)]
    for i in range(G):
        generated_tokens[i].append(y[i, 0].item())
        
    # 5. Batched Generation Loop
    for step in range(1, max_tokens):
        # Forward pass on just the single new token! 
        # The cache automatically updates with the new keys/values for all G paths
        logits = model(y, cache=cache)
        next_logits = logits[:, -1, :]
        
        # Sample the next tokens
        y = sample(next_logits)[:, None]
        
        # Save tokens
        for i in range(G):
            generated_tokens[i].append(y[i, 0].item())
            
        # Optimization: We could check if all paths have emitted an EOS token and break early
        # but for simplicity we'll just run to max_tokens.
        
    # Decode all paths
    responses = [tokenizer.decode(tokens) for tokens in generated_tokens]
    return responses

def main():
    print("Loading Model...")
    model_path = "storage/models/gemma-3-270m-it"
    model, tokenizer = load(model_path)
    
    # Create a test prompt
    prompt = "Solve the following math problem step-by-step. What is 12 + 4 * 3?"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    G = 8
    print(f"\nRunning EFFICIENT batched rollout (G={G})...")
    start_time = time.time()
    
    responses = generate_batched(model, tokenizer, formatted_prompt, G=G, max_tokens=50, temperature=0.7)
    
    elapsed = time.time() - start_time
    print(f"Generated {G} responses in {elapsed:.2f} seconds!")
    print("\n" + "="*50)
    
    for i, res in enumerate(responses):
        print(f"--- Response {i+1} ---")
        # Truncate response for display
        print(res.strip()[:100] + "...")
        print("-" * 50)

if __name__ == "__main__":
    main()
