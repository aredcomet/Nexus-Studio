import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from grpo_dataset import generate_math_dataset, reward_function

# GRPO Hyperparameters
G = 4               # Group size (number of rollouts per prompt)
BETA = 0.04         # KL divergence penalty coefficient
EPSILON = 0.2       # Clipping parameter for probability ratios
LR = 1e-5           # Learning rate

def compute_logprobs(logits, token_ids):
    """
    Computes the log probability of the generated tokens.
    logits shape: (batch, seq_len, vocab_size)
    token_ids shape: (batch, seq_len)
    """
    # Shift logits and tokens for next-token prediction
    logits = logits[:, :-1, :]
    targets = token_ids[:, 1:]
    
    # Compute log_softmax
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    
    # Gather the logprobs for the actual target tokens
    # Using mx.take_along_axis
    gathered_logprobs = mx.take_along_axis(logprobs, targets[..., None], axis=-1).squeeze(-1)
    
    return gathered_logprobs

def grpo_loss(model, ref_model, input_ids_batch, advantages_batch, prompt_lengths):
    """
    The core GRPO loss function.
    input_ids_batch: Shape (G, max_seq_len)
    advantages_batch: Shape (G,)
    prompt_lengths: Shape (G,) - to mask out the prompt tokens from the loss
    """
    # 1. Get logits from both models
    logits = model(input_ids_batch)
    ref_logits = ref_model(input_ids_batch)
    
    # 2. Extract log probabilities for the generated tokens
    logprobs = compute_logprobs(logits, input_ids_batch)
    ref_logprobs = compute_logprobs(ref_logits, input_ids_batch)
    
    # We treat the old policy as the ref_model for this simplified loop.
    # In a full loop, old policy is the model from the start of the PPO epoch.
    old_logprobs = ref_logprobs 
    
    # 3. Create a mask to ignore prompt tokens and padding
    # (Simplified: we assume dense arrays here, but in reality we'd use mx.arange)
    seq_len = logprobs.shape[1]
    positions = mx.arange(seq_len)[None, :]
    # +1 because logprobs are shifted
    mask = positions >= (prompt_lengths[:, None] - 1) 
    
    # 4. Probability Ratio
    ratio = mx.exp(logprobs - old_logprobs)
    
    # 5. Clipped Objective
    # advantages_batch is (G,), reshape to (G, 1) for broadcasting
    adv = advantages_batch[:, None]
    unclipped_obj = ratio * adv
    clipped_ratio = mx.clip(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
    clipped_obj = clipped_ratio * adv
    
    # 6. KL Penalty (approximate KL divergence)
    # D_KL ≈ old_logprobs - logprobs
    kl_div = ref_logprobs - logprobs
    
    # 7. Final Loss calculation per token
    token_loss = - (mx.minimum(unclipped_obj, clipped_obj) - BETA * kl_div)
    
    # Apply mask and mean over generated tokens
    masked_loss = token_loss * mask
    loss = mx.sum(masked_loss) / mx.sum(mask)
    
    return loss

def main():
    print("Loading models (this requires enough VRAM for 2 copies!)...")
    model_path = "storage/models/gemma-3-270m-it"
    
    # Trainable Policy Model
    model, tokenizer = load(model_path)
    # Frozen Reference Model
    ref_model, _ = load(model_path)
    ref_model.freeze()
    
    dataset = generate_math_dataset(num_samples=5)
    
    for step, data in enumerate(dataset):
        prompt_text = data['prompt']
        ground_truth = data['ground_truth']
        
        print(f"\n--- STEP {step+1} ---")
        print(f"Problem: {ground_truth}")
        
        # Format prompt
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 1. Rollout Phase (Generate G responses)
        responses = []
        for _ in range(G):
            # Define a custom sampler inside the loop
            def sampler(logits):
                return mx.random.categorical(logits / 0.7)
                
            response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=150, sampler=sampler, verbose=False).strip()
            responses.append(response)
            
        # 2. Score the responses
        rewards = [reward_function(r, ground_truth) for r in responses]
        print("Rewards:", rewards)
        
        # 3. Compute Advantages
        rewards_tensor = mx.array(rewards)
        mean_reward = mx.mean(rewards_tensor)
        std_reward = mx.std(rewards_tensor) + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
        print("Advantages:", advantages.tolist())
        
        # NOTE: In a complete training loop, we would now tokenize the 
        # (prompt + response) pairs, pad them to the same length, 
        # compute the loss using nn.value_and_grad(grpo_loss), 
        # and update the model weights using an optimizer (like AdamW).
        # We will pause here to review the architecture!

if __name__ == "__main__":
    main()
