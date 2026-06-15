import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from grpo_dataset import generate_math_dataset, reward_function
import time
from tqdm import tqdm
import logging

logging.basicConfig(
    filename='train_grpo_pipeline.log', 
    filemode='a', # 'a' to append, 'w' to overwrite
    level=logging.INFO, # Capture INFO level and more severe logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# GRPO Hyperparameters
G = 8               # Group size (number of rollouts per prompt)
BETA = 0.04         # KL divergence penalty coefficient
EPSILON = 0.2       # Clipping parameter for probability ratios
LR = 1e-6           # Learning rate
NUM_STEPS = 100     # Number of training steps

def compute_logprobs(logits, token_ids):
    """
    Computes the log probability of the generated tokens.
    logits shape: (batch, seq_len, vocab_size)
    token_ids shape: (batch, seq_len)
    """
    logits = logits[:, :-1, :]
    targets = token_ids[:, 1:]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    gathered_logprobs = mx.take_along_axis(logprobs, targets[..., None], axis=-1).squeeze(-1)
    return gathered_logprobs

def grpo_loss(model, ref_model, input_ids_batch, advantages_batch, prompt_lengths, loss_mask):
    """
    The core GRPO loss function.
    input_ids_batch: Shape (G, max_seq_len)
    advantages_batch: Shape (G,)
    prompt_lengths: Shape (G,)
    loss_mask: Shape (G, max_seq_len - 1)
    """
    logits = model(input_ids_batch)
    ref_logits = ref_model(input_ids_batch)
    
    logprobs = compute_logprobs(logits, input_ids_batch)
    
    # Detached logprobs for the reference and old policy
    # Since we do 1 update per rollout, old_policy == current policy before update
    # We use stop_gradient to treat them as constants
    old_logprobs = mx.stop_gradient(logprobs)
    ref_logprobs = mx.stop_gradient(compute_logprobs(ref_logits, input_ids_batch))
    
    # 1. Mask to ignore prompt tokens and padding
    seq_len = logprobs.shape[1]
    positions = mx.arange(seq_len)[None, :]
    prompt_mask = positions >= (prompt_lengths[:, None] - 1)
    final_mask = prompt_mask & loss_mask
    
    # 2. Probability Ratio
    ratio = mx.exp(logprobs - old_logprobs)
    
    # 3. Clipped Objective
    adv = advantages_batch[:, None]
    unclipped_obj = ratio * adv
    clipped_ratio = mx.clip(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
    clipped_obj = clipped_ratio * adv
    
    # 4. KL Penalty (approximate KL divergence)
    kl_div = ref_logprobs - logprobs
    
    # 5. Final Loss calculation
    token_loss = - (mx.minimum(unclipped_obj, clipped_obj) - BETA * kl_div)
    
    masked_loss = token_loss * final_mask
    loss = mx.sum(masked_loss) / mx.sum(final_mask)
    
    return loss

def generate_batched(model, tokenizer, prompt_text, G=8, max_tokens=150, temperature=0.7):
    # Format and tokenize prompt
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(formatted_prompt)
    prompt_len = len(prompt_tokens)
    prompt_array = mx.array([prompt_tokens])
    
    # 1. Prefix Cache
    cache = model.make_cache()
    logits = model(prompt_array, cache=cache)
    
    for c in cache:
        c.keys = mx.repeat(c.keys, G, axis=0)
        c.values = mx.repeat(c.values, G, axis=0)
        
    next_logits = mx.repeat(logits[:, -1, :], G, axis=0)
    
    def sample(logits):
        return mx.random.categorical(logits / temperature)
        
    y = sample(next_logits)[:, None]
    
    generated_tokens = [[] for _ in range(G)]
    active_seqs = [True] * G
    eos_id = tokenizer.eos_token_id
    
    for i in range(G):
        token = y[i, 0].item()
        generated_tokens[i].append(token)
        if token == eos_id:
            active_seqs[i] = False
            
    # 2. Generation Loop
    for step in range(1, max_tokens):
        if not any(active_seqs):
            break
            
        logits = model(y, cache=cache)
        next_logits = logits[:, -1, :]
        y = sample(next_logits)[:, None]
        
        for i in range(G):
            if active_seqs[i]:
                token = y[i, 0].item()
                generated_tokens[i].append(token)
                if token == eos_id:
                    active_seqs[i] = False
                    
    responses = [tokenizer.decode(tokens).strip() for tokens in generated_tokens]
    full_tokens = [prompt_tokens + tokens for tokens in generated_tokens]
    return responses, full_tokens, prompt_len

def pad_sequences(sequences, pad_id):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in sequences]
    mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in sequences]
    shifted_mask = [m[1:] for m in mask]
    return mx.array(padded), mx.array(shifted_mask)

def main():
    print("Loading models for full training pipeline...")
    model_path = "storage/models/gemma-3-270m-it"
    
    model, tokenizer = load(model_path)
    model.train()
    
    ref_model, _ = load(model_path)
    ref_model.eval()
    ref_model.freeze()
    
    optimizer = optim.AdamW(learning_rate=LR)
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss)
    
    dataset = generate_math_dataset(num_samples=NUM_STEPS)
    
    for step, data in tqdm(enumerate(dataset)):
        step_start = time.time()
        prompt_text = data['prompt']
        ground_truth = data['ground_truth']
        
        logging.info(f"[Step {step+1}/{NUM_STEPS}] Problem: What is {data['ground_truth']}?")
        
        # 1. Efficient Batched Rollout Phase
        responses, token_sequences, prompt_len = generate_batched(
            model, tokenizer, prompt_text, G=G, max_tokens=150, temperature=0.7
        )
        
        # 2. Score and Advantage
        rewards = [reward_function(r, ground_truth) for r in responses]
        rewards_tensor = mx.array(rewards)
        mean_reward = mx.mean(rewards_tensor)
        std_reward = mx.std(rewards_tensor) + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        # 3. Prepare Batch Tensors
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids_batch, loss_mask = pad_sequences(token_sequences, pad_id)
        prompt_lengths = mx.array([prompt_len] * G)
        
        # 4. Forward and Backward Pass
        loss, grads = loss_and_grad_fn(
            model, ref_model, input_ids_batch, advantages, prompt_lengths, loss_mask
        )
        
        # 5. Optimizer Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        step_time = time.time() - step_start
        logging.info(f"Rewards: {rewards} | Loss: {loss.item():.4f} | Time: {step_time:.2f}s")
        
        if (step + 1) % 10 == 0:
            save_path = f"storage/models/grpo_checkpoint_step_{step+1}"
            logging.info(f"Saving checkpoint to {save_path}...")
            model.save_weights(f"{save_path}.safetensors")

if __name__ == "__main__":
    main()
