import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import BatchKVCache
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

class MLXBatchGenerator:
    """
    A custom batched generator for MLX models that performs true batch generation.
    It left-pads input prompts of varying lengths, processes them in parallel,
    and dynamically filters out finished sequences mid-generation to optimize performance.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        verbose: bool = False,
    ) -> list[str]:
        if not prompts:
            return []

        # 1. Tokenize prompts if they are passed as strings
        tokenized = []
        for p in prompts:
            if isinstance(p, str):
                tokenized.append(self.tokenizer.encode(p))
            elif isinstance(p, list) and all(isinstance(x, int) for x in p):
                tokenized.append(p)
            else:
                raise ValueError("Prompts must be a list of strings or list of token IDs")

        batch_size = len(tokenized)

        # 2. Determine padding and create padded array
        max_len = max(len(t) for t in tokenized)
        left_padding = [max_len - len(t) for t in tokenized]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        padded_tokens = [[pad_id] * lp + t for lp, t in zip(left_padding, tokenized)]
        x = mx.array(padded_tokens)

        # 3. Create BatchKVCache objects for each layer
        cache = [BatchKVCache(left_padding) for _ in self.model.layers]

        # 4. Initialize outputs and tracking
        generated_tokens = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        active_to_original = list(range(batch_size))

        # Setup stopping EOS tokens
        eos_ids = self.tokenizer.eos_token_id
        if eos_ids is None:
            eos_ids = []
        elif isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        else:
            eos_ids = list(eos_ids)
        eos_set = set(eos_ids)

        # Create sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)

        # 5. Prefill step
        logits = self.model(x, cache=cache)
        next_logits = logits[:, -1, :]

        # Calculate logprobs and sample
        logprobs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
        next_tokens = sampler(logprobs)[:, None]

        # Record first generated token
        for active_idx, orig_idx in enumerate(active_to_original):
            tok = next_tokens[active_idx].item()
            generated_tokens[orig_idx].append(tok)

        # Use progress bar if verbose
        range_iter = range(1, max_tokens)
        if verbose:
            range_iter = tqdm(range_iter, desc="Generating batch tokens")

        # 6. Autoregressive decode loop
        for step in range_iter:
            # Evaluate current batch to execute the graph
            mx.eval(next_tokens)

            # Check stopping criteria for active sequences
            new_active_indices = []
            for active_idx, orig_idx in enumerate(active_to_original):
                tok = next_tokens[active_idx].item()
                if tok in eos_set:
                    finished[orig_idx] = True
                if not finished[orig_idx]:
                    new_active_indices.append(active_idx)

            if not new_active_indices:
                break

            # If some sequences finished, filter cache and tensors in-place
            if len(new_active_indices) < len(active_to_original):
                for c in cache:
                    c.filter(new_active_indices)
                next_tokens = next_tokens[new_active_indices]
                active_to_original = [active_to_original[i] for i in new_active_indices]
                
                # Evaluate cache state to compile filter and reclaim memory
                mx.eval([c.state for c in cache])

            # Forward pass on the next token
            logits = self.model(next_tokens, cache=cache)
            next_logits = logits[:, -1, :]

            logprobs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
            next_tokens = sampler(logprobs)[:, None]

            # Record generated tokens
            for active_idx, orig_idx in enumerate(active_to_original):
                tok = next_tokens[active_idx].item()
                generated_tokens[orig_idx].append(tok)
                
            # Periodically clear MLX cache to free unreferenced allocations
            if step % 256 == 0:
                mx.clear_cache()

        # 7. Decode generated token IDs back to strings
        responses = [self.tokenizer.decode(toks) for toks in generated_tokens]
        return responses

    def generate_with_diverse_rollouts(
        self,
        prompts: list[str] | list[list[int]],
        group_size: int = 1,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        verbose: bool = False,
    ) -> list[list[str]]:
        """
        Generate multiple diverse rollouts (group_size) for each prompt.
        Computes the prefill (prompt KV cache) exactly once per unique prompt, 
        then replicates it along the batch dimension to branch into group_size rollouts.
        This saves massive prefill compute during group rollout phases (e.g. GRPO/PPO).
        """
        if not prompts:
            return []

        if group_size <= 0:
            raise ValueError("group_size must be greater than 0")

        # 1. Tokenize prompts
        tokenized = []
        for p in prompts:
            if isinstance(p, str):
                tokenized.append(self.tokenizer.encode(p))
            elif isinstance(p, list) and all(isinstance(x, int) for x in p):
                tokenized.append(p)
            else:
                raise ValueError("Prompts must be a list of strings or list of token IDs")

        orig_batch_size = len(tokenized)
        total_batch_size = orig_batch_size * group_size

        # 2. Determine padding and create padded array for prefill
        max_len = max(len(t) for t in tokenized)
        left_padding = [max_len - len(t) for t in tokenized]

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        padded_tokens = [[pad_id] * lp + t for lp, t in zip(left_padding, tokenized)]
        x = mx.array(padded_tokens)

        # 3. Prefill (Compute prompt KV cache once per unique prompt)
        cache = [BatchKVCache(left_padding) for _ in self.model.layers]
        logits = self.model(x, cache=cache)
        next_logits = logits[:, -1, :]

        # 4. Duplicate/repeat cache and next_logits for group_size rollouts
        if group_size > 1:
            for c in cache:
                c.keys = mx.repeat(c.keys, group_size, axis=0)
                c.values = mx.repeat(c.values, group_size, axis=0)
                c.left_padding = mx.repeat(c.left_padding, group_size, axis=0)
                c.offset = mx.repeat(c.offset, group_size, axis=0)
            next_logits = mx.repeat(next_logits, group_size, axis=0)

        # 5. Initialize outputs and tracking for total_batch_size paths
        generated_tokens = [[] for _ in range(total_batch_size)]
        finished = [False] * total_batch_size
        active_to_original = list(range(total_batch_size))

        # Setup stopping EOS tokens
        eos_ids = self.tokenizer.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (list(eos_ids) if eos_ids else [])
        eos_set = set(eos_ids)

        # Create sampler (usually temperature > 0 for diverse rollouts)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        # Sample the first generated token
        logprobs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
        next_tokens = sampler(logprobs)[:, None]

        # Record first generated token
        for active_idx, orig_idx in enumerate(active_to_original):
            tok = next_tokens[active_idx].item()
            generated_tokens[orig_idx].append(tok)

        # Use progress bar if verbose
        range_iter = range(1, max_tokens)
        if verbose:
            range_iter = tqdm(range_iter, desc="Generating diverse rollouts")

        # 6. Autoregressive decode loop
        for step in range_iter:
            mx.eval(next_tokens)

            # Check stopping criteria for active sequences
            new_active_indices = []
            for active_idx, orig_idx in enumerate(active_to_original):
                tok = next_tokens[active_idx].item()
                if tok in eos_set:
                    finished[orig_idx] = True
                if not finished[orig_idx]:
                    new_active_indices.append(active_idx)

            if not new_active_indices:
                break

            # If some sequences finished, filter cache and tensors in-place
            if len(new_active_indices) < len(active_to_original):
                for c in cache:
                    c.filter(new_active_indices)
                next_tokens = next_tokens[new_active_indices]
                active_to_original = [active_to_original[i] for i in new_active_indices]
                
                # Evaluate cache state to compile filter and reclaim memory
                mx.eval([c.state for c in cache])

            # Forward pass on the next token
            logits = self.model(next_tokens, cache=cache)
            next_logits = logits[:, -1, :]

            logprobs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
            next_tokens = sampler(logprobs)[:, None]

            # Record generated tokens
            for active_idx, orig_idx in enumerate(active_to_original):
                tok = next_tokens[active_idx].item()
                generated_tokens[orig_idx].append(tok)
                
            # Periodically clear MLX cache to free unreferenced allocations
            if step % 256 == 0:
                mx.clear_cache()

        # 7. Decode generated token IDs back to strings
        all_responses = [self.tokenizer.decode(toks) for toks in generated_tokens]
        
        # 8. Group responses back into list[list[str]] (shape: [orig_batch_size, group_size])
        grouped_responses = []
        for i in range(orig_batch_size):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            grouped_responses.append(all_responses[start_idx:end_idx])
            
        return grouped_responses

