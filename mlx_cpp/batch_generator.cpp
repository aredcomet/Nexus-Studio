#include "batch_generator.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <limits>

KVCache::KVCache(const std::vector<int>& left_padding_vec)
    : keys(0.0f), values(0.0f), left_padding(0.0f), offset(0.0f) {
    left_padding = mlx::core::array(left_padding_vec.data(), { (int)left_padding_vec.size() }, mlx::core::int32);
    offset = -left_padding;
}

std::pair<mlx::core::array, mlx::core::array> KVCache::update_and_fetch(const mlx::core::array& new_keys, const mlx::core::array& new_values) {
    int prev = idx;
    int num_steps = new_keys.shape(2);
    int B = new_keys.shape(0);
    int H = new_keys.shape(1);
    int Dk = new_keys.shape(3);
    int Dv = new_values.shape(3);

    if (keys.ndim() == 0) {
        int initial_steps = ((num_steps + 255) / 256) * 256;
        keys = mlx::core::zeros({B, H, initial_steps, Dk}, new_keys.dtype());
        values = mlx::core::zeros({B, H, initial_steps, Dv}, new_values.dtype());
    } else if (prev + num_steps > keys.shape(2)) {
        int new_size = ((prev + num_steps + 255) / 256) * 256;
        auto pad_k = mlx::core::zeros({B, H, new_size - keys.shape(2), Dk}, keys.dtype());
        auto pad_v = mlx::core::zeros({B, H, new_size - values.shape(2), Dv}, values.dtype());
        keys = mlx::core::concatenate({keys, pad_k}, 2);
        values = mlx::core::concatenate({values, pad_v}, 2);
    }

    auto slice_update = [](const mlx::core::array& src, const mlx::core::array& update, int start) {
        int B = src.shape(0);
        int H = src.shape(1);
        int L = src.shape(2);
        int D = src.shape(3);
        int S = update.shape(2);
        
        std::vector<mlx::core::array> parts;
        if (start > 0) {
            parts.push_back(mlx::core::slice(src, {0, 0, 0, 0}, {B, H, start, D}));
        }
        parts.push_back(update);
        if (start + S < L) {
            parts.push_back(mlx::core::slice(src, {0, 0, start + S, 0}, {B, H, L, D}));
        }
        return mlx::core::concatenate(parts, 2);
    };

    keys = slice_update(keys, new_keys, prev);
    values = slice_update(values, new_values, prev);
    
    offset = offset + num_steps;
    idx += num_steps;

    auto k_fetched = mlx::core::slice(keys, {0, 0, 0, 0}, {B, H, idx, Dk});
    auto v_fetched = mlx::core::slice(values, {0, 0, 0, 0}, {B, H, idx, Dv});
    return {k_fetched, v_fetched};
}

void KVCache::filter(const mlx::core::array& batch_indices) {
    if (keys.ndim() > 0) {
        keys = mlx::core::take(keys, batch_indices, 0);
        values = mlx::core::take(values, batch_indices, 0);
    }
    offset = mlx::core::take(offset, batch_indices, 0);
    left_padding = mlx::core::take(left_padding, batch_indices, 0);
    
    auto min_left_pad_arr = mlx::core::min(left_padding);
    mlx::core::eval(min_left_pad_arr);
    int min_left_pad = min_left_pad_arr.item<int>();
    
    if (min_left_pad > 0) {
        if (keys.ndim() > 0) {
            int B = keys.shape(0);
            int H = keys.shape(1);
            int L = keys.shape(2);
            int Dk = keys.shape(3);
            int Dv = values.shape(3);
            keys = mlx::core::slice(keys, {0, 0, min_left_pad, 0}, {B, H, L, Dk});
            values = mlx::core::slice(values, {0, 0, min_left_pad, 0}, {B, H, L, Dv});
        }
        idx -= min_left_pad;
        left_padding = left_padding - min_left_pad;
    }
}

void KVCache::repeat(int group_size) {
    if (group_size > 1) {
        if (keys.ndim() > 0) {
            keys = mlx::core::repeat(keys, group_size, 0);
            values = mlx::core::repeat(values, group_size, 0);
        }
        if (left_padding.size() > 0) {
            left_padding = mlx::core::repeat(left_padding, group_size, 0);
        }
        if (offset.size() > 0) {
            offset = mlx::core::repeat(offset, group_size, 0);
        }
    }
}

mlx::core::array KVCache::make_mask(int N) {
    auto rinds = mlx::core::arange(0.0f, (float)(idx + N), 1.0f, mlx::core::float32); // (idx + N,)
    auto linds = (idx > 0) ? mlx::core::arange((float)idx, (float)(idx + N), 1.0f, mlx::core::float32) : rinds; // (N,)
    
    auto linds_expanded = mlx::core::expand_dims(linds, -1); // (N, 1)
    auto rinds_expanded = mlx::core::expand_dims(rinds, 0); // (1, idx + N)
    
    auto mask = linds_expanded >= rinds_expanded; // (N, idx + N)
    
    auto lp_expanded = mlx::core::expand_dims(mlx::core::expand_dims(mlx::core::expand_dims(left_padding, -1), -1), -1); // (B, 1, 1, 1)
    auto rinds_expanded_4d = mlx::core::expand_dims(mlx::core::expand_dims(rinds_expanded, 0), 0); // (1, 1, 1, idx + N)
    auto pad_mask = lp_expanded <= rinds_expanded_4d; // (B, 1, 1, idx + N)
    
    auto mask_4d = mlx::core::expand_dims(mlx::core::expand_dims(mask, 0), 0); // (1, 1, N, idx + N)
    auto final_mask = mask_4d & pad_mask; // (B, 1, N, idx + N)
    return final_mask;
}


MLXBatchGenerator::MLXBatchGenerator(Model& model, Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

static mlx::core::array sample_token(const mlx::core::array& logits, float temperature, float top_p) {
    if (temperature == 0.0f) {
        return mlx::core::argmax(logits, -1, true); // (B, 1)
    }

    auto scaled_logits = logits / temperature; // (B, V)

    if (top_p <= 0.0f || top_p >= 1.0f) {
        auto tokens = mlx::core::random::categorical(scaled_logits, -1, std::nullopt); // (B,)
        return mlx::core::expand_dims(tokens, -1); // (B, 1)
    }

    // Sort descending along the last axis (-1)
    auto sorted_indices = mlx::core::argsort(-scaled_logits, -1); // (B, V)
    auto sorted_logits = mlx::core::take_along_axis(scaled_logits, sorted_indices, -1); // (B, V)

    // Compute cumulative sum of probabilities
    auto probs = mlx::core::softmax(sorted_logits, -1); // (B, V)
    auto cumsum = mlx::core::cumsum(probs, -1); // (B, V)

    // Shift cumulative sum right by 1
    int B = logits.shape(0);
    int V = logits.shape(1);
    auto zeros = mlx::core::zeros({B, 1}, cumsum.dtype());
    auto slice_cumsum = mlx::core::slice(cumsum, {0, 0}, {B, V - 1});
    auto shifted_cumsum = mlx::core::concatenate({zeros, slice_cumsum}, -1);

    auto exclude_mask = shifted_cumsum > top_p; // (B, V)

    // Apply exclude mask (set excluded logits to -inf)
    auto masked_sorted_logits = mlx::core::where(
        exclude_mask,
        mlx::core::array(-std::numeric_limits<float>::infinity()),
        sorted_logits
    );

    // Sample from masked sorted logits
    auto sampled_sorted_idx = mlx::core::random::categorical(masked_sorted_logits, -1, std::nullopt); // (B,)
    auto sampled_sorted_idx_expanded = mlx::core::expand_dims(sampled_sorted_idx, -1); // (B, 1)

    // Map back to original indices
    return mlx::core::take_along_axis(sorted_indices, sampled_sorted_idx_expanded, -1); // (B, 1)
}

static int get_token_item(const mlx::core::array& arr, int idx) {
    return mlx::core::reshape(mlx::core::slice(arr, {idx, 0}, {idx + 1, 1}), {}).item<int>();
}

std::vector<std::vector<int>> MLXBatchGenerator::generate(
    const std::vector<std::vector<int>>& prompts,
    int max_tokens,
    float temperature,
    float top_p,
    bool verbose
) {
    if (prompts.empty()) return {};

    int batch_size = prompts.size();

    // 1. Determine padding and create padded array
    int max_len = 0;
    for (const auto& p : prompts) {
        max_len = std::max(max_len, (int)p.size());
    }

    std::vector<int> left_padding(batch_size);
    std::vector<int> padded_tokens;
    padded_tokens.reserve(batch_size * max_len);

    int pad_id = tokenizer_.pad_token_id;

    for (int i = 0; i < batch_size; ++i) {
        left_padding[i] = max_len - prompts[i].size();
        for (int j = 0; j < left_padding[i]; ++j) {
            padded_tokens.push_back(pad_id);
        }
        for (int tok : prompts[i]) {
            padded_tokens.push_back(tok);
        }
    }

    auto x = mlx::core::array(padded_tokens.data(), {batch_size, max_len}, mlx::core::int32);

    // 2. Create BatchKVCache objects for each layer
    std::vector<std::shared_ptr<KVCache>> cache;
    for (int i = 0; i < model_.args.num_hidden_layers; ++i) {
        cache.push_back(std::make_shared<KVCache>(left_padding));
    }

    // 3. Initialize outputs and tracking
    std::vector<std::vector<int>> generated_tokens(batch_size);
    std::vector<bool> finished(batch_size, false);
    std::vector<int> active_to_original(batch_size);
    for (int i = 0; i < batch_size; ++i) active_to_original[i] = i;

    std::set<int> eos_set(tokenizer_.eos_token_ids.begin(), tokenizer_.eos_token_ids.end());

    // 4. Prefill step
    auto logits = model_(x, cache);
    auto next_logits = mlx::core::slice(logits, {0, max_len - 1, 0}, {batch_size, max_len, logits.shape(2)});
    next_logits = mlx::core::reshape(next_logits, {batch_size, -1}); // (B, V)

    auto next_tokens = sample_token(next_logits, temperature, top_p); // (B, 1)

    // Record first generated token
    for (int i = 0; i < batch_size; ++i) {
        int tok = get_token_item(next_tokens, i);
        generated_tokens[active_to_original[i]].push_back(tok);
    }

    // 5. Autoregressive decode loop
    for (int step = 1; step < max_tokens; ++step) {
        // Evaluate current batch to execute the graph
        mlx::core::eval(next_tokens);

        // Check stopping criteria and record tokens
        std::vector<int> new_active_indices;
        for (size_t active_idx = 0; active_idx < active_to_original.size(); ++active_idx) {
            int orig_idx = active_to_original[active_idx];
            int tok = get_token_item(next_tokens, (int)active_idx);
            
            if (step == 1) {
                if (eos_set.count(tok)) {
                    finished[orig_idx] = true;
                }
            } else {
                if (eos_set.count(tok)) {
                    finished[orig_idx] = true;
                    generated_tokens[orig_idx].push_back(tok);
                } else {
                    generated_tokens[orig_idx].push_back(tok);
                }
            }
            
            if (!finished[orig_idx]) {
                new_active_indices.push_back(active_idx);
            }
        }

        if (new_active_indices.empty()) {
            break;
        }

        // If some sequences finished, filter cache and tensors in-place
        if (new_active_indices.size() < active_to_original.size()) {
            auto idx_arr = mlx::core::array(new_active_indices.data(), { (int)new_active_indices.size() }, mlx::core::int32);
            for (auto& c : cache) {
                c->filter(idx_arr);
            }
            next_tokens = mlx::core::take(next_tokens, idx_arr, 0);
            
            std::vector<int> new_active_to_orig;
            for (int idx : new_active_indices) {
                new_active_to_orig.push_back(active_to_original[idx]);
            }
            active_to_original = new_active_to_orig;
        }

        // Forward pass on the next token
        logits = model_(next_tokens, cache);
        next_logits = mlx::core::reshape(logits, {(int)active_to_original.size(), -1}); // (B, V)

        next_tokens = sample_token(next_logits, temperature, top_p);

        // Periodically clear MLX cache
        if (step % 256 == 0) {
            mlx::core::clear_cache();
        }
    }

    return generated_tokens;
}

std::vector<std::vector<std::vector<int>>> MLXBatchGenerator::generate_with_diverse_rollouts(
    const std::vector<std::vector<int>>& prompts,
    int group_size,
    int max_tokens,
    float temperature,
    float top_p,
    bool verbose
) {
    if (prompts.empty()) return {};
    if (group_size <= 0) throw std::invalid_argument("group_size must be greater than 0");

    int orig_batch_size = prompts.size();
    int total_batch_size = orig_batch_size * group_size;

    // 1. Determine padding and create padded array for prefill
    int max_len = 0;
    for (const auto& p : prompts) {
        max_len = std::max(max_len, (int)p.size());
    }

    std::vector<int> left_padding(orig_batch_size);
    std::vector<int> padded_tokens;
    padded_tokens.reserve(orig_batch_size * max_len);

    int pad_id = tokenizer_.pad_token_id;

    for (int i = 0; i < orig_batch_size; ++i) {
        left_padding[i] = max_len - prompts[i].size();
        for (int j = 0; j < left_padding[i]; ++j) {
            padded_tokens.push_back(pad_id);
        }
        for (int tok : prompts[i]) {
            padded_tokens.push_back(tok);
        }
    }

    auto x = mlx::core::array(padded_tokens.data(), {orig_batch_size, max_len}, mlx::core::int32);

    // 2. Prefill (Compute prompt KV cache once per unique prompt)
    std::vector<std::shared_ptr<KVCache>> cache;
    for (int i = 0; i < model_.args.num_hidden_layers; ++i) {
        cache.push_back(std::make_shared<KVCache>(left_padding));
    }

    auto logits = model_(x, cache);
    auto next_logits = mlx::core::slice(logits, {0, max_len - 1, 0}, {orig_batch_size, max_len, logits.shape(2)});
    next_logits = mlx::core::reshape(next_logits, {orig_batch_size, -1}); // (orig_batch_size, V)

    // 3. Duplicate/repeat cache and next_logits for group_size rollouts
    if (group_size > 1) {
        for (auto& c : cache) {
            c->repeat(group_size);
        }
        next_logits = mlx::core::repeat(next_logits, group_size, 0);
    }

    // 4. Initialize outputs and tracking for total_batch_size paths
    std::vector<std::vector<int>> generated_tokens(total_batch_size);
    std::vector<bool> finished(total_batch_size, false);
    std::vector<int> active_to_original(total_batch_size);
    for (int i = 0; i < total_batch_size; ++i) active_to_original[i] = i;

    std::set<int> eos_set(tokenizer_.eos_token_ids.begin(), tokenizer_.eos_token_ids.end());

    // Sample the first generated token
    auto next_tokens = sample_token(next_logits, temperature, top_p); // (total_batch_size, 1)

    // Record first generated token
    for (int i = 0; i < total_batch_size; ++i) {
        int tok = get_token_item(next_tokens, i);
        generated_tokens[active_to_original[i]].push_back(tok);
    }

    // 5. Autoregressive decode loop
    for (int step = 1; step < max_tokens; ++step) {
        mlx::core::eval(next_tokens);

        // Check stopping criteria and record tokens
        std::vector<int> new_active_indices;
        for (size_t active_idx = 0; active_idx < active_to_original.size(); ++active_idx) {
            int orig_idx = active_to_original[active_idx];
            int tok = get_token_item(next_tokens, (int)active_idx);
            
            if (step == 1) {
                if (eos_set.count(tok)) {
                    finished[orig_idx] = true;
                }
            } else {
                if (eos_set.count(tok)) {
                    finished[orig_idx] = true;
                    generated_tokens[orig_idx].push_back(tok);
                } else {
                    generated_tokens[orig_idx].push_back(tok);
                }
            }
            
            if (!finished[orig_idx]) {
                new_active_indices.push_back(active_idx);
            }
        }

        if (new_active_indices.empty()) {
            break;
        }

        // If some sequences finished, filter cache and tensors in-place
        if (new_active_indices.size() < active_to_original.size()) {
            auto idx_arr = mlx::core::array(new_active_indices.data(), { (int)new_active_indices.size() }, mlx::core::int32);
            for (auto& c : cache) {
                c->filter(idx_arr);
            }
            next_tokens = mlx::core::take(next_tokens, idx_arr, 0);
            
            std::vector<int> new_active_to_orig;
            for (int idx : new_active_indices) {
                new_active_to_orig.push_back(active_to_original[idx]);
            }
            active_to_original = new_active_to_orig;
        }

        // Forward pass on the next token
        logits = model_(next_tokens, cache);
        next_logits = mlx::core::reshape(logits, {(int)active_to_original.size(), -1}); // (B, V)

        next_tokens = sample_token(next_logits, temperature, top_p);

        if (step % 256 == 0) {
            mlx::core::clear_cache();
        }
    }

    // 6. Group responses back into shape: [orig_batch_size, group_size]
    std::vector<std::vector<std::vector<int>>> grouped_responses(orig_batch_size);
    for (int i = 0; i < orig_batch_size; ++i) {
        grouped_responses[i].resize(group_size);
        for (int g = 0; g < group_size; ++g) {
            int orig_idx = i * group_size + g;
            grouped_responses[i][g] = generated_tokens[orig_idx];
        }
    }

    return grouped_responses;
}
