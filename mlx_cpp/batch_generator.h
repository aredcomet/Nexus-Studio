#pragma once
#include <vector>
#include <memory>
#include <set>
#include "mlx/mlx.h"
#include "model.h"
#include "tokenizer.h"

class KVCache {
public:
    mlx::core::array keys;
    mlx::core::array values;
    mlx::core::array left_padding;
    mlx::core::array offset;
    int idx = 0;

    KVCache(const std::vector<int>& left_padding_vec);
    std::pair<mlx::core::array, mlx::core::array> update_and_fetch(const mlx::core::array& new_keys, const mlx::core::array& new_values);
    void filter(const mlx::core::array& batch_indices);
    void repeat(int group_size);
    mlx::core::array make_mask(int N);
};

class MLXBatchGenerator {
public:
    MLXBatchGenerator(Model& model, Tokenizer& tokenizer);

    std::vector<std::vector<int>> generate(
        const std::vector<std::vector<int>>& prompts,
        int max_tokens = 1024,
        float temperature = 0.0f,
        float top_p = 1.0f,
        bool verbose = false
    );

    std::vector<std::vector<std::vector<int>>> generate_with_diverse_rollouts(
        const std::vector<std::vector<int>>& prompts,
        int group_size = 1,
        int max_tokens = 1024,
        float temperature = 0.7f,
        float top_p = 1.0f,
        bool verbose = false
    );

private:
    Model& model_;
    Tokenizer& tokenizer_;
};
