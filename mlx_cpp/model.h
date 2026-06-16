#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include "mlx/mlx.h"

struct ModelArgs {
    std::string model_type = "";
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int intermediate_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    float rms_norm_eps = 1e-05f;
    int vocab_size = 0;
    int head_dim = 0;
    int max_position_embeddings = 0;
    bool tie_word_embeddings = false;
    
    // RoPE parameters
    float rope_theta = 0.0f;
    float rope_factor = 0.0f;
    int original_max_position_embeddings = 0;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
    float llama_4_scaling_beta = 0.0f;

    static ModelArgs load_from_config(const std::string& config_path);
};

class QuantizedEmbedding {
public:
    mlx::core::array weight;
    mlx::core::array scales;
    int group_size = 32;
    int bits = 4;
    std::string mode = "mxfp4";

    QuantizedEmbedding() : weight(0.0f), scales(0.0f) {}
    mlx::core::array operator()(const mlx::core::array& x);
};

class QuantizedLinear {
public:
    mlx::core::array weight;
    mlx::core::array scales;
    std::optional<mlx::core::array> biases;
    std::optional<mlx::core::array> bias;
    int group_size = 32;
    int bits = 4;
    std::string mode = "mxfp4";

    QuantizedLinear() : weight(0.0f), scales(0.0f) {}
    mlx::core::array operator()(const mlx::core::array& x);
};

class Attention {
public:
    Attention(const ModelArgs& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& attn_scale,
        const std::optional<mlx::core::array>& mask,
        std::shared_ptr<class KVCache>& cache
    );

    int n_heads;
    int n_kv_heads;
    int head_dim;
    float scale;
    QuantizedLinear q_proj;
    QuantizedLinear k_proj;
    QuantizedLinear v_proj;
    QuantizedLinear o_proj;
    mlx::core::array rope_freqs;
};

class MLP {
public:
    MLP(const ModelArgs& args);
    mlx::core::array operator()(const mlx::core::array& x);

    QuantizedLinear gate_proj;
    QuantizedLinear down_proj;
    QuantizedLinear up_proj;
};

class TransformerBlock {
public:
    TransformerBlock(const ModelArgs& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& attn_scale,
        const std::optional<mlx::core::array>& mask,
        std::shared_ptr<class KVCache>& cache
    );

    Attention self_attn;
    MLP mlp;
    mlx::core::array input_layernorm_weight;
    mlx::core::array post_attention_layernorm_weight;
    float rms_norm_eps;
};

class LanguageModel {
public:
    LanguageModel(const ModelArgs& args);
    mlx::core::array operator()(
        const mlx::core::array& inputs,
        std::vector<std::shared_ptr<class KVCache>>& cache
    );

    ModelArgs args;
    QuantizedEmbedding embed_tokens;
    std::vector<TransformerBlock> layers;
    mlx::core::array norm_weight;
};

class Model {
public:
    Model(const ModelArgs& args);
    mlx::core::array operator()(
        const mlx::core::array& inputs,
        std::vector<std::shared_ptr<class KVCache>>& cache
    );

    void load_weights(const std::string& safetensors_path);

    ModelArgs args;
    LanguageModel model;
    QuantizedLinear lm_head;
};
