#include "model.h"
#include "batch_generator.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>

// Helper to compute Yarn RoPE frequencies
static mlx::core::array compute_yarn_freqs(int dims, float base, float scaling_factor, int original_max_position_embeddings, float beta_fast, float beta_slow) {
    auto yarn_find_correction_dim = [&](float num_rotations) {
        return (dims * std::log(original_max_position_embeddings / (num_rotations * 2.0f * M_PI))) / (2.0f * std::log(base));
    };
    
    int low = std::max((int)std::floor(yarn_find_correction_dim(beta_fast)), 0);
    int high = std::min((int)std::ceil(yarn_find_correction_dim(beta_slow)), dims - 1);
    
    auto arange = mlx::core::arange(0.0f, (float)dims, 2.0f, mlx::core::float32);
    auto exponents = arange / (float)dims;
    auto freq_extra = mlx::core::power(mlx::core::array(base), exponents);
    auto freq_inter = scaling_factor * freq_extra;
    
    float min_val = low;
    float max_val = high;
    if (min_val == max_val) max_val += 0.001f;
    
    auto ramp = (mlx::core::arange(0.0f, (float)(dims / 2), 1.0f, mlx::core::float32) - min_val) / (max_val - min_val);
    auto freq_mask = 1.0f - mlx::core::clip(ramp, mlx::core::array(0.0f), mlx::core::array(1.0f));
    
    auto freqs = (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0f - freq_mask));
    return freqs;
}

// SuScaled/Yarn RoPE scale helper
static mlx::core::array get_llama_4_attn_scale(int size, const mlx::core::array& offset, float beta, int max_position_embeddings) {
    auto offset_expanded = mlx::core::expand_dims(offset, -1); // (B, 1)
    auto arange = mlx::core::arange(0.0f, (float)size, 1.0f, mlx::core::float32); // (size,)
    auto positions = arange + offset_expanded; // (B, size)
    
    auto scaling = 1.0f + beta * mlx::core::log(
        1.0f + mlx::core::floor(positions / (float)max_position_embeddings)
    );
    
    return mlx::core::expand_dims(mlx::core::expand_dims(scaling, 1), -1); // (B, 1, size, 1)
}

mlx::core::array QuantizedEmbedding::operator()(const mlx::core::array& x) {
    auto w = mlx::core::take(weight, x, 0);
    auto s = mlx::core::take(scales, x, 0);
    return mlx::core::dequantize(w, s, std::nullopt, group_size, bits, mode);
}

mlx::core::array QuantizedLinear::operator()(const mlx::core::array& x) {
    auto out = mlx::core::quantized_matmul(
        x, weight, scales, biases, true, group_size, bits, mode
    );
    if (bias.has_value()) {
        out = out + bias.value();
    }
    return out;
}

Attention::Attention(const ModelArgs& args) : rope_freqs(0.0f) {
    n_heads = args.num_attention_heads;
    n_kv_heads = args.num_key_value_heads;
    head_dim = args.head_dim;
    scale = 1.0f / std::sqrt((float)head_dim);
    
    rope_freqs = compute_yarn_freqs(
        head_dim,
        args.rope_theta,
        args.rope_factor,
        args.original_max_position_embeddings,
        args.beta_fast,
        args.beta_slow
    );
}

mlx::core::array Attention::operator()(
    const mlx::core::array& x,
    const mlx::core::array& attn_scale,
    const std::optional<mlx::core::array>& mask,
    std::shared_ptr<KVCache>& cache
) {
    // x shape: (B, L, D)
    int B = x.shape(0);
    int L = x.shape(1);
    
    auto queries = q_proj(x);
    auto keys = k_proj(x);
    auto values = v_proj(x);
    
    // Reshape & transpose for multi-head attention: (B, n_heads, L, head_dim)
    queries = mlx::core::transpose(mlx::core::reshape(queries, {B, L, n_heads, head_dim}), {0, 2, 1, 3});
    keys = mlx::core::transpose(mlx::core::reshape(keys, {B, L, n_kv_heads, head_dim}), {0, 2, 1, 3});
    values = mlx::core::transpose(mlx::core::reshape(values, {B, L, n_kv_heads, head_dim}), {0, 2, 1, 3});
    
    if (cache) {
        queries = mlx::core::fast::rope(queries, head_dim, false, std::nullopt, 1.0f, cache->offset, rope_freqs);
        keys = mlx::core::fast::rope(keys, head_dim, false, std::nullopt, 1.0f, cache->offset, rope_freqs);
        auto updated = cache->update_and_fetch(keys, values);
        keys = updated.first;
        values = updated.second;
    } else {
        // Fallback for no cache (though we always use cache in SFT)
        queries = mlx::core::fast::rope(queries, head_dim, false, std::nullopt, 1.0f, 0, rope_freqs);
        keys = mlx::core::fast::rope(keys, head_dim, false, std::nullopt, 1.0f, 0, rope_freqs);
    }
    
    queries = queries * attn_scale;
    
    auto output = mlx::core::fast::scaled_dot_product_attention(
        queries, keys, values, scale, "", mask
    );
    
    output = mlx::core::reshape(mlx::core::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return o_proj(output);
}

MLP::MLP(const ModelArgs& args) {}

mlx::core::array MLP::operator()(const mlx::core::array& x) {
    auto silu = [](const mlx::core::array& a) { return a * mlx::core::sigmoid(a); };
    return down_proj(silu(gate_proj(x)) * up_proj(x));
}

TransformerBlock::TransformerBlock(const ModelArgs& args)
    : self_attn(args), mlp(args), input_layernorm_weight(0.0f), post_attention_layernorm_weight(0.0f) {
    rms_norm_eps = args.rms_norm_eps;
}

mlx::core::array TransformerBlock::operator()(
    const mlx::core::array& x,
    const mlx::core::array& attn_scale,
    const std::optional<mlx::core::array>& mask,
    std::shared_ptr<KVCache>& cache
) {
    auto r = self_attn(mlx::core::fast::rms_norm(x, input_layernorm_weight, rms_norm_eps), attn_scale, mask, cache);
    auto h = x + r;
    r = mlp(mlx::core::fast::rms_norm(h, post_attention_layernorm_weight, rms_norm_eps));
    return h + r;
}

LanguageModel::LanguageModel(const ModelArgs& args) : norm_weight(0.0f) {
    this->args = args;
    for (int i = 0; i < args.num_hidden_layers; ++i) {
        layers.emplace_back(args);
    }
}

mlx::core::array LanguageModel::operator()(
    const mlx::core::array& inputs,
    std::vector<std::shared_ptr<KVCache>>& cache
) {
    auto h = embed_tokens(inputs);
    
    int L = inputs.shape(1);
    
    // Calculate attention mask
    std::optional<mlx::core::array> mask = std::nullopt;
    if (cache.size() > 0 && cache[0]) {
        mask = cache[0]->make_mask(L);
    }
    
    // Calculate llama_4 scaling attn_scale
    auto offset = cache.size() > 0 && cache[0] ? cache[0]->offset : mlx::core::array(0);
    auto attn_scale = mlx::core::astype(get_llama_4_attn_scale(
        L,
        offset,
        args.llama_4_scaling_beta,
        args.original_max_position_embeddings
    ), h.dtype());
    
    for (size_t i = 0; i < layers.size(); ++i) {
        h = layers[i](h, attn_scale, mask, cache[i]);
    }
    
    return mlx::core::fast::rms_norm(h, norm_weight, args.rms_norm_eps);
}

Model::Model(const ModelArgs& args) : args(args), model(args) {}

mlx::core::array Model::operator()(
    const mlx::core::array& inputs,
    std::vector<std::shared_ptr<KVCache>>& cache
) {
    auto out = model(inputs, cache);
    return lm_head(out);
}

static mlx::core::array get_or_throw(const std::unordered_map<std::string, mlx::core::array>& weights, const std::string& key) {
    auto it = weights.find(key);
    if (it == weights.end()) {
        throw std::runtime_error("Weight key not found: " + key);
    }
    return it->second;
}

static std::optional<mlx::core::array> get_optional(const std::unordered_map<std::string, mlx::core::array>& weights, const std::string& key) {
    auto it = weights.find(key);
    if (it == weights.end()) {
        return std::nullopt;
    }
    return it->second;
}

void Model::load_weights(const std::string& safetensors_path) {
    auto weights = mlx::core::load_safetensors(safetensors_path).first;
    
    // 1. Load Embedding
    model.embed_tokens.weight = get_or_throw(weights, "language_model.model.embed_tokens.weight");
    model.embed_tokens.scales = get_or_throw(weights, "language_model.model.embed_tokens.scales");
    
    // 2. Load layers
    for (int i = 0; i < args.num_hidden_layers; ++i) {
        std::string layer_prefix = "language_model.model.layers." + std::to_string(i) + ".";
        
        // Input layernorm
        model.layers[i].input_layernorm_weight = get_or_throw(weights, layer_prefix + "input_layernorm.weight");
        model.layers[i].post_attention_layernorm_weight = get_or_throw(weights, layer_prefix + "post_attention_layernorm.weight");
        
        // Q, K, V, O projections
        model.layers[i].self_attn.q_proj.weight = get_or_throw(weights, layer_prefix + "self_attn.q_proj.weight");
        model.layers[i].self_attn.q_proj.scales = get_or_throw(weights, layer_prefix + "self_attn.q_proj.scales");
        model.layers[i].self_attn.q_proj.biases = get_optional(weights, layer_prefix + "self_attn.q_proj.biases");
        model.layers[i].self_attn.q_proj.bias = get_optional(weights, layer_prefix + "self_attn.q_proj.bias");
        
        model.layers[i].self_attn.k_proj.weight = get_or_throw(weights, layer_prefix + "self_attn.k_proj.weight");
        model.layers[i].self_attn.k_proj.scales = get_or_throw(weights, layer_prefix + "self_attn.k_proj.scales");
        model.layers[i].self_attn.k_proj.biases = get_optional(weights, layer_prefix + "self_attn.k_proj.biases");
        model.layers[i].self_attn.k_proj.bias = get_optional(weights, layer_prefix + "self_attn.k_proj.bias");
        
        model.layers[i].self_attn.v_proj.weight = get_or_throw(weights, layer_prefix + "self_attn.v_proj.weight");
        model.layers[i].self_attn.v_proj.scales = get_or_throw(weights, layer_prefix + "self_attn.v_proj.scales");
        model.layers[i].self_attn.v_proj.biases = get_optional(weights, layer_prefix + "self_attn.v_proj.biases");
        model.layers[i].self_attn.v_proj.bias = get_optional(weights, layer_prefix + "self_attn.v_proj.bias");
        
        model.layers[i].self_attn.o_proj.weight = get_or_throw(weights, layer_prefix + "self_attn.o_proj.weight");
        model.layers[i].self_attn.o_proj.scales = get_or_throw(weights, layer_prefix + "self_attn.o_proj.scales");
        model.layers[i].self_attn.o_proj.biases = get_optional(weights, layer_prefix + "self_attn.o_proj.biases");
        model.layers[i].self_attn.o_proj.bias = get_optional(weights, layer_prefix + "self_attn.o_proj.bias");
        
        // MLP projections
        model.layers[i].mlp.gate_proj.weight = get_or_throw(weights, layer_prefix + "mlp.gate_proj.weight");
        model.layers[i].mlp.gate_proj.scales = get_or_throw(weights, layer_prefix + "mlp.gate_proj.scales");
        model.layers[i].mlp.gate_proj.biases = get_optional(weights, layer_prefix + "mlp.gate_proj.biases");
        model.layers[i].mlp.gate_proj.bias = get_optional(weights, layer_prefix + "mlp.gate_proj.bias");
        
        model.layers[i].mlp.up_proj.weight = get_or_throw(weights, layer_prefix + "mlp.up_proj.weight");
        model.layers[i].mlp.up_proj.scales = get_or_throw(weights, layer_prefix + "mlp.up_proj.scales");
        model.layers[i].mlp.up_proj.biases = get_optional(weights, layer_prefix + "mlp.up_proj.biases");
        model.layers[i].mlp.up_proj.bias = get_optional(weights, layer_prefix + "mlp.up_proj.bias");
        
        model.layers[i].mlp.down_proj.weight = get_or_throw(weights, layer_prefix + "mlp.down_proj.weight");
        model.layers[i].mlp.down_proj.scales = get_or_throw(weights, layer_prefix + "mlp.down_proj.scales");
        model.layers[i].mlp.down_proj.biases = get_optional(weights, layer_prefix + "mlp.down_proj.biases");
        model.layers[i].mlp.down_proj.bias = get_optional(weights, layer_prefix + "mlp.down_proj.bias");
    }
    
    // 3. Load final norm and head
    model.norm_weight = get_or_throw(weights, "language_model.model.norm.weight");
    lm_head.weight = get_or_throw(weights, "language_model.lm_head.weight");
    lm_head.scales = get_or_throw(weights, "language_model.lm_head.scales");
    lm_head.biases = get_optional(weights, "language_model.lm_head.biases");
    lm_head.bias = get_optional(weights, "language_model.lm_head.bias");
}

ModelArgs ModelArgs::load_from_config(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }
    
    nlohmann::json root;
    f >> root;

    // Check if we have "text_config" (common for multimodal/conditional architectures)
    nlohmann::json config = root;
    if (root.contains("text_config") && root["text_config"].is_object()) {
        config = root["text_config"];
    }

    ModelArgs args;
    
    // Model type
    if (root.contains("model_type") && root["model_type"].is_string()) {
        args.model_type = root["model_type"].get<std::string>();
    } else if (config.contains("model_type") && config["model_type"].is_string()) {
        args.model_type = config["model_type"].get<std::string>();
    }

    // Basic parameters
    if (config.contains("hidden_size")) args.hidden_size = config["hidden_size"].get<int>();
    if (config.contains("num_hidden_layers")) args.num_hidden_layers = config["num_hidden_layers"].get<int>();
    if (config.contains("intermediate_size")) args.intermediate_size = config["intermediate_size"].get<int>();
    if (config.contains("num_attention_heads")) args.num_attention_heads = config["num_attention_heads"].get<int>();
    
    if (config.contains("num_key_value_heads")) {
        args.num_key_value_heads = config["num_key_value_heads"].get<int>();
    } else {
        args.num_key_value_heads = args.num_attention_heads;
    }
    
    if (config.contains("rms_norm_eps")) args.rms_norm_eps = config["rms_norm_eps"].get<float>();
    if (config.contains("vocab_size")) args.vocab_size = config["vocab_size"].get<int>();
    if (config.contains("head_dim")) args.head_dim = config["head_dim"].get<int>();
    if (config.contains("max_position_embeddings")) args.max_position_embeddings = config["max_position_embeddings"].get<int>();
    if (config.contains("tie_word_embeddings")) args.tie_word_embeddings = config["tie_word_embeddings"].get<bool>();

    // RoPE parameters
    if (config.contains("rope_parameters") && config["rope_parameters"].is_object()) {
        auto rope = config["rope_parameters"];
        if (rope.contains("rope_theta")) args.rope_theta = rope["rope_theta"].get<float>();
        if (rope.contains("factor")) args.rope_factor = rope["factor"].get<float>();
        if (rope.contains("original_max_position_embeddings")) args.original_max_position_embeddings = rope["original_max_position_embeddings"].get<int>();
        if (rope.contains("beta_fast")) args.beta_fast = rope["beta_fast"].get<float>();
        if (rope.contains("beta_slow")) args.beta_slow = rope["beta_slow"].get<float>();
        if (rope.contains("llama_4_scaling_beta")) args.llama_4_scaling_beta = rope["llama_4_scaling_beta"].get<float>();
    } else {
        // Flat rope configuration
        if (config.contains("rope_theta")) args.rope_theta = config["rope_theta"].get<float>();
        if (config.contains("rope_factor")) args.rope_factor = config["rope_factor"].get<float>();
        if (config.contains("original_max_position_embeddings")) args.original_max_position_embeddings = config["original_max_position_embeddings"].get<int>();
        if (config.contains("beta_fast")) args.beta_fast = config["beta_fast"].get<float>();
        if (config.contains("beta_slow")) args.beta_slow = config["beta_slow"].get<float>();
        if (config.contains("llama_4_scaling_beta")) args.llama_4_scaling_beta = config["llama_4_scaling_beta"].get<float>();
    }

    return args;
}
