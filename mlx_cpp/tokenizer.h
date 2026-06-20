#pragma once
#include <string>
#include <vector>
#include <unistd.h>

class Tokenizer {
public:
    Tokenizer(const std::string& model_path = "/Users/bran/.lmstudio/models/local/ministral-3-8B-reasoning-2512-mxfp4");
    ~Tokenizer();

    std::vector<std::vector<int>> encode_chat_prompts(const std::vector<std::string>& problems);
    std::vector<std::string> decode(const std::vector<std::vector<int>>& ids);
    std::string decode(const std::vector<int>& ids);
    
    int pad_token_id = 0;
    std::vector<int> eos_token_ids;

private:
    std::string model_path_;
    pid_t pid_ = -1;
    int write_fd_ = -1;
    int read_fd_ = -1;

    void start_helper();
    std::string send_command(const std::string& cmd_json);
};
