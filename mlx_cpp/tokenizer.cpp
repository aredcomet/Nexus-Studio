#include "tokenizer.h"
#include <iostream>
#include <sstream>
#include <signal.h>
#include <stdexcept>
#include <cstring>
#include <sys/wait.h>
#include <fcntl.h>

static std::string escape_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

static std::string unescape_json(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next = s[i + 1];
            if (next == '"') out += '"';
            else if (next == '\\') out += '\\';
            else if (next == 'n') out += '\n';
            else if (next == 'r') out += '\r';
            else if (next == 't') out += '\t';
            else out += next;
            i++;
        } else {
            out += s[i];
        }
    }
    return out;
}

Tokenizer::Tokenizer(const std::string& model_path) : model_path_(model_path) {
    start_helper();
    // Retrieve special tokens
    std::string response = send_command("{\"command\": \"get_special_tokens\"}\n");
    
    // Simple manual parsing of get_special_tokens
    // Expected: {"status": "ok", "pad_token_id": X, "eos_token_ids": [A, B]}
    size_t pad_pos = response.find("\"pad_token_id\":");
    if (pad_pos != std::string::npos) {
        pad_token_id = std::stoi(response.substr(pad_pos + 15));
    }
    
    size_t eos_pos = response.find("\"eos_token_ids\":");
    if (eos_pos != std::string::npos) {
        size_t start_bracket = response.find('[', eos_pos);
        size_t end_bracket = response.find(']', start_bracket);
        if (start_bracket != std::string::npos && end_bracket != std::string::npos) {
            std::string list_str = response.substr(start_bracket + 1, end_bracket - start_bracket - 1);
            std::stringstream ss(list_str);
            std::string val;
            while (std::getline(ss, val, ',')) {
                if (!val.empty()) {
                    eos_token_ids.push_back(std::stoi(val));
                }
            }
        }
    }
}

Tokenizer::~Tokenizer() {
    if (write_fd_ != -1) {
        close(write_fd_);
    }
    if (read_fd_ != -1) {
        close(read_fd_);
    }
    if (pid_ != -1) {
        kill(pid_, SIGTERM);
        int status;
        waitpid(pid_, &status, 0);
    }
}

void Tokenizer::start_helper() {
    int pipe_in[2];  // C++ writes, Python reads (Python's stdin)
    int pipe_out[2]; // Python writes, C++ reads (Python's stdout)

    if (pipe(pipe_in) == -1 || pipe(pipe_out) == -1) {
        throw std::runtime_error("Failed to create pipes for tokenizer helper");
    }

    pid_ = fork();
    if (pid_ == -1) {
        throw std::runtime_error("Failed to fork tokenizer helper");
    }

    if (pid_ == 0) {
        // Child process (Python helper)
        dup2(pipe_in[0], STDIN_FILENO);
        dup2(pipe_out[1], STDOUT_FILENO);

        close(pipe_in[0]);
        close(pipe_in[1]);
        close(pipe_out[0]);
        close(pipe_out[1]);

        // Exec Python helper using virtualenv's python
        execl("/Users/bran/src/play/llm/.venv/bin/python3", "python3", "/Users/bran/src/play/llm/mlx_cpp/tokenizer_helper.py", model_path_.c_str(), nullptr);
        
        // If exec fails:
        std::cerr << "Child process failed to execute python tokenizer_helper" << std::endl;
        exit(1);
    } else {
        // Parent process (C++)
        close(pipe_in[0]);
        close(pipe_out[1]);
        write_fd_ = pipe_in[1];
        read_fd_ = pipe_out[0];
    }
}

std::string Tokenizer::send_command(const std::string& cmd_json) {
    if (write_fd_ == -1 || read_fd_ == -1) {
        throw std::runtime_error("Tokenizer helper process is not running");
    }

    // Write command
    ssize_t bytes_written = write(write_fd_, cmd_json.c_str(), cmd_json.size());
    if (bytes_written != (ssize_t)cmd_json.size()) {
        throw std::runtime_error("Failed to write to tokenizer helper");
    }

    // Read response line (terminated by newline)
    std::string response;
    char c;
    while (true) {
        ssize_t bytes_read = read(read_fd_, &c, 1);
        if (bytes_read <= 0) {
            throw std::runtime_error("Failed to read from tokenizer helper (EOF or Error)");
        }
        if (c == '\n') {
            break;
        }
        response += c;
    }
    return response;
}

std::vector<std::vector<int>> Tokenizer::encode_chat_prompts(const std::vector<std::string>& problems) {
    std::string cmd = "{\"command\": \"encode_chat_prompts\", \"problems\": [";
    for (size_t i = 0; i < problems.size(); ++i) {
        cmd += "\"" + escape_json(problems[i]) + "\"";
        if (i + 1 < problems.size()) cmd += ", ";
    }
    cmd += "]}\n";

    std::string response = send_command(cmd);

    // Simple manual parsing of: {"status": "ok", "ids": [[1,2,3], [4,5]]}
    std::vector<std::vector<int>> result;
    size_t ids_pos = response.find("\"ids\":");
    if (ids_pos == std::string::npos) {
        throw std::runtime_error("Tokenizer response error: " + response);
    }

    size_t i = response.find('[', ids_pos);
    if (i == std::string::npos) return result;
    i++; // Skip the outer [

    while (i < response.size()) {
        // Find next inner list start
        size_t start = response.find('[', i);
        if (start == std::string::npos || start > response.find(']', i)) {
            break; // No more lists (reached outer ])
        }
        size_t end = response.find(']', start);
        if (end == std::string::npos) {
            break;
        }

        std::string inner = response.substr(start + 1, end - start - 1);
        std::vector<int> tokens;
        std::stringstream ss(inner);
        std::string val;
        while (std::getline(ss, val, ',')) {
            if (!val.empty()) {
                tokens.push_back(std::stoi(val));
            }
        }
        result.push_back(tokens);
        i = end + 1;
    }

    return result;
}

std::vector<std::string> Tokenizer::decode(const std::vector<std::vector<int>>& ids) {
    std::string cmd = "{\"command\": \"decode\", \"ids\": [";
    for (size_t i = 0; i < ids.size(); ++i) {
        cmd += "[";
        for (size_t j = 0; j < ids[i].size(); ++j) {
            cmd += std::to_string(ids[i][j]);
            if (j + 1 < ids[i].size()) cmd += ",";
        }
        cmd += "]";
        if (i + 1 < ids.size()) cmd += ", ";
    }
    cmd += "]}\n";

    std::string response = send_command(cmd);

    // Simple manual parsing of: {"status": "ok", "texts": ["text1", "text2"]}
    std::vector<std::string> result;
    size_t texts_pos = response.find("\"texts\":");
    if (texts_pos == std::string::npos) {
        throw std::runtime_error("Tokenizer response error: " + response);
    }

    size_t i = response.find('[', texts_pos);
    if (i == std::string::npos) return result;
    i++; // Skip the outer [

    while (i < response.size()) {
        size_t start = response.find('"', i);
        if (start == std::string::npos || start > response.find(']', i)) {
            break;
        }
        // Scan until matching unescaped quote
        size_t end = start + 1;
        while (end < response.size()) {
            if (response[end] == '"' && response[end - 1] != '\\') {
                break;
            }
            end++;
        }
        if (end >= response.size()) {
            break;
        }

        std::string escaped_str = response.substr(start + 1, end - start - 1);
        result.push_back(unescape_json(escaped_str));
        i = end + 1;
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::vector<std::vector<int>> batch = {ids};
    std::vector<std::string> res = decode(batch);
    if (res.empty()) return "";
    return res[0];
}
