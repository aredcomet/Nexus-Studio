#include "batch_generator.h"
#include "grpo_dataset.h"
#include "model.h"
#include "tokenizer.h"
#include <chrono>
#include <future>
#include <iomanip>
#include <iostream>
#include <libpq-fe.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <set>
#include <functional>
#include <limits>

// FTXUI Headers
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/component/event.hpp>

void print_progress_bar(int current, int total, double elapsed_seconds,
                        const std::string &prefix = "",
                        int last_batch_time = -1) {
  int bar_width = 30;
  double progress = (double)current / total;
  int pos = bar_width * progress;

  std::cout << "\r\033[K" << prefix << " [";
  for (int i = 0; i < bar_width; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total
            << ")";

  if (current > 0 && elapsed_seconds > 0) {
    double time_per_item = elapsed_seconds / current;
    double remaining_seconds = time_per_item * (total - current);

    int rem_min = (int)remaining_seconds / 60;
    int rem_sec = (int)remaining_seconds % 60;

    std::cout << " | " << std::fixed << std::setprecision(1) << time_per_item
              << "s/it | remaining: " << std::setw(2) << std::setfill('0')
              << rem_min << "m " << std::setw(2) << std::setfill('0') << rem_sec
              << "s";
  }

  if (last_batch_time >= 0) {
    std::cout << " | last: " << last_batch_time << "s";
  }

  std::cout << std::flush;
}

struct Problem {
  int id;
  std::string problem;
  std::string answer;
  std::string problem_type;
};

struct SolutionUpdate {
  std::string solution;
  int time_to_solve;
  int id;
};

// Thread-Safe Shared State
struct SharedState {
  std::mutex mutex;
  std::vector<std::string> logs;
  std::string current_prompt = "No active problem.";
  std::string current_response = "";
  size_t main_progress = 0;
  size_t main_max = 0;
  size_t batch_progress = 0;
  size_t batch_max = 5;
  double tok_sec = 0.0;
  long long ttft_ms = 0;
  size_t token_count = 0;
  std::string status = "Initializing";
  size_t total_problems = 0;
  size_t solved_problems = 0;

  std::atomic<bool> is_paused{false};
  std::atomic<bool> should_exit{false};
  std::condition_variable pause_cv;
  std::mutex pause_mutex;
};

SharedState g_state;
std::function<void()> trigger_redraw = nullptr;

void add_log(const std::string &msg) {
  std::lock_guard<std::mutex> lock(g_state.mutex);
  g_state.logs.push_back(msg);
  if (g_state.logs.size() > 200) {
    g_state.logs.erase(g_state.logs.begin());
  }
  if (trigger_redraw) {
    trigger_redraw();
  }
}

void check_pause_and_exit() {
  if (g_state.should_exit.load()) {
    throw std::runtime_error("aborted");
  }
  if (g_state.is_paused.load()) {
    {
      std::lock_guard<std::mutex> lock(g_state.mutex);
      g_state.status = "Paused";
    }
    if (trigger_redraw) trigger_redraw();

    std::unique_lock<std::mutex> lk(g_state.pause_mutex);
    g_state.pause_cv.wait(lk, [] {
      return !g_state.is_paused.load() || g_state.should_exit.load();
    });

    if (g_state.should_exit.load()) {
      throw std::runtime_error("aborted");
    }

    {
      std::lock_guard<std::mutex> lock(g_state.mutex);
      g_state.status = "Generating";
    }
    if (trigger_redraw) trigger_redraw();
  }
}

void cooldown_sleep(int seconds) {
  for (int i = 0; i < seconds * 10; ++i) {
    check_pause_and_exit();

    int remaining = seconds - (i / 10);
    {
      std::lock_guard<std::mutex> lock(g_state.mutex);
      g_state.status = "Cooldown (" + std::to_string(remaining) + "s)";
    }
    if (trigger_redraw) trigger_redraw();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

int get_total_count(PGconn *conn) {
  PGresult *res = PQexec(conn, "SELECT count(*) FROM problemset");
  if (PQresultStatus(res) != PGRES_TUPLES_OK) {
    std::string err = PQerrorMessage(conn);
    PQclear(res);
    throw std::runtime_error("Failed to query total count: " + err);
  }
  int count = std::stoi(PQgetvalue(res, 0, 0));
  PQclear(res);
  return count;
}

int get_solved_count(PGconn *conn) {
  PGresult *res = PQexec(conn, "SELECT count(*) FROM problemset WHERE solution IS NOT NULL");
  if (PQresultStatus(res) != PGRES_TUPLES_OK) {
    std::string err = PQerrorMessage(conn);
    PQclear(res);
    throw std::runtime_error("Failed to query solved count: " + err);
  }
  int count = std::stoi(PQgetvalue(res, 0, 0));
  PQclear(res);
  return count;
}

void create_problemset(PGconn *conn, int num_samples = 2200) {
  int total_count = get_total_count(conn);
  if (total_count >= num_samples) {
    std::cout << "Generated " << total_count << " problems already."
              << std::endl;
    return;
  }

  int needed = num_samples - total_count;
  std::cout << "Generating " << needed << " problems to reach " << num_samples
            << "..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  PQexec(conn, "BEGIN");
  for (int i = 0; i < needed; ++i) {
    auto [problem, answer, problem_type] = generate_math_problem();

    const char *paramValues[3];
    paramValues[0] = problem.c_str();
    paramValues[1] = answer.c_str();
    paramValues[2] = problem_type.c_str();

    PGresult *res = PQexecParams(conn,
                                 "INSERT INTO problemset (problem, answer, "
                                 "problem_type) VALUES ($1, $2, $3)",
                                 3, nullptr, paramValues, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
      std::string err = PQerrorMessage(conn);
      PQclear(res);
      PQexec(conn, "ROLLBACK");
      throw std::runtime_error("Failed to insert problem: " + err);
    }
    PQclear(res);

    if ((i + 1) % 50 == 0 || i + 1 == needed) {
      auto current_time = std::chrono::high_resolution_clock::now();
      double elapsed =
          std::chrono::duration<double>(current_time - start_time).count();
      print_progress_bar(i + 1, needed, elapsed, "Generating problems:");
    }
  }
  PQexec(conn, "COMMIT");
  std::cout << "\nGenerated " << num_samples << " problems" << std::endl;
}

std::vector<Problem> get_unsolved_problems(PGconn *conn, int limit = 200) {
  const char *paramValues[1];
  std::string limit_str = std::to_string(limit);
  paramValues[0] = limit_str.c_str();

  PGresult *res =
      PQexecParams(conn,
                   "SELECT id, problem, answer, problem_type FROM problemset "
                   "WHERE solution IS NULL ORDER BY id ASC LIMIT $1",
                   1, nullptr, paramValues, nullptr, nullptr, 0);

  if (PQresultStatus(res) != PGRES_TUPLES_OK) {
    std::string err = PQerrorMessage(conn);
    PQclear(res);
    throw std::runtime_error("Failed to select unsolved problems: " + err);
  }

  int num_rows = PQntuples(res);
  std::vector<Problem> problems;
  for (int i = 0; i < num_rows; ++i) {
    Problem p;
    p.id = std::stoi(PQgetvalue(res, i, 0));
    p.problem = PQgetvalue(res, i, 1);
    p.answer = PQgetvalue(res, i, 2);
    p.problem_type = PQgetvalue(res, i, 3);
    problems.push_back(p);
  }
  PQclear(res);
  return problems;
}

void save_batch_solutions(PGconn *conn,
                          const std::vector<SolutionUpdate> &updates) {
  PQexec(conn, "BEGIN");
  for (const auto &u : updates) {
    const char *paramValues[3];
    paramValues[0] = u.solution.c_str();
    std::string time_str = std::to_string(u.time_to_solve);
    paramValues[1] = time_str.c_str();
    std::string id_str = std::to_string(u.id);
    paramValues[2] = id_str.c_str();

    PGresult *res = PQexecParams(
        conn,
        "UPDATE problemset SET solution = $1, time_to_solve = $2 WHERE id = $3",
        3, nullptr, paramValues, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
      std::cerr << "Failed to update problem id: " << u.id
                << ", error: " << PQerrorMessage(conn) << std::endl;
    }
    PQclear(res);
  }
  PQexec(conn, "COMMIT");
}

std::vector<int> generate_single(Model &model, Tokenizer &tokenizer,
                                 const std::vector<int> &prompt, int max_tokens,
                                 float temperature, float top_p) {
  auto start_time = std::chrono::high_resolution_clock::now();

  {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.current_response = "";
    g_state.token_count = 0;
    g_state.tok_sec = 0.0;
    g_state.ttft_ms = 0;
    g_state.status = "Generating (Prefill)";
  }
  if (trigger_redraw) trigger_redraw();

  // 1. Initialize KV Cache for each layer (left_padding is all 0 since no
  // padding is needed for batch_size=1)
  std::vector<std::shared_ptr<KVCache>> cache;
  std::vector<int> left_padding = {0};
  for (int i = 0; i < model.args.num_hidden_layers; ++i) {
    cache.push_back(std::make_shared<KVCache>(left_padding));
  }

  // 2. Prepare inputs (batch_size=1, sequence_length = prompt.size())
  auto x = mlx::core::array(prompt.data(), {1, (int)prompt.size()},
                            mlx::core::int32);

  // 3. Prefill step
  auto logits = model(x, cache);
  // Get the logits for the last token of the prompt: shape is (1, V)
  auto next_logits = mlx::core::slice(logits, {0, (int)prompt.size() - 1, 0},
                                      {1, (int)prompt.size(), logits.shape(2)});
  next_logits = mlx::core::reshape(next_logits, {1, -1}); // (1, V)

  // 4. Helper for sampling
  auto sample_token = [top_p](const mlx::core::array &lgt, float temp) {
    if (temp == 0.0f) {
      return mlx::core::argmax(lgt, -1, true); // (1, 1)
    }

    // Scale logits
    auto scaled_logits = lgt / temp; // (1, V)

    if (top_p <= 0.0f || top_p >= 1.0f) {
      auto tokens = mlx::core::random::categorical(scaled_logits, -1, std::nullopt);
      return mlx::core::expand_dims(tokens, -1);
    }

    // Squeeze to 1D for sorting
    auto logits_1d = mlx::core::reshape(scaled_logits, {-1});
    int V = logits_1d.shape(0);

    // Sort descending
    auto sorted_indices = mlx::core::argsort(-logits_1d, 0);
    auto sorted_logits = mlx::core::take(logits_1d, sorted_indices, 0);

    // Calculate cumulative probabilities
    auto probs = mlx::core::softmax(sorted_logits, 0);
    auto cumsum = mlx::core::cumsum(probs, 0);

    // Create exclude mask (shift cumsum right by 1)
    auto zero = mlx::core::array({0.0f});
    auto slice_cumsum = mlx::core::slice(cumsum, {0}, {V - 1});
    auto shifted_cumsum = mlx::core::concatenate({zero, slice_cumsum}, 0);

    auto exclude_mask = shifted_cumsum > top_p;

    // Apply exclude mask (set excluded logits to -inf)
    auto masked_sorted_logits = mlx::core::where(
        exclude_mask,
        mlx::core::array(-std::numeric_limits<float>::infinity()),
        sorted_logits
    );

    // Sample from masked sorted logits
    auto sampled_sorted_idx = mlx::core::random::categorical(masked_sorted_logits, -1, std::nullopt);

    // Map back to original index
    auto sampled_token = mlx::core::take(sorted_indices, sampled_sorted_idx, 0);

    // Expand to (1, 1) to match expected shape
    return mlx::core::expand_dims(mlx::core::expand_dims(sampled_token, 0), 0);
  };

  auto next_token = sample_token(next_logits, temperature); // (1, 1)

  std::vector<int> generated;
  int first_tok = mlx::core::reshape(next_token, {}).item<int>();
  generated.push_back(first_tok);

  auto prefill_end_time = std::chrono::high_resolution_clock::now();
  auto ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end_time - start_time).count();
  
  {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.token_count = 1;
    g_state.ttft_ms = ttft_ms;
    g_state.status = "Generating";
    g_state.current_response = tokenizer.decode(generated);
  }
  if (trigger_redraw) trigger_redraw();

  std::set<int> eos_set(tokenizer.eos_token_ids.begin(), tokenizer.eos_token_ids.end());

  // 5. Decode loop
  for (int step = 1; step < max_tokens; ++step) {
    check_pause_and_exit();

    mlx::core::eval(next_token);

    int tok = mlx::core::reshape(next_token, {}).item<int>();
    if (eos_set.count(tok)) {
      break;
    }

    // Forward pass on the single new token
    logits = model(next_token, cache);
    next_logits = mlx::core::reshape(logits, {1, -1});
    next_token = sample_token(next_logits, temperature);

    int new_tok = mlx::core::reshape(next_token, {}).item<int>();
    generated.push_back(new_tok);

    auto current_time = std::chrono::high_resolution_clock::now();
    double decode_elapsed = std::chrono::duration<double>(current_time - prefill_end_time).count();
    double tok_sec = (decode_elapsed > 0) ? (step / decode_elapsed) : 0.0;

    {
      std::lock_guard<std::mutex> lock(g_state.mutex);
      g_state.token_count = step + 1;
      g_state.tok_sec = tok_sec;
      g_state.current_response = tokenizer.decode(generated);
    }
    if (trigger_redraw) trigger_redraw();

    if (step % 256 == 0) {
      mlx::core::clear_cache();
    }
  }

  return generated;
}

void worker_thread_func(PGconn *conn, int number, int batch_size, int max_tokens, int cooldown_time) {
  try {
    std::vector<Problem> problems = get_unsolved_problems(conn, number);
    if (problems.empty()) {
      add_log("No unsolved problems found.");
      {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        g_state.status = "Done";
      }
      if (trigger_redraw) trigger_redraw();
      return;
    }

    add_log("Found " + std::to_string(problems.size()) + " unsolved problems. Initializing model...");

    std::string model_dir =
        "/Users/bran/.lmstudio/models/local/ministral-3-8B-reasoning-2512-mxfp4";
    std::string config_path = model_dir + "/config.json";
    std::string weights_path = model_dir + "/model.safetensors";

    add_log("Loading model arguments from " + config_path + "...");
    ModelArgs args = ModelArgs::load_from_config(config_path);

    Model model(args);
    add_log("Loading weights from " + weights_path + "...");
    model.load_weights(weights_path);
    add_log("Model loaded successfully.");

    Tokenizer tokenizer(model_dir);

    size_t num_batches = (problems.size() + batch_size - 1) / batch_size;
    
    {
      std::lock_guard<std::mutex> lock(g_state.mutex);
      g_state.main_max = num_batches;
      g_state.main_progress = 0;
      g_state.batch_max = batch_size;
    }
    if (trigger_redraw) trigger_redraw();

    std::future<void> db_write_future;

    for (size_t b = 0; b < num_batches; ++b) {
      check_pause_and_exit();

      size_t start_idx = b * batch_size;
      size_t end_idx = std::min(start_idx + batch_size, problems.size());

      std::vector<Problem> batch(problems.begin() + start_idx,
                                 problems.begin() + end_idx);

      std::vector<SolutionUpdate> updates;

      for (size_t i = 0; i < batch.size(); ++i) {
        check_pause_and_exit();

        {
          std::lock_guard<std::mutex> lock(g_state.mutex);
          g_state.batch_progress = i;
          g_state.current_prompt = "Problem " + std::to_string(batch[i].id) + ":\n" + batch[i].problem;
        }
        if (trigger_redraw) trigger_redraw();

        std::vector<std::string> problems_texts = {batch[i].problem};

        // Tokenize
        auto prompts = tokenizer.encode_chat_prompts(problems_texts);
        if (prompts.empty() || prompts[0].empty()) {
          continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Generate tokens sequentially (batch size 1) using custom
        // generate_single function
        auto generated_tokens = generate_single(model, tokenizer, prompts[0],
                                                 max_tokens,
                                                 0.10f, // temperature
                                                 0.95f  // top_p
        );

        auto end_time = std::chrono::high_resolution_clock::now();
        int item_time = std::chrono::duration_cast<std::chrono::seconds>(
                            end_time - start_time)
                            .count();

        // Decode solutions
        auto solution = tokenizer.decode(generated_tokens);

        bool ends_with_eos = false;
        if (!generated_tokens.empty()) {
          int last_token = generated_tokens.back();
          for (int eos_id : tokenizer.eos_token_ids) {
            if (last_token == eos_id) {
              ends_with_eos = true;
              break;
            }
          }
        }

        if (!ends_with_eos) {
          add_log("Problem ID " + std::to_string(batch[i].id) + " generation truncated/incomplete. Skipping DB save.");
          continue;
        }

        // Replace [THINK] tags
        size_t pos = 0;
        while ((pos = solution.find("[THINK]", pos)) != std::string::npos) {
          solution.replace(pos, 7, "<think>");
          pos += 7;
        }
        pos = 0;
        while ((pos = solution.find("[/THINK]", pos)) != std::string::npos) {
          solution.replace(pos, 8, "</think>");
          pos += 8;
        }

        updates.push_back({solution, item_time, batch[i].id});
      }

      check_pause_and_exit();

      if (db_write_future.valid()) {
        db_write_future.get();
      }
      db_write_future = std::async(std::launch::async, [conn, updates]() {
        save_batch_solutions(conn, updates);
        {
          std::lock_guard<std::mutex> lock(g_state.mutex);
          g_state.solved_problems += updates.size();
        }
        if (trigger_redraw) trigger_redraw();
      });

      {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        g_state.main_progress = b + 1;
        g_state.batch_progress = batch.size();
      }
      if (trigger_redraw) trigger_redraw();

      // Cool down after each batch
      if (b + 1 < num_batches) {
        cooldown_sleep(cooldown_time);
        mlx::core::clear_cache();
      }
    }

    if (db_write_future.valid()) {
      db_write_future.get();
    }

    {
      std::lock_guard<std::mutex> lock(g_state.mutex);
      g_state.status = "Done";
    }
    if (trigger_redraw) trigger_redraw();
    add_log("Sequential solver worker thread finished successfully.");
  } catch (const std::exception &e) {
    add_log("Worker thread aborted: " + std::string(e.what()));
    {
      std::lock_guard<std::mutex> lock(g_state.mutex);
      g_state.status = "Aborted";
    }
    if (trigger_redraw) trigger_redraw();
  }
}

ftxui::Element paragraph_multiline(const std::string& text_block, bool dim_yellow = false) {
  using namespace ftxui;
  Elements lines;
  std::stringstream ss(text_block);
  std::string line;
  while (std::getline(ss, line, '\n')) {
    Element p = paragraph(line);
    if (dim_yellow) {
      p = p | dim | color(Color::Yellow);
    }
    lines.push_back(p);
  }
  if (!text_block.empty() && text_block.back() == '\n') {
    lines.push_back(text(" "));
  }
  return vbox(std::move(lines));
}

ftxui::Element format_response_stream(const std::string &resp) {
  using namespace ftxui;
  Elements lines;
  size_t pos = 0;
  bool inside_think = false;

  while (pos < resp.size()) {
    if (!inside_think) {
      size_t think_start = resp.find("[THINK]", pos);
      size_t tag_len = 7;
      std::string tag_str = "[THINK]";

      size_t alternative_start = resp.find("<think>", pos);
      if (alternative_start != std::string::npos && (think_start == std::string::npos || alternative_start < think_start)) {
        think_start = alternative_start;
        tag_len = 7;
        tag_str = "<think>";
      }

      if (think_start == std::string::npos) {
        lines.push_back(paragraph_multiline(resp.substr(pos), false));
        break;
      } else {
        if (think_start > pos) {
          lines.push_back(paragraph_multiline(resp.substr(pos, think_start - pos), false));
        }
        lines.push_back(text(tag_str) | dim | color(Color::Yellow));
        inside_think = true;
        pos = think_start + tag_len;
      }
    } else {
      size_t think_end = resp.find("[/THINK]", pos);
      size_t tag_len = 8;
      std::string tag_str = "[/THINK]";

      size_t alternative_end = resp.find("</think>", pos);
      if (alternative_end != std::string::npos && (think_end == std::string::npos || alternative_end < think_end)) {
        think_end = alternative_end;
        tag_len = 8;
        tag_str = "</think>";
      }

      if (think_end == std::string::npos) {
        lines.push_back(paragraph_multiline(resp.substr(pos), true));
        break;
      } else {
        if (think_end > pos) {
          lines.push_back(paragraph_multiline(resp.substr(pos, think_end - pos), true));
        }
        lines.push_back(text(tag_str) | dim | color(Color::Yellow));
        inside_think = false;
        pos = think_end + tag_len;
      }
    }
  }
  if (!lines.empty()) {
    lines.back() = lines.back() | focus;
  }
  return vbox(std::move(lines)) | yframe;
}


int main(int argc, char* argv[]) {
  int batch_size = 5;
  int max_tokens = 8192;
  int cooldown_time = 60;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--batch-size" || arg == "-b") {
      if (i + 1 < argc) {
        try {
          batch_size = std::stoi(argv[++i]);
          if (batch_size <= 0) {
            std::cerr << "Error: --batch-size must be a positive integer.\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for --batch-size.\n";
          return 1;
        }
      } else {
        std::cerr << "Error: --batch-size requires an argument.\n";
        return 1;
      }
    } else if (arg == "--max-tokens" || arg == "-m") {
      if (i + 1 < argc) {
        try {
          max_tokens = std::stoi(argv[++i]);
          if (max_tokens <= 0) {
            std::cerr << "Error: --max-tokens must be a positive integer.\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for --max-tokens.\n";
          return 1;
        }
      } else {
        std::cerr << "Error: --max-tokens requires an argument.\n";
        return 1;
      }
    } else if (arg == "--cooldown" || arg == "-c") {
      if (i + 1 < argc) {
        try {
          cooldown_time = std::stoi(argv[++i]);
          if (cooldown_time < 0) {
            std::cerr << "Error: --cooldown must be a non-negative integer.\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for --cooldown.\n";
          return 1;
        }
      } else {
        std::cerr << "Error: --cooldown requires an argument.\n";
        return 1;
      }
    } else {
      std::cerr << "Usage: " << argv[0] << " [--batch-size|-b <size>] [--max-tokens|-m <tokens>] [--cooldown|-c <seconds>]\n";
      return 1;
    }
  }

  const char *dsn = std::getenv("POSTGRES_GO_DSN");
  if (!dsn) {
    std::cerr << "POSTGRES_GO_DSN environment variable is not set."
              << std::endl;
    return 1;
  }

  PGconn *conn = PQconnectdb(dsn);
  if (PQstatus(conn) != CONNECTION_OK) {
    std::cerr << "Connection to database failed: " << PQerrorMessage(conn)
              << std::endl;
    PQfinish(conn);
    return 1;
  }

  // Pre-seed the DB before interactive view starts
  create_problemset(conn, 2200);

  g_state.total_problems = get_total_count(conn);
  g_state.solved_problems = get_solved_count(conn);

  // Initialize FTXUI interactive screen
  using namespace ftxui;
  auto screen = ScreenInteractive::Fullscreen();

  trigger_redraw = [&screen]() {
    screen.PostEvent(Event::Custom);
  };

  // Launch sequential solver worker thread
  std::thread worker(worker_thread_func, conn, 2200, batch_size, max_tokens, cooldown_time);

  auto component = Renderer([&] {
    Element logs_title = text("System Logs") | bold | color(Color::Blue);
    Element logs_pane = vbox({
        logs_title,
        separator(),
        [&]() {
            std::lock_guard<std::mutex> lock(g_state.mutex);
            Elements log_elements;
            for (const auto& log : g_state.logs) {
                log_elements.push_back(text(log));
            }
            if (!log_elements.empty()) {
                log_elements.back() = log_elements.back() | focus;
            }
            return vbox(std::move(log_elements)) | yframe;
        }() | flex
    }) | border;

    Element gauges_pane = vbox({
        text("Progress Meters") | bold | color(Color::Blue),
        separator(),
        [&]() {
            float overall_ratio = 0.0f;
            float batch_ratio = 0.0f;
            float db_ratio = 0.0f;
            std::string main_progress_str = "0 / 0";
            std::string batch_progress_str = "0 / 5";
            std::string db_progress_str = "0 / 0";
            
            std::lock_guard<std::mutex> lock(g_state.mutex);
            if (g_state.main_max > 0) {
                overall_ratio = (float)g_state.main_progress / g_state.main_max;
                main_progress_str = std::to_string(g_state.main_progress) + " / " + std::to_string(g_state.main_max);
            }
            if (g_state.batch_max > 0) {
                batch_ratio = (float)g_state.batch_progress / g_state.batch_max;
                batch_progress_str = std::to_string(g_state.batch_progress) + " / " + std::to_string(g_state.batch_max);
            }
            if (g_state.total_problems > 0) {
                db_ratio = (float)g_state.solved_problems / g_state.total_problems;
                db_progress_str = std::to_string(g_state.solved_problems) + " / " + std::to_string(g_state.total_problems);
            }
            
            return vbox({
                hbox({
                    text("Overall Progress: "),
                    gauge(overall_ratio) | color(Color::Green) | flex,
                    text(" " + main_progress_str)
                }),
                separator(),
                hbox({
                    text("Current Batch:    "),
                    gauge(batch_ratio) | color(Color::Cyan) | flex,
                    text(" " + batch_progress_str)
                }),
                separator(),
                hbox({
                    text("Database Solved:  "),
                    gauge(db_ratio) | color(Color::Magenta) | flex,
                    text(" " + db_progress_str)
                })
            });
        }()
    }) | border;

    Element left_panel = vbox({
        logs_pane | flex,
        gauges_pane
    });

    Element prompt_pane = vbox({
        text("Current Math Problem") | bold | color(Color::Blue),
        separator(),
        [&]() {
            std::string prompt_str;
            std::lock_guard<std::mutex> lock(g_state.mutex);
            prompt_str = g_state.current_prompt;
            return paragraph_multiline(prompt_str);
        }() | yframe
    }) | border;

    Element response_pane = vbox({
        text("Live Model Generation Stream") | bold | color(Color::Blue),
        separator(),
        [&]() {
            std::string resp_str;
            std::lock_guard<std::mutex> lock(g_state.mutex);
            resp_str = g_state.current_response;
            return format_response_stream(resp_str);
        }() | flex
    }) | border;

    Element stats_pane = vbox({
        [&]() {
            double tok_sec = 0.0;
            long long ttft = 0;
            size_t tokens = 0;
            std::string status;
            std::lock_guard<std::mutex> lock(g_state.mutex);
            tok_sec = g_state.tok_sec;
            ttft = g_state.ttft_ms;
            tokens = g_state.token_count;
            status = g_state.status;

            std::stringstream stats_ss;
            stats_ss << "Status: " << status
                     << " | Speed: " << std::fixed << std::setprecision(1) << tok_sec << " tok/s"
                     << " | TTFT: " << ttft << "ms"
                     << " | Tokens: " << tokens;

            return hbox({
                text(stats_ss.str()) | bold | color(Color::Green),
                filler(),
                text("[P] Pause/Resume  [Q/Esc] Quit") | dim
            });
        }()
    }) | border;

    Element right_panel = vbox({
        prompt_pane | size(HEIGHT, EQUAL, 6),
        response_pane | flex,
        stats_pane
    });

    return hbox({
        left_panel | size(WIDTH, EQUAL, 50),
        right_panel | flex
    });
  });

  auto catch_key = CatchEvent(component, [&](Event event) {
    if (event == Event::Character('q') || event == Event::Character('Q') || event == Event::Escape) {
      g_state.should_exit = true;
      g_state.pause_cv.notify_all();
      screen.ExitLoopClosure()();
      return true;
    }
    if (event == Event::Character('p') || event == Event::Character('P')) {
      bool was_paused = g_state.is_paused.load();
      g_state.is_paused = !was_paused;
      if (was_paused) {
        g_state.pause_cv.notify_all();
        add_log("Worker thread resumed.");
      } else {
        add_log("Worker thread paused.");
      }
      return true;
    }
    return false;
  });

  screen.Loop(catch_key);

  // Shutdown sequence
  g_state.should_exit = true;
  g_state.pause_cv.notify_all();
  if (worker.joinable()) {
    worker.join();
  }

  PQfinish(conn);
  std::cout << "\nSafe exit. TUI closed." << std::endl;
  return 0;
}
