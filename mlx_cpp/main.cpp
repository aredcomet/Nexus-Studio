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
#include <indicators/progress_bar.hpp>
#include <indicators/multi_progress.hpp>


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

  // Start transaction for insertion
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
                                 float temperature, indicators::ProgressBar &sub_bar,
                                 indicators::MultiProgress<indicators::ProgressBar, 2> &bars) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // 1. Initialize KV Cache for each layer (left_padding is all 0 since no
  // padding is needed for batch_size=1)
  std::vector<std::shared_ptr<KVCache>> cache;
  std::vector<int> left_padding = {0};
  for (int i = 0; i < model.args.num_hidden_layers; ++i) {
    cache.push_back(std::make_shared<KVCache>(left_padding));
  }

  // Set sub_bar options
  sub_bar.set_option(indicators::option::MaxProgress{static_cast<size_t>(max_tokens)});
  sub_bar.set_option(indicators::option::PostfixText{"Preparing..."});
  bars.set_progress<1>(size_t(0));

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
  auto sample_token = [](const mlx::core::array &lgt, float temp) {
    if (temp == 0.0f) {
      return mlx::core::argmax(lgt, -1, true); // (1, 1)
    } else {
      auto scaled_logits = lgt / temp;
      auto tokens = mlx::core::random::categorical(scaled_logits, -1,
                                                   std::nullopt); // (1,)
      return mlx::core::expand_dims(tokens, -1);                  // (1, 1)
    }
  };

  auto next_token = sample_token(next_logits, temperature); // (1, 1)

  std::vector<int> generated;
  int first_tok = mlx::core::reshape(next_token, {}).item<int>();
  generated.push_back(first_tok);

  auto prefill_end_time = std::chrono::high_resolution_clock::now();
  auto ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end_time - start_time).count();
  
  sub_bar.set_option(indicators::option::PostfixText{"TTFT: " + std::to_string(ttft_ms) + "ms | tokens: 1"});
  bars.set_progress<1>(size_t(1));

  // 5. Decode loop
  for (int step = 1; step < max_tokens; ++step) {
    mlx::core::eval(next_token);

    int tok = mlx::core::reshape(next_token, {}).item<int>();
    bool is_eos = false;
    for (int eos_id : tokenizer.eos_token_ids) {
      if (tok == eos_id) {
        is_eos = true;
        break;
      }
    }
    if (is_eos) {
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

    std::stringstream ss;
    ss << std::fixed << std::setprecision(1)
       << "tok/s: " << tok_sec 
       << " | TTFT: " << ttft_ms << "ms"
       << " | tokens: " << (step + 1);
    sub_bar.set_option(indicators::option::PostfixText{ss.str()});
    bars.set_progress<1>(size_t(step + 1));

    if (step % 256 == 0) {
      mlx::core::clear_cache();
    }
  }

  return generated;
}

void solve_problems(PGconn *conn, int number = 200) {
  std::vector<Problem> problems = get_unsolved_problems(conn, number);
  if (problems.empty()) {
    std::cout << "No unsolved problems found." << std::endl;
    return;
  }

  std::cout << "Found " << problems.size()
            << " unsolved problems. Initializing model..." << std::endl;

  std::string model_dir =
      "/Users/bran/.lmstudio/models/local/ministral-3-8B-reasoning-2512-mxfp4";
  std::string config_path = model_dir + "/config.json";
  std::string weights_path = model_dir + "/model.safetensors";

  std::cout << "Loading model arguments from " << config_path << "..."
            << std::endl;
  ModelArgs args = ModelArgs::load_from_config(config_path);

  Model model(args);
  std::cout << "Loading weights from " << weights_path << "..." << std::endl;
  model.load_weights(weights_path);
  std::cout << "Model loaded successfully." << std::endl;

  Tokenizer tokenizer(model_dir);

  int batch_size = 5;
  size_t num_batches = (problems.size() + batch_size - 1) / batch_size;
  std::cout << "Processing " << problems.size() << " problems in "
            << num_batches << " batches (batch size " << batch_size << ")..."
            << std::endl;

  // Initialize Indicators progress bars
  indicators::ProgressBar main_bar{
      indicators::option::BarWidth{40},
      indicators::option::Start{"["},
      indicators::option::Fill{"█"},
      indicators::option::Lead{"█"},
      indicators::option::Remainder{"-"},
      indicators::option::End{"]"},
      indicators::option::PrefixText{"Main Batch Progress  "},
      indicators::option::ForegroundColor{indicators::Color::green},
      indicators::option::ShowPercentage{true},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true},
      indicators::option::MaxProgress{num_batches}
  };

  indicators::ProgressBar sub_bar{
      indicators::option::BarWidth{40},
      indicators::option::Start{"["},
      indicators::option::Fill{"█"},
      indicators::option::Lead{"█"},
      indicators::option::Remainder{"-"},
      indicators::option::End{"]"},
      indicators::option::PrefixText{"Current Problem Gen  "},
      indicators::option::ForegroundColor{indicators::Color::cyan},
      indicators::option::ShowPercentage{false},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{false},
      indicators::option::MaxProgress{8192}
  };

  indicators::MultiProgress<indicators::ProgressBar, 2> bars(main_bar, sub_bar);

  std::future<void> db_write_future;

  for (size_t b = 0; b < num_batches; ++b) {
    size_t start_idx = b * batch_size;
    size_t end_idx = std::min(start_idx + batch_size, problems.size());

    std::vector<Problem> batch(problems.begin() + start_idx,
                               problems.begin() + end_idx);

    std::vector<SolutionUpdate> updates;

    for (size_t i = 0; i < batch.size(); ++i) {
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
                                              8192, // max_tokens
                                              0.0f, // temperature
                                              sub_bar,
                                              bars
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

    if (db_write_future.valid()) {
      db_write_future.get();
    }
    db_write_future = std::async(std::launch::async, [conn, updates]() {
      save_batch_solutions(conn, updates);
    });

    // Update main progress bar
    bars.tick<0>();

    // Cool down after each batch of 5
    if (b + 1 < num_batches) {
      sub_bar.set_option(indicators::option::PostfixText{"Cooling down for 60s..."});
      bars.set_progress<1>(size_t(0));
      mlx::core::clear_cache(); // Reclaim all cached Metal VRAM back to the system immediately
      std::this_thread::sleep_for(std::chrono::seconds(60));
    }
  }

  if (db_write_future.valid()) {
    db_write_future.get();
  }
}

int main() {
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

  std::cout << "Connected to database successfully." << std::endl;

  create_problemset(conn, 2200);
  solve_problems(conn, 2200);

  PQfinish(conn);
  std::cout << "Done!" << std::endl;
  return 0;
}
