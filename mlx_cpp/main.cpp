#include "batch_generator.h"
#include "grpo_dataset.h"
#include "model.h"
#include "tokenizer.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <libpq-fe.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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

void solve_problems(PGconn *conn, int number = 200) {
  std::vector<Problem> problems = get_unsolved_problems(conn, number);
  if (problems.empty()) {
    std::cout << "No unsolved problems found." << std::endl;
    return;
  }

  std::cout << "Found " << problems.size()
            << " unsolved problems. Initializing model..." << std::endl;

  std::string model_dir = "/Users/bran/.lmstudio/models/local/ministral-3-8B-reasoning-2512-mxfp4";
  std::string config_path = model_dir + "/config.json";
  std::string weights_path = model_dir + "/model.safetensors";

  std::cout << "Loading model arguments from " << config_path << "..." << std::endl;
  ModelArgs args = ModelArgs::load_from_config(config_path);

  Model model(args);
  std::cout << "Loading weights from " << weights_path << "..." << std::endl;
  model.load_weights(weights_path);
  std::cout << "Model loaded successfully." << std::endl;

  Tokenizer tokenizer(model_dir);
  MLXBatchGenerator batch_generator(model, tokenizer);

  int batch_size = 6;
  size_t num_batches = (problems.size() + batch_size - 1) / batch_size;
  std::cout << "Processing " << problems.size() << " problems in "
            << num_batches << " batches (batch size " << batch_size << ")..."
            << std::endl;

  auto total_start_time = std::chrono::high_resolution_clock::now();

  // Print initial batch progress bar
  print_progress_bar(0, num_batches, 0.0, "Solving batches:");

  for (size_t b = 0; b < num_batches; ++b) {
    size_t start_idx = b * batch_size;
    size_t end_idx = std::min(start_idx + batch_size, problems.size());

    std::vector<Problem> batch(problems.begin() + start_idx,
                               problems.begin() + end_idx);
    std::vector<std::string> problems_texts;
    for (const auto &p : batch) {
      problems_texts.push_back(p.problem);
    }

    // Tokenize
    auto prompts = tokenizer.encode_chat_prompts(problems_texts);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Generate tokens
    auto generated_tokens = batch_generator.generate(prompts,
                                                     2048, // max_tokens
                                                     0.0f, // temperature
                                                     1.0f, // top_p
                                                     false // verbose
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    int batch_time =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
            .count();
    int time_per_problem = batch_time / (int)batch.size();

    // Decode solutions
    auto solutions = tokenizer.decode(generated_tokens);

    std::vector<SolutionUpdate> updates;
    for (size_t i = 0; i < batch.size(); ++i) {
      // Check if the response ends with an EOS token
      const auto &tokens = generated_tokens[i];
      bool ends_with_eos = false;
      if (!tokens.empty()) {
        int last_token = tokens.back();
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

      std::string solution = solutions[i];
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

      updates.push_back({solution, time_per_problem, batch[i].id});
    }

    save_batch_solutions(conn, updates);

    auto current_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec =
        std::chrono::duration<double>(current_time - total_start_time).count();
    print_progress_bar(b + 1, num_batches, elapsed_sec,
                       "Solving batches:", batch_time);
  }
  std::cout << std::endl;
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
