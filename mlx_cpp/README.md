# MLX C++ Inference Engine & Math Problem Solver (`mlx_cpp`)

> [!NOTE]
> **LLM Agent Context**: This README is structured specifically for agentic parsing. It contains precise details on architecture, hardcoded local paths, JSON-RPC protocols, database schemas, compilation requirements, and optimization patterns to allow immediate navigation, execution, or modification of the code.

---

## 1. Overview & Architecture

The `mlx_cpp` subdirectory implements a high-performance, native C++ LLM inference runner using the Apple Silicon-optimized **MLX C++ API** (with Metal backend acceleration). 

Its primary application is to:
1. Connect to a PostgreSQL database containing a problem-set table.
2. Check if a minimum number of math reasoning problems exist (automatically generating more if needed).
3. Query unsolved problems.
4. Tokenize and format prompts into Chat Templates using a HuggingFace Python tokenizer helper subprocess.
5. Perform batched autoregressive text generation with key-value cache (KVCache) acceleration.
6. Parse, post-process (e.g., converting `[THINK]` tags to `<think>` elements), and persist generated solutions back to the database.

---

## 2. Directory Structure & Component Roles

```text
mlx_cpp/
├── CMakeLists.txt         # Build definition (links MLX, FetchContent JSON, PostgreSQL/libpq)
├── main.cpp               # System orchestrator (DB interactions, generation batch loop, main)
├── model.h / .cpp         # C++ MLX Neural Network definitions (Yarn RoPE, Layers, Quantized linear/embeddings)
├── tokenizer.h / .cpp     # Tokenizer C++ class (spawns Python helper subprocess via pipe)
├── tokenizer_helper.py    # Python helper using transformers library to manage Chat Templates & decoding
├── batch_generator.h/.cpp # batched autoregressive generation loop, KVCache handling, diverse rollouts
└── grpo_dataset.h / .cpp  # Random math question generator (Mixed & BODMAS templates)
```

### File Summary
* **[CMakeLists.txt](file:///Users/bran/src/play/llm/mlx_cpp/CMakeLists.txt)**: Configures C++20, specifies local include/library paths for MLX and `postgresql@18`, downloads `nlohmann/json` (v3.11.3), and targets the executable `mlx_cpp_runner`.
* **[main.cpp](file:///Users/bran/src/play/llm/mlx_cpp/main.cpp)**: Entry point. Connects to the database using `POSTGRES_GO_DSN`. Automatically ensures at least 2200 problems are in the database. Processes unsolved problems in batches of 6, calls the MLX generator, cleans up the reasoning markers, and writes results.
* **[model.h](file:///Users/bran/src/play/llm/mlx_cpp/model.h) / [model.cpp](file:///Users/bran/src/play/llm/mlx_cpp/model.cpp)**:
  * Implements `ModelArgs` to parse parameters from HuggingFace configurations.
  * Implements `QuantizedEmbedding` and `QuantizedLinear` utilizing `mlx::core::dequantize` and `mlx::core::quantized_matmul` for `mxfp4` (4-bit MX format) weights.
  * Implements `Attention` with Yarn (Yet Another RoPE Extrapolation) RoPE frequencies (`compute_yarn_freqs`) and Llama 4 SuScaled attention scale scaling.
  * Implements `MLP`, `TransformerBlock`, `LanguageModel`, and `Model` wrappers.
* **[tokenizer.h](file:///Users/bran/src/play/llm/mlx_cpp/tokenizer.h) / [tokenizer.cpp](file:///Users/bran/src/play/llm/mlx_cpp/tokenizer.cpp)**: Establishes double-pipe redirection (`fork`/`exec`) to interact with `tokenizer_helper.py` through line-delimited JSON messages.
* **[tokenizer_helper.py](file:///Users/bran/src/play/llm/mlx_cpp/tokenizer_helper.py)**: A lightweight Python script that loads the HuggingFace `AutoTokenizer` and processes requests via standard I/O (handling prompt formats, chat template application, and decoding).
* **[batch_generator.h](file:///Users/bran/src/play/llm/mlx_cpp/batch_generator.h) / [batch_generator.cpp](file:///Users/bran/src/play/llm/mlx_cpp/batch_generator.cpp)**:
  * `KVCache`: Manages key-value arrays per attention layer. Supports left-padding adjustment (`offset = -left_padding`), repetition (for group rollouts), and dynamic sequence filtering (removing completed batch entries and resizing/shifting keys/values to reclaim VRAM).
  * `MLXBatchGenerator`: Implements batch decoding (`generate`) and diverse rollouts (`generate_with_diverse_rollouts`).
* **[grpo_dataset.h](file:///Users/bran/src/play/llm/mlx_cpp/grpo_dataset.h) / [grpo_dataset.cpp](file:///Users/bran/src/play/llm/mlx_cpp/grpo_dataset.cpp)**: Procedurally generates simple arithmetic and order-of-operation tasks.

---

## 3. Environment Dependencies & Hardcoded Local Paths

This subdirectory compiles on MacOS (Apple Silicon) and assumes the following filesystem structure:

| Parameter / Library | Hardcoded Path in Code | File / Line Reference |
|---|---|---|
| **Python VirtualEnv** | `/Users/bran/src/play/llm/.venv/bin/python3` | [tokenizer.cpp#L109](file:///Users/bran/src/play/llm/mlx_cpp/tokenizer.cpp#L109) |
| **MLX Library Path** | `/Users/bran/src/play/llm/.venv/lib/python3.14/site-packages/mlx` | [CMakeLists.txt#L14-L16](file:///Users/bran/src/play/llm/mlx_cpp/CMakeLists.txt#L14-L16) |
| **PostgreSQL Includes** | `/opt/homebrew/include/postgresql@18` | [CMakeLists.txt#L10](file:///Users/bran/src/play/llm/mlx_cpp/CMakeLists.txt#L10) |
| **PostgreSQL Libraries**| `/opt/homebrew/lib/postgresql@18/libpq.dylib` | [CMakeLists.txt#L11](file:///Users/bran/src/play/llm/mlx_cpp/CMakeLists.txt#L11) |
| **Default Model Dir** | `/Users/bran/.lmstudio/models/local/ministral-3-8B-reasoning-2512-mxfp4` | [main.cpp#L192](file:///Users/bran/src/play/llm/mlx_cpp/main.cpp#L192), [tokenizer.h#L8](file:///Users/bran/src/play/llm/mlx_cpp/tokenizer.h#L8), [tokenizer_helper.py#L12](file:///Users/bran/src/play/llm/mlx_cpp/tokenizer_helper.py#L12) |

> [!WARNING]
> If building or running on a different system or under a different virtual environment, you **MUST** update these hardcoded absolute paths in `CMakeLists.txt`, `tokenizer.cpp`, `main.cpp`, and `tokenizer_helper.py`.

---

## 4. Subprocess Communication Protocol (JSON-RPC)

The C++ class `Tokenizer` talks to `tokenizer_helper.py` using a custom JSON protocol over stdin/stdout. Each message must be exactly one line, terminated by `\n`.

### A. Get Special Tokens
* **Request**:
  ```json
  {"command": "get_special_tokens"}
  ```
* **Response**:
  ```json
  {"status": "ok", "pad_token_id": 0, "eos_token_ids": [2, 3]}
  ```

### B. Encode Chat Prompts
* **Request**:
  ```json
  {"command": "encode_chat_prompts", "problems": ["2 + 2", "3 * 5"]}
  ```
* **Response** (returns token arrays with applied prompt templates):
  ```json
  {"status": "ok", "ids": [[101, 1023, 203, 102], [101, 1024, 205, 102]]}
  ```

### C. Decode Token Sequences
* **Request**:
  ```json
  {"command": "decode", "ids": [[101, 1023], [101, 1024]]}
  ```
* **Response**:
  ```json
  {"status": "ok", "texts": ["Solution text 1", "Solution text 2"]}
  ```

---

## 5. Database Schema Integration

The runner connects to PostgreSQL using the DSN from the environment. It operates on a table called `problemset`.

### Expected Table Schema
```sql
CREATE TABLE problemset (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    problem TEXT NOT NULL,
    answer TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    solution TEXT,
    time_to_solve INT, -- time taken in seconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

* **Inputs**: The runner selects rows `WHERE solution IS NULL ORDER BY id ASC LIMIT <batch_limit>`.
* **Outputs**: Upon generation completion, it issues:
  ```sql
  UPDATE problemset SET solution = $1, time_to_solve = $2 WHERE id = $3;
  ```
* **Replacement Logic**: The generator maps model outputs containing `[THINK]` and `[/THINK]` reasoning tags to standard `<think>` and `</think>` tags before saving.

---

## 6. How to Build & Run

### Prerequisites
1. Ensure the Python virtualenv is set up at `/Users/bran/src/play/llm/.venv` and has `transformers` and `mlx` installed.
2. Install `postgresql@18` via homebrew (to provide `libpq` header/libraries).

### Compilation
From the `mlx_cpp` directory:
```bash
mkdir -p build && cd build
cmake ..
make -j4
```
This generates the executable `mlx_cpp_runner` inside the `build` directory.

### Execution
Provide the DSN of the running PostgreSQL instance:
```bash
export POSTGRES_GO_DSN="postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
./build/mlx_cpp_runner
```

---

## 7. Key Algorithms & Traps for Agents

### A. Autoregressive Batch Generation
* **Padding**: Tokens are left-padded using the tokenizer's `pad_token_id`. The initial key-value cache offset is set to `-left_padding`.
* **Cache Compaction**: During generation, some sequences in the batch will hit the EOS token before others. The loop filters the batch in-place to stop running finished sequences. The KVCache has a `.filter(batch_indices)` method that uses `mlx::core::take` and removes prepended padding once all remaining sequences have moved past the initial padding size (`min_left_pad`).
* **Diverse Rollouts**: `generate_with_diverse_rollouts()` takes a single prompt batch, performs the model prefill step *once*, replicates the cache keys/values and logits `group_size` times, and executes distinct generation paths using high-temperature (`0.7`) sampling.

### B. Common Traps for Agents
1. **Dynamic Library Paths**: Running the executable requires that `libmlx.dylib` can be located. `CMakeLists.txt` sets `CMAKE_INSTALL_RPATH` to `${MLX_DIR}/lib`, which embeds the runpath into the binary. Do not move the built binary outside `build/` unless you copy/link the dylib or configure `DYLD_LIBRARY_PATH`.
2. **Tokenizer Helper Exit Code**: If `AutoTokenizer` fails to load (e.g. invalid model path or missing python libraries), the python helper exits immediately. The C++ class constructor will throw a `runtime_error` during special token fetching since the read pipe will reach EOF.
3. **Regex Fix**: `AutoTokenizer.from_pretrained` inside `tokenizer_helper.py` utilizes `fix_mistral_regex=True`. If utilizing a non-mistral tokenizer model, this parameter might need to be adjusted.
