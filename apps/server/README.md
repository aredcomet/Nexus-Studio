# MLX Multi-Model Gateway & Chat Workspace

This directory contains a hot-swapping unified model server and a dual-sidebar chat dashboard designed to run and switch between various MLX architectures natively on macOS (Apple Silicon).

It provides a local alternative to LM Studio, adding support for:
1. **Recurrent/Hybrid Architectures**: e.g., RecurrentGemma/Griffin.
2. **Encoder-Decoder Models**: e.g., T5Gemma-2.
3. **Dynamic LoRA Merging**: Apply LoRA adapters on top of base models on the fly.
4. **VRAM Clearing**: Full unload capabilities to clear GPU unified memory instantaneously.

---

## File Structure

* **[server.py](file:///Users/bran/src/play/llm/server/server.py)**: Dynamic FastAPI gateway hosting:
  - `POST /v1/model/load`: Hot-loads the chosen model driver (RecurrentGemma, T5v2, or standard LLM) and merges adapters.
  - `POST /v1/model/unload`: Clears the model references, calls garbage collection, and flushes GPU cache (`mx.metal.clear_cache()`).
  - `GET /v1/model/status`: Returns current server/model state.
  - `POST /v1/chat/completions`: Handles streaming/non-streaming tokens for the active model.
* **[index.html](file:///Users/bran/src/play/llm/server/index.html)**: Legacy raw HTML client providing basic controls.
* **[frontend/](file:///Users/bran/src/play/llm/frontend)**: Modern, high-performance workspace client built with Svelte 5 and Tailwind CSS v4.

---

## Getting Started

### 1. Launch the Server
Start the gateway server:
```bash
.venv/bin/python server/server.py
```
*(The server launches on `http://127.0.0.1:8089` by default. It remains empty, consuming 0 VRAM, until a model load is requested).*

### 2. Run the Svelte Workspace UI
Navigate to the frontend directory and start the Vite development server:
```bash
cd frontend
npm run dev
```
Open the localhost address displayed in your terminal (usually `http://localhost:5173`) to open the workspace.

### 3. Load a Preset and Chat
- Use the **Quick Presets** dropdown on the left panel of the UI to populate paths for RecurrentGemma or T5 models.
- Click **Load Selected Model**. A loading overlay will display while the model registers and warms up the GPU.
- Once loaded, type your message in the chat box. Chunks stream back in real-time.
- Hover over messages to access advanced actions:
  - **Edit (✏️)** on User prompts to modify and regenerate conversation threads.
  - **Retry (🔄)** on Assistant responses to request a fresh generation.
- Click **Unload Active Model** to immediately reclaim VRAM for other macOS processes.
