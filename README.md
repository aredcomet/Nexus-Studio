# Nexus-Studio

A unified, multi-architecture LLM inference gateway and studio UI optimized for Apple Silicon (MLX). This project serves as a local playground to run, train, and test models like Gemma 4, RecurrentGemma, and T5v2 natively on your Mac's unified memory, complete with dynamic VRAM management and an MCP plugin system.

## Directory Structure

```text
.
├── apps/                 # Core applications
│   ├── frontend/         # Svelte/Vite User Interface
│   └── server/           # FastAPI Gateway & MLX Engine Router
├── config/               # Declarative configurations
│   ├── models_registry.json # Dynamic model loader registry
│   ├── mcp.json          # MCP server plugin configurations
│   └── presets/          # Saved chat parameter UI presets
├── engines/              # Native MLX architecture engines
│   ├── recurrentgemma/   # Engine for RecurrentGemma models
│   └── t5v2/             # Engine for T5v2 encoder-decoder models
├── scripts/              # Utility scripts
│   ├── convert/          # Data and model conversion tools
│   ├── test/             # Inference testing scripts
│   ├── train/            # Training and fine-tuning loops
│   └── utils/            # Bash and patch utilities
├── storage/              # Heavy data (Ignored by Git)
│   ├── adapters/         # LoRA adapters
│   ├── data/             # Training datasets
│   ├── models/           # Tokenizers and config files
│   └── weights/          # .safetensors files
└── notebooks/            # Jupyter notebook scratchpads (Ignored by Git)
```

## Setup & Execution

We use `uv` to manage the Python environment and dependencies for fast package management. Do not use `pip`; use `uv add <package>` instead.

**1. Start the Backend API**
```bash
python apps/server/server.py
```

**2. Start the Frontend UI (In a new terminal)**
```bash
cd apps/frontend
npm run dev
```

## Configuration & Environment Variables

If you are using MCP plugins that require API tokens (e.g., HuggingFace), you can inject environment variables directly into `config/mcp.json` using the `${VARIABLE_NAME}` syntax. The backend will automatically expand them on startup.
