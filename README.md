# 🚀 Universal AI Workspace for Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neutron84/universal-ai-workspace/blob/main/Workspace_Launcher.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12 & 13](https://img.shields.io/badge/CUDA-12.8%20%7C%2013.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A production-grade, 100% shareable, multi-backend AI inference workspace designed for Google Colab, cloud GPUs, and local Linux workstations.

---

## 🌟 Key Features

1. **True GPU Acceleration on CUDA 12 & 13 (No Silent CPU Fallback!)**:
   - Automated detection and installation of native `llama.cpp` CUDA 12.8 binaries paired with required `cudart` (`libcublas.so.12`, `libcudart.so.12`, `libcublasLt.so.12`) libraries, ensuring full GPU offloading on modern Colab runtimes (Tesla T4, L4, A100, H100).
2. **True Multi-Mode Storage & Decoupled Execution**:
   - **`hybrid`**: Ephemeral fast Local SSD runtime (`/content/runtime`) + persistent Google Drive archive.
   - **`local_ssd` (`ephemeral`)**: 100% Google Drive-free operation. Anyone opening the shared `.ipynb` can run without mounting Google Drive!
   - **`google_drive` / `gdrive_relative`**: Complete persistent storage on Google Drive.
   - **`custom_path`**: Support for Shared Drives, Kaggle (`/kaggle/working`), or arbitrary mounted volumes.
3. **Universal Input Source Resolver (`IOResolver`)**:
   - `auto`: Automatic URI detection.
   - `huggingface`: `hf://org/repo` or standard `org/repo`.
   - `gdrive_share`: `gshare://<file_or_folder_id>` or public Google Drive share URLs via `gdown`.
   - `civitai`: `civitai://<model_id>` or CivitAI model URLs with API resolution.
   - `direct`: `direct:///path/to/model` or filesystem paths.
   - `url`: Direct HTTP/HTTPS download with 16 parallel connections via `aria2c`.
   - `git_repo`: Shallow git cloning.
4. **Universal Artifact & Output Sinks (`OutputManager`)**:
   - Export models, re-quantized weights, diffusion images, audio transcripts, and benchmarks to Local SSD, Google Drive, Hugging Face Hub (`HfApi`), or direct browser download (`google.colab.files.download`).
5. **Cross-Cell Process & Tunnel State Persistence**:
   - Live server state, active models, API keys, and Cloudflare tunnel endpoints persist to `/content/runtime/runtime_state.json`. Notebook cells and the 6-tab Gradio dashboard seamlessly share live states without losing handles.
6. **Multi-Engine Support**:
   - Native **`llama.cpp`** (GGUF LLM, VLM with `mmproj`, Embeddings).
   - Hugging Face **`transformers`** with BitsAndBytes 4-bit / 8-bit quantization.
   - **`vLLM`** PagedAttention server.
   - Real **`diffusers`** server (OpenAI `/v1/images/generations` returning base64 PNG).
   - Real **`faster-whisper`** server (OpenAI `/v1/audio/transcriptions`).

---

## 📂 Architecture Overview

```
universal-ai-workspace/
├── Workspace_Launcher.ipynb   # Thin interactive Colab controller notebook
├── pyproject.toml             # Python package configuration & dependencies
├── .gitignore                 # Clean repository exclusions
├── README.md                  # Documentation and quickstart guide
├── configs/
│   ├── presets.yaml           # Inference presets (GGUF Balanced, Long Ctx, 4-bit, vLLM)
│   ├── settings.yaml          # Workspace default ports, host, and preferences
│   └── registry.json          # Machine-independent model registry
└── workspace_core/            # Decoupled core Python module
    ├── io_resolver.py         # Universal Input / Workspace / Output Resolver
    ├── hardware.py            # Safe hardware telemetry & dynamic CUDA library profiler
    ├── env_manager.py         # Fast uv package manager & native binary installer
    ├── model_inspector.py     # Multi-source inspection & split-GGUF grouping
    ├── auto_selector.py       # VRAM budget calculator & safe offload optimizer
    ├── downloader.py          # High-speed aria2c & Hugging Face downloader
    ├── registry.py            # Model metadata, shard management, and cache cleaner
    ├── process_manager.py     # Process lifecycle, deadlock-free tunnels, & JSON state persistence
    ├── dashboard.py           # Complete 6-tab interactive Gradio dashboard
    ├── security.py            # API key generation and token management
    └── engines/               # Modular inference backends
        ├── base.py            # Abstract engine base class & benchmarking
        ├── llama_cpp.py       # Native llama-server executor
        ├── transformers_engine.py  # FastAPI Transformers server
        ├── vllm_engine.py     # vLLM OpenAI API server
        ├── diffusers_engine.py# Real FastAPI Stable Diffusion server
        └── audio_engine.py    # Real FastAPI Faster-Whisper server
```

---

## 🚀 Quickstart in Google Colab

1. Open `Workspace_Launcher.ipynb` in Google Colab.
2. In **Cell 1**, select your desired **Workspace Mode**:
   - Choose `hybrid` to persist downloads and models to Google Drive.
   - Choose `local_ssd` to run completely ephemerally without mounting Google Drive.
3. Run **Cell 1** to bootstrap system dependencies, native CUDA llama.cpp binaries, and environment profiles.
4. Run **Cell 3** to launch the full 6-tab interactive Gradio Dashboard, or run **Cell 4** for headless 1-click execution.
5. In **Cell 5**, test live streaming completions directly against the OpenAI-compatible API or public Cloudflare Tunnel.

---

## 🌐 Universal I/O URI Cheat Sheet

| Source Type | Example URI Format | Description |
| :--- | :--- | :--- |
| **Hugging Face** | `hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF` or `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | Resolves GGUF variants, split shards, or Safetensors |
| **Google Drive Share** | `gshare://1a2b3c4d5e6f7g8h9i` or `https://drive.google.com/file/d/...` | Downloads public shared Drive files/folders via `gdown` |
| **CivitAI** | `civitai://123456` or `https://civitai.com/models/123456` | Resolves primary model file via CivitAI API |
| **Direct URL** | `https://example.com/models/custom-model.gguf` | High-speed 16-connection download via `aria2c` |
| **Local File** | `direct:///content/my_model.gguf` or `/content/my_model.gguf` | Instant access to existing local filesystem file |
| **Drive Relative** | `gdrive://AI_Workspace/models/gguf/model.gguf` | Resolves relative to `/content/drive/MyDrive/` |
| **Git Repository** | `https://github.com/org/custom-model.git` | Shallow clone of model / code repository |

---

## 🛠️ Publishing to GitHub

To publish this workspace to your own GitHub repository:

```bash
# 1. Initialize git in the workspace folder
cd /content/drive/MyDrive/AI_Workspace
git init

# 2. Add remote origin
git remote add origin https://github.com/Neutron84/universal-ai-workspace.git

# 3. Add tracked code (large weights, caches, and logs are automatically ignored)
git add pyproject.toml .gitignore README.md Workspace_Launcher.ipynb workspace_core/ configs/presets.yaml configs/settings.yaml

# 4. Commit and push
git commit -m "feat: universal AI workspace v1.0.0 with native CUDA acceleration and multi-mode I/O"
git branch -M main
git push -u origin main
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
