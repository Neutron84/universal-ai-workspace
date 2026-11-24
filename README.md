# Universal AI Workspace

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neutron84/universal-ai-workspace/blob/main/Workspace_Launcher.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Universal AI Workspace provides a modular environment for running, converting, quantizing, and benchmarking machine learning models on Google Colab, cloud GPUs, and Linux workstations. Every layer supports both capability-driven `auto` routing and granular manual overrides via the Colab notebook, the 8-tab Gradio dashboard, or the CLI.

---

## Setup and Quickstart

### Google Colab
1. Open `Workspace_Launcher.ipynb` in Google Colab.
2. Leave defaults in Cell 1 (`Setup Configuration`) for automatic hardware/model detection, or expand manual parameters for fine-grained control.
3. Select **Runtime > Run All** (`Ctrl + F9`) or run individual cells in sequence.
4. For serving tasks, use the printed client snippet, local endpoint, or public tunnel URL.

### Command-Line Interface (Linux / Local)
```bash
# Run pre-flight hardware and binary diagnostics
python -m workspace_core doctor

# Dry-run inspect and VRAM/engine planning without downloading weights
python -m workspace_core inspect Qwen/Qwen2.5-0.5B-Instruct-GGUF --strategy balanced

# Serve a model with automatic engine/quant selection
python -m workspace_core serve Qwen/Qwen2.5-0.5B-Instruct-GGUF --ctx auto --flash-attn true

# Convert and quantize a Safetensors repo with imatrix calibration and 2GB shard splitting
python -m workspace_core convert Qwen/Qwen2.5-0.5B-Instruct --target-quant Q4_K_M --imatrix --split-max-size 2G

# Launch the 8-tab Gradio management dashboard
python -m workspace_core dashboard --port 7860
```

---

## Supported Engines and Converters

| Engine / Converter | Type | Supported Formats | Hardware Support | Description |
| :--- | :--- | :--- | :--- | :--- |
| **llama.cpp** | Engine & Converter | GGUF (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ4_XS`, etc.) | CPU, CUDA (`sm_75`+) | Native inference, Safetensors-to-GGUF conversion, `llama-imatrix`, and shard splitting. |
| **ik_llama.cpp** | Engine & Converter | Non-uniform GGUF (`IQ4_K`, `IQ4_KS`, `IQ3_K`, `IQ2_K`, etc.) | CUDA (`sm_75`+) | Optimized non-uniform IQ quants and hybrid MoE CPU/GPU tensor offload (`-ot`). |
| **unsloth** | Engine & Converter | Dynamic GGUF, 4-bit LoRA Safetensors | CUDA (`sm_75`+) | LoRA adapter merging, merged Safetensors export, and Dynamic GGUF (`q4_k_xl`, `q4_k_m`). |
| **vllm** | Engine | Safetensors (AWQ, GPTQ, FP8, FP16) | CUDA (`sm_89`+, L4, A100, H100) | High-throughput PagedAttention server with KV dtype and tensor-parallel controls. |
| **transformers** | Engine | Safetensors (BnB 4-bit, 8-bit, FP16, BF16, LoRA) | CPU, CUDA | Flexible OpenAI-compatible server with automatic PEFT LoRA adapter loading. |
| **exllamav2** | Engine & Converter | EXL2 (fractional bpw), GPTQ | CUDA (`sm_75`+) | Fractional bits-per-weight quantization and native ExLlamaV2 streaming server. |
| **gptqmodel** | Converter | GPTQ (4-bit, 8-bit) | CUDA | Quantizes Safetensors models using GPTQ with configurable group size and desc_act. |
| **autoawq** | Converter | AWQ (4-bit), FP8 | CUDA (`sm_75`+) | Activation-aware Weight Quantization for vLLM and Transformers. |
| **faster-whisper / ct2** | Engine & Converter | CTranslate2 | CPU, CUDA | Converts HF Whisper models and serves `/v1/audio/transcriptions` and `/translations`. |
| **diffusers** | Engine | Safetensors, single-file `.ckpt`/`.safetensors` | CUDA, CPU | Image generation server (`from_pretrained` & `from_single_file`) with LoRA & CPU offload. |

---

## Workspace Storage Modes

| Mode | Target Location | Google Drive Mount | Description |
| :--- | :--- | :--- | :--- |
| `local_ssd` | `/content/ai_workspace` | Not Mounted | Ephemeral local SSD storage. Recommended for public shared notebooks. |
| `hybrid` | `/content/drive/MyDrive/AI_Workspace` | Mounted | Fast SSD runtime cache with Drive model and config sync. |
| `google_drive` | Absolute Drive path | Mounted | Stores workspace directly in the specified Google Drive folder. |
| `gdrive_relative` | Relative to `MyDrive/` | Mounted | Resolves workspace path relative to `/content/drive/MyDrive/`. |
| `custom_path` | Custom mount or directory | Optional | Arbitrary local, network, or container storage path. |

---

## Input Sources and Output Sinks

### Input Sources (`input_type`)
- `auto`: Automatically detects `hf://`, `gdrive://`, `gshare://`, `civitai://`, `git://`, HTTP URLs, local paths, and `org/repo`.
- `huggingface`, `direct_upload`, `local_path`, `google_drive`, `gdrive_relative`, `gdrive_share`, `url`, `civitai`, `git_repo`.

### Output Sinks (`output_type`)
- `none`, `local_ssd`, `direct_download`, `google_drive`, `gdrive_relative`, `gdrive_direct`, `huggingface_hub`, `custom_path`.

---

## Tasks (`task`)

- `serve`: Inspect, resolve/download (auto-requantizing or auto-converting if needed), and start the OpenAI-compatible server.
- `convert_and_quantize`: Convert or quantize weights with optional imatrix calibration and shard splitting, then register and export.
- `convert_and_serve`: Convert Safetensors/LoRA to GGUF first, then immediately launch the GGUF server.
- `benchmark`: Launch the model, measure TTFT and generation throughput (tok/s), and record metrics in the registry.
- `inspect_only`: Dry-run metadata inspection and AutoSelector VRAM budget planning without downloading model weights.
- `dashboard_only`: Launch the 8-tab Gradio management dashboard directly.
- `serve_and_dashboard`: Start the model inference server in the background and launch the management dashboard.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
