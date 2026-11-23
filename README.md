# Universal AI Workspace

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neutron84/universal-ai-workspace/blob/main/Workspace_Launcher.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Universal AI Workspace provides a modular environment for running, converting, and quantizing machine learning models on Google Colab, cloud GPUs, and Linux workstations. It integrates native CUDA execution backends, dynamic hardware budget allocation, and decoupled storage modes without mandatory cloud drive dependencies.

---

## Setup and Quickstart

1. Open `Workspace_Launcher.ipynb` in Google Colab.
2. In Cell 1 (`Setup Configuration`), choose your desired code source, storage mode, input model, engine, and output destination.
3. Select **Runtime > Run All** (`Ctrl + F9`) or run the individual cells in sequence.
4. If `task="serve"` was selected, use the generated client snippet or public Cloudflare tunnel to send OpenAI-compatible API requests.

---

## Supported Engines and Converters

| Engine / Converter | Type | Supported Formats | Hardware Support | Description |
| :--- | :--- | :--- | :--- | :--- |
| **llama.cpp** | Engine & Converter | GGUF (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ4_XS`, etc.) | CPU, CUDA (`sm_75` to `sm_90`) | Native inference, Safetensors-to-GGUF conversion, imatrix calibration, and shard splitting. |
| **ik_llama.cpp** | Engine & Converter | Non-uniform GGUF (`IQ4_K`, `IQ4_KS`, `IQ3_K`, `IQ2_K`, etc.) | CUDA (`sm_75` to `sm_90`) | Optimized non-uniform IQ quants and hybrid MoE CPU/GPU tensor offload (`-ot`). |
| **unsloth** | Engine & Converter | Dynamic GGUF, 4-bit LoRA Safetensors | CUDA (`sm_75`+) | Fast LoRA adapter merging and Dynamic GGUF export (`q4_k_xl`, `q4_k_m`, `q5_k_m`). |
| **vllm** | Engine | Safetensors (AWQ, GPTQ, FP8, FP16) | CUDA (`sm_89`+, L4, A100, H100) | High-throughput PagedAttention server for quantized and unquantized Safetensors. |
| **transformers** | Engine | Safetensors (BitsAndBytes 4-bit, 8-bit, FP16) | CPU, CUDA (T4, L4, A100) | General fallback server for Hugging Face transformer models. |
| **exllamav2** | Engine & Converter | EXL2 (fractional bpw) | CUDA (`sm_75`+) | Fractional bits-per-weight quantization and low-overhead inference. |
| **gptqmodel** | Converter | GPTQ (4-bit, 8-bit) | CUDA | Quantizes Safetensors models using the GPTQ algorithm. |
| **autoawq** | Converter | AWQ (4-bit), FP8 | CUDA (`sm_75` to `sm_90`) | Activation-aware Weight Quantization for vLLM and Transformers. |
| **faster-whisper / ct2** | Engine & Converter | CTranslate2 | CPU, CUDA | Converts Hugging Face Whisper models and serves transcription endpoints. |
| **diffusers** | Engine | Safetensors, Checkpoints | CUDA | Stable Diffusion image generation server with `/v1/images/generations` endpoint. |

---

## Workspace Storage Modes

| Mode | Target Location | Google Drive Mount | Description |
| :--- | :--- | :--- | :--- |
| `local_ssd` | `/content/ai_workspace` | Not Mounted | Operates entirely on ephemeral local SSD. Recommended for public notebooks. |
| `hybrid` | `/content/drive/MyDrive/AI_Workspace` | Mounted | Writes runtime cache to local SSD and syncs registered models and configs to Drive. |
| `google_drive` | Absolute Drive path | Mounted | Keeps models and configuration directly in the specified Google Drive folder. |
| `gdrive_relative`| Relative to `MyDrive/` | Mounted | Resolves paths relative to Google Drive root directory. |
| `custom_path` | Custom mount or directory | Optional | Allows arbitrary mount paths (e.g., shared network volumes or external drives). |

---

## Input and Output Options

### Input Sources (`input_type`)

| Input Type | Format / Example | Description |
| :--- | :--- | :--- |
| `auto` | URI or repo ID | Automatically identifies source based on URL scheme or syntax. |
| `huggingface` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | Resolves models from Hugging Face Hub. Supports optional `input_file`. |
| `direct_upload` | Interactive Colab file upload | Uploads file directly into `/content/runtime/inputs/`. |
| `local_path` | `/content/models/model.gguf` | Loads an existing file or directory from local filesystem. |
| `google_drive` | `/content/drive/MyDrive/models/model.gguf` | Loads from an absolute Google Drive path. |
| `gdrive_relative`| `models/model.gguf` or folder name | Copies relative file or folder from `MyDrive/` into local working directory. |
| `gdrive_share` | `gshare://<id>` or share URL | Downloads public Google Drive shared files or folders using `gdown`. |
| `url` | Direct HTTP/HTTPS link | Downloads via `aria2c` with parallel segmented connections. |
| `civitai` | `civitai://<id>` or CivitAI URL | Queries CivitAI API and downloads model checkpoint. |
| `git_repo` | `https://github.com/org/repo.git` | Shallow clones repository into local runtime. |

### Output Sinks (`output_type`)

| Output Type | Behavior | Description |
| :--- | :--- | :--- |
| `none` | Retains artifact at source | No export or copy is performed. |
| `local_ssd` | Saves to `/content/runtime/outputs/` | Fast local SSD storage. |
| `direct_download` | Triggers browser download | Auto-downloads through browser via `google.colab.files.download`. |
| `google_drive` | Staged to local SSD then copied | Backs up artifact to persistent Google Drive path. |
| `gdrive_relative` | Staged then copied to `MyDrive/` | Copies output relative to `/content/drive/MyDrive/`. |
| `gdrive_direct` | Direct write to Google Drive | Writes directly to Drive without local copy. Useful when local disk is constrained. |
| `huggingface_hub` | Uploads via `HfApi` | Publishes converted or quantized model directly to Hugging Face repository. |
| `custom_path` | Copies to directory | Exports to arbitrary filesystem path. |

---

## Setup Form Parameters

- `code_source`: Source of workspace code (`github`, `google_drive`, `gdrive_relative`, `url`, `local_dir`).
- `code_path`: Path or URL for code repository. Default: `https://github.com/Neutron84/universal-ai-workspace.git`.
- `workspace_mode`: Storage strategy (`local_ssd`, `hybrid`, `google_drive`, `gdrive_relative`, `custom_path`).
- `workspace_path`: Path relative to `MyDrive` for hybrid/relative modes, or absolute path for custom modes.
- `input_type`: Source type for the model or archive.
- `input_path`: Model identifier, repo ID, URL, or filesystem path. Ignored if `input_type` is `direct_upload`.
- `input_file`: Specific file inside repo or folder. Left blank for automatic selection.
- `unpack_archive`: Automatically extracts `.zip`, `.tar.gz`, `.tar.xz`, and `.tar.bz2` input archives.
- `task`: Operation mode (`serve` to host an API server, `convert_and_quantize` to run conversion, or `dashboard_only`).
- `engine`: Inference engine (`auto`, `llama.cpp`, `ik_llama`, `vllm`, `transformers`, `unsloth`, `exllamav2`, `diffusers`, `faster-whisper`).
- `quantizer`: Quantization backend (`auto`, `llama.cpp`, `ik_llama`, `unsloth`, `gptqmodel`, `autoawq`, `exllamav2`, `ct2`).
- `target_quant`: Desired quantization format (`auto`, `Q4_K_M`, `IQ4_KS`, `Q8_0`, `AWQ`, `GPTQ`, `EXL2`, etc.).
- `context_size`: Context window size in tokens. Default: `4096`.
- `use_imatrix`: Enables importance matrix calibration during `llama.cpp` quantization.
- `output_type`: Destination for artifacts (`none`, `local_ssd`, `direct_download`, `google_drive`, `gdrive_relative`, `gdrive_direct`, `huggingface_hub`, `custom_path`).
- `output_path`: Destination path or Hugging Face repository ID (`username/repo-name`).
- `output_name`: Custom filename or folder name. Derived automatically if left blank.
- `zip_output`: Compresses exported artifacts into a `.zip` archive before saving or downloading.
- `cleanup_source`: Deletes oversized source files after successful conversion to reclaim local disk space.
- `enable_tunnel`: Launches a Cloudflare tunnel for public API access.
- `enable_api_auth`: Generates and enforces an API key for server access.
- `port`: Local port for the inference server. Default: `8000`.
- `hf_token`: Hugging Face access token for gated models and repository uploads.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
