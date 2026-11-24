"""
Workspace Configuration & Path Management
Provides structured settings, rich server presets supporting both Auto and Manual tuning,
and unified CLI/config serialization helpers.
"""

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from pydantic import BaseModel, Field, model_validator

from .io_resolver import (
    IOResolver,
    ResolvedWorkspacePaths,
    WorkspaceMode,
    get_default_local_root,
    get_default_runtime_root,
)


DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/AI_Workspace"
FALLBACK_LOCAL_ROOT = str(get_default_local_root())
DEFAULT_RUNTIME_ROOT = str(get_default_runtime_root())


def parse_cli_args(raw: Union[str, List[str], Dict[str, Any], None]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalizes arbitrary custom server arguments from string, list, or dict
    into (dict_args, cli_token_list).
    """
    dict_args: Dict[str, Any] = {}
    cli_tokens: List[str] = []

    if not raw:
        return dict_args, cli_tokens

    if isinstance(raw, dict):
        return dict(raw), []

    if isinstance(raw, list):
        return {}, [str(x) for x in raw if str(x).strip()]

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return dict_args, cli_tokens
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return parsed, []
            except Exception:
                pass
        try:
            tokens = shlex.split(stripped)
        except ValueError:
            tokens = stripped.split()
        return {}, tokens

    return dict_args, cli_tokens


class ServerPreset(BaseModel):
    name: str
    backend: str = "llama.cpp"
    ctx_size: int = 4096  # 0 means auto
    n_gpu_layers: int = -1
    gpu_offload_mode: str = "auto"  # auto, all, cpu, manual
    flash_attn: Optional[bool] = True  # None means auto
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    parallel: int = 1
    batch_size: int = 512
    ubatch_size: int = 512
    threads: int = 0  # 0 means auto
    threads_batch: int = 0  # 0 means auto
    split_mode: str = "layer"  # layer, row, none
    tensor_split: Optional[str] = None
    main_gpu: int = 0
    override_tensor: Optional[str] = None
    mlock: bool = False
    mmap: bool = True
    cont_batching: bool = True
    jinja: bool = True
    chat_template: Optional[str] = None
    reasoning_format: Optional[str] = None
    rope_scaling: Optional[str] = None
    rope_freq_base: Optional[float] = None
    rope_freq_scale: Optional[float] = None
    seed: Optional[int] = None
    enable_webui: bool = True
    embedding: Optional[bool] = None
    reranking: bool = False
    draft_model_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    lora_paths: List[str] = Field(default_factory=list)
    quantization_mode: str = "auto"  # auto, bnb_4bit, bnb_8bit, fp16, bf16, fp32, fp8, awq, gptq
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    kv_cache_dtype: str = "auto"
    diffusion_steps: int = 25
    diffusion_guidance_scale: float = 7.5
    diffusion_attention_slicing: bool = True
    diffusion_cpu_offload: bool = False
    whisper_compute_type: str = "auto"
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True
    custom_args: Dict[str, Any] = Field(default_factory=dict)
    extra_cli_args: List[str] = Field(default_factory=list)
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_custom_args(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw_custom = data.get("custom_args")
        if isinstance(raw_custom, (str, list)):
            dict_args, cli_tokens = parse_cli_args(raw_custom)
            data = dict(data)
            data["custom_args"] = dict_args
            existing_extra = list(data.get("extra_cli_args") or [])
            data["extra_cli_args"] = existing_extra + cli_tokens
        raw_extra = data.get("extra_cli_args")
        if isinstance(raw_extra, str):
            _, cli_tokens = parse_cli_args(raw_extra)
            data = dict(data)
            data["extra_cli_args"] = cli_tokens
        return data


class WorkspaceSettings(BaseModel):
    workspace_name: str = "Universal AI Workspace"
    workspace_mode: str = "local_ssd"  # local_ssd, hybrid, google_drive, gdrive_relative, custom_path
    drive_root: str = DEFAULT_DRIVE_ROOT
    runtime_root: str = DEFAULT_RUNTIME_ROOT
    default_backend: str = "llama.cpp"
    port: int = 8000
    dashboard_port: int = 7860
    host: str = "0.0.0.0"
    tunnel_type: str = "cloudflared"  # cloudflared, ngrok, none
    tunnel_hostname: str = ""
    enforce_auth: bool = True
    auto_sync_to_drive: bool = False
    ssd_low_space_warning_gb: float = 10.0
    drive_low_space_warning_gb: float = 5.0
    auto_cleanup_temp_on_low_space: bool = True
    active_profile: str = "core"
    env_isolation_mode: str = "auto"  # auto, isolated_venv, system
    venv_system_site_packages: bool = True
    auto_install_dependencies: bool = True
    llama_cpp_release_tag: str = "latest"
    cuda_target: str = "auto"  # auto, cuda-12, cuda-13, cpu
    download_backend: str = "auto"  # auto, aria2c, hf_transfer, urllib
    aria2_connections: int = 16
    selection_strategy: str = "balanced"  # balanced, max_quality, max_speed, max_context, low_vram_safe
    cuda_overhead_mb: float = 1200.0
    vram_safety_margin_pct: float = 0.05
    auto_requantize: bool = True
    auto_partial_offload: bool = True
    prefer_ik_llama_for_moe: bool = True
    default_quantizer: str = "auto"
    default_target_quant: str = "auto"
    default_outtype: str = "f16"
    use_imatrix: bool = False
    imatrix_chunks: int = 100
    split_max_size: str = ""
    cleanup_intermediate_files: bool = True
    healthcheck_timeout_sec: int = 120
    cors_allow_origins: str = "*"
    extra_settings: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceConfig:
    """Manages workspace paths, directory initialization, settings persistence, and presets."""

    def __init__(
        self,
        drive_root: Optional[str] = None,
        runtime_root: Optional[str] = None,
        workspace_mode: Optional[Union[str, WorkspaceMode]] = None,
        workspace_path: Optional[str] = None,
    ):
        # Determine mode
        if workspace_mode:
            mode = workspace_mode
            ws_path = workspace_path or drive_root
        elif drive_root:
            if drive_root.startswith("/content/drive/MyDrive"):
                mode = WorkspaceMode.GOOGLE_DRIVE
            else:
                mode = WorkspaceMode.CUSTOM_PATH
            ws_path = drive_root
        else:
            mode = WorkspaceMode.LOCAL_SSD
            ws_path = workspace_path or FALLBACK_LOCAL_ROOT

        # Resolve paths using IOResolver
        self.paths: ResolvedWorkspacePaths = IOResolver.resolve_workspace_paths(
            mode=mode,
            workspace_path=ws_path,
            runtime_root=runtime_root or DEFAULT_RUNTIME_ROOT,
        )

        self.mode = self.paths.mode
        self.drive_root = self.paths.workspace_root
        self.runtime_root = self.paths.runtime_root

        # Core directories
        self.workspace_core_dir = self.paths.workspace_core_dir
        self.configs_dir = self.paths.configs_dir
        self.cache_dir = self.paths.cache_dir
        self.uv_cache_dir = self.paths.uv_cache_dir
        self.bin_dir = self.paths.bin_dir
        self.models_dir = self.paths.models_dir
        self.models_gguf_dir = self.paths.models_gguf_dir
        self.models_safetensors_dir = self.paths.models_safetensors_dir
        self.models_diffusion_dir = self.paths.models_diffusion_dir
        self.models_audio_dir = self.paths.models_audio_dir

        # Config files
        self.settings_file = self.paths.settings_file
        self.presets_file = self.paths.presets_file
        self.registry_file = self.paths.registry_file

        # Runtime directories
        self.runtime_envs_dir = self.paths.runtime_envs_dir
        self.runtime_models_cache_dir = self.paths.runtime_models_cache_dir
        self.runtime_logs_dir = self.paths.runtime_logs_dir
        self.runtime_outputs_dir = self.paths.runtime_outputs_dir
        self.drive_mounted = self.paths.drive_mounted

        # Ensure python path includes workspace_core directory parent
        parent_dir = str(self.drive_root)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Environment variable overrides
        os.environ["UV_CACHE_DIR"] = str(self.uv_cache_dir)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        self.ensure_directories()
        self.settings = self.load_settings()
        self.presets = self.load_presets()

    def ensure_directories(self) -> None:
        """Create all required persistent and runtime directories if they do not exist."""
        for directory in [
            self.drive_root,
            self.workspace_core_dir,
            self.configs_dir,
            self.cache_dir,
            self.uv_cache_dir,
            self.bin_dir,
            self.models_dir,
            self.models_gguf_dir,
            self.models_safetensors_dir,
            self.models_diffusion_dir,
            self.models_audio_dir,
            self.runtime_root,
            self.runtime_envs_dir,
            self.runtime_models_cache_dir,
            self.runtime_logs_dir,
            self.runtime_outputs_dir,
        ]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def load_settings(self) -> WorkspaceSettings:
        """Load settings from settings.yaml or create defaults."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                return WorkspaceSettings(**data)
            except Exception as e:
                print(f"[WorkspaceConfig] Warning: Failed to parse settings.yaml ({e}), using defaults.")

        # Check for packaged template configs
        repo_settings = Path(__file__).parent.parent / "configs" / "settings.yaml"
        if repo_settings.exists() and repo_settings.resolve() != self.settings_file.resolve():
            try:
                with open(repo_settings, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                data["workspace_mode"] = self.mode.value if hasattr(self.mode, "value") else str(self.mode)
                data["drive_root"] = str(self.drive_root)
                data["runtime_root"] = str(self.runtime_root)
                default_settings = WorkspaceSettings(**data)
            except Exception:
                default_settings = WorkspaceSettings(
                    workspace_mode=self.mode.value if hasattr(self.mode, "value") else str(self.mode),
                    drive_root=str(self.drive_root),
                    runtime_root=str(self.runtime_root),
                )
        else:
            default_settings = WorkspaceSettings(
                workspace_mode=self.mode.value if hasattr(self.mode, "value") else str(self.mode),
                drive_root=str(self.drive_root),
                runtime_root=str(self.runtime_root),
            )
        self.save_settings(default_settings)
        return default_settings

    def save_settings(self, settings: Optional[WorkspaceSettings] = None) -> None:
        """Save settings to settings.yaml safely."""
        if settings is not None:
            self.settings = settings
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_file, "w", encoding="utf-8") as f:
                yaml.dump(self.settings.model_dump(), f, default_flow_style=False, sort_keys=True)
        except (PermissionError, OSError):
            pass

    def update_settings(self, **kwargs: Any) -> WorkspaceSettings:
        """Updates one or more settings fields and persists settings.yaml."""
        data = self.settings.model_dump()
        extra = dict(data.get("extra_settings") or {})
        valid_fields = set(WorkspaceSettings.model_fields.keys())
        for key, val in kwargs.items():
            if key in valid_fields:
                data[key] = val
            else:
                extra[key] = val
        data["extra_settings"] = extra
        updated = WorkspaceSettings(**data)
        self.save_settings(updated)
        return updated

    def _builtin_default_presets(self) -> Dict[str, ServerPreset]:
        return {
            "gguf_balanced": ServerPreset(
                name="GGUF Balanced (4K FlashAttn)",
                backend="llama.cpp",
                ctx_size=4096,
                n_gpu_layers=-1,
                gpu_offload_mode="auto",
                flash_attn=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                parallel=1,
                description="Fast and memory-efficient for 7B-14B models on T4/L4 GPUs.",
            ),
            "gguf_long_ctx": ServerPreset(
                name="GGUF Long Context (16K FlashAttn + Q4 Cache)",
                backend="llama.cpp",
                ctx_size=16384,
                n_gpu_layers=-1,
                gpu_offload_mode="auto",
                flash_attn=True,
                cache_type_k="q4_0",
                cache_type_v="q4_0",
                parallel=1,
                description="High context window with quantized KV cache for long document processing.",
            ),
            "vlm_multimodal": ServerPreset(
                name="VLM Multimodal (Vision-Language)",
                backend="llama.cpp",
                ctx_size=4096,
                n_gpu_layers=-1,
                gpu_offload_mode="auto",
                flash_attn=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                parallel=1,
                description="Multimodal LLM with mmproj projector enabled.",
            ),
            "vllm_high_throughput": ServerPreset(
                name="vLLM High Throughput (L4/A100)",
                backend="vllm",
                ctx_size=4096,
                dtype="auto",
                gpu_memory_utilization=0.90,
                custom_args={"gpu_memory_utilization": 0.90, "dtype": "auto"},
                description="PagedAttention server for unquantized / AWQ / GPTQ Safetensors.",
            ),
            "transformers_4bit": ServerPreset(
                name="Transformers + BitsAndBytes 4-Bit",
                backend="transformers",
                ctx_size=4096,
                quantization_mode="bnb_4bit",
                dtype="auto",
                custom_args={"load_in_4bit": True, "device_map": "auto"},
                description="Flexible fallback for Safetensors models on T4 GPUs.",
            ),
            "ik_llama_moe": ServerPreset(
                name="ik_llama.cpp MoE & IQ Hybrid",
                backend="ik_llama",
                ctx_size=4096,
                n_gpu_layers=-1,
                gpu_offload_mode="auto",
                flash_attn=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                description="Optimized engine for non-uniform IQ quants and MoE hybrid CPU/GPU offload.",
            ),
            "unsloth_fast": ServerPreset(
                name="Unsloth 4-Bit Fast Inference",
                backend="unsloth",
                ctx_size=4096,
                quantization_mode="bnb_4bit",
                custom_args={"load_in_4bit": True},
                description="Unsloth FastLanguageModel engine for LoRA and 4-bit Safetensors.",
            ),
            "exllamav2_fast": ServerPreset(
                name="ExLlamaV2 EXL2 Fast Inference",
                backend="exllamav2",
                ctx_size=4096,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                description="Fractional bpw EXL2 execution engine for NVIDIA GPUs.",
            ),
            "diffusers_sd": ServerPreset(
                name="Diffusers Image Generation",
                backend="diffusers",
                ctx_size=512,
                diffusion_steps=25,
                diffusion_guidance_scale=7.5,
                diffusion_attention_slicing=True,
                description="Stable Diffusion / SDXL / Flux image generation API server.",
            ),
            "whisper_audio": ServerPreset(
                name="Faster-Whisper Speech-to-Text",
                backend="faster-whisper",
                whisper_compute_type="auto",
                whisper_beam_size=5,
                whisper_vad_filter=True,
                description="Faster-Whisper CTranslate2 speech-to-text transcription API server.",
            ),
        }

    def load_presets(self) -> Dict[str, ServerPreset]:
        """Load server presets from presets.yaml or create defaults (merging any new built-ins)."""
        defaults = self._builtin_default_presets()
        if self.presets_file.exists():
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f) or {}
                presets = {}
                for key, val in raw_data.items():
                    presets[key] = ServerPreset(**val)
                # Merge any missing new built-in presets without clobbering user presets
                merged = False
                for k, v in defaults.items():
                    if k not in presets:
                        presets[k] = v
                        merged = True
                if merged:
                    self.save_presets(presets)
                return presets
            except Exception as e:
                print(f"[WorkspaceConfig] Warning: Failed to parse presets.yaml ({e}), creating defaults.")

        # Check for packaged template presets
        repo_presets = Path(__file__).parent.parent / "configs" / "presets.yaml"
        if repo_presets.exists() and repo_presets.resolve() != self.presets_file.resolve():
            try:
                with open(repo_presets, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f) or {}
                presets = {}
                for key, val in raw_data.items():
                    presets[key] = ServerPreset(**val)
                for k, v in defaults.items():
                    presets.setdefault(k, v)
                self.save_presets(presets)
                return presets
            except Exception:
                pass

        self.save_presets(defaults)
        return defaults

    def save_presets(self, presets: Dict[str, ServerPreset]) -> None:
        """Save server presets to presets.yaml safely."""
        self.presets = presets
        serializable = {k: v.model_dump() for k, v in presets.items()}
        try:
            self.presets_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.presets_file, "w", encoding="utf-8") as f:
                yaml.dump(serializable, f, default_flow_style=False)
        except (PermissionError, OSError):
            pass

    def upsert_preset(self, key: str, preset: ServerPreset) -> Dict[str, ServerPreset]:
        """Creates or updates a named preset and persists presets.yaml."""
        clean_key = key.strip().replace(" ", "_")
        if not clean_key:
            raise ValueError("Preset key cannot be empty.")
        self.presets[clean_key] = preset
        self.save_presets(self.presets)
        return self.presets

    def delete_preset(self, key: str) -> bool:
        """Deletes a custom preset from presets.yaml."""
        if key not in self.presets:
            return False
        del self.presets[key]
        self.save_presets(self.presets)
        return True

    def export_config_bundle(self) -> Dict[str, Any]:
        """Exports current settings and presets as a portable dictionary."""
        return {
            "settings": self.settings.model_dump(),
            "presets": {k: v.model_dump() for k, v in self.presets.items()},
        }

    def import_config_bundle(self, data: Dict[str, Any]) -> None:
        """Imports settings and presets from a dictionary."""
        if "settings" in data and isinstance(data["settings"], dict):
            self.save_settings(WorkspaceSettings(**data["settings"]))
        if "presets" in data and isinstance(data["presets"], dict):
            imported = {k: ServerPreset(**v) for k, v in data["presets"].items()}
            self.save_presets(imported)


_CONFIG_INSTANCE: Optional[WorkspaceConfig] = None


def get_config(
    drive_root: Optional[str] = None,
    runtime_root: Optional[str] = None,
    workspace_mode: Optional[Union[str, WorkspaceMode]] = None,
    workspace_path: Optional[str] = None,
    force_reload: bool = False,
) -> WorkspaceConfig:
    global _CONFIG_INSTANCE
    if (
        _CONFIG_INSTANCE is None
        or drive_root is not None
        or workspace_mode is not None
        or workspace_path is not None
        or force_reload
    ):
        _CONFIG_INSTANCE = WorkspaceConfig(
            drive_root=drive_root,
            runtime_root=runtime_root,
            workspace_mode=workspace_mode,
            workspace_path=workspace_path,
        )
    return _CONFIG_INSTANCE
