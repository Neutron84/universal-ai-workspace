"""
Workspace Configuration & Path Management
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from pydantic import BaseModel, Field

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


class ServerPreset(BaseModel):
    name: str
    backend: str = "llama.cpp"
    ctx_size: int = 4096
    n_gpu_layers: int = -1
    flash_attn: bool = True
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    parallel: int = 1
    custom_args: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class WorkspaceSettings(BaseModel):
    workspace_name: str = "Universal AI Workspace"
    workspace_mode: str = "local_ssd"  # local_ssd, hybrid, google_drive, gdrive_relative, custom_path
    drive_root: str = DEFAULT_DRIVE_ROOT
    runtime_root: str = DEFAULT_RUNTIME_ROOT
    default_backend: str = "llama.cpp"
    port: int = 8000
    host: str = "0.0.0.0"
    tunnel_type: str = "cloudflared"  # cloudflared, ngrok, none
    enforce_auth: bool = True
    auto_sync_to_drive: bool = True
    ssd_low_space_warning_gb: float = 10.0
    drive_low_space_warning_gb: float = 5.0
    active_profile: str = "core"
    extra_settings: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceConfig:
    """Manages workspace paths, directory initialization, and settings persistence."""

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
                    runtime_root=str(self.runtime_root)
                )
        else:
            default_settings = WorkspaceSettings(
                workspace_mode=self.mode.value if hasattr(self.mode, "value") else str(self.mode),
                drive_root=str(self.drive_root),
                runtime_root=str(self.runtime_root)
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
                yaml.dump(self.settings.model_dump(), f, default_flow_style=False)
        except (PermissionError, OSError):
            pass

    def load_presets(self) -> Dict[str, ServerPreset]:
        """Load server presets from presets.yaml or create defaults."""
        if self.presets_file.exists():
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f) or {}
                presets = {}
                for key, val in raw_data.items():
                    presets[key] = ServerPreset(**val)
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
                self.save_presets(presets)
                return presets
            except Exception:
                pass

        defaults = {
            "gguf_balanced": ServerPreset(
                name="GGUF Balanced (4K FlashAttn)",
                backend="llama.cpp",
                ctx_size=4096,
                n_gpu_layers=-1,
                flash_attn=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                parallel=1,
                description="Fast and memory-efficient for 7B-14B models on T4/L4 GPUs."
            ),
            "gguf_long_ctx": ServerPreset(
                name="GGUF Long Context (16K FlashAttn + Q4 Cache)",
                backend="llama.cpp",
                ctx_size=16384,
                n_gpu_layers=-1,
                flash_attn=True,
                cache_type_k="q4_0",
                cache_type_v="q4_0",
                parallel=1,
                description="High context window with quantized KV cache for long document processing."
            ),
            "vlm_multimodal": ServerPreset(
                name="VLM Multimodal (Vision-Language)",
                backend="llama.cpp",
                ctx_size=4096,
                n_gpu_layers=-1,
                flash_attn=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                parallel=1,
                description="Multimodal LLM with mmproj projector enabled."
            ),
            "vllm_high_throughput": ServerPreset(
                name="vLLM High Throughput (L4/A100)",
                backend="vllm",
                ctx_size=4096,
                custom_args={"gpu_memory_utilization": 0.90, "dtype": "auto"},
                description="PagedAttention server for unquantized / AWQ / GPTQ Safetensors."
            ),
            "transformers_4bit": ServerPreset(
                name="Transformers + BitsAndBytes 4-Bit",
                backend="transformers",
                ctx_size=4096,
                custom_args={"load_in_4bit": True, "device_map": "auto"},
                description="Flexible fallback for Safetensors models on T4 GPUs."
            ),
            "ik_llama_moe": ServerPreset(
                name="ik_llama.cpp MoE & IQ Hybrid",
                backend="ik_llama",
                ctx_size=4096,
                n_gpu_layers=-1,
                flash_attn=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                description="Optimized engine for non-uniform IQ quants and MoE hybrid CPU/GPU offload."
            ),
            "unsloth_fast": ServerPreset(
                name="Unsloth 4-Bit Fast Inference",
                backend="unsloth",
                ctx_size=4096,
                custom_args={"load_in_4bit": True},
                description="Unsloth FastLanguageModel engine for LoRA and 4-bit Safetensors."
            )
        }
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
