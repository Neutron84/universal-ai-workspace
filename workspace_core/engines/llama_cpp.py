"""
Native Llama.cpp Engine
Executes official ggml-org/llama-server directly with full hardware acceleration,
OpenAI-compatible REST API, Flash Attention, KV Cache Quantization, RoPE scaling,
tensor offloading, LoRA adapters, speculative decoding, and VLM mmproj support.
"""

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from .base import BaseEngine
from ..config import ServerPreset, WorkspaceConfig
from ..hardware import get_hardware_profile
from ..registry import ModelEntry


class LlamaCppEngine(BaseEngine):
    """Controls the native llama-server binary lifecycle."""

    backend_id = "llama.cpp"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.log_file = self.config.runtime_logs_dir / "llama_server.log"
        self.bin_path = Path("/usr/local/bin/llama-server")

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides: Any,
    ) -> bool:
        """Validates model path, merges preset parameters with dynamic overrides, and ensures binary."""
        self.model_entry = model_entry
        self.preset = self._merge_preset(preset, "gguf_balanced", overrides)

        local_p = Path(model_entry.local_path)
        if not local_p.exists():
            raise FileNotFoundError(f"Model file not found at {local_p}")

        if not self.bin_path.exists():
            found = shutil.which("llama-server")
            if found:
                self.bin_path = Path(found)
            else:
                from ..env_manager import EnvironmentManager
                em = EnvironmentManager(self.config)
                bins = em.fetch_llama_cpp()
                if "llama-server" in bins:
                    self.bin_path = Path(bins["llama-server"])
                else:
                    raise RuntimeError("Failed to locate or download llama-server binary.")

        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        """Constructs the native llama-server command line arguments."""
        if not self.model_entry:
            raise ValueError("No model prepared for LlamaCppEngine.")

        p = self.preset or ServerPreset(name="default")
        hw = get_hardware_profile()

        ctx_val = p.ctx_size if (p.ctx_size and int(p.ctx_size) > 0) else 4096
        if p.gpu_offload_mode == "cpu":
            ngl_val = 0
        elif p.gpu_offload_mode == "all":
            ngl_val = -1
        else:
            ngl_val = p.n_gpu_layers

        cmd = [
            str(self.bin_path),
            "-m", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "-c", str(ctx_val),
            "-ngl", str(ngl_val),
            "--parallel", str(max(1, int(p.parallel or 1))),
            "-v",
        ]

        if p.jinja:
            cmd.append("--jinja")
        else:
            cmd.append("--no-jinja")

        if p.batch_size and int(p.batch_size) > 0:
            cmd.extend(["-b", str(int(p.batch_size))])
        if p.ubatch_size and int(p.ubatch_size) > 0:
            cmd.extend(["-ub", str(int(p.ubatch_size))])
        if p.threads and int(p.threads) > 0:
            cmd.extend(["-t", str(int(p.threads))])
        if p.threads_batch and int(p.threads_batch) > 0:
            cmd.extend(["-tb", str(int(p.threads_batch))])

        # Multi-GPU / Offload tuning
        if p.split_mode and p.split_mode != "layer":
            cmd.extend(["--split-mode", str(p.split_mode)])
        if p.tensor_split:
            cmd.extend(["--tensor-split", str(p.tensor_split)])
        if p.main_gpu:
            cmd.extend(["--main-gpu", str(int(p.main_gpu))])
        if p.override_tensor:
            cmd.extend(["-ot", str(p.override_tensor)])

        # Memory mapping
        if p.mlock:
            cmd.append("--mlock")
        if not p.mmap:
            cmd.append("--no-mmap")

        # Flash Attention
        fa_enabled = p.flash_attn if p.flash_attn is not None else (bool(hw.primary_gpu and hw.primary_gpu.supports_flash_attn))
        if fa_enabled and hw.primary_gpu and hw.primary_gpu.supports_flash_attn:
            cmd.extend(["-fa", "on"])
        else:
            cmd.extend(["-fa", "off"])

        # KV Cache Quantization
        if p.cache_type_k and p.cache_type_k.lower() != "auto":
            cmd.extend(["-ctk", p.cache_type_k])
        if p.cache_type_v and p.cache_type_v.lower() != "auto":
            cmd.extend(["-ctv", p.cache_type_v])

        # Continuous batching & WebUI
        if not p.cont_batching:
            cmd.append("--no-cont-batching")
        if not p.enable_webui:
            cmd.append("--no-webui")

        # RoPE & Context Scaling
        if p.rope_scaling:
            cmd.extend(["--rope-scaling", str(p.rope_scaling)])
        if p.rope_freq_base is not None:
            cmd.extend(["--rope-freq-base", str(p.rope_freq_base)])
        if p.rope_freq_scale is not None:
            cmd.extend(["--rope-freq-scale", str(p.rope_freq_scale)])

        # Chat template & reasoning format
        if p.chat_template:
            cmd.extend(["--chat-template", str(p.chat_template)])
        if p.reasoning_format:
            cmd.extend(["--reasoning-format", str(p.reasoning_format)])
        if p.seed is not None:
            cmd.extend(["-s", str(int(p.seed))])

        # Embedding / Reranking
        if p.embedding is True or (p.embedding is None and self.model_entry.type == "embedding-gguf"):
            cmd.append("--embedding")
        if p.reranking:
            cmd.append("--reranking")

        # Vision Projector (VLM)
        mmproj = p.mmproj_path or self.model_entry.mmproj_path
        if mmproj and Path(mmproj).exists():
            cmd.extend(["-mm", str(mmproj)])

        # Speculative decoding draft model & LoRAs
        if p.draft_model_path and Path(p.draft_model_path).exists():
            cmd.extend(["-md", str(p.draft_model_path)])
        for lora in (p.lora_paths or []):
            if Path(lora).exists():
                cmd.extend(["--lora", str(lora)])

        # Security API Key
        if api_key:
            cmd.extend(["--api-key", str(api_key)])

        return self._append_extra_cli_args(cmd, p)

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        """Starts native llama-server in the background with file logging."""
        if not self.model_entry:
            raise ValueError("Model entry not configured. Call prepare() first.")
        cmd = self.build_command(host, port, api_key)
        return self._spawn_and_wait(cmd, host, port, api_key, self.backend_id)

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "models": f"{base}/v1/models",
            "chat_completions": f"{base}/v1/chat/completions",
            "completions": f"{base}/v1/completions",
            "embeddings": f"{base}/v1/embeddings",
            "web_ui": base,
            "health": f"{base}/health",
        }
