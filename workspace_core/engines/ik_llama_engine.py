"""
ik_llama.cpp Engine
Executes ikawrakow/ik_llama.cpp with support for non-uniform IQ quants,
optimized Mixture-of-Experts (MoE) tensor offload rules (-ot), VLM mmproj, and Flash Attention.
"""

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from .llama_cpp import LlamaCppEngine
from ..config import ServerPreset, WorkspaceConfig
from ..registry import ModelEntry


class IKLlamaEngine(LlamaCppEngine):
    """Controls the ik_llama-server runtime lifecycle with MoE tensor offload support."""

    backend_id = "ik_llama"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.log_file = self.config.runtime_logs_dir / "ik_llama_server.log"
        self.bin_path = self.config.runtime_root / "bin" / "ik_llama_cpp" / "llama-server"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides: Any,
    ) -> bool:
        """Prepares model entry, merges preset + overrides, and locates ik_llama-server binary."""
        self.model_entry = model_entry
        self.preset = self._merge_preset(preset, "ik_llama_moe", overrides)

        local_p = Path(model_entry.local_path)
        if not local_p.exists():
            raise FileNotFoundError(f"Model file not found at {local_p}")

        if not self.bin_path.exists():
            from ..env_manager import EnvironmentManager
            em = EnvironmentManager(self.config)
            bins = em.fetch_ik_llama_cpp()
            if "llama-server" in bins and Path(bins["llama-server"]).exists():
                self.bin_path = Path(bins["llama-server"])
            else:
                system_bin = shutil.which("llama-server") or "/usr/local/bin/llama-server"
                if Path(system_bin).exists():
                    self.bin_path = Path(system_bin)
                else:
                    bins = em.fetch_llama_cpp()
                    self.bin_path = Path(bins.get("llama-server", "/usr/local/bin/llama-server"))

        return True

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        """Starts ik_llama-server in the background with CUDA library paths & file logging."""
        if not self.model_entry:
            raise ValueError("No model prepared for IKLlamaEngine.")
        cmd = self.build_command(host, port, api_key)
        return self._spawn_and_wait(cmd, host, port, api_key, self.backend_id)

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "health": f"{base}/health",
            "models": f"{base}/v1/models",
            "chat_completions": f"{base}/v1/chat/completions",
            "completions": f"{base}/v1/completions",
            "embeddings": f"{base}/v1/embeddings",
        }
