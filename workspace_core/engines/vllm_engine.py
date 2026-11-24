"""
vLLM Engine
High-throughput PagedAttention server executing vllm.entrypoints.openai.api_server.
Supports full manual overrides (--max-model-len, --gpu-memory-utilization, --dtype,
--quantization, --kv-cache-dtype, --tensor-parallel-size, --enforce-eager, etc.).
"""

from typing import Any, Dict, List, Optional

from .base import BaseEngine
from ..config import ServerPreset, WorkspaceConfig
from ..env_manager import EnvironmentManager
from ..registry import ModelEntry


class VLLMEngine(BaseEngine):
    """Executes official vLLM OpenAI API server."""

    backend_id = "vllm"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.log_file = self.config.runtime_logs_dir / "vllm_server.log"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides: Any,
    ) -> bool:
        self.model_entry = model_entry
        self.preset = self._merge_preset(preset, "vllm_high_throughput", overrides)
        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        if not self.model_entry:
            raise ValueError("No model prepared for VLLMEngine.")

        p = self.preset or ServerPreset(name="vllm_default", backend="vllm")
        py_exec = EnvironmentManager(self.config).get_profile_python("vllm")

        cmd = [
            py_exec,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "--trust-remote-code",
        ]

        if api_key:
            cmd.extend(["--api-key", str(api_key)])

        if p.ctx_size and int(p.ctx_size) > 0:
            cmd.extend(["--max-model-len", str(int(p.ctx_size))])

        gpu_util = p.custom_args.get("gpu_memory_utilization", p.gpu_memory_utilization)
        if gpu_util is not None:
            cmd.extend(["--gpu-memory-utilization", str(gpu_util)])

        dtype_val = p.custom_args.get("dtype", p.dtype)
        if dtype_val:
            cmd.extend(["--dtype", str(dtype_val)])

        if p.quantization_mode and p.quantization_mode not in ("auto", "fp16", "bf16", "fp32", "bnb_4bit"):
            cmd.extend(["--quantization", str(p.quantization_mode)])
        elif p.quantization_mode == "bnb_4bit":
            cmd.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])

        if p.kv_cache_dtype and p.kv_cache_dtype.lower() != "auto":
            cmd.extend(["--kv-cache-dtype", str(p.kv_cache_dtype)])

        if p.parallel and int(p.parallel) > 1:
            cmd.extend(["--max-num-seqs", str(int(p.parallel))])

        if p.tensor_parallel_size and int(p.tensor_parallel_size) > 1:
            cmd.extend(["--tensor-parallel-size", str(int(p.tensor_parallel_size))])

        if p.enforce_eager:
            cmd.append("--enforce-eager")

        # Avoid duplicating gpu_memory_utilization / dtype from custom_args
        filtered_preset = p.model_copy(
            update={
                "custom_args": {
                    k: v for k, v in (p.custom_args or {}).items()
                    if k not in ("gpu_memory_utilization", "dtype")
                }
            }
        )
        return self._append_extra_cli_args(cmd, filtered_preset)

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if not self.model_entry:
            raise ValueError("No model prepared for VLLMEngine.")
        EnvironmentManager(self.config).ensure_ready("vllm")
        cmd = self.build_command(host, port, api_key)
        return self._spawn_and_wait(cmd, host, port, api_key, self.backend_id)

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "models": f"{base}/v1/models",
            "chat_completions": f"{base}/v1/chat/completions",
            "completions": f"{base}/v1/completions",
            "health": f"{base}/health",
        }
