"""
vLLM Engine
High-throughput PagedAttention server executing vllm.entrypoints.openai.api_server.
Enabled when GPU Compute Capability >= 7.5 and sufficient VRAM exists.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from .base import BaseEngine, EngineStatus
from ..config import ServerPreset, WorkspaceConfig, get_config
from ..registry import ModelEntry


class VLLMEngine(BaseEngine):
    """Executes official vLLM OpenAI API server."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self.log_file = self.config.runtime_logs_dir / "vllm_server.log"

    def prepare(self, model_entry: ModelEntry, preset: Optional[ServerPreset] = None) -> bool:
        self.model_entry = model_entry
        self.preset = preset or self.config.presets.get("vllm_high_throughput")
        return True

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if self.process and self.process.poll() is None:
            return True

        if not self.model_entry:
            raise ValueError("No model prepared for VLLMEngine.")

        self.api_key = api_key

        # Prefer isolated venv Python if available
        venv_python = self.config.runtime_envs_dir / "vllm" / "bin" / "python"
        py_exec = str(venv_python) if venv_python.exists() else sys.executable

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

        custom = (self.preset.custom_args if self.preset else {}) or {}
        if "gpu_memory_utilization" in custom:
            cmd.extend(["--gpu-memory-utilization", str(custom["gpu_memory_utilization"])])
        if "dtype" in custom:
            cmd.extend(["--dtype", str(custom["dtype"])])

        log_f = open(self.log_file, "w", encoding="utf-8")
        self.process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        log_f.close()

        self.status = EngineStatus(
            is_running=True,
            backend_name="vllm",
            pid=self.process.pid,
            host=host,
            port=port,
            active_model_id=self.model_entry.model_id,
            api_key_configured=bool(api_key),
            endpoints=self._generate_endpoints(host, port),
            log_file_path=str(self.log_file)
        )

        for _ in range(60):
            time.sleep(1)
            if self.process.poll() is not None:
                self.status.is_running = False
                return False
            if self.healthcheck():
                return True
        return False

    def stop(self) -> bool:
        pid_to_kill = self.process.pid if self.process else self.status.pid
        if not pid_to_kill:
            self.status.is_running = False
            return True

        try:
            try:
                os.killpg(os.getpgid(pid_to_kill), signal.SIGTERM)
            except Exception:
                os.kill(pid_to_kill, signal.SIGTERM)

            if self.process:
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(pid_to_kill), signal.SIGKILL)
                    except Exception:
                        os.kill(pid_to_kill, signal.SIGKILL)
            else:
                for _ in range(10):
                    time.sleep(0.2)
                    try:
                        os.kill(pid_to_kill, 0)
                    except OSError:
                        break
                else:
                    try:
                        os.killpg(os.getpgid(pid_to_kill), signal.SIGKILL)
                    except Exception:
                        os.kill(pid_to_kill, signal.SIGKILL)
        except Exception:
            pass

        self.process = None
        self.status.pid = None
        self.status.is_running = False
        return True

    def healthcheck(self) -> bool:
        if self.process:
            if self.process.poll() is not None:
                return False
        elif self.status.pid:
            try:
                os.kill(self.status.pid, 0)
            except OSError:
                return False
        else:
            return False
        try:
            r = requests.get(f"http://127.0.0.1:{self.status.port}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def get_endpoints(self) -> Dict[str, str]:
        return self._generate_endpoints(self.status.host, self.status.port)

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "models": f"{base}/v1/models",
            "chat_completions": f"{base}/v1/chat/completions",
            "completions": f"{base}/v1/completions",
            "health": f"{base}/health",
        }
