"""
ik_llama.cpp Engine
Executes ikawrakow/ik_llama.cpp with support for non-uniform IQ quants,
optimized Mixture-of-Experts (MoE) tensor offload rules, and Flash Attention.
"""

import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import requests

from .base import BaseEngine, EngineStatus
from ..config import ServerPreset, WorkspaceConfig, get_config
from ..hardware import get_hardware_profile
from ..registry import ModelEntry


class IKLlamaEngine(BaseEngine):
    """Controls the ik_llama-server runtime lifecycle with MoE tensor offload support."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self.log_file = self.config.runtime_logs_dir / "ik_llama_server.log"
        self.bin_path = self.config.runtime_root / "bin" / "ik_llama_cpp" / "llama-server"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides
    ) -> bool:
        """Prepares model entry, applies preset, and locates ik_llama-server binary."""
        self.model_entry = model_entry
        base_preset = preset or self.config.presets.get("ik_llama_moe") or self.config.presets.get("gguf_balanced")
        if base_preset:
            p_dict = base_preset.model_dump()
            for k, v in overrides.items():
                if v is not None and hasattr(base_preset, k):
                    p_dict[k] = v
            self.preset = ServerPreset(**p_dict)
        else:
            self.preset = ServerPreset(name="custom_ik", **{k: v for k, v in overrides.items() if v is not None and hasattr(ServerPreset, k)})

        local_p = Path(model_entry.local_path)
        if not local_p.exists():
            raise FileNotFoundError(f"Model file not found at {local_p}")

        if not self.bin_path.exists():
            system_bin = shutil.which("llama-server") or "/usr/local/bin/llama-server"
            if Path(system_bin).exists():
                self.bin_path = Path(system_bin)
            else:
                from ..env_manager import EnvironmentManager
                em = EnvironmentManager(self.config)
                em.fetch_ik_llama_cpp()
                if not self.bin_path.exists():
                    bins = em.fetch_llama_cpp()
                    self.bin_path = Path(bins.get("llama-server", "/usr/local/bin/llama-server"))

        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        """Constructs command line arguments for ik_llama-server."""
        if not self.model_entry:
            raise ValueError("No model prepared for IKLlamaEngine.")

        p = self.preset or ServerPreset(name="ik_default")
        hw = get_hardware_profile()

        cmd = [
            str(self.bin_path),
            "-m", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "-c", str(p.ctx_size),
            "-ngl", str(p.n_gpu_layers),
            "--parallel", str(p.parallel),
            "--jinja",
            "-v",
        ]

        # Flash Attention
        if p.flash_attn and hw.primary_gpu and hw.primary_gpu.supports_flash_attn:
            cmd.extend(["-fa", "on"])
        else:
            cmd.extend(["-fa", "off"])

        # KV Cache Quantization
        if p.cache_type_k:
            cmd.extend(["-ctk", p.cache_type_k])
        if p.cache_type_v:
            cmd.extend(["-ctv", p.cache_type_v])

        # Security API Key
        if api_key:
            cmd.extend(["--api-key", str(api_key)])

        # Additional custom args (e.g. tensor offloading -ot)
        for k, v in p.custom_args.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            elif v is not None:
                cmd.extend([flag, str(v)])

        return cmd

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        """Starts ik_llama-server in the background."""
        if self.process and self.process.poll() is None:
            self.stop()

        self.api_key = api_key
        cmd = self.build_command(host, port, api_key)
        print(f"[IKLlamaEngine] Starting server: {' '.join(cmd[:6])} ...")

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(self.log_file, "w", encoding="utf-8")

        self.process = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True
        )

        self.status = EngineStatus(
            is_running=False,
            backend_name="ik_llama",
            pid=self.process.pid,
            host=host,
            port=port,
            active_model_id=self.model_entry.model_id if self.model_entry else None,
            api_key_configured=bool(api_key),
            endpoints=self.get_endpoints(),
            log_file_path=str(self.log_file)
        )

        healthy = False
        start_wait = time.time()
        while time.time() - start_wait < 60:
            if self.process.poll() is not None:
                print(f"[IKLlamaEngine] Server exited prematurely with code {self.process.returncode}.")
                return False
            if self.healthcheck():
                healthy = True
                break
            time.sleep(1.0)

        self.status.is_running = healthy
        return healthy

    def stop(self) -> bool:
        """Stops the ik_llama-server process."""
        if self.process and self.process.poll() is None:
            print(f"[IKLlamaEngine] Stopping server process (PID {self.process.pid})...")
            try:
                import signal
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except Exception:
                try:
                    import signal
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.process = None

        self.status.is_running = False
        self.status.pid = None
        return True

    def healthcheck(self) -> bool:
        """Checks if the server responds on /health or /v1/models."""
        port = self.status.port
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", headers=headers, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", headers=headers, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        return False

    def get_endpoints(self) -> Dict[str, str]:
        port = self.status.port
        return {
            "health": f"http://127.0.0.1:{port}/health",
            "models": f"http://127.0.0.1:{port}/v1/models",
            "chat_completions": f"http://127.0.0.1:{port}/v1/chat/completions",
            "completions": f"http://127.0.0.1:{port}/v1/completions",
        }
