"""
Native Llama.cpp Engine
Executes official ggml-org/llama-server directly with full hardware acceleration,
OpenAI-compatible REST API, Flash Attention, KV Cache Quantization, and VLM mmproj support.
"""

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from .base import BaseEngine, EngineStatus
from ..config import ServerPreset, WorkspaceConfig, get_config
from ..hardware import get_hardware_profile
from ..registry import ModelEntry


class LlamaCppEngine(BaseEngine):
    """Controls the native llama-server binary lifecycle."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self.log_file = self.config.runtime_logs_dir / "llama_server.log"
        self.bin_path = Path("/usr/local/bin/llama-server")

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides
    ) -> bool:
        """Validates model path and applies preset parameters with optional dynamic overrides."""
        self.model_entry = model_entry
        base_preset = preset or self.config.presets.get("gguf_balanced")
        if base_preset:
            p_dict = base_preset.model_dump()
            for k, v in overrides.items():
                if v is not None and hasattr(base_preset, k):
                    p_dict[k] = v
            self.preset = ServerPreset(**p_dict)
        else:
            self.preset = ServerPreset(name="custom", **{k: v for k, v in overrides.items() if v is not None and hasattr(ServerPreset, k)})

        local_p = Path(model_entry.local_path)
        if not local_p.exists():
            raise FileNotFoundError(f"Model file not found at {local_p}")

        if not self.bin_path.exists():
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

        # Embedding flag for embedding models
        if self.model_entry.type == "embedding-gguf":
            cmd.append("--embedding")

        # Vision Projector (VLM)
        if self.model_entry.mmproj_path and Path(self.model_entry.mmproj_path).exists():
            cmd.extend(["-mm", str(self.model_entry.mmproj_path)])

        # Security API Key
        if api_key:
            cmd.extend(["--api-key", str(api_key)])

        # Additional custom args from preset
        for k, v in p.custom_args.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            elif v is not None:
                cmd.extend([flag, str(v)])

        return cmd

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        """Starts native llama-server in the background with file logging."""
        if self.process and self.process.poll() is None:
            print("[LlamaCppEngine] Server is already running.")
            return True

        if not self.model_entry:
            raise ValueError("Model entry not configured. Call prepare() first.")

        self.api_key = api_key
        cmd = self.build_command(host, port, api_key)
        self.config.runtime_logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"[LlamaCppEngine] Launching llama-server on {host}:{port}...")
        log_f = open(self.log_file, "w", encoding="utf-8")

        # Set up dynamic library path environment
        env = os.environ.copy()
        lib_dirs = set()
        for p in self.config.runtime_root.glob("bin/**/*.so*"):
            lib_dirs.add(str(p.parent.resolve()))
        for p in self.config.runtime_root.glob("bin/**/llama_cpp*"):
            if p.is_dir():
                lib_dirs.add(str(p.resolve()))
        current_ld = env.get("LD_LIBRARY_PATH", "")
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (":" + current_ld if current_ld else "")

        self.process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid
        )
        log_f.close()

        self.status = EngineStatus(
            is_running=True,
            backend_name="llama.cpp",
            pid=self.process.pid,
            host=host,
            port=port,
            active_model_id=self.model_entry.model_id,
            api_key_configured=bool(api_key),
            endpoints=self._generate_endpoints(host, port),
            log_file_path=str(self.log_file)
        )

        # Wait for server to become healthy (up to 90 seconds)
        healthy = False
        for _ in range(90):
            time.sleep(1)
            if self.process.poll() is not None:
                print(f"[LlamaCppEngine] Server exited prematurely with code {self.process.returncode}.")
                self.status.is_running = False
                return False
            if self.healthcheck():
                healthy = True
                break

        if healthy:
            print(f"[LlamaCppEngine] llama-server is healthy and ready on port {port}!")
            return True
        else:
            print("[LlamaCppEngine] Warning: Server started but healthcheck timed out.")
            return False

    def stop(self) -> bool:
        """Stops the llama-server process cleanly."""
        pid_to_kill = self.process.pid if self.process else self.status.pid
        if not pid_to_kill:
            self.status.is_running = False
            return True

        print(f"[LlamaCppEngine] Stopping server process PID {pid_to_kill}...")
        try:
            # Terminate process group
            try:
                os.killpg(os.getpgid(pid_to_kill), signal.SIGTERM)
            except Exception:
                os.kill(pid_to_kill, signal.SIGTERM)

            if self.process:
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[LlamaCppEngine] Force killing process...")
                    try:
                        os.killpg(os.getpgid(pid_to_kill), signal.SIGKILL)
                    except Exception:
                        os.kill(pid_to_kill, signal.SIGKILL)
                    self.process.wait(timeout=3)
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
        except Exception as e:
            print(f"[LlamaCppEngine] Error during process termination: {e}")

        self.process = None
        self.status.pid = None
        self.status.is_running = False
        return True

    def healthcheck(self) -> bool:
        """Checks whether llama-server is responding."""
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

        url = f"http://127.0.0.1:{self.status.port}/health"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            r = requests.get(url, headers=headers, timeout=2)
            if r.status_code == 200:
                data = r.json()
                return data.get("status") in ["ok", "success"]
        except Exception:
            pass

        # Fallback check /v1/models
        try:
            r = requests.get(f"http://127.0.0.1:{self.status.port}/v1/models", headers=headers, timeout=2)
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
            "embeddings": f"{base}/v1/embeddings",
            "web_ui": base,
            "health": f"{base}/health",
        }
