"""
Process & Tunnel Manager
Orchestrates inference server and public tunnel lifecycles.
Implements pipe-deadlock-free Cloudflare tunneling, port cleanup, security token enforcement,
and robust cross-cell runtime state persistence to runtime_state.json.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from .config import ServerPreset, WorkspaceConfig, get_config
from .engines.base import BaseEngine
from .engines.llama_cpp import LlamaCppEngine
from .engines.transformers_engine import TransformersEngine
from .engines.vllm_engine import VLLMEngine
from .engines.diffusers_engine import DiffusersEngine
from .engines.audio_engine import AudioEngine
from .registry import ModelEntry, ModelRegistry
from .security import SecurityManager
from .auto_selector import SelectionRecommendation


@dataclass
class TunnelInfo:
    is_running: bool
    tunnel_type: str  # "cloudflared", "ngrok", "none"
    public_url: Optional[str] = None
    pid: Optional[int] = None
    log_file_path: Optional[str] = None


@dataclass
class SystemState:
    engine_name: str
    is_server_running: bool
    server_pid: Optional[int]
    port: int
    active_model: Optional[str]
    api_key: Optional[str]
    is_tunnel_running: bool
    tunnel_type: str
    public_url: Optional[str]
    local_endpoints: Dict[str, str]
    public_endpoints: Dict[str, str]


class ProcessManager:
    """Controls server backends, tunnels, port reclamation, and persistent runtime endpoints."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.registry = ModelRegistry(self.config)
        self.state_file = self.config.runtime_root / "runtime_state.json"
        self.current_engine: Optional[BaseEngine] = None
        self.tunnel_process: Optional[subprocess.Popen] = None
        self.tunnel_info = TunnelInfo(is_running=False, tunnel_type="none")
        self.cloudflared_log = self.config.runtime_logs_dir / "cloudflared.log"
        self.api_key: Optional[str] = None

        # Reconstruct state from runtime_state.json across notebook cells
        self._load_and_reconstruct_state()

    def _is_pid_alive(self, pid: Optional[int]) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _check_http_health(self, port: int, api_key: Optional[str] = None) -> bool:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
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

    def _save_state(self) -> None:
        """Persists live runtime state to /content/runtime/runtime_state.json."""
        server_pid = None
        backend_name = "None"
        port = 8000
        active_model = None

        if self.current_engine and self.current_engine.status.is_running:
            server_pid = self.current_engine.status.pid
            backend_name = self.current_engine.status.backend_name
            port = self.current_engine.status.port
            active_model = self.current_engine.status.active_model_id

        tunnel_pid = self.tunnel_info.pid if (self.tunnel_info and self.tunnel_info.is_running) else None
        tunnel_type = self.tunnel_info.tunnel_type if self.tunnel_info else "none"
        public_url = self.tunnel_info.public_url if self.tunnel_info else None

        state_data = {
            "server_pid": server_pid,
            "backend_name": backend_name,
            "port": port,
            "active_model": active_model,
            "api_key": self.api_key,
            "tunnel_pid": tunnel_pid,
            "tunnel_type": tunnel_type,
            "public_url": public_url,
            "log_files": {
                "server": str(self.current_engine.status.log_file_path) if (self.current_engine and self.current_engine.status.log_file_path) else None,
                "tunnel": str(self.cloudflared_log),
            },
            "updated_at": time.time(),
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            print(f"[ProcessManager] Warning saving runtime state: {e}")

    def _load_and_reconstruct_state(self) -> None:
        """Loads runtime_state.json, verifies live PIDs and health, and reconstructs endpoints."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        server_pid = data.get("server_pid")
        backend_name = data.get("backend_name", "llama.cpp")
        port = data.get("port", 8000)
        active_model = data.get("active_model")
        api_key = data.get("api_key")
        tunnel_pid = data.get("tunnel_pid")
        tunnel_type = data.get("tunnel_type", "none")
        public_url = data.get("public_url")
        log_files = data.get("log_files", {})

        # Verify server PID & health
        server_alive = False
        if server_pid and self._is_pid_alive(server_pid):
            if self._check_http_health(port, api_key):
                server_alive = True

        if server_alive:
            self.api_key = api_key
            if not self.current_engine:
                if backend_name == "llama.cpp":
                    self.current_engine = LlamaCppEngine(self.config)
                elif backend_name == "transformers":
                    self.current_engine = TransformersEngine(self.config)
                elif backend_name == "vllm":
                    self.current_engine = VLLMEngine(self.config)
                elif backend_name == "diffusers":
                    self.current_engine = DiffusersEngine(self.config)
                elif backend_name == "audio":
                    self.current_engine = AudioEngine(self.config)
                else:
                    self.current_engine = LlamaCppEngine(self.config)

                self.current_engine.status.is_running = True
                self.current_engine.status.pid = server_pid
                self.current_engine.status.port = port
                self.current_engine.status.active_model_id = active_model
                self.current_engine.status.api_key_configured = bool(api_key)
                self.current_engine.status.log_file_path = log_files.get("server")
                self.current_engine.status.endpoints = self.current_engine.get_endpoints()
                self.current_engine.api_key = api_key

        # Verify tunnel PID
        tunnel_alive = False
        if tunnel_pid and self._is_pid_alive(tunnel_pid) and public_url:
            tunnel_alive = True
            self.tunnel_info = TunnelInfo(
                is_running=True,
                tunnel_type=tunnel_type,
                public_url=public_url,
                pid=tunnel_pid,
                log_file_path=log_files.get("tunnel")
            )
        else:
            self.tunnel_info = TunnelInfo(is_running=False, tunnel_type="none")

    def cleanup_port(self, port: int) -> None:
        """Kills any process currently occupying the specified port (protecting system services)."""
        if port in [8080, 9000, 6000, 2222]:
            return
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            time.sleep(0.5)
        except Exception:
            pass

    def start_server(
        self,
        model_id: Optional[str] = None,
        preset_name: Optional[str] = None,
        port: int = 8000,
        enable_auth: bool = True,
        recommendation: Optional[SelectionRecommendation] = None,
        **preset_overrides
    ) -> bool:
        """Starts the appropriate inference engine for the selected or active model with dynamic overrides."""
        self.stop_server()
        self.cleanup_port(port)

        # Select model
        if model_id:
            entry = self.registry.get_model(model_id)
            if not entry:
                raise ValueError(f"Model ID '{model_id}' not found in registry.")
            self.registry.set_active_model(model_id)
        else:
            entry = self.registry.get_active_model()
            if not entry:
                raise ValueError("No model specified and no active model found in registry.")

        # Determine preset
        preset = None
        if preset_name:
            preset = self.config.presets.get(preset_name)

        # Merge recommendation overrides if provided
        overrides = dict(preset_overrides)
        if recommendation:
            if recommendation.recommended_ctx:
                overrides["ctx_size"] = recommendation.recommended_ctx
            if recommendation.recommended_n_gpu_layers is not None:
                overrides["n_gpu_layers"] = recommendation.recommended_n_gpu_layers
            if recommendation.flash_attn is not None:
                overrides["flash_attn"] = recommendation.flash_attn
            if recommendation.cache_type_k:
                overrides["cache_type_k"] = recommendation.cache_type_k
            if recommendation.cache_type_v:
                overrides["cache_type_v"] = recommendation.cache_type_v

        # Select engine backend
        backend = preset.backend if preset else "llama.cpp"
        if entry.type in ["llm-gguf", "vlm-gguf", "embedding-gguf"]:
            backend = "llama.cpp"
        elif entry.type == "diffusion":
            backend = "diffusers"
        elif entry.type == "audio":
            backend = "audio"

        if backend == "llama.cpp":
            self.current_engine = LlamaCppEngine(self.config)
        elif backend == "vllm":
            self.current_engine = VLLMEngine(self.config)
        elif backend == "transformers":
            self.current_engine = TransformersEngine(self.config)
        elif backend == "diffusers":
            self.current_engine = DiffusersEngine(self.config)
        elif backend == "audio":
            self.current_engine = AudioEngine(self.config)
        else:
            self.current_engine = LlamaCppEngine(self.config)

        # Configure API Key security
        if enable_auth:
            self.api_key = SecurityManager.get_or_create_workspace_api_key()
        else:
            self.api_key = None

        print(f"[ProcessManager] Preparing {backend} engine for model '{entry.name}' with overrides {overrides}...")
        self.current_engine.prepare(entry, preset, **overrides)
        success = self.current_engine.start(host="0.0.0.0", port=port, api_key=self.api_key)

        # Persist live state to runtime_state.json
        self._save_state()
        return success

    def stop_server(self) -> bool:
        """Stops the current inference server and releases port/VRAM."""
        if self.current_engine:
            self.current_engine.stop()
            self.current_engine = None

        # Check state file PID in case engine instance was lost across notebook cells
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                old_pid = data.get("server_pid")
                if old_pid and self._is_pid_alive(old_pid):
                    print(f"[ProcessManager] Terminating persisted server PID {old_pid}...")
                    try:
                        os.killpg(os.getpgid(old_pid), signal.SIGTERM)
                    except Exception:
                        try:
                            os.kill(old_pid, signal.SIGTERM)
                        except Exception:
                            pass
            except Exception:
                pass

        self._save_state()
        return True

    def start_tunnel(self, port: int = 8000, tunnel_type: str = "cloudflared") -> Optional[str]:
        """
        Starts public tunnel without subprocess pipe deadlocks.
        Redirects output directly to cloudflared.log and extracts public URL.
        """
        self.stop_tunnel()
        self.config.runtime_logs_dir.mkdir(parents=True, exist_ok=True)

        if tunnel_type == "cloudflared":
            cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
            if not Path(cf_bin).exists():
                from .env_manager import EnvironmentManager
                em = EnvironmentManager(self.config)
                cf_bin = em.fetch_cloudflared()

            # Check for persistent Named Tunnel token
            named_token = SecurityManager.get_cf_tunnel_token()
            if named_token:
                cmd = [cf_bin, "tunnel", "run", "--token", named_token]
            else:
                # Quick tunnel
                cmd = [
                    cf_bin,
                    "tunnel",
                    "--url", f"http://127.0.0.1:{port}",
                    "--no-autoupdate"
                ]

            # Redirect stdout/stderr directly to file handle to prevent pipe buffer deadlock
            log_f = open(self.cloudflared_log, "w", encoding="utf-8")
            print(f"[ProcessManager] Launching cloudflared tunnel on port {port}...")

            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            # Close file descriptor in parent to prevent FD leak
            log_f.close()

            # Poll the log file to extract the public URL
            public_url = None
            url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

            for _ in range(35):
                time.sleep(1)
                if self.tunnel_process.poll() is not None:
                    print(f"[ProcessManager] cloudflared exited unexpectedly with code {self.tunnel_process.returncode}.")
                    break

                if self.cloudflared_log.exists():
                    try:
                        with open(self.cloudflared_log, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        match = url_pattern.search(content)
                        if match:
                            public_url = match.group(0)
                            break
                    except Exception:
                        pass

            if public_url:
                print(f"[ProcessManager] Cloudflare Tunnel established: {public_url}")
                self.tunnel_info = TunnelInfo(
                    is_running=True,
                    tunnel_type="cloudflared",
                    public_url=public_url,
                    pid=self.tunnel_process.pid,
                    log_file_path=str(self.cloudflared_log)
                )
                self._save_state()
                return public_url
            else:
                print("[ProcessManager] Warning: Could not detect public URL from cloudflared logs.")
                self.tunnel_info = TunnelInfo(
                    is_running=self.tunnel_process.poll() is None,
                    tunnel_type="cloudflared",
                    public_url=None,
                    pid=self.tunnel_process.pid if self.tunnel_process else None,
                    log_file_path=str(self.cloudflared_log)
                )
                self._save_state()
                return None

        elif tunnel_type == "ngrok":
            ngrok_token = SecurityManager.get_ngrok_token()
            if not ngrok_token:
                print("[ProcessManager] Ngrok token not found in secrets/env.")
                return None
            try:
                from pyngrok import ngrok  # type: ignore
                ngrok.set_auth_token(ngrok_token)
                tunnel = ngrok.connect(port)
                self.tunnel_info = TunnelInfo(
                    is_running=True,
                    tunnel_type="ngrok",
                    public_url=tunnel.public_url
                )
                self._save_state()
                return tunnel.public_url
            except Exception as e:
                print(f"[ProcessManager] Ngrok error: {e}")
                return None

        return None

    def stop_tunnel(self) -> bool:
        """Stops the active tunnel process cleanly."""
        if self.tunnel_process:
            print(f"[ProcessManager] Stopping tunnel process PID {self.tunnel_process.pid}...")
            try:
                os.killpg(os.getpgid(self.tunnel_process.pid), signal.SIGTERM)
                self.tunnel_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.tunnel_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.tunnel_process = None

        # Check persisted tunnel PID
        if self.tunnel_info and self.tunnel_info.pid and self._is_pid_alive(self.tunnel_info.pid):
            try:
                os.killpg(os.getpgid(self.tunnel_info.pid), signal.SIGTERM)
            except Exception:
                try:
                    os.kill(self.tunnel_info.pid, signal.SIGTERM)
                except Exception:
                    pass

        if self.tunnel_info.tunnel_type == "ngrok":
            try:
                from pyngrok import ngrok  # type: ignore
                ngrok.kill()
            except Exception:
                pass

        self.tunnel_info = TunnelInfo(is_running=False, tunnel_type="none")
        self._save_state()
        return True

    def get_state(self) -> SystemState:
        """Returns unified system, server, and tunnel state (syncing with persisted state)."""
        self._load_and_reconstruct_state()

        server_running = self.current_engine.healthcheck() if self.current_engine else False
        port = self.current_engine.status.port if self.current_engine else 8000
        backend_name = self.current_engine.status.backend_name if self.current_engine else "None"
        server_pid = self.current_engine.status.pid if self.current_engine else None
        active_model = self.current_engine.status.active_model_id if self.current_engine else None

        local_endpoints = self.current_engine.get_endpoints() if self.current_engine else {}
        public_endpoints = {}

        if self.tunnel_info.is_running and self.tunnel_info.public_url:
            base_pub = self.tunnel_info.public_url
            public_endpoints = {
                "base_url": base_pub,
                "models": f"{base_pub}/v1/models",
                "chat_completions": f"{base_pub}/v1/chat/completions",
                "completions": f"{base_pub}/v1/completions",
                "web_ui": base_pub,
            }

        return SystemState(
            engine_name=backend_name,
            is_server_running=server_running,
            server_pid=server_pid,
            port=port,
            active_model=active_model,
            api_key=self.api_key,
            is_tunnel_running=self.tunnel_info.is_running,
            tunnel_type=self.tunnel_info.tunnel_type,
            public_url=self.tunnel_info.public_url,
            local_endpoints=local_endpoints,
            public_endpoints=public_endpoints,
        )

    def shutdown_all(self) -> None:
        """Full system shutdown: stops server, stops tunnel, and updates persisted state."""
        self.stop_server()
        self.stop_tunnel()
        self._save_state()
