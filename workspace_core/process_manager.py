"""
Process & Tunnel Manager
Orchestrates inference server and public tunnel lifecycles across all 8 engines.
Implements pipe-deadlock-free Cloudflare Quick & Named tunneling, port reclamation,
security token enforcement, code snippet generation, and cross-cell state persistence.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import requests

from .config import WorkspaceConfig, get_config
from .engines import get_engine, normalize_engine_name
from .engines.base import BaseEngine
from .hardware import get_hardware_profile
from .registry import ModelRegistry
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
        """Persists live runtime state to runtime_state.json."""
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

        server_alive = bool(server_pid and self._is_pid_alive(server_pid) and self._check_http_health(port, api_key))

        if server_alive:
            self.api_key = api_key
            if not self.current_engine or self.current_engine.status.pid != server_pid:
                try:
                    self.current_engine = get_engine(backend_name, self.config)
                except ValueError:
                    self.current_engine = get_engine("llama.cpp", self.config)

            if active_model:
                self.current_engine.model_entry = self.registry.get_model(active_model)
            self.current_engine.status.is_running = True
            self.current_engine.status.backend_name = normalize_engine_name(backend_name)
            self.current_engine.status.pid = server_pid
            self.current_engine.status.port = port
            self.current_engine.status.active_model_id = active_model
            self.current_engine.status.api_key_configured = bool(api_key)
            self.current_engine.status.log_file_path = log_files.get("server")
            self.current_engine.status.endpoints = self.current_engine.get_endpoints()
            self.current_engine.api_key = api_key
        else:
            # Clear stale engine reference if persisted server died
            if self.current_engine and self.current_engine.status.pid == server_pid:
                self.current_engine.status.is_running = False
                self.current_engine = None

        if tunnel_type == "ngrok" and public_url:
            self.tunnel_info = TunnelInfo(
                is_running=True,
                tunnel_type="ngrok",
                public_url=public_url,
                pid=tunnel_pid,
            )
        elif tunnel_pid and self._is_pid_alive(tunnel_pid):
            self.tunnel_info = TunnelInfo(
                is_running=True,
                tunnel_type=tunnel_type,
                public_url=public_url,
                pid=tunnel_pid,
                log_file_path=log_files.get("tunnel"),
            )
        else:
            self.tunnel_info = TunnelInfo(is_running=False, tunnel_type="none")

    def cleanup_port(self, port: int) -> None:
        """Kills any process currently occupying the specified port (protecting system services)."""
        if port in [8080, 9000, 6000, 2222]:
            return
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            time.sleep(0.4)
        except Exception:
            pass

    def _infer_backend_for_entry(self, entry) -> str:
        """Selects an appropriate backend for a ModelEntry when no explicit override was provided."""
        if getattr(entry, "backend_hint", None):
            return normalize_engine_name(entry.backend_hint)
        etype = (entry.type or "").lower()
        q_upper = (entry.quantization or "").upper()
        if etype == "diffusion":
            return "diffusers"
        if etype == "audio":
            return "faster-whisper"
        if "EXL2" in q_upper:
            return "exllamav2"
        if any(tag in q_upper for tag in ("IQ4_K", "IQ4_KS", "IQ3_K", "IQ3_KS", "IQ2_K", "IQ2_KS", "IQ5_K", "IQ6_K")):
            return "ik_llama"
        if etype in ("llm-gguf", "vlm-gguf", "embedding-gguf"):
            return "llama.cpp"
        if etype == "llm-safetensors":
            hw = get_hardware_profile()
            if hw.primary_gpu and hw.primary_gpu.supports_vllm and hw.primary_gpu.compute_capability >= (8, 9):
                return "vllm"
            return "transformers"
        return "llama.cpp"

    def start_server(
        self,
        model_id: Optional[str] = None,
        preset_name: Optional[str] = None,
        port: int = 8000,
        enable_auth: bool = True,
        recommendation: Optional[SelectionRecommendation] = None,
        api_key_override: Optional[str] = None,
        **preset_overrides: Any,
    ) -> bool:
        """Starts the appropriate inference engine for the selected or active model with dynamic overrides."""
        self.stop_server()
        self.cleanup_port(port)

        self.registry.load()
        if model_id:
            entry = self.registry.get_model(model_id)
            if not entry:
                raise ValueError(f"Model ID '{model_id}' not found in registry.")
            self.registry.set_active_model(model_id)
        else:
            entry = self.registry.get_active_model()
            if not entry:
                raise ValueError("No model specified and no active model found in registry.")

        preset = self.config.presets.get(preset_name) if preset_name else None

        overrides: Dict[str, Any] = {}
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
            if recommendation.parallel:
                overrides["parallel"] = recommendation.parallel
            if recommendation.quantization_mode and recommendation.quantization_mode != "auto":
                overrides["quantization_mode"] = recommendation.quantization_mode
            if recommendation.suggested_extra_args:
                for sk, sv in recommendation.suggested_extra_args.items():
                    overrides.setdefault(sk, sv)

        for k, v in preset_overrides.items():
            if v is not None:
                overrides[k] = v

        raw_backend = overrides.pop("backend", None)
        if raw_backend and str(raw_backend).strip().lower() == "auto":
            raw_backend = None

        backend = raw_backend
        if not backend and recommendation and getattr(recommendation, "recommended_engine", None):
            backend = recommendation.recommended_engine
        if not backend and preset:
            backend = preset.backend
        if not backend:
            backend = self._infer_backend_for_entry(entry)

        backend_clean = normalize_engine_name(backend)
        self.current_engine = get_engine(backend_clean, self.config)

        if enable_auth:
            if api_key_override and api_key_override.strip():
                SecurityManager.set_cached_secret("WORKSPACE_API_KEY", api_key_override.strip())
                self.api_key = api_key_override.strip()
            else:
                self.api_key = SecurityManager.get_or_create_workspace_api_key()
        else:
            self.api_key = None

        print(f"[ProcessManager] Preparing {backend_clean} engine for model '{entry.name}'...")
        self.current_engine.prepare(entry, preset, **overrides)
        success = self.current_engine.start(
            host=self.config.settings.host or "0.0.0.0",
            port=int(port),
            api_key=self.api_key,
        )

        self._save_state()
        return success

    def stop_server(self) -> bool:
        """Stops the current inference server and releases port/VRAM."""
        if self.current_engine:
            self.current_engine.stop()
            self.current_engine = None

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

    def start_tunnel(
        self,
        port: int = 8000,
        tunnel_type: str = "cloudflared",
        tunnel_token: Optional[str] = None,
        tunnel_hostname: Optional[str] = None,
    ) -> Optional[str]:
        """
        Starts public tunnel without subprocess pipe deadlocks.
        Supports Cloudflare Quick Tunnels, Cloudflare Named Tunnels (instant connection detection),
        and Ngrok fallback.
        """
        self.stop_tunnel()
        self.config.runtime_logs_dir.mkdir(parents=True, exist_ok=True)
        t_type = (tunnel_type or "none").lower().strip()

        if t_type == "cloudflared":
            cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
            if not Path(cf_bin).exists():
                from .env_manager import EnvironmentManager
                cf_bin = EnvironmentManager(self.config).fetch_cloudflared()

            named_token = (tunnel_token.strip() if tunnel_token and tunnel_token.strip() else None) or SecurityManager.get_cf_tunnel_token()
            custom_host = (tunnel_hostname.strip() if tunnel_hostname and tunnel_hostname.strip() else None) or (self.config.settings.tunnel_hostname or None)
            if custom_host and not custom_host.startswith("http"):
                custom_host = f"https://{custom_host}"

            if named_token:
                cmd = [cf_bin, "tunnel", "run", "--token", named_token]
            else:
                cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]

            log_f = open(self.cloudflared_log, "w", encoding="utf-8")
            print(f"[ProcessManager] Launching cloudflared {'named ' if named_token else 'quick '}tunnel on port {port}...")
            try:
                self.tunnel_process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
            finally:
                log_f.close()

            public_url: Optional[str] = None
            connected = False
            quick_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            named_pattern = re.compile(r"Registered tunnel connection|Connection .* registered", re.IGNORECASE)

            for _ in range(35):
                time.sleep(1)
                if self.tunnel_process.poll() is not None:
                    print(f"[ProcessManager] cloudflared exited unexpectedly with code {self.tunnel_process.returncode}.")
                    break

                if self.cloudflared_log.exists():
                    try:
                        content = self.cloudflared_log.read_text(encoding="utf-8", errors="ignore")
                        if named_token:
                            if named_pattern.search(content):
                                connected = True
                                public_url = custom_host
                                break
                        else:
                            match = quick_pattern.search(content)
                            if match:
                                public_url = match.group(0)
                                connected = True
                                break
                    except Exception:
                        pass

            if connected:
                print(f"[ProcessManager] Cloudflare Tunnel established: {public_url or 'Named Tunnel Active'}")
                self.tunnel_info = TunnelInfo(
                    is_running=True,
                    tunnel_type="cloudflared",
                    public_url=public_url,
                    pid=self.tunnel_process.pid,
                    log_file_path=str(self.cloudflared_log),
                )
                self._save_state()
                return public_url
            else:
                print("[ProcessManager] Warning: Could not confirm Cloudflare tunnel URL/connection in logs.")
                self.tunnel_info = TunnelInfo(
                    is_running=self.tunnel_process.poll() is None,
                    tunnel_type="cloudflared",
                    public_url=custom_host if named_token else None,
                    pid=self.tunnel_process.pid if self.tunnel_process else None,
                    log_file_path=str(self.cloudflared_log),
                )
                self._save_state()
                return self.tunnel_info.public_url

        if t_type == "ngrok":
            ngrok_token = (tunnel_token.strip() if tunnel_token and tunnel_token.strip() else None) or SecurityManager.get_ngrok_token()
            if not ngrok_token:
                print("[ProcessManager] Ngrok token not found in secrets/env.")
                return None
            try:
                try:
                    from pyngrok import ngrok  # type: ignore
                except ImportError:
                    if self.config.settings.auto_install_dependencies:
                        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)
                        from pyngrok import ngrok  # type: ignore
                    else:
                        raise
                ngrok.set_auth_token(ngrok_token)
                tunnel = ngrok.connect(port)
                self.tunnel_info = TunnelInfo(
                    is_running=True,
                    tunnel_type="ngrok",
                    public_url=tunnel.public_url,
                )
                self._save_state()
                print(f"[ProcessManager] Ngrok Tunnel established: {tunnel.public_url}")
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
        public_endpoints: Dict[str, str] = {}

        if self.tunnel_info.is_running and self.tunnel_info.public_url:
            base_pub = self.tunnel_info.public_url.rstrip("/")
            public_endpoints = {
                key: (base_pub + val.split(f":{port}", 1)[-1] if f":{port}" in val else base_pub)
                for key, val in local_endpoints.items()
            } or {"base_url": base_pub, "models": f"{base_pub}/v1/models"}

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

    def get_log_tail(self, log_type: str = "llama_server", lines: int = 100) -> str:
        """Reads the last N lines of any engine or tunnel log file."""
        log_map = {
            "llama_server": self.config.runtime_logs_dir / "llama_server.log",
            "ik_llama_server": self.config.runtime_logs_dir / "ik_llama_server.log",
            "cloudflared": self.config.runtime_logs_dir / "cloudflared.log",
            "transformers": self.config.runtime_logs_dir / "transformers_server.log",
            "unsloth": self.config.runtime_logs_dir / "unsloth_server.log",
            "vllm": self.config.runtime_logs_dir / "vllm_server.log",
            "exllamav2": self.config.runtime_logs_dir / "exllamav2_server.log",
            "diffusers": self.config.runtime_logs_dir / "diffusers_server.log",
            "audio": self.config.runtime_logs_dir / "audio_server.log",
        }
        target = log_map.get(log_type, self.config.runtime_logs_dir / f"{log_type}.log")
        if not target.exists():
            return f"Log file '{target.name}' does not exist yet."
        try:
            content_lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
            return "\n".join(content_lines[-lines:])
        except Exception as e:
            return f"Error reading logs: {e}"

    @staticmethod
    def generate_code_snippets(state: SystemState) -> tuple[str, str]:
        """Generates ready-to-run Python and cURL client snippets matching the active engine."""
        base_url = (state.public_url or f"http://127.0.0.1:{state.port}").rstrip("/")
        model_name = state.active_model or "active-model"
        api_key = state.api_key or "your-api-key"
        engine = (state.engine_name or "").lower()

        if engine == "diffusers":
            py_snippet = f"""import base64
import requests

resp = requests.post(
    "{base_url}/v1/images/generations",
    headers={{"Authorization": "Bearer {api_key}"}},
    json={{
        "prompt": "A serene alpine lake at sunrise, detailed photography",
        "size": "512x512",
        "num_inference_steps": 25
    }},
)
img_b64 = resp.json()["data"][0]["b64_json"]
with open("output.png", "wb") as f:
    f.write(base64.b64decode(img_b64))
print("Saved output.png")
"""
            curl_snippet = f"""curl -X POST "{base_url}/v1/images/generations" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {api_key}" \\
  -d '{{"prompt": "A serene alpine lake at sunrise", "size": "512x512", "num_inference_steps": 25}}'
"""
            return py_snippet, curl_snippet

        if engine in ("faster-whisper", "audio"):
            py_snippet = f"""import requests

with open("audio.mp3", "rb") as f:
    resp = requests.post(
        "{base_url}/v1/audio/transcriptions",
        headers={{"Authorization": "Bearer {api_key}"}},
        files={{"file": f}},
    )
print(resp.json())
"""
            curl_snippet = f"""curl -X POST "{base_url}/v1/audio/transcriptions" \\
  -H "Authorization: Bearer {api_key}" \\
  -F "file=@audio.mp3"
"""
            return py_snippet, curl_snippet

        py_snippet = f"""from openai import OpenAI

client = OpenAI(
    base_url="{base_url}/v1",
    api_key="{api_key}"
)

response = client.chat.completions.create(
    model="{model_name}",
    messages=[
        {{"role": "system", "content": "You are a helpful assistant."}},
        {{"role": "user", "content": "Explain quantum computing in two sentences."}}
    ],
    temperature=0.7,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
"""
        curl_snippet = f"""curl -X POST "{base_url}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {api_key}" \\
  -d '{{
    "model": "{model_name}",
    "messages": [
      {{"role": "system", "content": "You are a helpful assistant."}},
      {{"role": "user", "content": "Explain quantum computing in two sentences."}}
    ],
    "temperature": 0.7,
    "stream": false
  }}'
"""
        return py_snippet, curl_snippet

    def shutdown_all(self) -> None:
        """Full system shutdown: stops server, stops tunnel, and updates persisted state."""
        self.stop_server()
        self.stop_tunnel()
        self._save_state()
