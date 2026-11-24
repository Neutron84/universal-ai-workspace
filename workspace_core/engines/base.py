"""
Base Inference Engine Abstract Base Class
Defines the unified interface and shared subprocess/healthcheck/preset/benchmark helpers
for all model execution backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import wave
import requests

from ..config import ServerPreset, WorkspaceConfig, get_config, parse_cli_args
from ..registry import ModelEntry


@dataclass
class EngineStatus:
    is_running: bool
    backend_name: str
    pid: Optional[int] = None
    host: str = "0.0.0.0"
    port: int = 8000
    active_model_id: Optional[str] = None
    api_key_configured: bool = False
    endpoints: Dict[str, str] = None
    log_file_path: Optional[str] = None


class BaseEngine(ABC):
    """Abstract Base Class for all inference engines."""

    backend_id: str = "base"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.status = EngineStatus(
            is_running=False,
            backend_name=self.backend_id if self.backend_id != "base" else self.__class__.__name__,
            endpoints={},
        )
        self.model_entry: Optional[ModelEntry] = None
        self.preset: Optional[ServerPreset] = None
        self.api_key: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.log_file: Path = self.config.runtime_logs_dir / f"{self.backend_id}_server.log"

    def _merge_preset(
        self,
        preset: Optional[ServerPreset],
        default_preset_key: str,
        overrides: Dict[str, Any],
    ) -> ServerPreset:
        """Merges base preset with dynamic keyword/CLI overrides without losing unknown custom flags."""
        base = preset or self.config.presets.get(default_preset_key)
        p_dict = base.model_dump() if base else {"name": f"custom_{default_preset_key}", "backend": self.backend_id}

        valid_fields = set(ServerPreset.model_fields.keys())
        custom_dict = dict(p_dict.get("custom_args") or {})
        extra_cli = list(p_dict.get("extra_cli_args") or [])

        for k, v in overrides.items():
            if v is None:
                continue
            if k in ("custom_args", "custom_server_args"):
                d_args, c_tokens = parse_cli_args(v)
                custom_dict.update(d_args)
                extra_cli.extend(c_tokens)
            elif k == "extra_cli_args":
                _, c_tokens = parse_cli_args(v)
                extra_cli.extend(c_tokens)
            elif k in valid_fields:
                p_dict[k] = v
            else:
                custom_dict[k] = v

        p_dict["custom_args"] = custom_dict
        p_dict["extra_cli_args"] = extra_cli
        return ServerPreset(**p_dict)

    def _build_runtime_env(self) -> Dict[str, str]:
        """Constructs process environment with CUDA & local bin shared library directories."""
        env = os.environ.copy()
        lib_dirs = set()
        if self.config.runtime_root.exists():
            for p in self.config.runtime_root.glob("bin/**/*.so*"):
                lib_dirs.add(str(p.parent.resolve()))
            for p in self.config.runtime_root.glob("bin/**/llama_cpp*"):
                if p.is_dir():
                    lib_dirs.add(str(p.resolve()))
            ik_dir = self.config.runtime_root / "bin" / "ik_llama_cpp"
            if ik_dir.is_dir():
                lib_dirs.add(str(ik_dir.resolve()))
        current_ld = env.get("LD_LIBRARY_PATH", "")
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = ":".join(sorted(lib_dirs)) + (":" + current_ld if current_ld else "")
        return env

    @staticmethod
    def _append_extra_cli_args(cmd: List[str], preset: Optional[ServerPreset]) -> List[str]:
        """Appends structured custom_args and raw extra_cli_args tokens to command."""
        if not preset:
            return cmd
        for k, v in (preset.custom_args or {}).items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            elif v is not None and v != "":
                cmd.extend([flag, str(v)])
        for tok in (preset.extra_cli_args or []):
            if str(tok).strip():
                cmd.append(str(tok).strip())
        return cmd

    def _spawn_and_wait(
        self,
        cmd: List[str],
        host: str,
        port: int,
        api_key: Optional[str],
        backend_name: str,
        timeout_sec: Optional[int] = None,
    ) -> bool:
        """Spawns subprocess server cleanly (closing parent log FD) and waits for healthcheck."""
        if self.process and self.process.poll() is None:
            self.stop()

        self.api_key = api_key
        self.config.runtime_logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{self.__class__.__name__}] Launching {backend_name} on {host}:{port}...")
        log_f = open(self.log_file, "w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=self._build_runtime_env(),
                preexec_fn=os.setsid,
            )
        finally:
            log_f.close()

        self.status = EngineStatus(
            is_running=True,
            backend_name=backend_name,
            pid=self.process.pid,
            host=host,
            port=port,
            active_model_id=self.model_entry.model_id if self.model_entry else None,
            api_key_configured=bool(api_key),
            endpoints=self._generate_endpoints(host, port),
            log_file_path=str(self.log_file),
        )

        wait_limit = int(timeout_sec or self.config.settings.healthcheck_timeout_sec or 90)
        for _ in range(wait_limit):
            time.sleep(1)
            if self.process.poll() is not None:
                print(f"[{self.__class__.__name__}] Server exited prematurely with code {self.process.returncode}.")
                self.status.is_running = False
                return False
            if self.healthcheck():
                print(f"[{self.__class__.__name__}] {backend_name} is healthy and ready on port {port}.")
                self.status.is_running = True
                return True

        print(f"[{self.__class__.__name__}] Warning: Server started but healthcheck timed out after {wait_limit}s.")
        self.status.is_running = False
        return False

    def stop(self) -> bool:
        """Gracefully stops the server process (including reconstructed cross-cell PIDs)."""
        pid_to_kill = self.process.pid if self.process else self.status.pid
        if not pid_to_kill:
            self.status.is_running = False
            return True

        print(f"[{self.__class__.__name__}] Stopping server process PID {pid_to_kill}...")
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
                    try:
                        self.process.wait(timeout=3)
                    except Exception:
                        pass
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
                        try:
                            os.kill(pid_to_kill, signal.SIGKILL)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[{self.__class__.__name__}] Note during process termination: {e}")

        self.process = None
        self.status.pid = None
        self.status.is_running = False
        return True

    def _is_pid_running(self) -> bool:
        if self.process:
            return self.process.poll() is None
        if self.status.pid:
            try:
                os.kill(self.status.pid, 0)
                return True
            except OSError:
                return False
        return False

    def healthcheck(self) -> bool:
        """Verifies if the server process is alive and responding on HTTP endpoints."""
        if not self._is_pid_running():
            return False

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            r = requests.get(f"http://127.0.0.1:{self.status.port}/health", headers=headers, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass

        try:
            r = requests.get(f"http://127.0.0.1:{self.status.port}/v1/models", headers=headers, timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "models": f"{base}/v1/models",
            "chat_completions": f"{base}/v1/chat/completions",
            "completions": f"{base}/v1/completions",
            "health": f"{base}/health",
        }

    def get_endpoints(self) -> Dict[str, str]:
        return self._generate_endpoints(self.status.host, self.status.port)

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        return []

    def get_command_preview(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> str:
        cmd = self.build_command(host, port, api_key)
        return " ".join(cmd)

    @abstractmethod
    def prepare(self, model_entry: ModelEntry, preset: Optional[ServerPreset] = None, **overrides: Any) -> bool:
        """Prepares environment, dependencies, and command line arguments for the model."""
        pass

    @abstractmethod
    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        """Starts the backend inference server in the background."""
        pass

    def benchmark(
        self,
        prompt: str = "Explain quantum computing in one short paragraph.",
        max_tokens: int = 128,
        runs: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes a real multi-modal benchmark against the active engine endpoint.
        Supports LLM Chat Completions, Embeddings, Diffusers Images, and Faster-Whisper Audio.
        """
        if not self.healthcheck():
            return {"error": "Server is not healthy or running."}

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoints = self.get_endpoints()
        model_name = self.model_entry.name if self.model_entry else "default"
        model_type = self.model_entry.type if self.model_entry else "llm-gguf"

        # 1. Diffusers image generation benchmark
        if self.status.backend_name == "diffusers" or model_type == "diffusion":
            img_url = endpoints.get("images_generations", f"http://127.0.0.1:{self.status.port}/v1/images/generations")
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    img_url,
                    json={"prompt": prompt, "n": 1, "size": "512x512", "num_inference_steps": 15},
                    headers={"Content-Type": "application/json", **headers},
                    timeout=120,
                )
                elapsed = time.perf_counter() - t0
                if resp.status_code != 200:
                    return {"error": f"Image benchmark failed ({resp.status_code}): {resp.text}"}
                return {
                    "mode": "diffusion",
                    "ttft_ms": round(elapsed * 1000.0, 2),
                    "tokens_per_sec": round(15.0 / max(0.01, elapsed), 2),
                    "total_tokens": 15,
                    "total_time_sec": round(elapsed, 2),
                    "response_preview": "Generated 512x512 PNG (15 steps)",
                }
            except Exception as e:
                return {"error": f"Diffusion benchmark error: {e}"}

        # 2. Audio transcription benchmark (1-second silent WAV)
        if self.status.backend_name in ("audio", "faster-whisper") or model_type == "audio":
            audio_url = endpoints.get("audio_transcriptions", f"http://127.0.0.1:{self.status.port}/v1/audio/transcriptions")
            try:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"\x00\x00" * 16000)
                buf.seek(0)
                t0 = time.perf_counter()
                resp = requests.post(
                    audio_url,
                    files={"file": ("bench.wav", buf, "audio/wav")},
                    headers=headers,
                    timeout=60,
                )
                elapsed = time.perf_counter() - t0
                if resp.status_code != 200:
                    return {"error": f"Audio benchmark failed ({resp.status_code}): {resp.text}"}
                return {
                    "mode": "audio",
                    "ttft_ms": round(elapsed * 1000.0, 2),
                    "tokens_per_sec": round(1.0 / max(0.01, elapsed), 2),
                    "total_tokens": 1,
                    "total_time_sec": round(elapsed, 2),
                    "response_preview": f"Transcribed 1.0s WAV in {elapsed:.2f}s",
                }
            except Exception as e:
                return {"error": f"Audio benchmark error: {e}"}

        # 3. Embeddings benchmark
        if model_type == "embedding-gguf":
            emb_url = endpoints.get("embeddings", f"http://127.0.0.1:{self.status.port}/v1/embeddings")
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    emb_url,
                    json={"model": model_name, "input": prompt},
                    headers={"Content-Type": "application/json", **headers},
                    timeout=30,
                )
                elapsed = time.perf_counter() - t0
                if resp.status_code != 200:
                    return {"error": f"Embeddings benchmark failed ({resp.status_code}): {resp.text}"}
                dim = len(resp.json().get("data", [{}])[0].get("embedding", []))
                return {
                    "mode": "embeddings",
                    "ttft_ms": round(elapsed * 1000.0, 2),
                    "tokens_per_sec": round(1.0 / max(0.001, elapsed), 2),
                    "total_tokens": dim,
                    "total_time_sec": round(elapsed, 3),
                    "response_preview": f"Embedding vector dimension: {dim}",
                }
            except Exception as e:
                return {"error": f"Embeddings benchmark error: {e}"}

        # 4. LLM / VLM Chat Completions streaming benchmark
        chat_url = endpoints.get("chat_completions", f"http://127.0.0.1:{self.status.port}/v1/chat/completions")
        headers["Content-Type"] = "application/json"

        ttfts: List[float] = []
        tps_list: List[float] = []
        total_times: List[float] = []
        last_tokens = 0
        last_preview = ""

        num_runs = max(1, int(runs))
        for _ in range(num_runs):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0.2,
            }

            try:
                start_time = time.perf_counter()
                first_token_time = None
                chunk_tokens = 0
                exact_completion_tokens = None
                full_response: List[str] = []

                resp = requests.post(chat_url, json=payload, headers=headers, stream=True, timeout=90)
                if resp.status_code != 200:
                    return {"error": f"Benchmark request failed with status {resp.status_code}: {resp.text}"}

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8", errors="ignore")
                    if decoded.startswith("data: "):
                        data_str = decoded[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            usage = chunk.get("usage")
                            if isinstance(usage, dict) and usage.get("completion_tokens"):
                                exact_completion_tokens = int(usage["completion_tokens"])
                            choices = chunk.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta", {}).get("content") or ""
                                if delta:
                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                    full_response.append(delta)
                                    chunk_tokens += 1
                        except Exception:
                            pass

                end_time = time.perf_counter()
                if first_token_time is None:
                    first_token_time = end_time

                total_tokens = exact_completion_tokens or chunk_tokens
                ttft_ms = (first_token_time - start_time) * 1000.0
                gen_time = max(0.001, end_time - first_token_time)
                tokens_per_sec = total_tokens / gen_time if total_tokens > 0 else 0.0

                ttfts.append(ttft_ms)
                tps_list.append(tokens_per_sec)
                total_times.append(end_time - start_time)
                last_tokens = total_tokens
                last_preview = "".join(full_response)[:150] + ("..." if len("".join(full_response)) > 150 else "")
            except Exception as e:
                return {"error": f"Benchmark execution error: {str(e)}"}

        return {
            "mode": "chat",
            "runs": num_runs,
            "ttft_ms": round(sum(ttfts) / len(ttfts), 2),
            "tokens_per_sec": round(sum(tps_list) / len(tps_list), 2),
            "total_tokens": last_tokens,
            "total_time_sec": round(sum(total_times) / len(total_times), 2),
            "response_preview": last_preview,
        }
