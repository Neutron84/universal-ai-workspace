"""
Base Inference Engine Abstract Base Class
Defines the standard interface for all model execution backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from ..config import ServerPreset, WorkspaceConfig, get_config
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

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.status = EngineStatus(
            is_running=False,
            backend_name=self.__class__.__name__,
            endpoints={},
        )
        self.model_entry: Optional[ModelEntry] = None
        self.preset: Optional[ServerPreset] = None
        self.api_key: Optional[str] = None

    @abstractmethod
    def prepare(self, model_entry: ModelEntry, preset: Optional[ServerPreset] = None) -> bool:
        """Prepares environment, dependencies, and command line arguments for the model."""
        pass

    @abstractmethod
    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        """Starts the backend inference server in the background."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Gracefully stops the server process."""
        pass

    @abstractmethod
    def healthcheck(self) -> bool:
        """Verifies if the server is healthy and responding to requests."""
        pass

    @abstractmethod
    def get_endpoints(self) -> Dict[str, str]:
        """Returns map of API endpoints exposed by the engine."""
        pass

    def benchmark(
        self,
        prompt: str = "Explain quantum computing in one short paragraph.",
        max_tokens: int = 128
    ) -> Dict[str, Any]:
        """
        Executes a real benchmark using the OpenAI chat completions API.
        Measures Time-To-First-Token (TTFT) and generation tokens/sec.
        """
        import time

        if not self.healthcheck():
            return {"error": "Server is not healthy or running."}

        endpoints = self.get_endpoints()
        chat_url = endpoints.get("chat_completions", f"http://127.0.0.1:{self.status.port}/v1/chat/completions")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        model_name = self.model_entry.name if self.model_entry else "default"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.2,
        }

        try:
            start_time = time.perf_counter()
            first_token_time = None
            total_tokens = 0
            full_response = []

            resp = requests.post(chat_url, json=payload, headers=headers, stream=True, timeout=60)
            if resp.status_code != 200:
                return {
                    "error": f"Benchmark request failed with status {resp.status_code}: {resp.text}"
                }

            import json
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data_str = decoded[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            full_response.append(delta)
                            total_tokens += 1
                    except Exception:
                        pass

            end_time = time.perf_counter()

            if first_token_time is None:
                first_token_time = end_time

            ttft_ms = (first_token_time - start_time) * 1000.0
            gen_time = max(0.001, end_time - first_token_time)
            tokens_per_sec = total_tokens / gen_time if total_tokens > 0 else 0.0
            total_latency_sec = end_time - start_time

            metrics = {
                "ttft_ms": round(ttft_ms, 2),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "total_tokens": total_tokens,
                "total_time_sec": round(total_latency_sec, 2),
                "response_preview": "".join(full_response)[:150] + "...",
            }
            return metrics
        except Exception as e:
            return {"error": f"Benchmark execution error: {str(e)}"}
