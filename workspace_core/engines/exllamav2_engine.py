"""
ExLlamaV2 Engine
Executes EXL2 / GPTQ models using ExLlamaV2 in a dedicated OpenAI-compatible
FastAPI streaming subprocess server with configurable context length and KV cache dtype.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseEngine
from ..config import ServerPreset, WorkspaceConfig
from ..env_manager import EnvironmentManager
from ..registry import ModelEntry


EXLLAMA_SERVER_SCRIPT = """
import argparse
import json
import os
import sys
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--ctx-size", type=int, default=4096)
parser.add_argument("--cache-mode", type=str, default="q8")
args = parser.parse_args()

print(f"[ExLlamaV2Server] Loading EXL2 model from {args.model_path} (ctx={args.ctx_size}, cache={args.cache_mode})...")
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

config = ExLlamaV2Config(args.model_path)
config.max_seq_len = args.ctx_size
model = ExLlamaV2(config)

cache_mode = args.cache_mode.lower()
if "q4" in cache_mode:
    cache = ExLlamaV2Cache_Q4(model, lazy=True)
elif "q6" in cache_mode:
    cache = ExLlamaV2Cache_Q6(model, lazy=True)
elif "q8" in cache_mode or "8bit" in cache_mode:
    cache = ExLlamaV2Cache_8bit(model, lazy=True)
else:
    cache = ExLlamaV2Cache(model, lazy=True)

model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.warmup()

model_name = os.path.basename(args.model_path.rstrip("/"))
print("[ExLlamaV2Server] Model loaded successfully.")

app = FastAPI(title="ExLlamaV2 Inference API")

def check_auth(authorization: Optional[str] = Header(None)):
    if args.api_key and authorization != f"Bearer {args.api_key}":
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name, "backend": "exllamav2"}

@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "exllamav2"}],
    }

def format_messages(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\\n")
    return "\\n".join(parts)

@app.post("/v1/chat/completions")
async def chat_completions(req: Request, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 512))
    stream = bool(body.get("stream", False))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.95))

    prompt = format_messages(messages)
    input_ids = tokenizer.encode(prompt)
    prompt_tokens = int(input_ids.shape[-1])

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = max(0.01, temperature)
    settings.top_p = top_p
    generator.set_stop_conditions([tokenizer.eos_token_id])

    if stream:
        generator.begin_stream_ex(input_ids, settings)

        def stream_generator():
            generated = 0
            while generated < max_tokens:
                res = generator.stream_ex()
                chunk = res.get("chunk", "")
                eos = res.get("eos", False)
                if chunk:
                    generated += 1
                    payload = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"delta": {"content": chunk}, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\\n\\n"
                if eos:
                    break
            final_chunk = {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": generated,
                    "total_tokens": prompt_tokens + generated,
                },
            }
            yield f"data: {json.dumps(final_chunk)}\\n\\n"
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    output = generator.generate_simple(prompt, settings, max_tokens, stop_token=tokenizer.eos_token_id)
    text = output[len(prompt):] if output.startswith(prompt) else output
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"message": {"role": "assistant", "content": text}, "index": 0, "finish_reason": "stop"}],
    }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
"""


class ExLlamaV2Engine(BaseEngine):
    """Controls the ExLlamaV2 EXL2 inference server process."""

    backend_id = "exllamav2"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.log_file = self.config.runtime_logs_dir / "exllamav2_server.log"
        self.server_script_path = self.config.runtime_root / "exllamav2_server.py"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides: Any,
    ) -> bool:
        self.model_entry = model_entry
        self.preset = self._merge_preset(preset, "exllamav2_fast", overrides)
        local_p = Path(model_entry.local_path)
        if not local_p.exists():
            raise FileNotFoundError(f"Model path not found: {local_p}")
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self.server_script_path.write_text(EXLLAMA_SERVER_SCRIPT, encoding="utf-8")
        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        if not self.model_entry:
            raise ValueError("No model prepared for ExLlamaV2Engine.")

        p = self.preset or ServerPreset(name="exllamav2_default", backend="exllamav2")
        py_exec = EnvironmentManager(self.config).get_profile_python("exllamav2")
        ctx_size = p.ctx_size if (p.ctx_size and p.ctx_size > 0) else 4096
        cache_mode = (p.cache_type_k or "q8_0").lower()
        if cache_mode == "auto":
            cache_mode = "q8_0"

        cmd = [
            py_exec,
            str(self.server_script_path),
            "--model-path", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "--ctx-size", str(ctx_size),
            "--cache-mode", str(cache_mode),
        ]
        if api_key:
            cmd.extend(["--api-key", str(api_key)])
        return self._append_extra_cli_args(cmd, p)

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if not self.model_entry:
            raise ValueError("No model prepared for ExLlamaV2Engine.")
        EnvironmentManager(self.config).ensure_ready("exllamav2")
        cmd = self.build_command(host, port, api_key)
        return self._spawn_and_wait(cmd, host, port, api_key, self.backend_id)

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "models": f"{base}/v1/models",
            "chat_completions": f"{base}/v1/chat/completions",
            "health": f"{base}/health",
        }
