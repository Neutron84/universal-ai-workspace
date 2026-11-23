"""
Unsloth Engine
FastLanguageModel inference server providing OpenAI-compatible endpoints
for 4-bit Unsloth weights and fine-tuned LoRA adapters.
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


UNSLOTH_SERVER_SCRIPT = """
import argparse
import json
import os
import sys
import time
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
import uvicorn
from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--ctx-size", type=int, default=4096)
args = parser.parse_args()

print(f"[UnslothServer] Loading model from {args.model_path}...")

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.ctx_size,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
except Exception as e:
    print(f"[UnslothServer] FastLanguageModel note ({e}), falling back to standard 4-bit Transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

from transformers import TextIteratorStreamer

app = FastAPI(title="Unsloth Inference Server")

def verify_token(authorization: Optional[str] = Header(None)):
    if not args.api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing API Key")
    token = authorization[7:].strip()
    if token != args.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")

@app.get("/health")
def health():
    return {"status": "ok", "backend": "unsloth"}

@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    verify_token(authorization)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "unsloth"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    verify_token(authorization)
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)

    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<|im_start|>{role}\\n{content}<|im_end|>\\n"
    prompt += "<|im_start|>assistant\\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        do_sample=temperature > 0.05,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    if stream:
        def stream_generator():
            for text_chunk in streamer:
                payload = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": os.path.basename(args.model_path),
                    "choices": [{"delta": {"content": text_chunk}, "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(payload)}\\n\\n"
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        thread.join()
        generated_text = "".join(list(streamer))
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(args.model_path),
            "choices": [{"message": {"role": "assistant", "content": generated_text}, "index": 0, "finish_reason": "stop"}]
        }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
"""


class UnslothEngine(BaseEngine):
    """Controls the FastLanguageModel Unsloth server process."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self.log_file = self.config.runtime_logs_dir / "unsloth_server.log"
        self.script_path = self.config.runtime_root / "bin" / "unsloth_server.py"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides
    ) -> bool:
        self.model_entry = model_entry
        self.preset = preset or self.config.presets.get("unsloth_fast") or ServerPreset(name="unsloth_default", backend="unsloth")

        local_p = Path(model_entry.local_path)
        if not local_p.exists():
            raise FileNotFoundError(f"Model path not found: {local_p}")

        self.script_path.parent.mkdir(parents=True, exist_ok=True)
        self.script_path.write_text(UNSLOTH_SERVER_SCRIPT, encoding="utf-8")
        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        if not self.model_entry:
            raise ValueError("No model prepared for UnslothEngine.")

        ctx_size = getattr(self.preset, "ctx_size", 4096)
        cmd = [
            sys.executable,
            str(self.script_path),
            "--model-path", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "--ctx-size", str(ctx_size),
        ]
        if api_key:
            cmd.extend(["--api-key", str(api_key)])
        return cmd

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if self.process and self.process.poll() is None:
            self.stop()

        self.api_key = api_key
        cmd = self.build_command(host, port, api_key)
        print(f"[UnslothEngine] Starting server: {' '.join(cmd[:6])}...")

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
            backend_name="unsloth",
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
        while time.time() - start_wait < 90:
            if self.process.poll() is not None:
                print(f"[UnslothEngine] Server exited prematurely with code {self.process.returncode}.")
                return False
            if self.healthcheck():
                healthy = True
                break
            time.sleep(1.0)

        self.status.is_running = healthy
        return healthy

    def stop(self) -> bool:
        if self.process and self.process.poll() is None:
            print(f"[UnslothEngine] Stopping server process (PID {self.process.pid})...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.process = None

        self.status.is_running = False
        self.status.pid = None
        return True

    def healthcheck(self) -> bool:
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
        return False

    def get_endpoints(self) -> Dict[str, str]:
        port = self.status.port
        return {
            "health": f"http://127.0.0.1:{port}/health",
            "models": f"http://127.0.0.1:{port}/v1/models",
            "chat_completions": f"http://127.0.0.1:{port}/v1/chat/completions",
        }
