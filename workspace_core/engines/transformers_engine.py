"""
Transformers & BitsAndBytes Engine
Lightweight, OpenAI-compatible server utilizing Hugging Face Transformers
with optional 4-bit / 8-bit quantization for Safetensors models.
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


SERVER_SCRIPT = """
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--load-in-4bit", action="store_true")
args = parser.parse_args()

print(f"[TransformersServer] Loading model from {args.model_path}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

kwargs = {
    "device_map": "auto",
    "trust_remote_code": True,
}
if args.load_in_4bit:
    try:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    except Exception as e:
        print(f"[TransformersServer] BitsAndBytes error, falling back to fp16: {e}")
        kwargs["torch_dtype"] = torch.float16
else:
    kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
model_name = os.path.basename(args.model_path)
print("[TransformersServer] Model loaded successfully.")

app = FastAPI(title="Transformers Inference API")

def check_auth(authorization: Optional[str] = Header(None)):
    if args.api_key:
        expected = f"Bearer {args.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}

@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "workspace"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: Request, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)

    prompt = ""
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        for m in messages:
            prompt += f"{m.get('role', 'user')}: {m.get('content', '')}\\n"
        prompt += "assistant:\\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(0.01, temperature),
            do_sample=temperature > 0.0,
            streamer=streamer,
        )
        t = Thread(target=model.generate, kwargs=gen_kwargs)
        t.start()

        def stream_generator():
            for text_chunk in streamer:
                chunk_obj = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"delta": {"content": text_chunk}, "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_obj)}\\n\\n"
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(0.01, temperature),
            do_sample=temperature > 0.0,
        )
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"message": {"role": "assistant", "content": output_text}, "index": 0, "finish_reason": "stop"}]
    }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
"""


class TransformersEngine(BaseEngine):
    """Executes Hugging Face Transformers models with OpenAI-compatible FastAPI endpoints."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self.log_file = self.config.runtime_logs_dir / "transformers_server.log"
        self.server_script_path = self.config.runtime_root / "transformers_server.py"

    def prepare(self, model_entry: ModelEntry, preset: Optional[ServerPreset] = None) -> bool:
        self.model_entry = model_entry
        self.preset = preset or self.config.presets.get("transformers_4bit")
        self.server_script_path.write_text(SERVER_SCRIPT)
        return True

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if self.process and self.process.poll() is None:
            return True

        if not self.model_entry:
            raise ValueError("No model prepared for TransformersEngine.")

        self.api_key = api_key

        # Prefer isolated venv Python if available
        venv_python = self.config.runtime_envs_dir / "transformers" / "bin" / "python"
        py_exec = str(venv_python) if venv_python.exists() else sys.executable

        cmd = [
            py_exec,
            str(self.server_script_path),
            "--model-path", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
        ]
        if api_key:
            cmd.extend(["--api-key", str(api_key)])

        custom = (self.preset.custom_args if self.preset else {}) or {}
        if custom.get("load_in_4bit", True):
            cmd.append("--load-in-4bit")

        log_f = open(self.log_file, "w", encoding="utf-8")
        self.process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        log_f.close()

        self.status = EngineStatus(
            is_running=True,
            backend_name="transformers",
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
            "health": f"{base}/health",
        }
