"""
Unsloth Engine
FastLanguageModel inference server providing OpenAI-compatible endpoints
for 4-bit Unsloth weights and fine-tuned LoRA adapters.
Executes inside isolated unsloth virtualenv when available.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseEngine
from ..config import ServerPreset, WorkspaceConfig
from ..env_manager import EnvironmentManager
from ..registry import ModelEntry


UNSLOTH_SERVER_SCRIPT = """
import argparse
import json
import os
import sys
import time
from threading import Thread
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--ctx-size", type=int, default=4096)
parser.add_argument("--load-in-4bit", action="store_true")
parser.add_argument("--dtype", type=str, default="auto")
args = parser.parse_args()

print(f"[UnslothServer] Loading model from {args.model_path}...")

dtype_map = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
torch_dtype = dtype_map.get(args.dtype.lower(), None)

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.ctx_size,
        dtype=torch_dtype,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
except Exception as e:
    print(f"[UnslothServer] FastLanguageModel note ({e}), falling back to standard Transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["torch_dtype"] = torch_dtype or torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

from transformers import TextIteratorStreamer

app = FastAPI(title="Unsloth Inference Server")

def verify_token(authorization: Optional[str] = Header(None)):
    if not args.api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing API Key")
    if authorization[7:].strip() != args.api_key:
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
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.95))

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg.get('role', 'user')}\\n{msg.get('content', '')}<|im_end|>\\n"
        prompt += "<|im_start|>assistant\\n"

    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=args.ctx_size).to(model.device)
    prompt_tokens = int(inputs.input_ids.shape[1])
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        top_p=top_p,
        do_sample=temperature > 0.05,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    if stream:
        def stream_generator():
            comp_tokens = 0
            for text_chunk in streamer:
                if not text_chunk:
                    continue
                comp_tokens += 1
                payload = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": os.path.basename(args.model_path),
                    "choices": [{"delta": {"content": text_chunk}, "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(payload)}\\n\\n"
            end_payload = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": os.path.basename(args.model_path),
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": comp_tokens,
                    "total_tokens": prompt_tokens + comp_tokens,
                },
            }
            yield f"data: {json.dumps(end_payload)}\\n\\n"
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

    backend_id = "unsloth"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.log_file = self.config.runtime_logs_dir / "unsloth_server.log"
        self.script_path = self.config.runtime_root / "bin" / "unsloth_server.py"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides: Any,
    ) -> bool:
        self.model_entry = model_entry
        self.preset = self._merge_preset(preset, "unsloth_fast", overrides)

        local_p = Path(model_entry.local_path)
        if not local_p.exists():
            raise FileNotFoundError(f"Model path not found: {local_p}")

        self.script_path.parent.mkdir(parents=True, exist_ok=True)
        self.script_path.write_text(UNSLOTH_SERVER_SCRIPT, encoding="utf-8")
        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        if not self.model_entry:
            raise ValueError("No model prepared for UnslothEngine.")

        p = self.preset or ServerPreset(name="unsloth_default", backend="unsloth")
        py_exec = EnvironmentManager(self.config).get_profile_python("unsloth")
        ctx_size = p.ctx_size if (p.ctx_size and p.ctx_size > 0) else 4096

        cmd = [
            py_exec,
            str(self.script_path),
            "--model-path", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "--ctx-size", str(ctx_size),
            "--dtype", str(p.dtype or "auto"),
        ]
        if p.quantization_mode in ("auto", "bnb_4bit", "4bit"):
            cmd.append("--load-in-4bit")
        if api_key:
            cmd.extend(["--api-key", str(api_key)])
        return self._append_extra_cli_args(cmd, p)

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if not self.model_entry:
            raise ValueError("No model prepared for UnslothEngine.")
        EnvironmentManager(self.config).ensure_ready("unsloth")
        cmd = self.build_command(host, port, api_key)
        return self._spawn_and_wait(cmd, host, port, api_key, self.backend_id)

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "health": f"{base}/health",
            "models": f"{base}/v1/models",
            "chat_completions": f"{base}/v1/chat/completions",
        }
