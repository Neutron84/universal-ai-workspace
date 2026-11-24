"""
Transformers & BitsAndBytes Engine
OpenAI-compatible FastAPI server utilizing Hugging Face Transformers
with configurable 4-bit / 8-bit / FP16 / BF16 precision, Flash Attention 2 / SDPA,
and automatic PEFT LoRA adapter loading.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseEngine
from ..config import ServerPreset, WorkspaceConfig
from ..env_manager import EnvironmentManager
from ..registry import ModelEntry


SERVER_SCRIPT = """
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--quant-mode", type=str, default="bnb_4bit")
parser.add_argument("--dtype", type=str, default="auto")
parser.add_argument("--ctx-size", type=int, default=4096)
parser.add_argument("--flash-attn", action="store_true")
args = parser.parse_args()

print(f"[TransformersServer] Loading model from {args.model_path} (quant={args.quant_mode}, dtype={args.dtype})...")

is_lora = os.path.isfile(os.path.join(args.model_path, "adapter_config.json"))
base_model_path = args.model_path
if is_lora:
    try:
        with open(os.path.join(args.model_path, "adapter_config.json"), "r", encoding="utf-8") as af:
            base_model_path = json.load(af).get("base_model_name_or_path") or args.model_path
    except Exception:
        pass

tokenizer = AutoTokenizer.from_pretrained(args.model_path if not is_lora else base_model_path, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

dtype_map = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
torch_dtype = dtype_map.get(args.dtype.lower(), torch.float16 if torch.cuda.is_available() else torch.float32)

kwargs = {
    "device_map": "auto",
    "trust_remote_code": True,
}
if args.flash_attn and torch.cuda.is_available():
    kwargs["attn_implementation"] = "sdpa"

qmode = args.quant_mode.lower()
if qmode in ("bnb_4bit", "4bit"):
    try:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
    except Exception as e:
        print(f"[TransformersServer] BitsAndBytes 4-bit unavailable ({e}), using {torch_dtype}")
        kwargs["torch_dtype"] = torch_dtype
elif qmode in ("bnb_8bit", "8bit"):
    try:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    except Exception as e:
        print(f"[TransformersServer] BitsAndBytes 8-bit unavailable ({e}), using {torch_dtype}")
        kwargs["torch_dtype"] = torch_dtype
else:
    kwargs["torch_dtype"] = torch_dtype

model = AutoModelForCausalLM.from_pretrained(base_model_path if is_lora else args.model_path, **kwargs)
if is_lora:
    try:
        from peft import PeftModel
        print(f"[TransformersServer] Attaching LoRA adapter from {args.model_path}...")
        model = PeftModel.from_pretrained(model, args.model_path)
    except Exception as e:
        print(f"[TransformersServer] Warning attaching LoRA adapter: {e}")

model_name = os.path.basename(args.model_path.rstrip("/"))
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
    max_tokens = int(body.get("max_tokens", 512))
    stream = bool(body.get("stream", False))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.95))

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ""
        for m in messages:
            prompt += f"{m.get('role', 'user')}: {m.get('content', '')}\\n"
        prompt += "assistant:\\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.ctx_size).to(model.device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    gen_cfg = {
        "max_new_tokens": max_tokens,
        "temperature": max(0.01, temperature),
        "top_p": top_p,
        "do_sample": temperature > 0.05,
    }

    if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        t = Thread(target=model.generate, kwargs={**inputs, **gen_cfg, "streamer": streamer})
        t.start()

        def stream_generator():
            completion_tokens = 0
            for text_chunk in streamer:
                if not text_chunk:
                    continue
                completion_tokens += 1
                chunk_obj = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"delta": {"content": text_chunk}, "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_obj)}\\n\\n"
            final_chunk = {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(final_chunk)}\\n\\n"
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_cfg)
    gen_ids = outputs[0][prompt_tokens:]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"message": {"role": "assistant", "content": output_text}, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(len(gen_ids)),
            "total_tokens": prompt_tokens + int(len(gen_ids)),
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
"""


class TransformersEngine(BaseEngine):
    """Executes Hugging Face Transformers models with OpenAI-compatible FastAPI endpoints."""

    backend_id = "transformers"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.log_file = self.config.runtime_logs_dir / "transformers_server.log"
        self.server_script_path = self.config.runtime_root / "transformers_server.py"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides: Any,
    ) -> bool:
        self.model_entry = model_entry
        self.preset = self._merge_preset(preset, "transformers_4bit", overrides)
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self.server_script_path.write_text(SERVER_SCRIPT, encoding="utf-8")
        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        if not self.model_entry:
            raise ValueError("No model prepared for TransformersEngine.")

        p = self.preset or ServerPreset(name="transformers_default", backend="transformers")
        em = EnvironmentManager(self.config)
        py_exec = em.get_profile_python("transformers")

        quant_mode = p.quantization_mode
        if quant_mode == "auto":
            quant_mode = "bnb_4bit" if p.custom_args.get("load_in_4bit", True) else "fp16"

        cmd = [
            py_exec,
            str(self.server_script_path),
            "--model-path", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "--quant-mode", str(quant_mode),
            "--dtype", str(p.dtype or "auto"),
            "--ctx-size", str(p.ctx_size if (p.ctx_size and p.ctx_size > 0) else 4096),
        ]
        if p.flash_attn:
            cmd.append("--flash-attn")
        if api_key:
            cmd.extend(["--api-key", str(api_key)])
        return self._append_extra_cli_args(cmd, p)

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if not self.model_entry:
            raise ValueError("No model prepared for TransformersEngine.")
        EnvironmentManager(self.config).ensure_ready("transformers")
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
