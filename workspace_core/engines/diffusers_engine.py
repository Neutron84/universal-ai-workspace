"""
Diffusers Engine (Image Generation)
Executes Stable Diffusion / SDXL / Flux models (both HF directories and single-file checkpoints)
in a dedicated subprocess server exposing OpenAI-compatible /v1/images/generations returning base64 PNG.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseEngine
from ..config import ServerPreset, WorkspaceConfig
from ..env_manager import EnvironmentManager
from ..registry import ModelEntry


DIFFUSERS_SERVER_SCRIPT = """
import argparse
import base64
import io
import json
import os
import time
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException, Request, Header
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--dtype", type=str, default="auto")
parser.add_argument("--default-steps", type=int, default=25)
parser.add_argument("--default-guidance", type=float, default=7.5)
parser.add_argument("--attention-slicing", action="store_true")
parser.add_argument("--cpu-offload", action="store_true")
parser.add_argument("--lora-paths", type=str, default="[]")
args = parser.parse_args()

print(f"[DiffusersServer] Loading diffusion model from {args.model_path}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype_map = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
dtype = dtype_map.get(args.dtype.lower(), torch.float16 if torch.cuda.is_available() else torch.float32)

pipe = None
is_single_file = os.path.isfile(args.model_path)

if is_single_file:
    for cls_name in ("StableDiffusionXLPipeline", "StableDiffusionPipeline", "FluxPipeline"):
        try:
            import diffusers
            cls = getattr(diffusers, cls_name, None)
            if cls is not None and hasattr(cls, "from_single_file"):
                pipe = cls.from_single_file(args.model_path, torch_dtype=dtype)
                print(f"[DiffusersServer] Loaded single-file checkpoint using {cls_name}.")
                break
        except Exception as e:
            print(f"[DiffusersServer] {cls_name}.from_single_file skipped ({e})")
    if pipe is None:
        raise RuntimeError(f"Could not load single-file checkpoint: {args.model_path}")
else:
    try:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            safety_checker=None,
        )
    except Exception as e:
        print(f"[DiffusersServer] AutoPipeline fallback to StableDiffusionPipeline: {e}")
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            safety_checker=None,
        )

try:
    lora_list = json.loads(args.lora_paths)
    for lp in lora_list:
        if os.path.exists(lp):
            pipe.load_lora_weights(lp)
            print(f"[DiffusersServer] Loaded LoRA weights: {lp}")
except Exception as e:
    print(f"[DiffusersServer] Note loading LoRA weights: {e}")

if args.cpu_offload and device == "cuda":
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pipe = pipe.to(device)
else:
    pipe = pipe.to(device)

if device == "cuda" and args.attention_slicing:
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

model_name = os.path.basename(args.model_path)
print("[DiffusersServer] Model loaded successfully.")

app = FastAPI(title="Diffusers Image Generation API")

def check_auth(authorization: Optional[str] = Header(None)):
    if args.api_key:
        expected = f"Bearer {args.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name, "backend": "diffusers"}

@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "workspace"}]
    }

@app.post("/v1/images/generations")
async def generate_images(req: Request, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    body = await req.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' field.")

    negative_prompt = body.get("negative_prompt") or None
    n = max(1, min(4, int(body.get("n", 1))))
    size_str = body.get("size", "512x512")
    w, h = 512, 512
    if "x" in size_str:
        try:
            parts = size_str.lower().split("x")
            w = int(parts[0])
            h = int(parts[1])
        except Exception:
            w, h = 512, 512

    steps = int(body.get("num_inference_steps", args.default_steps))
    guidance = float(body.get("guidance_scale", args.default_guidance))
    seed = body.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    call_kwargs = {
        "prompt": prompt,
        "num_images_per_prompt": n,
        "width": w,
        "height": h,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    if negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt
    if generator is not None:
        call_kwargs["generator"] = generator

    with torch.inference_mode():
        output = pipe(**call_kwargs)

    b64_data = []
    for img in output.images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        b64_data.append({"b64_json": img_b64})

    return {
        "created": int(time.time()),
        "data": b64_data,
    }

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
"""


class DiffusersEngine(BaseEngine):
    """Executes diffusion image generation models with OpenAI-compatible FastAPI endpoints."""

    backend_id = "diffusers"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.log_file = self.config.runtime_logs_dir / "diffusers_server.log"
        self.server_script_path = self.config.runtime_root / "diffusers_server.py"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides: Any,
    ) -> bool:
        self.model_entry = model_entry
        self.preset = self._merge_preset(preset, "diffusers_sd", overrides)
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self.server_script_path.write_text(DIFFUSERS_SERVER_SCRIPT, encoding="utf-8")
        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        if not self.model_entry:
            raise ValueError("No model prepared for DiffusersEngine.")

        import json
        p = self.preset or ServerPreset(name="diffusers_default", backend="diffusers")
        py_exec = EnvironmentManager(self.config).get_profile_python("diffusion")

        cmd = [
            py_exec,
            str(self.server_script_path),
            "--model-path", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "--dtype", str(p.dtype or "auto"),
            "--default-steps", str(int(p.diffusion_steps or 25)),
            "--default-guidance", str(float(p.diffusion_guidance_scale or 7.5)),
            "--lora-paths", json.dumps(p.lora_paths or []),
        ]
        if p.diffusion_attention_slicing:
            cmd.append("--attention-slicing")
        if p.diffusion_cpu_offload:
            cmd.append("--cpu-offload")
        if api_key:
            cmd.extend(["--api-key", str(api_key)])
        return self._append_extra_cli_args(cmd, p)

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if not self.model_entry:
            raise ValueError("No model prepared for DiffusersEngine.")
        EnvironmentManager(self.config).ensure_ready("diffusers")
        cmd = self.build_command(host, port, api_key)
        return self._spawn_and_wait(cmd, host, port, api_key, self.backend_id)

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "images_generations": f"{base}/v1/images/generations",
            "models": f"{base}/v1/models",
            "health": f"{base}/health",
        }
