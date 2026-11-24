"""
Audio Engine (Whisper / Speech-to-Text / Audio Translation)
Executes Faster-Whisper models in a dedicated subprocess server.
Exposes OpenAI-compatible /v1/audio/transcriptions and /v1/audio/translations endpoints.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseEngine
from ..config import ServerPreset, WorkspaceConfig
from ..env_manager import EnvironmentManager
from ..registry import ModelEntry


AUDIO_SERVER_SCRIPT = """
import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--compute-type", type=str, default="auto")
parser.add_argument("--beam-size", type=int, default=5)
parser.add_argument("--vad-filter", action="store_true")
args = parser.parse_args()

print(f"[AudioServer] Loading Faster-Whisper model from {args.model_path}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = args.compute_type
if compute_type == "auto":
    compute_type = "float16" if torch.cuda.is_available() else "int8"

from faster_whisper import WhisperModel
model = WhisperModel(args.model_path, device=device, compute_type=compute_type)

model_name = os.path.basename(args.model_path)
print("[AudioServer] Model loaded successfully.")

app = FastAPI(title="Faster-Whisper Audio Transcription API")

def check_auth(authorization: Optional[str] = Header(None)):
    if args.api_key:
        expected = f"Bearer {args.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name, "backend": "audio"}

@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "workspace"}]
    }

async def _run_whisper(
    task_name: str,
    file: UploadFile,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    response_format: str,
):
    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments_gen, info = model.transcribe(
            tmp_path,
            task=task_name,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
        )
        segments = list(segments_gen)
        full_text = "".join([seg.text for seg in segments]).strip()

        fmt = response_format.lower()
        if fmt == "text":
            return PlainTextResponse(full_text)
        if fmt == "verbose_json":
            return {
                "task": task_name,
                "language": info.language,
                "duration": round(info.duration, 2),
                "text": full_text,
                "segments": [
                    {"id": idx, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                    for idx, s in enumerate(segments)
                ],
            }
        return {
            "text": full_text,
            "language": info.language,
            "duration": round(info.duration, 2),
        }
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model_param: Optional[str] = Form(None, alias="model"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    response_format: str = Form("json"),
    authorization: Optional[str] = Header(None),
):
    check_auth(authorization)
    return await _run_whisper("transcribe", file, language, prompt, temperature, response_format)

@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model_param: Optional[str] = Form(None, alias="model"),
    prompt: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    response_format: str = Form("json"),
    authorization: Optional[str] = Header(None),
):
    check_auth(authorization)
    return await _run_whisper("translate", file, None, prompt, temperature, response_format)

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
"""


class AudioEngine(BaseEngine):
    """Executes faster-whisper speech-to-text models with OpenAI-compatible FastAPI endpoints."""

    backend_id = "faster-whisper"

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.log_file = self.config.runtime_logs_dir / "audio_server.log"
        self.server_script_path = self.config.runtime_root / "audio_server.py"

    def prepare(
        self,
        model_entry: ModelEntry,
        preset: Optional[ServerPreset] = None,
        **overrides: Any,
    ) -> bool:
        self.model_entry = model_entry
        self.preset = self._merge_preset(preset, "whisper_audio", overrides)
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self.server_script_path.write_text(AUDIO_SERVER_SCRIPT, encoding="utf-8")
        return True

    def build_command(self, host: str, port: int, api_key: Optional[str] = None) -> List[str]:
        if not self.model_entry:
            raise ValueError("No model prepared for AudioEngine.")

        p = self.preset or ServerPreset(name="audio_default", backend="faster-whisper")
        py_exec = EnvironmentManager(self.config).get_profile_python("audio")

        cmd = [
            py_exec,
            str(self.server_script_path),
            "--model-path", str(self.model_entry.local_path),
            "--host", str(host),
            "--port", str(port),
            "--compute-type", str(p.whisper_compute_type or "auto"),
            "--beam-size", str(int(p.whisper_beam_size or 5)),
        ]
        if p.whisper_vad_filter:
            cmd.append("--vad-filter")
        if api_key:
            cmd.extend(["--api-key", str(api_key)])
        return self._append_extra_cli_args(cmd, p)

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if not self.model_entry:
            raise ValueError("No model prepared for AudioEngine.")
        EnvironmentManager(self.config).ensure_ready("audio")
        cmd = self.build_command(host, port, api_key)
        return self._spawn_and_wait(cmd, host, port, api_key, self.backend_id)

    def _generate_endpoints(self, host: str, port: int) -> Dict[str, str]:
        base = f"http://127.0.0.1:{port}"
        return {
            "base_url": base,
            "audio_transcriptions": f"{base}/v1/audio/transcriptions",
            "audio_translations": f"{base}/v1/audio/translations",
            "models": f"{base}/v1/models",
            "health": f"{base}/health",
        }
