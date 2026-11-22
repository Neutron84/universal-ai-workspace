"""
Audio Engine (Whisper / Speech-to-Text / Audio Processing)
Executes Faster-Whisper models in a dedicated subprocess server.
Exposes standard OpenAI-compatible /v1/audio/transcriptions endpoint.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from .base import BaseEngine, EngineStatus
from ..config import ServerPreset, WorkspaceConfig, get_config
from ..registry import ModelEntry


AUDIO_SERVER_SCRIPT = """
import argparse
import io
import json
import os
import sys
import tempfile
import time
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException, Request, Header, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", type=str, default=None)
args = parser.parse_args()

print(f"[AudioServer] Loading Faster-Whisper model from {args.model_path}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"

model = None
try:
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model_path, device=device, compute_type=compute_type)
except Exception as e:
    print(f"[AudioServer] Faster-Whisper initialization error: {e}")
    raise

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
    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            beam_size=5,
        )
        full_text = "".join([segment.text for segment in segments]).strip()

        if response_format.lower() == "text":
            return PlainTextResponse(full_text)

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

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
"""


class AudioEngine(BaseEngine):
    """Executes faster-whisper speech-to-text models with OpenAI-compatible FastAPI endpoints."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self.log_file = self.config.runtime_logs_dir / "audio_server.log"
        self.server_script_path = self.config.runtime_root / "audio_server.py"

    def prepare(self, model_entry: ModelEntry, preset: Optional[ServerPreset] = None, **overrides) -> bool:
        self.model_entry = model_entry
        self.preset = preset
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self.server_script_path.write_text(AUDIO_SERVER_SCRIPT, encoding="utf-8")
        return True

    def start(self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None) -> bool:
        if self.process and self.process.poll() is None:
            return True

        if not self.model_entry:
            raise ValueError("No model prepared for AudioEngine.")

        self.api_key = api_key

        # Prefer isolated venv Python if available
        venv_python = self.config.runtime_envs_dir / "audio" / "bin" / "python"
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

        self.config.runtime_logs_dir.mkdir(parents=True, exist_ok=True)
        log_f = open(self.log_file, "w", encoding="utf-8")
        self.process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        log_f.close()

        self.status = EngineStatus(
            is_running=True,
            backend_name="audio",
            pid=self.process.pid,
            host=host,
            port=port,
            active_model_id=self.model_entry.model_id,
            api_key_configured=bool(api_key),
            endpoints=self._generate_endpoints(host, port),
            log_file_path=str(self.log_file),
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
            "audio_transcriptions": f"{base}/v1/audio/transcriptions",
            "models": f"{base}/v1/models",
            "health": f"{base}/health",
        }
