"""
Universal I/O & Storage Mode Resolver
Provides unified URI abstraction, multi-source input resolution,
flexible workspace storage modes, and multi-sink output/artifact export.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse
import zipfile
import tarfile

from .security import SecurityManager


class WorkspaceMode(str, Enum):
    HYBRID = "hybrid"                    # Fast Local SSD runtime + persistent sync to Google Drive
    LOCAL_SSD = "local_ssd"              # 100% ephemeral SSD (/content/ai_workspace), zero Drive requirement
    GOOGLE_DRIVE = "google_drive"        # Persistent workspace at explicit Drive path
    GDRIVE_RELATIVE = "gdrive_relative"  # Relative path under /content/drive/MyDrive/<path>
    CUSTOM_PATH = "custom_path"          # Arbitrary directory (Shared Drive, rclone mount, Kaggle, etc.)


class InputMode(str, Enum):
    AUTO = "auto"                        # Auto-detect from URI scheme or pattern
    HUGGINGFACE = "huggingface"          # hf://org/repo, https://huggingface.co/..., or org/repo
    DIRECT = "direct"                    # direct:///abs/path or existing local path
    GOOGLE_DRIVE = "google_drive"        # /content/drive/MyDrive/...
    GDRIVE_RELATIVE = "gdrive_relative"  # gdrive://folder/model.gguf
    GDRIVE_SHARE = "gdrive_share"        # gshare://<id> or https://drive.google.com/file/d/<id>/...
    URL = "url"                          # http:// or https:// direct download
    CIVITAI = "civitai"                  # civitai://<id> or https://civitai.com/...
    GIT_REPO = "git_repo"                # git://... or .git URL
    COLAB_UPLOAD = "colab_upload"        # Interactive file upload via Colab/Gradio


class OutputMode(str, Enum):
    LOCAL_SSD = "local_ssd"              # Save/copy to /content/runtime/outputs
    GOOGLE_DRIVE = "google_drive"        # Copy to Google Drive persistent storage
    GDRIVE_RELATIVE = "gdrive_relative"  # Copy relative to /content/drive/MyDrive/
    HUGGINGFACE_UPLOAD = "huggingface_upload" # Upload to HF Hub repo via HfApi
    BROWSER_DOWNLOAD = "browser_download"     # Trigger Colab browser download (auto-zipping folders)
    CUSTOM_PATH = "custom_path"          # Copy to arbitrary mounted directory


@dataclass
class ResolvedWorkspacePaths:
    mode: WorkspaceMode
    workspace_root: Path
    runtime_root: Path
    workspace_core_dir: Path
    configs_dir: Path
    cache_dir: Path
    uv_cache_dir: Path
    bin_dir: Path
    models_dir: Path
    models_gguf_dir: Path
    models_safetensors_dir: Path
    models_diffusion_dir: Path
    models_audio_dir: Path
    runtime_envs_dir: Path
    runtime_models_cache_dir: Path
    runtime_logs_dir: Path
    runtime_outputs_dir: Path
    settings_file: Path
    presets_file: Path
    registry_file: Path
    drive_mounted: bool = False


@dataclass
class ResolvedInput:
    mode: InputMode
    raw_uri: str
    resolved_type: str                   # "hf_repo", "hf_file", "direct_local", "gdrive_share", "civitai", "url", "git_repo"
    identifier: str                      # repo_id, model_id, or filename
    local_path: Optional[Path] = None    # Set if resolved/downloaded locally
    subpath_or_file: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


class IOResolver:
    """Resolves workspace storage modes, input URIs, and artifact sinks."""

    @staticmethod
    def ensure_google_drive_mounted(mount_point: str = "/content/drive") -> bool:
        """
        Attempts to mount Google Drive if running in Google Colab.
        Returns True if /content/drive/MyDrive exists, False otherwise.
        """
        if Path("/content/drive/MyDrive").is_dir():
            return True

        try:
            from google.colab import drive  # type: ignore
            print("[IOResolver] Mounting Google Drive to /content/drive...")
            drive.mount(mount_point)
            return Path("/content/drive/MyDrive").is_dir()
        except Exception as e:
            print(f"[IOResolver] Note: Google Drive not mounted ({e}). Operating in Local SSD mode.")
            return False

    @classmethod
    def resolve_workspace_paths(
        cls,
        mode: Union[str, WorkspaceMode] = WorkspaceMode.HYBRID,
        workspace_path: Optional[str] = None,
        runtime_root: Optional[str] = None,
    ) -> ResolvedWorkspacePaths:
        """
        Resolves workspace directories based on the selected WorkspaceMode.
        Auto-mounts Google Drive if required, gracefully falling back to local SSD.
        """
        if isinstance(mode, str):
            try:
                w_mode = WorkspaceMode(mode.lower())
            except ValueError:
                w_mode = WorkspaceMode.HYBRID
        else:
            w_mode = mode

        rt_root = Path(runtime_root or "/content/runtime").resolve()
        drive_mounted = False

        if w_mode == WorkspaceMode.HYBRID:
            drive_mounted = cls.ensure_google_drive_mounted()
            if drive_mounted:
                ws_str = (workspace_path or "AI_Workspace").strip()
                if ws_str.startswith("/content/drive/MyDrive"):
                    ws_root = Path(ws_str).resolve()
                else:
                    rel = ws_str.lstrip("/")
                    ws_root = Path("/content/drive/MyDrive") / rel
            else:
                # Graceful degradation to local SSD
                print("[IOResolver] Warning: Google Drive unavailable for Hybrid mode. Falling back to local_ssd mode.")
                w_mode = WorkspaceMode.LOCAL_SSD
                ws_root = Path(workspace_path or "/content/ai_workspace").resolve()

        elif w_mode == WorkspaceMode.LOCAL_SSD:
            # 100% ephemeral local SSD — zero Drive dependency
            ws_root = Path(workspace_path or "/content/ai_workspace").resolve()
            drive_mounted = False

        elif w_mode == WorkspaceMode.GOOGLE_DRIVE:
            drive_mounted = cls.ensure_google_drive_mounted()
            ws_root = Path(workspace_path or "/content/drive/MyDrive/AI_Workspace").resolve()

        elif w_mode == WorkspaceMode.GDRIVE_RELATIVE:
            drive_mounted = cls.ensure_google_drive_mounted()
            ws_str = (workspace_path or "AI_Workspace").strip()
            if ws_str.startswith("/content/drive/MyDrive"):
                ws_root = Path(ws_str).resolve()
            else:
                rel = ws_str.lstrip("/")
                ws_root = Path("/content/drive/MyDrive") / rel

        elif w_mode == WorkspaceMode.CUSTOM_PATH:
            ws_root = Path(workspace_path or "/content/custom_workspace").resolve()
            drive_mounted = Path("/content/drive/MyDrive").is_dir()

        else:
            ws_root = Path(workspace_path or "/content/ai_workspace").resolve()

        # Build paths
        configs_dir = ws_root / "configs"
        cache_dir = ws_root / "cache"
        models_dir = ws_root / "models"

        paths = ResolvedWorkspacePaths(
            mode=w_mode,
            workspace_root=ws_root,
            runtime_root=rt_root,
            workspace_core_dir=ws_root / "workspace_core",
            configs_dir=configs_dir,
            cache_dir=cache_dir,
            uv_cache_dir=cache_dir / "uv_cache",
            bin_dir=cache_dir / "bin",
            models_dir=models_dir,
            models_gguf_dir=models_dir / "gguf",
            models_safetensors_dir=models_dir / "hf_safetensors",
            models_diffusion_dir=models_dir / "diffusion",
            models_audio_dir=models_dir / "audio",
            runtime_envs_dir=rt_root / "envs",
            runtime_models_cache_dir=rt_root / "models_cache",
            runtime_logs_dir=rt_root / "logs",
            runtime_outputs_dir=rt_root / "outputs",
            settings_file=configs_dir / "settings.yaml",
            presets_file=configs_dir / "presets.yaml",
            registry_file=configs_dir / "registry.json",
            drive_mounted=drive_mounted,
        )

        # Create necessary directories
        for d in [
            paths.workspace_root,
            paths.configs_dir,
            paths.cache_dir,
            paths.uv_cache_dir,
            paths.bin_dir,
            paths.models_dir,
            paths.models_gguf_dir,
            paths.models_safetensors_dir,
            paths.models_diffusion_dir,
            paths.models_audio_dir,
            paths.runtime_root,
            paths.runtime_envs_dir,
            paths.runtime_models_cache_dir,
            paths.runtime_logs_dir,
            paths.runtime_outputs_dir,
        ]:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        return paths

    @classmethod
    def detect_input_mode(cls, uri: str) -> InputMode:
        """Auto-detects the InputMode from an arbitrary URI string."""
        raw = uri.strip()
        if not raw:
            return InputMode.AUTO

        if raw.lower() == "colab_upload":
            return InputMode.COLAB_UPLOAD

        if raw.startswith("hf://"):
            return InputMode.HUGGINGFACE

        if raw.startswith("gshare://"):
            return InputMode.GDRIVE_SHARE

        if "drive.google.com" in raw:
            if "/file/d/" in raw or "/folders/" in raw or "id=" in raw:
                return InputMode.GDRIVE_SHARE
            return InputMode.GOOGLE_DRIVE

        if raw.startswith("gdrive://"):
            return InputMode.GDRIVE_RELATIVE

        if raw.startswith("direct://"):
            return InputMode.DIRECT

        if raw.startswith("civitai://") or "civitai.com" in raw:
            return InputMode.CIVITAI

        if raw.startswith("git://") or raw.endswith(".git"):
            return InputMode.GIT_REPO

        if raw.startswith("http://") or raw.startswith("https://"):
            if "huggingface.co" in raw:
                return InputMode.HUGGINGFACE
            if "civitai.com" in raw:
                return InputMode.CIVITAI
            return InputMode.URL

        if raw.startswith("/content/drive/MyDrive"):
            return InputMode.GOOGLE_DRIVE

        # Check existing filesystem path
        if Path(raw).exists():
            return InputMode.DIRECT

        # Standard Hugging Face repo format (e.g., "org/model-name")
        if re.match(r"^[a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.]+$", raw):
            return InputMode.HUGGINGFACE

        return InputMode.AUTO

    @classmethod
    def resolve_input(
        cls,
        uri: str,
        mode: Union[str, InputMode] = InputMode.AUTO,
        download_dir: Optional[Path] = None,
        hf_token: Optional[str] = None,
        civitai_token: Optional[str] = None,
    ) -> ResolvedInput:
        """
        Parses and resolves any input URI into a structured ResolvedInput.
        Supports downloading from Google Drive shares via gdown, CivitAI API,
        HuggingFace Hub, direct URLs via aria2c, and git shallow clones.
        """
        raw = uri.strip()
        if isinstance(mode, str):
            try:
                in_mode = InputMode(mode.lower())
            except ValueError:
                in_mode = InputMode.AUTO
        else:
            in_mode = mode

        if in_mode == InputMode.AUTO:
            in_mode = cls.detect_input_mode(raw)

        # 1. Hugging Face
        if in_mode == InputMode.HUGGINGFACE:
            clean = raw
            if clean.startswith("hf://"):
                clean = clean[5:]
            # Parse possible URL
            if "huggingface.co" in clean:
                m = re.search(r"huggingface\.co/([^/]+/[^/]+)(?:/(?:resolve|blob)/[^/]+/(.+))?", clean)
                if m:
                    repo_id = m.group(1)
                    subpath = m.group(2)
                    return ResolvedInput(
                        mode=InputMode.HUGGINGFACE,
                        raw_uri=raw,
                        resolved_type="hf_file" if subpath else "hf_repo",
                        identifier=repo_id,
                        subpath_or_file=subpath
                    )
            parts = clean.split("/", 2)
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                subpath = parts[2] if len(parts) > 2 else None
                return ResolvedInput(
                    mode=InputMode.HUGGINGFACE,
                    raw_uri=raw,
                    resolved_type="hf_file" if subpath else "hf_repo",
                    identifier=repo_id,
                    subpath_or_file=subpath
                )
            return ResolvedInput(
                mode=InputMode.HUGGINGFACE,
                raw_uri=raw,
                resolved_type="hf_repo",
                identifier=clean
            )

        # 2. Direct Local File / Directory
        if in_mode == InputMode.DIRECT:
            clean = raw
            if clean.startswith("direct://"):
                clean = clean[9:]
            p = Path(clean).resolve()
            return ResolvedInput(
                mode=InputMode.DIRECT,
                raw_uri=raw,
                resolved_type="direct_local",
                identifier=p.name,
                local_path=p
            )

        # 3. Google Drive Absolute Path
        if in_mode == InputMode.GOOGLE_DRIVE:
            cls.ensure_google_drive_mounted()
            p = Path(raw).resolve()
            return ResolvedInput(
                mode=InputMode.GOOGLE_DRIVE,
                raw_uri=raw,
                resolved_type="direct_local",
                identifier=p.name,
                local_path=p
            )

        # 4. Google Drive Relative Path
        if in_mode == InputMode.GDRIVE_RELATIVE:
            cls.ensure_google_drive_mounted()
            clean = raw
            if clean.startswith("gdrive://"):
                clean = clean[9:]
            if clean.startswith("/content/drive/MyDrive"):
                p = Path(clean).resolve()
            else:
                rel = clean.lstrip("/")
                p = Path("/content/drive/MyDrive") / rel
            return ResolvedInput(
                mode=InputMode.GDRIVE_RELATIVE,
                raw_uri=raw,
                resolved_type="direct_local",
                identifier=p.name,
                local_path=p
            )

        # 5. Google Drive Public Share via gdown
        if in_mode == InputMode.GDRIVE_SHARE:
            file_or_folder_id = cls._extract_gdrive_id(raw)
            is_folder = ("/folders/" in raw) or raw.endswith("/edit") or ("folder" in raw.lower())
            return ResolvedInput(
                mode=InputMode.GDRIVE_SHARE,
                raw_uri=raw,
                resolved_type="gdrive_share",
                identifier=file_or_folder_id,
                extra_metadata={"is_folder": is_folder, "id": file_or_folder_id}
            )

        # 6. CivitAI Model Resolution
        if in_mode == InputMode.CIVITAI:
            model_id, version_id = cls._extract_civitai_ids(raw)
            return ResolvedInput(
                mode=InputMode.CIVITAI,
                raw_uri=raw,
                resolved_type="civitai",
                identifier=str(model_id),
                extra_metadata={"model_id": model_id, "version_id": version_id}
            )

        # 7. Git Repository
        if in_mode == InputMode.GIT_REPO:
            clean = raw
            if clean.startswith("git://"):
                clean = "https://" + clean[6:]
            repo_name = Path(clean).stem
            return ResolvedInput(
                mode=InputMode.GIT_REPO,
                raw_uri=clean,
                resolved_type="git_repo",
                identifier=repo_name,
                extra_metadata={"clone_url": clean}
            )

        # 8. Direct HTTP / HTTPS URL
        if in_mode == InputMode.URL:
            fname = Path(urlparse(raw).path).name or "downloaded_file"
            return ResolvedInput(
                mode=InputMode.URL,
                raw_uri=raw,
                resolved_type="url",
                identifier=fname
            )

        # 9. Interactive Colab / UI Upload
        if in_mode == InputMode.COLAB_UPLOAD:
            return ResolvedInput(
                mode=InputMode.COLAB_UPLOAD,
                raw_uri=raw,
                resolved_type="colab_upload",
                identifier="colab_upload"
            )

        return ResolvedInput(
            mode=InputMode.AUTO,
            raw_uri=raw,
            resolved_type="unknown",
            identifier=raw
        )

    @staticmethod
    def _extract_gdrive_id(uri: str) -> str:
        """Extracts Google Drive file or folder ID from various link formats."""
        if uri.startswith("gshare://"):
            return uri[9:].strip("/")

        # Format: /file/d/<id>/
        m = re.search(r"/file/d/([a-zA-Z0-9_\-]+)", uri)
        if m:
            return m.group(1)

        # Format: /drive/folders/<id>
        m = re.search(r"/folders/([a-zA-Z0-9_\-]+)", uri)
        if m:
            return m.group(1)

        # Format: ?id=<id>
        parsed = urlparse(uri)
        qs = parse_qs(parsed.query)
        if "id" in qs:
            return qs["id"][0]

        return uri.strip("/")

    @staticmethod
    def _extract_civitai_ids(uri: str) -> Tuple[str, Optional[str]]:
        """Extracts model_id and optional modelVersionId from CivitAI URLs or URIs."""
        clean = uri
        if clean.startswith("civitai://"):
            clean = clean[10:]

        parsed = urlparse(clean)
        qs = parse_qs(parsed.query)
        version_id = qs.get("modelVersionId", [None])[0]

        # e.g. /models/12345 or /models/12345/model-name
        m = re.search(r"(?:models/)?([0-9]+)", clean)
        model_id = m.group(1) if m else clean.split("?")[0].strip("/")

        return model_id, version_id

    @classmethod
    def download_gdrive_share(cls, share_id: str, out_dir: Path, is_folder: bool = False) -> Path:
        """Downloads a public Google Drive file or folder using gdown."""
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import gdown  # type: ignore
        except ImportError:
            print("[IOResolver] Installing gdown for Google Drive share resolution...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            import gdown

        print(f"[IOResolver] Downloading Google Drive share (ID: {share_id}, is_folder={is_folder})...")
        if is_folder:
            dest_dir = out_dir / f"gdrive_folder_{share_id}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            gdown.download_folder(id=share_id, output=str(dest_dir), quiet=False, use_cookies=False)
            return dest_dir
        else:
            dest_file = gdown.download(id=share_id, output=str(out_dir) + "/", quiet=False, fuzzy=True)
            return Path(dest_file) if dest_file else out_dir

    @classmethod
    def resolve_civitai_model(
        cls,
        model_id: str,
        version_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Queries CivitAI API (/api/v1/models/<id>) to retrieve download URLs,
        metadata, files, and format specifications.
        """
        import urllib.request
        key = api_key or os.environ.get("CIVITAI_API_KEY")

        url = f"https://civitai.com/api/v1/models/{model_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "AI-Workspace/1.0"})
        if key:
            req.add_header("Authorization", f"Bearer {key}")

        print(f"[IOResolver] Querying CivitAI API: {url}...")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        versions = data.get("modelVersions", [])
        if not versions:
            raise ValueError(f"No model versions found on CivitAI for model ID {model_id}.")

        selected_ver = versions[0]
        if version_id:
            for v in versions:
                if str(v.get("id")) == str(version_id):
                    selected_ver = v
                    break

        files = selected_ver.get("files", [])
        primary_file = None
        for f in files:
            if f.get("primary", False):
                primary_file = f
                break
        if not primary_file and files:
            primary_file = files[0]

        download_url = selected_ver.get("downloadUrl", "")
        if key and download_url:
            separator = "&" if "?" in download_url else "?"
            download_url = f"{download_url}{separator}token={key}"

        return {
            "model_id": model_id,
            "name": data.get("name", f"civitai-{model_id}"),
            "type": data.get("type", "Checkpoint"),
            "version_id": selected_ver.get("id"),
            "version_name": selected_ver.get("name"),
            "primary_file_name": primary_file.get("name", f"model-{model_id}.safetensors") if primary_file else "model.safetensors",
            "size_kb": primary_file.get("sizeKB", 0) if primary_file else 0,
            "download_url": download_url,
            "images": [img.get("url") for img in selected_ver.get("images", [])[:3]],
        }

    @classmethod
    def clone_git_repo(cls, clone_url: str, out_dir: Path) -> Path:
        """Shallow clones a git repository into out_dir."""
        repo_name = Path(clone_url).stem
        dest = out_dir / repo_name
        if dest.exists() and (dest / ".git").is_dir():
            print(f"[IOResolver] Git repo already cloned at {dest}.")
            return dest

        print(f"[IOResolver] Shallow cloning git repo {clone_url} -> {dest}...")
        cmd = ["git", "clone", "--depth", "1", clone_url, str(dest)]
        subprocess.run(cmd, check=True)
        return dest


class OutputManager:
    """Manages universal artifact export to local SSD, Google Drive, HuggingFace Hub, or browser."""

    def __init__(self, paths: ResolvedWorkspacePaths, hf_token: Optional[str] = None):
        self.paths = paths
        self.hf_token = hf_token or SecurityManager.get_hf_token()

    def export_artifact(
        self,
        source_path: Union[str, Path],
        output_mode: Union[str, OutputMode] = OutputMode.LOCAL_SSD,
        destination: Optional[str] = None,
        hf_repo_id: Optional[str] = None,
        subpath_in_dest: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Exports an artifact (model file, folder, image, benchmark log) to the chosen sink.
        Returns a dict summarizing the export result and destination URL or path.
        """
        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Artifact source does not exist: {src}")

        if isinstance(output_mode, str):
            try:
                mode = OutputMode(output_mode.lower())
            except ValueError:
                mode = OutputMode.LOCAL_SSD
        else:
            mode = output_mode

        result: Dict[str, Any] = {
            "source": str(src),
            "output_mode": mode.value,
            "success": False,
            "exported_path": None,
            "message": "",
        }

        # 1. Local SSD export
        if mode == OutputMode.LOCAL_SSD:
            out_dir = Path(destination).resolve() if destination else self.paths.runtime_outputs_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / src.name
            if src.is_dir():
                shutil.copytree(str(src), str(target), dirs_exist_ok=True)
            else:
                shutil.copy2(str(src), str(target))
            result.update({
                "success": True,
                "exported_path": str(target),
                "message": f"Artifact saved to Local SSD: {target}",
            })
            return result

        # 2. Google Drive / GDrive Relative export
        if mode in (OutputMode.GOOGLE_DRIVE, OutputMode.GDRIVE_RELATIVE):
            IOResolver.ensure_google_drive_mounted()
            if mode == OutputMode.GDRIVE_RELATIVE:
                rel = (destination or "AI_Workspace/outputs").strip()
                if rel.startswith("/content/drive/MyDrive"):
                    dest_dir = Path(rel).resolve()
                else:
                    dest_dir = Path("/content/drive/MyDrive") / rel.lstrip("/")
            else:
                dest_dir = Path(destination or str(self.paths.models_dir)).resolve()

            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / src.name
            print(f"[OutputManager] Syncing artifact to Drive: {src} -> {target}...")
            if src.is_dir():
                shutil.copytree(str(src), str(target), dirs_exist_ok=True)
            else:
                shutil.copy2(str(src), str(target))

            result.update({
                "success": True,
                "exported_path": str(target),
                "message": f"Artifact safely backed up to Google Drive: {target}",
            })
            return result

        # 3. Hugging Face Hub Upload
        if mode == OutputMode.HUGGINGFACE_UPLOAD:
            target_repo = hf_repo_id or destination
            if not target_repo:
                raise ValueError("hf_repo_id or destination must be provided for Hugging Face upload.")

            from huggingface_hub import HfApi  # type: ignore
            api = HfApi(token=self.hf_token)
            path_in_repo = subpath_in_dest or src.name

            print(f"[OutputManager] Uploading {src} to Hugging Face Hub ({target_repo})...")
            if src.is_dir():
                api.upload_folder(
                    folder_path=str(src),
                    repo_id=target_repo,
                    path_in_repo=path_in_repo,
                    repo_type="model",
                )
            else:
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=path_in_repo,
                    repo_id=target_repo,
                    repo_type="model",
                )

            hf_url = f"https://huggingface.co/{target_repo}"
            result.update({
                "success": True,
                "exported_path": hf_url,
                "message": f"Uploaded successfully to Hugging Face Hub: {hf_url}",
            })
            return result

        # 4. Browser Download (Colab files.download)
        if mode == OutputMode.BROWSER_DOWNLOAD:
            download_target = src
            if src.is_dir():
                zip_path = self.paths.runtime_outputs_dir / f"{src.name}.zip"
                print(f"[OutputManager] Packaging folder into zip: {zip_path}...")
                shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(src))
                download_target = zip_path

            try:
                from google.colab import files  # type: ignore
                print(f"[OutputManager] Triggering browser download for {download_target}...")
                files.download(str(download_target))
                result.update({
                    "success": True,
                    "exported_path": str(download_target),
                    "message": f"Browser download initiated for {download_target.name}.",
                })
            except Exception as e:
                result.update({
                    "success": False,
                    "exported_path": str(download_target),
                    "message": f"Colab files.download not available ({e}). Artifact at: {download_target}",
                })
            return result

        # 5. Custom Path export
        if mode == OutputMode.CUSTOM_PATH:
            dest_dir = Path(destination or "/content/outputs").resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / src.name
            if src.is_dir():
                shutil.copytree(str(src), str(target), dirs_exist_ok=True)
            else:
                shutil.copy2(str(src), str(target))
            result.update({
                "success": True,
                "exported_path": str(target),
                "message": f"Exported to custom path: {target}",
            })
            return result

        return result
