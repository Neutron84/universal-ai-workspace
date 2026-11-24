"""
Universal I/O & Storage Mode Resolver
Provides URI abstraction, multi-source input resolution,
workspace storage modes, safe archive extraction, and artifact export sinks.
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
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse
import zipfile
import tarfile

from .security import SecurityManager


class WorkspaceMode(str, Enum):
    LOCAL_SSD = "local_ssd"              # Local SSD storage (/content/ai_workspace), zero Drive requirement
    HYBRID = "hybrid"                    # Local SSD runtime with Google Drive persistence
    GOOGLE_DRIVE = "google_drive"        # Persistent workspace at explicit Drive path
    GDRIVE_RELATIVE = "gdrive_relative"  # Relative path under /content/drive/MyDrive/<path>
    CUSTOM_PATH = "custom_path"          # Custom mounted directory


class InputMode(str, Enum):
    AUTO = "auto"                        # Auto-detect from URI scheme or pattern
    HUGGINGFACE = "huggingface"          # hf://org/repo, https://huggingface.co/..., or org/repo
    DIRECT_UPLOAD = "direct_upload"      # Interactive file upload via Colab/Gradio
    LOCAL_PATH = "local_path"            # direct:///abs/path or existing local path
    GOOGLE_DRIVE = "google_drive"        # /content/drive/MyDrive/...
    GDRIVE_RELATIVE = "gdrive_relative"  # gdrive://folder/model.gguf or relative path
    GDRIVE_SHARE = "gdrive_share"        # gshare://<id> or https://drive.google.com/file/d/<id>/...
    URL = "url"                          # http:// or https:// direct download
    CIVITAI = "civitai"                  # civitai://<id> or https://civitai.com/...
    GIT_REPO = "git_repo"                # git://... or .git URL

    # Backward compatibility aliases
    DIRECT = "local_path"
    COLAB_UPLOAD = "direct_upload"


class OutputMode(str, Enum):
    NONE = "none"                        # No export
    LOCAL_SSD = "local_ssd"              # Save/copy to /content/runtime/outputs
    DIRECT_DOWNLOAD = "direct_download"  # Trigger Colab browser download (auto-zipping if requested/folder)
    GOOGLE_DRIVE = "google_drive"        # Copy to Google Drive persistent storage
    GDRIVE_RELATIVE = "gdrive_relative"  # Copy relative to /content/drive/MyDrive/
    GDRIVE_DIRECT = "gdrive_direct"      # Write directly into Google Drive path without staging
    HUGGINGFACE_HUB = "huggingface_hub"  # Upload to HF Hub repo via HfApi
    CUSTOM_PATH = "custom_path"          # Copy to arbitrary mounted directory

    # Backward compatibility aliases
    BROWSER_DOWNLOAD = "direct_download"
    HUGGINGFACE_UPLOAD = "huggingface_hub"


def get_default_local_root() -> Path:
    """Returns a valid, writable local workspace root for Colab or local environments."""
    if os.environ.get("WORKSPACE_ROOT"):
        return Path(os.environ["WORKSPACE_ROOT"]).resolve()
    content_dir = Path("/content")
    if content_dir.is_dir() and os.access(str(content_dir), os.W_OK):
        return Path("/content/ai_workspace")
    if (Path.cwd() / "configs" / "settings.yaml").exists():
        return Path.cwd().resolve()
    return (Path.cwd() / "ai_workspace").resolve()


def get_default_runtime_root() -> Path:
    """Returns a valid, writable runtime root for Colab or local environments."""
    if os.environ.get("RUNTIME_ROOT"):
        return Path(os.environ["RUNTIME_ROOT"]).resolve()
    content_dir = Path("/content")
    if content_dir.is_dir() and os.access(str(content_dir), os.W_OK):
        return Path("/content/runtime")
    return (Path.cwd() / "runtime").resolve()


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
    hf_revision: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


class IOResolver:
    """Resolves workspace storage modes, input URIs, and artifact sinks."""

    @staticmethod
    def needs_drive(
        code_source: Optional[str] = None,
        workspace_mode: Optional[str] = None,
        input_type: Optional[str] = None,
        output_type: Optional[str] = None,
    ) -> bool:
        """Determines whether Google Drive must be mounted based on chosen modes."""
        drive_indicators = {"google_drive", "gdrive_relative", "gdrive_direct", "hybrid"}
        for val in (code_source, workspace_mode, input_type, output_type):
            if val and str(val).strip().lower() in drive_indicators:
                return True
        return False

    @staticmethod
    def ensure_google_drive_mounted(mount_point: str = "/content/drive") -> bool:
        """Attempts to mount Google Drive if running in Google Colab."""
        if Path("/content/drive/MyDrive").is_dir():
            return True

        try:
            from google.colab import drive  # type: ignore
            print("[IOResolver] Mounting Google Drive to /content/drive...")
            drive.mount(mount_point)
            return Path("/content/drive/MyDrive").is_dir()
        except Exception as e:
            print(f"[IOResolver] Google Drive not mounted ({e}). Operating in Local SSD mode.")
            return False

    @classmethod
    def resolve_workspace_paths(
        cls,
        mode: Union[str, WorkspaceMode] = WorkspaceMode.LOCAL_SSD,
        workspace_path: Optional[str] = None,
        runtime_root: Optional[str] = None,
    ) -> ResolvedWorkspacePaths:
        """Resolves workspace directories based on the selected WorkspaceMode."""
        if isinstance(mode, str):
            clean_mode = mode.lower().strip()
            try:
                w_mode = WorkspaceMode(clean_mode)
            except ValueError:
                w_mode = WorkspaceMode.LOCAL_SSD
        else:
            w_mode = mode

        rt_root = Path(runtime_root or get_default_runtime_root()).resolve()
        drive_mounted = False

        if w_mode == WorkspaceMode.LOCAL_SSD:
            ws_root = Path(workspace_path or get_default_local_root()).resolve()
            drive_mounted = False

        elif w_mode == WorkspaceMode.HYBRID:
            drive_mounted = cls.ensure_google_drive_mounted()
            if drive_mounted:
                ws_str = (workspace_path or "AI_Workspace").strip()
                if ws_str.startswith("/content/drive/MyDrive"):
                    ws_root = Path(ws_str).resolve()
                else:
                    rel = ws_str.lstrip("/")
                    ws_root = Path("/content/drive/MyDrive") / rel
            else:
                print("[IOResolver] Google Drive unavailable for Hybrid mode. Operating in Local SSD mode.")
                w_mode = WorkspaceMode.LOCAL_SSD
                ws_root = Path(workspace_path or get_default_local_root()).resolve()

        elif w_mode == WorkspaceMode.GOOGLE_DRIVE:
            drive_mounted = cls.ensure_google_drive_mounted()
            if drive_mounted:
                ws_root = Path(workspace_path or "/content/drive/MyDrive/AI_Workspace").resolve()
            else:
                print("[IOResolver] Google Drive unavailable. Falling back to Local SSD mode.")
                w_mode = WorkspaceMode.LOCAL_SSD
                ws_root = Path(get_default_local_root()).resolve()

        elif w_mode == WorkspaceMode.GDRIVE_RELATIVE:
            drive_mounted = cls.ensure_google_drive_mounted()
            if drive_mounted:
                ws_str = (workspace_path or "AI_Workspace").strip()
                if ws_str.startswith("/content/drive/MyDrive"):
                    ws_root = Path(ws_str).resolve()
                else:
                    rel = ws_str.lstrip("/")
                    ws_root = Path("/content/drive/MyDrive") / rel
            else:
                print("[IOResolver] Google Drive unavailable. Falling back to Local SSD mode.")
                w_mode = WorkspaceMode.LOCAL_SSD
                ws_root = Path(get_default_local_root()).resolve()

        elif w_mode == WorkspaceMode.CUSTOM_PATH:
            ws_root = Path(workspace_path or "/content/custom_workspace").resolve()
            drive_mounted = Path("/content/drive/MyDrive").is_dir()

        else:
            ws_root = Path(workspace_path or get_default_local_root()).resolve()

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

        if raw.lower() in ("colab_upload", "direct_upload"):
            return InputMode.DIRECT_UPLOAD

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
            return InputMode.LOCAL_PATH

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

        if Path(raw).exists():
            return InputMode.LOCAL_PATH

        if re.match(r"^[a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.]+$", raw):
            return InputMode.HUGGINGFACE

        return InputMode.AUTO

    @staticmethod
    def _is_within_directory(base_dir: Path, target_path: Path) -> bool:
        try:
            target_path.resolve().relative_to(base_dir.resolve())
            return True
        except ValueError:
            return False

    @classmethod
    def unpack_if_archive(cls, archive_path: Path, dest_dir: Optional[Path] = None) -> Path:
        """Safely extracts zip or tar archives into dest_dir if applicable."""
        archive_path = Path(archive_path).resolve()
        if not archive_path.is_file():
            return archive_path

        name = archive_path.name.lower()
        target_dir = Path(dest_dir).resolve() if dest_dir else (archive_path.parent / archive_path.stem).resolve()

        if name.endswith(".zip"):
            print(f"[IOResolver] Extracting zip archive: {archive_path} -> {target_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(archive_path), "r") as z:
                for member in z.infolist():
                    member_dest = target_dir / member.filename
                    if not cls._is_within_directory(target_dir, member_dest):
                        raise ValueError(f"Unsafe zip member path blocked: {member.filename}")
                z.extractall(str(target_dir))
            return target_dir

        if name.endswith((".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2", ".tar")):
            print(f"[IOResolver] Extracting tar archive: {archive_path} -> {target_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(archive_path), "r:*") as t:
                for member in t.getmembers():
                    member_dest = target_dir / member.name
                    if not cls._is_within_directory(target_dir, member_dest):
                        raise ValueError(f"Unsafe tar member path blocked: {member.name}")
                t.extractall(path=str(target_dir))
            return target_dir

        return archive_path

    @classmethod
    def resolve_input(
        cls,
        uri: str,
        mode: Union[str, InputMode] = InputMode.AUTO,
        download_dir: Optional[Path] = None,
        hf_token: Optional[str] = None,
        civitai_token: Optional[str] = None,
        unpack_archive: bool = True,
        input_file: Optional[str] = None,
        hf_revision: Optional[str] = None,
    ) -> ResolvedInput:
        """Parses and resolves input URIs into a structured ResolvedInput."""
        raw = uri.strip()
        if isinstance(mode, str):
            clean_mode = mode.lower().strip()
            if clean_mode in ("direct", "local_dir", "local"):
                in_mode = InputMode.LOCAL_PATH
            elif clean_mode in ("colab_upload", "upload"):
                in_mode = InputMode.DIRECT_UPLOAD
            else:
                try:
                    in_mode = InputMode(clean_mode)
                except ValueError:
                    in_mode = InputMode.AUTO
        else:
            in_mode = mode

        if in_mode == InputMode.AUTO:
            in_mode = cls.detect_input_mode(raw)

        clean_input_file = input_file.strip() if (input_file and input_file.strip()) else None
        clean_rev = hf_revision.strip() if (hf_revision and hf_revision.strip()) else None

        # 1. Hugging Face
        if in_mode == InputMode.HUGGINGFACE:
            clean = raw
            if clean.startswith("hf://"):
                clean = clean[5:]
            if "huggingface.co" in clean:
                m = re.search(r"huggingface\.co/([^/]+/[^/]+)(?:/(?:resolve|blob|tree)/([^/]+)(?:/(.+))?)?", clean)
                if m:
                    repo_id = m.group(1)
                    url_rev = m.group(2)
                    url_sub = m.group(3)
                    subpath = clean_input_file or url_sub
                    rev = clean_rev or url_rev
                    return ResolvedInput(
                        mode=InputMode.HUGGINGFACE,
                        raw_uri=raw,
                        resolved_type="hf_file" if subpath else "hf_repo",
                        identifier=repo_id,
                        subpath_or_file=subpath,
                        hf_revision=rev,
                    )
            parts = clean.split("/", 2)
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                subpath = clean_input_file or (parts[2] if len(parts) > 2 else None)
                return ResolvedInput(
                    mode=InputMode.HUGGINGFACE,
                    raw_uri=raw,
                    resolved_type="hf_file" if subpath else "hf_repo",
                    identifier=repo_id,
                    subpath_or_file=subpath,
                    hf_revision=clean_rev,
                )
            return ResolvedInput(
                mode=InputMode.HUGGINGFACE,
                raw_uri=raw,
                resolved_type="hf_repo",
                identifier=clean,
                subpath_or_file=clean_input_file,
                hf_revision=clean_rev,
            )

        # 2. Local Path
        if in_mode in (InputMode.LOCAL_PATH, InputMode.DIRECT):
            clean = raw
            if clean.startswith("direct://"):
                clean = clean[9:]
            p = Path(clean).resolve()
            if unpack_archive and p.is_file():
                p = cls.unpack_if_archive(p)
            if clean_input_file and p.is_dir():
                candidate = p / clean_input_file
                if candidate.exists():
                    p = candidate
            return ResolvedInput(
                mode=InputMode.LOCAL_PATH,
                raw_uri=raw,
                resolved_type="direct_local",
                identifier=p.name,
                local_path=p,
                subpath_or_file=clean_input_file,
            )

        # 3. Google Drive Absolute Path
        if in_mode == InputMode.GOOGLE_DRIVE:
            cls.ensure_google_drive_mounted()
            p = Path(raw).resolve()
            if unpack_archive and p.is_file():
                extract_dest = (Path(download_dir).resolve() / p.stem) if download_dir else None
                p = cls.unpack_if_archive(p, dest_dir=extract_dest)
            if clean_input_file and p.is_dir():
                candidate = p / clean_input_file
                if candidate.exists():
                    p = candidate
            return ResolvedInput(
                mode=InputMode.GOOGLE_DRIVE,
                raw_uri=raw,
                resolved_type="direct_local",
                identifier=p.name,
                local_path=p,
                subpath_or_file=clean_input_file,
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

            target_p = p
            if p.exists() and download_dir:
                dl_dir = Path(download_dir).resolve()
                dl_dir.mkdir(parents=True, exist_ok=True)
                if p.is_dir():
                    dest_folder = dl_dir / p.name
                    if not dest_folder.exists():
                        print(f"[IOResolver] Copying Google Drive folder {p} to local directory {dest_folder}...")
                        shutil.copytree(str(p), str(dest_folder), dirs_exist_ok=True)
                    target_p = dest_folder
                elif p.is_file():
                    dest_file = dl_dir / p.name
                    if not dest_file.exists():
                        print(f"[IOResolver] Copying Google Drive file {p} to local directory {dest_file}...")
                        shutil.copy2(str(p), str(dest_file))
                    target_p = dest_file

            if unpack_archive and target_p.is_file():
                target_p = cls.unpack_if_archive(target_p)

            if clean_input_file and target_p.is_dir():
                candidate = target_p / clean_input_file
                if candidate.exists():
                    target_p = candidate

            return ResolvedInput(
                mode=InputMode.GDRIVE_RELATIVE,
                raw_uri=raw,
                resolved_type="direct_local",
                identifier=target_p.name,
                local_path=target_p,
                subpath_or_file=clean_input_file,
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
                subpath_or_file=clean_input_file,
                extra_metadata={"is_folder": is_folder, "id": file_or_folder_id, "unpack_archive": unpack_archive},
            )

        # 6. CivitAI Model Resolution
        if in_mode == InputMode.CIVITAI:
            model_id, version_id = cls._extract_civitai_ids(raw)
            return ResolvedInput(
                mode=InputMode.CIVITAI,
                raw_uri=raw,
                resolved_type="civitai",
                identifier=str(model_id),
                subpath_or_file=clean_input_file,
                extra_metadata={
                    "model_id": model_id,
                    "version_id": version_id,
                    "civitai_token": civitai_token,
                },
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
                subpath_or_file=clean_input_file,
                extra_metadata={"clone_url": clean},
            )

        # 8. Direct HTTP / HTTPS URL
        if in_mode == InputMode.URL:
            fname = clean_input_file or Path(urlparse(raw).path).name or "downloaded_file"
            return ResolvedInput(
                mode=InputMode.URL,
                raw_uri=raw,
                resolved_type="url",
                identifier=fname,
                subpath_or_file=clean_input_file,
                extra_metadata={"unpack_archive": unpack_archive},
            )

        # 9. Interactive Direct Upload
        if in_mode in (InputMode.DIRECT_UPLOAD, InputMode.COLAB_UPLOAD):
            return ResolvedInput(
                mode=InputMode.DIRECT_UPLOAD,
                raw_uri=raw,
                resolved_type="direct_upload",
                identifier="direct_upload",
                subpath_or_file=clean_input_file,
            )

        return ResolvedInput(
            mode=InputMode.AUTO,
            raw_uri=raw,
            resolved_type="unknown",
            identifier=raw,
            subpath_or_file=clean_input_file,
        )

    @staticmethod
    def _extract_gdrive_id(uri: str) -> str:
        """Extracts file or folder ID from Google Drive sharing URLs."""
        if uri.startswith("gshare://"):
            return uri[9:].strip("/")
        m = re.search(r"/(?:file/d|folders|open\?id=)/([a-zA-Z0-9_\-]+)", uri)
        if m:
            return m.group(1)
        parsed = urlparse(uri)
        qs = parse_qs(parsed.query)
        if "id" in qs:
            return qs["id"][0]
        return uri.strip()

    @staticmethod
    def _extract_civitai_ids(uri: str) -> Tuple[str, Optional[str]]:
        """Extracts model ID and optional version ID from CivitAI URLs."""
        clean = uri.strip()
        if clean.startswith("civitai://"):
            clean = clean[10:]
            parts = clean.split("/")
            return parts[0], parts[1] if len(parts) > 1 else None

        m = re.search(r"civitai\.com/models/(\d+)", clean)
        model_id = m.group(1) if m else clean

        version_id = None
        parsed = urlparse(clean)
        qs = parse_qs(parsed.query)
        if "modelVersionId" in qs:
            version_id = qs["modelVersionId"][0]

        return model_id, version_id

    @classmethod
    def download_gdrive_share(
        cls,
        share_id: str,
        out_dir: Path,
        is_folder: bool = False,
        unpack_archive: bool = True,
        input_file: Optional[str] = None,
    ) -> Path:
        """Downloads public Google Drive shares using gdown."""
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import gdown  # type: ignore
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            import gdown  # type: ignore

        print(f"[IOResolver] Downloading Google Drive share (ID: {share_id}, is_folder={is_folder})...")
        if is_folder:
            dest_dir = out_dir / f"gdrive_folder_{share_id}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            gdown.download_folder(id=share_id, output=str(dest_dir), quiet=False, use_cookies=False)
            result_path = dest_dir
        else:
            dest_file = gdown.download(id=share_id, output=str(out_dir) + "/", quiet=False, fuzzy=True)
            result_path = Path(dest_file) if dest_file else out_dir

        if unpack_archive and result_path.is_file():
            result_path = cls.unpack_if_archive(result_path)
        if input_file and result_path.is_dir():
            candidate = result_path / input_file
            if candidate.exists():
                result_path = candidate
        return result_path

    @classmethod
    def resolve_civitai_model(
        cls,
        model_id: str,
        version_id: Optional[str] = None,
        api_key: Optional[str] = None,
        preferred_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queries CivitAI API (/api/v1/models/<id>) to retrieve metadata and download URLs."""
        import urllib.request
        key = api_key or SecurityManager.get_civitai_api_key()

        url = f"https://civitai.com/api/v1/models/{model_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "AI-Workspace/1.1"})
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
        if preferred_filename:
            for f in files:
                if preferred_filename.lower() in str(f.get("name", "")).lower():
                    primary_file = f
                    break
        if not primary_file:
            for f in files:
                if f.get("primary", False):
                    primary_file = f
                    break
        if not primary_file and files:
            primary_file = files[0]

        download_url = (primary_file.get("downloadUrl") if primary_file else None) or selected_ver.get("downloadUrl", "")
        if key and download_url and "token=" not in download_url:
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
            print(f"[IOResolver] Git repository already present at {dest}.")
            return dest

        print(f"[IOResolver] Shallow cloning git repository {clone_url} -> {dest}...")
        cmd = ["git", "clone", "--depth", "1", clone_url, str(dest)]
        subprocess.run(cmd, check=True)
        return dest


class OutputManager:
    """Manages artifact export to Local SSD, Google Drive, HuggingFace Hub, or browser."""

    def __init__(self, paths: ResolvedWorkspacePaths, hf_token: Optional[str] = None):
        self.paths = paths
        self.hf_token = hf_token or SecurityManager.get_hf_token()

    def _collect_all_sources(
        self,
        src: Path,
        companion_files: Optional[List[Union[str, Path]]] = None,
    ) -> List[Path]:
        items: List[Path] = [src]
        seen = {src.resolve()}
        if companion_files:
            for c in companion_files:
                cp = Path(c).resolve()
                if cp.exists() and cp not in seen:
                    items.append(cp)
                    seen.add(cp)
        return items

    def _package_zip(self, sources: List[Path], zip_path: Path) -> Path:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if len(sources) == 1 and sources[0].is_dir():
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(sources[0]))
            return zip_path

        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sources:
                if item.is_file():
                    zf.write(str(item), arcname=item.name)
                elif item.is_dir():
                    for f in item.rglob("*"):
                        if f.is_file():
                            zf.write(str(f), arcname=str(Path(item.name) / f.relative_to(item)))
        return zip_path

    def export_artifact(
        self,
        source_path: Union[str, Path],
        output_mode: Union[str, OutputMode] = OutputMode.LOCAL_SSD,
        destination: Optional[str] = None,
        output_name: Optional[str] = None,
        zip_output: bool = False,
        hf_repo_id: Optional[str] = None,
        subpath_in_dest: Optional[str] = None,
        cleanup_source: bool = False,
        companion_files: Optional[List[Union[str, Path]]] = None,
        hf_private: bool = False,
        hf_commit_message: Optional[str] = None,
        protected_paths: Optional[List[Union[str, Path]]] = None,
    ) -> Dict[str, Any]:
        """Exports an artifact (plus optional shard/mmproj companions) to the designated sink."""
        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Artifact source does not exist: {src}")

        all_sources = self._collect_all_sources(src, companion_files)

        if isinstance(output_mode, str):
            clean_mode = output_mode.lower().strip()
            if clean_mode in ("browser_download", "download"):
                mode = OutputMode.DIRECT_DOWNLOAD
            elif clean_mode in ("huggingface_upload", "hf_hub"):
                mode = OutputMode.HUGGINGFACE_HUB
            else:
                try:
                    mode = OutputMode(clean_mode)
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

        if mode == OutputMode.NONE:
            result.update({
                "success": True,
                "exported_path": str(src),
                "message": f"Artifact retained at source: {src}",
            })
            return result

        target_name = output_name or src.name

        # 1. Local SSD export
        if mode == OutputMode.LOCAL_SSD:
            out_dir = Path(destination).resolve() if destination else self.paths.runtime_outputs_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / target_name

            if zip_output and not target_name.endswith(".zip"):
                zip_target = out_dir / f"{target_name}.zip"
                print(f"[OutputManager] Zipping artifact to Local SSD: {zip_target}...")
                self._package_zip(all_sources, zip_target)
                target = zip_target
            else:
                for idx, item in enumerate(all_sources):
                    dest_item = target if idx == 0 else (out_dir / item.name)
                    if item.is_dir():
                        shutil.copytree(str(item), str(dest_item), dirs_exist_ok=True)
                    elif item != dest_item:
                        shutil.copy2(str(item), str(dest_item))

            result.update({
                "success": True,
                "exported_path": str(target),
                "message": f"Artifact saved to Local SSD: {target}",
            })
            self._handle_cleanup(all_sources, target, cleanup_source, protected_paths)
            return result

        # 2. Google Drive direct write (no local copy staging, zipping disabled)
        if mode == OutputMode.GDRIVE_DIRECT:
            IOResolver.ensure_google_drive_mounted()
            rel = (destination or "AI_Workspace/outputs").strip()
            if rel.startswith("/content/drive/MyDrive"):
                dest_dir = Path(rel).resolve()
            else:
                dest_dir = Path("/content/drive/MyDrive") / rel.lstrip("/")

            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / target_name

            print(f"[OutputManager] Writing artifact directly to Google Drive: {src} -> {target}...")
            for idx, item in enumerate(all_sources):
                dest_item = target if idx == 0 else (dest_dir / item.name)
                if item.is_dir():
                    shutil.copytree(str(item), str(dest_item), dirs_exist_ok=True)
                elif item != dest_item:
                    shutil.copy2(str(item), str(dest_item))

            result.update({
                "success": True,
                "exported_path": str(target),
                "message": f"Artifact written directly to Google Drive: {target}",
            })
            self._handle_cleanup(all_sources, target, cleanup_source, protected_paths)
            return result

        # 3. Google Drive / GDrive Relative export
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

            if zip_output and not target_name.endswith(".zip"):
                zip_name = f"{target_name}.zip"
                temp_zip = self.paths.runtime_outputs_dir / zip_name
                print(f"[OutputManager] Packaging artifact into zip: {temp_zip}...")
                self._package_zip(all_sources, temp_zip)
                target = dest_dir / zip_name
                shutil.copy2(str(temp_zip), str(target))
                try:
                    temp_zip.unlink()
                except Exception:
                    pass
            else:
                target = dest_dir / target_name
                print(f"[OutputManager] Copying artifact to Google Drive: {src} -> {target}...")
                for idx, item in enumerate(all_sources):
                    dest_item = target if idx == 0 else (dest_dir / item.name)
                    if item.is_dir():
                        shutil.copytree(str(item), str(dest_item), dirs_exist_ok=True)
                    elif item != dest_item:
                        shutil.copy2(str(item), str(dest_item))

            result.update({
                "success": True,
                "exported_path": str(target),
                "message": f"Artifact saved to Google Drive: {target}",
            })
            self._handle_cleanup(all_sources, target, cleanup_source, protected_paths)
            return result

        # 4. Hugging Face Hub Upload
        if mode == OutputMode.HUGGINGFACE_HUB:
            target_repo = hf_repo_id or destination
            if not target_repo:
                raise ValueError("hf_repo_id or destination (username/repo-name) must be provided for Hugging Face upload.")

            from huggingface_hub import HfApi  # type: ignore
            api = HfApi(token=self.hf_token)
            try:
                api.create_repo(repo_id=target_repo, repo_type="model", private=hf_private, exist_ok=True)
            except Exception as e:
                print(f"[OutputManager] Note creating/checking repo {target_repo}: {e}")

            commit_msg = hf_commit_message or f"Upload {target_name} via Universal AI Workspace"
            print(f"[OutputManager] Uploading {len(all_sources)} item(s) to Hugging Face Hub ({target_repo})...")

            for idx, item in enumerate(all_sources):
                path_in_repo = (subpath_in_dest or target_name) if idx == 0 else item.name
                if item.is_dir():
                    api.upload_folder(
                        folder_path=str(item),
                        repo_id=target_repo,
                        path_in_repo=path_in_repo,
                        repo_type="model",
                        commit_message=commit_msg,
                    )
                else:
                    api.upload_file(
                        path_or_fileobj=str(item),
                        path_in_repo=path_in_repo,
                        repo_id=target_repo,
                        repo_type="model",
                        commit_message=commit_msg,
                    )

            hf_url = f"https://huggingface.co/{target_repo}"
            result.update({
                "success": True,
                "exported_path": hf_url,
                "message": f"Uploaded successfully to Hugging Face Hub: {hf_url}",
            })
            self._handle_cleanup(all_sources, None, cleanup_source, protected_paths)
            return result

        # 5. Direct Browser Download (Colab files.download)
        if mode == OutputMode.DIRECT_DOWNLOAD:
            download_target = src
            if src.is_dir() or zip_output or len(all_sources) > 1:
                zip_path = self.paths.runtime_outputs_dir / f"{target_name}.zip"
                print(f"[OutputManager] Packaging into zip archive for download: {zip_path}...")
                download_target = self._package_zip(all_sources, zip_path)

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
                    "message": f"Colab files.download unavailable ({e}). Artifact at: {download_target}",
                })
            self._handle_cleanup(all_sources, download_target, cleanup_source, protected_paths)
            return result

        # 6. Custom Path export
        if mode == OutputMode.CUSTOM_PATH:
            dest_dir = Path(destination or "/content/outputs").resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            if zip_output and not target_name.endswith(".zip"):
                target = self._package_zip(all_sources, dest_dir / f"{target_name}.zip")
            else:
                target = dest_dir / target_name
                for idx, item in enumerate(all_sources):
                    dest_item = target if idx == 0 else (dest_dir / item.name)
                    if item.is_dir():
                        shutil.copytree(str(item), str(dest_item), dirs_exist_ok=True)
                    elif item != dest_item:
                        shutil.copy2(str(item), str(dest_item))

            result.update({
                "success": True,
                "exported_path": str(target),
                "message": f"Exported to custom path: {target}",
            })
            self._handle_cleanup(all_sources, target, cleanup_source, protected_paths)
            return result

        return result

    @staticmethod
    def _handle_cleanup(
        sources: Union[Path, List[Path]],
        target: Optional[Path],
        cleanup_source: bool,
        protected_paths: Optional[List[Union[str, Path]]] = None,
    ) -> None:
        """Removes source files if cleanup_source is requested and not protected or identical to target."""
        if not cleanup_source:
            return
        src_list = [sources] if isinstance(sources, Path) else list(sources)
        protected = {Path(p).resolve() for p in (protected_paths or []) if p}

        for source in src_list:
            resolved_src = source.resolve()
            if target is not None and resolved_src == target.resolve():
                continue
            if resolved_src in protected:
                print(f"[OutputManager] Skipping cleanup for active protected path: {source.name}")
                continue
            try:
                if source.is_file():
                    print(f"[OutputManager] Cleaning up intermediate source file {source.name}...")
                    source.unlink()
                elif source.is_dir():
                    print(f"[OutputManager] Cleaning up intermediate source directory {source.name}...")
                    shutil.rmtree(str(source))
            except Exception as e:
                print(f"[OutputManager] Note during source cleanup: {e}")
