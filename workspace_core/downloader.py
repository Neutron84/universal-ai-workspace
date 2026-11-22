"""
High-Speed Model Downloader & Storage Hierarchy Resolver
Supports aria2c (16 parallel connections) and huggingface_hub with hf_transfer.
Implements Local SSD -> Google Drive -> Remote cache resolution.
Handles split/sharded GGUF downloads and VLM mmproj pairing.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download, snapshot_download
from .config import WorkspaceConfig, get_config
from .model_inspector import GGUFVariant, InspectionResult
from .security import SecurityManager
from .io_resolver import IOResolver


@dataclass
class DownloadResult:
    model_id: str
    primary_file_path: Path
    all_downloaded_files: List[Path]
    mmproj_file_path: Optional[Path] = None
    is_split: bool = False
    total_size_bytes: int = 0
    cached_from_drive: bool = False
    cached_from_local: bool = False


class Downloader:
    """Handles fast downloads and hierarchy resolution (Local SSD -> Drive -> Remote)."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.hf_token = SecurityManager.get_hf_token()
        self.aria2c_bin = shutil.which("aria2c")

    def resolve_or_download(
        self,
        inspection: InspectionResult,
        variant: Optional[GGUFVariant] = None,
        sync_to_drive: bool = True
    ) -> DownloadResult:
        """
        Resolves model files from Local SSD -> Google Drive -> Remote Download.
        Returns paths to the files ready on Local SSD for fast GPU inference.
        """
        # Determine category directory on Drive
        cat_map = {
            "llm-gguf": self.config.models_gguf_dir,
            "vlm-gguf": self.config.models_gguf_dir,
            "embedding-gguf": self.config.models_gguf_dir,
            "llm-safetensors": self.config.models_safetensors_dir,
            "diffusion": self.config.models_diffusion_dir,
            "audio": self.config.models_audio_dir,
        }
        drive_cat_dir = cat_map.get(inspection.model_type, self.config.models_gguf_dir)
        local_cache_dir = self.config.runtime_models_cache_dir

        # Check if local source
        if inspection.source_type == "local":
            primary_path = Path(inspection.raw_source)
            return DownloadResult(
                model_id=inspection.model_id,
                primary_file_path=primary_path,
                all_downloaded_files=[primary_path],
                is_split=False,
                total_size_bytes=primary_path.stat().st_size if primary_path.is_file() else 0,
                cached_from_local=True
            )

        # Case 1: GGUF Model Variant
        if variant:
            return self._download_gguf_variant(
                inspection=inspection,
                variant=variant,
                drive_dir=drive_cat_dir,
                local_dir=local_cache_dir,
                sync_to_drive=sync_to_drive
            )

        # Case 2: Full HuggingFace Safetensors or generic repo
        if inspection.source_type == "hf_repo" and inspection.repo_id:
            return self._download_hf_repo(
                repo_id=inspection.repo_id,
                drive_dir=drive_cat_dir,
                local_dir=local_cache_dir,
                sync_to_drive=sync_to_drive
            )

        # Case 3: Direct URL
        if inspection.source_type == "direct_url":
            return self._download_direct_url(
                url=inspection.raw_source,
                drive_dir=drive_cat_dir,
                local_dir=local_cache_dir,
                sync_to_drive=sync_to_drive
            )

        # Case 4: CivitAI Model Download
        if inspection.source_type == "civitai":
            civitai_meta = inspection.config_json or {}
            dl_url = civitai_meta.get("download_url")
            if not dl_url:
                civitai_meta = IOResolver.resolve_civitai_model(inspection.model_id)
                dl_url = civitai_meta.get("download_url")
            fname = civitai_meta.get("primary_file_name", f"{inspection.model_id}.safetensors")
            return self._download_direct_url(
                url=dl_url,
                drive_dir=drive_cat_dir,
                local_dir=local_cache_dir,
                sync_to_drive=sync_to_drive,
                filename=fname
            )

        # Case 5: Google Drive Share Download (via gdown)
        if inspection.source_type == "gdrive_share":
            share_meta = inspection.config_json or {}
            share_id = share_meta.get("id") or IOResolver._extract_gdrive_id(inspection.raw_source)
            is_folder = share_meta.get("is_folder", False)
            dl_path = IOResolver.download_gdrive_share(share_id=share_id, out_dir=local_cache_dir, is_folder=is_folder)
            if sync_to_drive:
                dest = drive_cat_dir / dl_path.name
                if dl_path.is_file():
                    shutil.copy2(str(dl_path), str(dest))
                elif dl_path.is_dir():
                    shutil.copytree(str(dl_path), str(dest), dirs_exist_ok=True)
            files = [dl_path] if dl_path.is_file() else list(dl_path.glob("**/*"))
            total_sz = dl_path.stat().st_size if dl_path.is_file() else sum(f.stat().st_size for f in files if f.is_file())
            return DownloadResult(
                model_id=dl_path.name,
                primary_file_path=dl_path,
                all_downloaded_files=files,
                total_size_bytes=total_sz,
            )

        # Case 6: Git Repo shallow clone
        if inspection.source_type == "git_repo":
            dl_path = IOResolver.clone_git_repo(clone_url=inspection.raw_source, out_dir=local_cache_dir)
            if sync_to_drive:
                shutil.copytree(str(dl_path), str(drive_cat_dir / dl_path.name), dirs_exist_ok=True)
            files = list(dl_path.glob("**/*"))
            return DownloadResult(
                model_id=dl_path.name,
                primary_file_path=dl_path,
                all_downloaded_files=files,
                total_size_bytes=sum(f.stat().st_size for f in files if f.is_file()),
            )

        raise ValueError(f"Unsupported download source: {inspection.source_type} ({inspection.raw_source})")

    def _download_gguf_variant(
        self,
        inspection: InspectionResult,
        variant: GGUFVariant,
        drive_dir: Path,
        local_dir: Path,
        sync_to_drive: bool
    ) -> DownloadResult:
        """Downloads all shards of a GGUF variant and optional mmproj file."""
        files_to_fetch = list(variant.shard_files)
        if variant.mmproj_file and variant.mmproj_file not in files_to_fetch:
            files_to_fetch.append(variant.mmproj_file)

        downloaded_local_paths: List[Path] = []
        cached_from_drive = False
        cached_from_local = False

        for fname in files_to_fetch:
            local_file = local_dir / fname
            drive_file = drive_dir / fname

            # 1. Check Local SSD
            if local_file.exists() and local_file.stat().st_size > 0:
                print(f"[Downloader] Found {fname} in Local SSD cache: {local_file}")
                downloaded_local_paths.append(local_file)
                cached_from_local = True
                continue

            # 2. Check Google Drive
            if drive_file.exists() and drive_file.stat().st_size > 0:
                print(f"[Downloader] Found {fname} on Drive. Copying to Local SSD for fast loading: {drive_file} -> {local_file}...")
                local_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(drive_file), str(local_file))
                downloaded_local_paths.append(local_file)
                cached_from_drive = True
                continue

            # 3. Download from Remote (Hugging Face or Direct URL)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Downloader] Downloading {fname} to Local SSD ({local_file})...")

            if inspection.repo_id:
                # Use aria2c with HuggingFace direct URL if possible for 16-connection speed
                hf_download_url = f"https://huggingface.co/{inspection.repo_id}/resolve/main/{fname}"
                success = False

                if self.aria2c_bin:
                    headers = []
                    if self.hf_token:
                        headers = [f"Authorization: Bearer {self.hf_token}"]
                    success = self._run_aria2c(hf_download_url, local_file.parent, local_file.name, headers)

                if not success:
                    # Fallback to hf_hub_download
                    print(f"[Downloader] Falling back to hf_hub_download for {fname}...")
                    cached_hf = hf_hub_download(
                        repo_id=inspection.repo_id,
                        filename=fname,
                        token=self.hf_token,
                        local_dir=str(local_file.parent),
                    )
                    local_file = Path(cached_hf)

            downloaded_local_paths.append(local_file)

            # Auto-sync to Drive if requested
            if sync_to_drive:
                print(f"[Downloader] Syncing {fname} to persistent Drive storage ({drive_file})...")
                drive_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(local_file), str(drive_file))

        # Identify primary model file (first shard or single file)
        primary = downloaded_local_paths[0]
        mmproj_path = None
        if variant.mmproj_file:
            for p in downloaded_local_paths:
                if variant.mmproj_file in p.name:
                    mmproj_path = p
                    break

        total_bytes = sum(p.stat().st_size for p in downloaded_local_paths if p.exists())

        return DownloadResult(
            model_id=variant.name,
            primary_file_path=primary,
            all_downloaded_files=downloaded_local_paths,
            mmproj_file_path=mmproj_path,
            is_split=variant.is_split,
            total_size_bytes=total_bytes,
            cached_from_drive=cached_from_drive,
            cached_from_local=cached_from_local,
        )

    def _download_hf_repo(
        self,
        repo_id: str,
        drive_dir: Path,
        local_dir: Path,
        sync_to_drive: bool
    ) -> DownloadResult:
        """Downloads full Safetensors / Transformers repository."""
        repo_name = repo_id.replace("/", "--")
        local_repo_dir = local_dir / repo_name
        drive_repo_dir = drive_dir / repo_name

        cached_from_drive = False
        cached_from_local = False

        if local_repo_dir.exists() and any(local_repo_dir.glob("*.safetensors")):
            cached_from_local = True
        elif drive_repo_dir.exists() and any(drive_repo_dir.glob("*.safetensors")):
            print(f"[Downloader] Found repo on Drive. Copying to Local SSD: {drive_repo_dir} -> {local_repo_dir}...")
            shutil.copytree(str(drive_repo_dir), str(local_repo_dir), dirs_exist_ok=True)
            cached_from_drive = True
        else:
            print(f"[Downloader] Downloading Hugging Face repository {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_repo_dir),
                token=self.hf_token,
            )
            if sync_to_drive:
                print(f"[Downloader] Syncing repository to Drive: {drive_repo_dir}...")
                shutil.copytree(str(local_repo_dir), str(drive_repo_dir), dirs_exist_ok=True)

        all_files = list(local_repo_dir.glob("**/*"))
        total_bytes = sum(f.stat().st_size for f in all_files if f.is_file())

        return DownloadResult(
            model_id=repo_name,
            primary_file_path=local_repo_dir,
            all_downloaded_files=all_files,
            is_split=False,
            total_size_bytes=total_bytes,
            cached_from_drive=cached_from_drive,
            cached_from_local=cached_from_local,
        )

    def _download_direct_url(
        self,
        url: str,
        drive_dir: Path,
        local_dir: Path,
        sync_to_drive: bool,
        filename: Optional[str] = None,
    ) -> DownloadResult:
        """Downloads a model directly from HTTP/HTTPS URL via aria2c."""
        fname = filename or Path(urlparse(url).path).name or "model.bin"
        local_file = local_dir / fname
        drive_file = drive_dir / fname

        cached_from_drive = False
        cached_from_local = False

        if local_file.exists() and local_file.stat().st_size > 0:
            cached_from_local = True
        elif drive_file.exists() and drive_file.stat().st_size > 0:
            shutil.copy2(str(drive_file), str(local_file))
            cached_from_drive = True
        else:
            local_file.parent.mkdir(parents=True, exist_ok=True)
            if self.aria2c_bin:
                self._run_aria2c(url, local_file.parent, fname)
            else:
                import urllib.request
                urllib.request.urlretrieve(url, str(local_file))

            if sync_to_drive:
                drive_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(local_file), str(drive_file))

        return DownloadResult(
            model_id=fname,
            primary_file_path=local_file,
            all_downloaded_files=[local_file],
            is_split=False,
            total_size_bytes=local_file.stat().st_size if local_file.exists() else 0,
            cached_from_drive=cached_from_drive,
            cached_from_local=cached_from_local,
        )

    def _run_aria2c(self, url: str, out_dir: Path, out_filename: str, headers: Optional[List[str]] = None) -> bool:
        """Executes aria2c with 16 connections and resume support."""
        cmd = [
            self.aria2c_bin or "aria2c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "-c",
            "--file-allocation=none",
            "--dir", str(out_dir),
            "-o", out_filename,
        ]
        if headers:
            for h in headers:
                cmd.extend(["--header", h])
        cmd.append(url)

        res = subprocess.run(cmd, check=False)
        dest = out_dir / out_filename
        return res.returncode == 0 and dest.exists() and dest.stat().st_size > 0
