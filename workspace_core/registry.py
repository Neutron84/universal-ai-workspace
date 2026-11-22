"""
Model Registry & Storage Manager
Tracks all imported models, active model status, Drive synchronization,
benchmark metrics, and disk space maintenance.
"""

import datetime
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import WorkspaceConfig, get_config


@dataclass
class ModelEntry:
    model_id: str
    name: str
    type: str  # "llm-gguf", "vlm-gguf", "embedding-gguf", "llm-safetensors", "diffusion", "audio"
    source_url: str
    quantization: str
    total_size_bytes: int
    local_path: str
    drive_path: Optional[str] = None
    is_synced_to_drive: bool = False
    mmproj_path: Optional[str] = None
    shard_files: List[str] = field(default_factory=list)
    is_active: bool = False
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_used_at: Optional[str] = None
    benchmark_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_size_gb(self) -> float:
        return round(self.total_size_bytes / (1024**3), 2)


class ModelRegistry:
    """Manages the model registry file and storage cleanup/sync."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.registry_file = self.config.registry_file
        self.entries: Dict[str, ModelEntry] = {}
        self.load()

    def load(self) -> None:
        """Loads registry from registry.json."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = {}
                for mid, entry_dict in data.items():
                    self.entries[mid] = ModelEntry(**entry_dict)
            except Exception as e:
                print(f"[ModelRegistry] Warning: Failed to load registry ({e}). Starting empty.")
                self.entries = {}
        else:
            self.entries = {}
            self.save()

    def save(self) -> None:
        """Saves registry to registry.json."""
        serializable = {mid: asdict(entry) for mid, entry in self.entries.items()}
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def register_model(
        self,
        model_id: str,
        name: str,
        type: str,
        source_url: str,
        quantization: str,
        total_size_bytes: int,
        local_path: str,
        drive_path: Optional[str] = None,
        is_synced_to_drive: bool = False,
        mmproj_path: Optional[str] = None,
        shard_files: Optional[List[str]] = None,
        set_active: bool = True
    ) -> ModelEntry:
        """Registers a model into registry and optionally sets it as active."""
        if set_active:
            for e in self.entries.values():
                e.is_active = False

        entry = ModelEntry(
            model_id=model_id,
            name=name,
            type=type,
            source_url=source_url,
            quantization=quantization,
            total_size_bytes=total_size_bytes,
            local_path=str(local_path),
            drive_path=str(drive_path) if drive_path else None,
            is_synced_to_drive=is_synced_to_drive,
            mmproj_path=str(mmproj_path) if mmproj_path else None,
            shard_files=[str(s) for s in (shard_files or [])],
            is_active=set_active,
            last_used_at=datetime.datetime.now().isoformat() if set_active else None,
        )
        self.entries[model_id] = entry
        self.save()
        print(f"[ModelRegistry] Registered model: {model_id} ({quantization}, {entry.total_size_gb}GB)")
        return entry

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        return self.entries.get(model_id)

    def list_models(self) -> List[ModelEntry]:
        return list(self.entries.values())

    def get_active_model(self) -> Optional[ModelEntry]:
        for entry in self.entries.values():
            if entry.is_active:
                return entry
        return None

    def set_active_model(self, model_id: str) -> Optional[ModelEntry]:
        """Sets the specified model as active (hot-swap)."""
        target = self.entries.get(model_id)
        if not target:
            return None

        for entry in self.entries.values():
            entry.is_active = False

        target.is_active = True
        target.last_used_at = datetime.datetime.now().isoformat()
        self.save()
        return target

    def update_benchmark(self, model_id: str, metrics: Dict[str, Any]) -> None:
        """Updates benchmark metrics for a model."""
        entry = self.entries.get(model_id)
        if entry:
            entry.benchmark_metrics = {
                **entry.benchmark_metrics,
                **metrics,
                "tested_at": datetime.datetime.now().isoformat()
            }
            self.save()

    def sync_to_drive(self, model_id: str) -> bool:
        """Copies local model files (including all shards and mmproj) to Google Drive persistent storage."""
        entry = self.entries.get(model_id)
        if not entry:
            return False

        local_p = Path(entry.local_path)
        if not local_p.exists():
            return False

        # Determine target category folder on Drive
        cat_map = {
            "llm-gguf": self.config.models_gguf_dir,
            "vlm-gguf": self.config.models_gguf_dir,
            "embedding-gguf": self.config.models_gguf_dir,
            "llm-safetensors": self.config.models_safetensors_dir,
            "diffusion": self.config.models_diffusion_dir,
            "audio": self.config.models_audio_dir,
        }
        dest_dir = cat_map.get(entry.type, self.config.models_gguf_dir)
        dest_path = dest_dir / local_p.name

        print(f"[ModelRegistry] Syncing {local_p} -> {dest_path}...")
        if local_p.is_file():
            shutil.copy2(str(local_p), str(dest_path))
        elif local_p.is_dir():
            shutil.copytree(str(local_p), str(dest_path), dirs_exist_ok=True)

        # Sync all additional shards if split GGUF
        if entry.shard_files:
            for s in entry.shard_files:
                sp = Path(s)
                if sp.exists() and sp.resolve() != local_p.resolve():
                    s_dest = dest_dir / sp.name
                    print(f"[ModelRegistry] Syncing shard {sp.name} -> {s_dest}...")
                    shutil.copy2(str(sp), str(s_dest))

        # Sync mmproj if present
        if entry.mmproj_path and Path(entry.mmproj_path).exists():
            mm_p = Path(entry.mmproj_path)
            shutil.copy2(str(mm_p), str(dest_dir / mm_p.name))

        entry.drive_path = str(dest_path)
        entry.is_synced_to_drive = True
        self.save()
        return True

    def delete_local_cache(self, model_id: str) -> bool:
        """Deletes model files (including all shards) from Local SSD to free space, keeping Drive archive intact."""
        entry = self.entries.get(model_id)
        if not entry:
            return False

        local_p = Path(entry.local_path)
        if local_p.exists():
            print(f"[ModelRegistry] Deleting local cache for {model_id} ({local_p})...")
            if local_p.is_file():
                local_p.unlink()
            elif local_p.is_dir():
                shutil.rmtree(str(local_p))

        # Delete all shards locally
        if entry.shard_files:
            for s in entry.shard_files:
                sp = Path(s)
                if sp.exists():
                    try:
                        sp.unlink()
                    except Exception:
                        pass

        # Delete mmproj if it lives in local cache
        if entry.mmproj_path:
            mmp = Path(entry.mmproj_path)
            if mmp.exists() and str(mmp).startswith(str(self.config.runtime_root)):
                try:
                    mmp.unlink()
                except Exception:
                    pass

        self.save()
        return True

    def delete_drive_copy(self, model_id: str) -> bool:
        """Deletes model copy (including all shards and mmproj) from Google Drive persistent storage."""
        entry = self.entries.get(model_id)
        if not entry or not entry.drive_path:
            return False

        drive_p = Path(entry.drive_path)
        drive_dir = drive_p.parent
        if drive_p.exists():
            print(f"[ModelRegistry] Deleting Drive copy for {model_id} ({drive_p})...")
            if drive_p.is_file():
                drive_p.unlink()
            elif drive_p.is_dir():
                shutil.rmtree(str(drive_p))

        # Delete Drive copies of shards
        if entry.shard_files:
            for s in entry.shard_files:
                s_name = Path(s).name
                s_drive = drive_dir / s_name
                if s_drive.exists():
                    try:
                        s_drive.unlink()
                    except Exception:
                        pass

        # Delete Drive copy of mmproj
        if entry.mmproj_path:
            mm_name = Path(entry.mmproj_path).name
            mm_drive = drive_dir / mm_name
            if mm_drive.exists():
                try:
                    mm_drive.unlink()
                except Exception:
                    pass

        entry.drive_path = None
        entry.is_synced_to_drive = False
        self.save()
        return True

    def remove_model(self, model_id: str, delete_files: bool = False) -> bool:
        """Removes a model from the registry and optionally purges files."""
        entry = self.entries.get(model_id)
        if not entry:
            return False

        if delete_files:
            self.delete_local_cache(model_id)
            self.delete_drive_copy(model_id)

        del self.entries[model_id]
        self.save()
        return True

    def purge_temp_files(self) -> int:
        """Purges leftover temporary files in runtime directory."""
        freed_bytes = 0
        temp_patterns = ["*.tmp*", "*.aria2"]
        for pattern in temp_patterns:
            for p in self.config.runtime_root.glob(f"**/{pattern}"):
                try:
                    sz = p.stat().st_size
                    p.unlink()
                    freed_bytes += sz
                except Exception:
                    pass
        return freed_bytes

    def check_storage_thresholds(self) -> Dict[str, Any]:
        """Checks if SSD or Drive free space is below configured thresholds."""
        from .hardware import get_hardware_profile
        hw = get_hardware_profile()
        settings = self.config.settings

        ssd_warn = hw.ssd_free_gb < settings.ssd_low_space_warning_gb
        drive_warn = hw.drive_free_gb < settings.drive_low_space_warning_gb

        return {
            "ssd_free_gb": hw.ssd_free_gb,
            "ssd_warning": ssd_warn,
            "drive_free_gb": hw.drive_free_gb,
            "drive_warning": drive_warn,
        }
