"""
Base Converter Interface
Defines the standard contract and shared lifecycle helpers for model converters,
quantizers, and weight repacking plugins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from ..config import WorkspaceConfig, get_config


@dataclass
class ConversionResult:
    converter_name: str
    input_path: Path
    output_path: Path
    output_quant: Optional[str] = None
    output_size_bytes: int = 0
    duration_seconds: float = 0.0
    shard_files: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseConverter(ABC):
    """Abstract base class for all model converters and quantizers."""

    name: str = "base"
    supported_inputs: List[str] = []
    supported_outputs: List[str] = []

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()

    def get_python_executable(self, profile: Optional[str] = None) -> str:
        from ..env_manager import EnvironmentManager
        return EnvironmentManager(self.config).get_profile_python(profile or self.name)

    @staticmethod
    def safe_cleanup(target: Optional[Path], keep_paths: Optional[List[Optional[Path]]] = None) -> None:
        if not target or not target.exists():
            return
        protected = {p.resolve() for p in (keep_paths or []) if p is not None}
        if target.resolve() in protected:
            return
        try:
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(str(target))
        except Exception as e:
            print(f"[Converter] Cleanup note for {target.name}: {e}")

    @abstractmethod
    def check_installed(self) -> bool:
        """Verifies if required binaries or Python packages are available."""
        pass

    @abstractmethod
    def install(self) -> bool:
        """Installs necessary binaries, libraries, or Python dependencies."""
        pass

    @abstractmethod
    def estimate_requirements(self, input_info: Any) -> Dict[str, Any]:
        """Estimates required RAM, VRAM, and disk space for conversion."""
        pass

    @abstractmethod
    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = None,
        cleanup_source: bool = False,
        **kwargs
    ) -> Path:
        """
        Executes conversion or quantization on the input artifact.
        Returns the Path to the converted artifact.
        """
        pass
