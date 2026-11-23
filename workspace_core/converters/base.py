"""
Base Converter Interface
Defines the standard contract for model converters, quantizers, and weight repacking plugins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseConverter(ABC):
    """Abstract base class for all model converters and quantizers."""

    name: str = "base"
    supported_inputs: List[str] = []
    supported_outputs: List[str] = []

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()

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
