"""
Universal AI Workspace for Google Colab
Modular Python Core Package
"""

__version__ = "1.0.0"

from .config import WorkspaceConfig, ServerPreset, WorkspaceSettings, get_config
from .io_resolver import (
    IOResolver,
    OutputManager,
    WorkspaceMode,
    InputMode,
    OutputMode,
    ResolvedWorkspacePaths,
    ResolvedInput,
)
from .hardware import HardwareProfiler, HardwareProfile, GPUInfo, get_hardware_profile
from .security import SecurityManager, get_security_manager
from .env_manager import EnvironmentManager
from .model_inspector import ModelInspector, InspectionResult, GGUFVariant
from .auto_selector import AutoSelector, SelectionRecommendation
from .downloader import Downloader, DownloadResult
from .registry import ModelRegistry, ModelEntry
from .process_manager import ProcessManager, SystemState, TunnelInfo
from .engines import (
    BaseEngine,
    EngineStatus,
    LlamaCppEngine,
    TransformersEngine,
    VLLMEngine,
    DiffusersEngine,
    AudioEngine,
)

__all__ = [
    "WorkspaceConfig",
    "ServerPreset",
    "WorkspaceSettings",
    "get_config",
    "IOResolver",
    "OutputManager",
    "WorkspaceMode",
    "InputMode",
    "OutputMode",
    "ResolvedWorkspacePaths",
    "ResolvedInput",
    "HardwareProfiler",
    "HardwareProfile",
    "GPUInfo",
    "get_hardware_profile",
    "SecurityManager",
    "get_security_manager",
    "EnvironmentManager",
    "ModelInspector",
    "InspectionResult",
    "GGUFVariant",
    "AutoSelector",
    "SelectionRecommendation",
    "Downloader",
    "DownloadResult",
    "ModelRegistry",
    "ModelEntry",
    "ProcessManager",
    "SystemState",
    "TunnelInfo",
    "BaseEngine",
    "EngineStatus",
    "LlamaCppEngine",
    "TransformersEngine",
    "VLLMEngine",
    "DiffusersEngine",
    "AudioEngine",
]
