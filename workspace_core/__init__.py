"""
Universal AI Workspace for Google Colab & Linux Workstations
Modular Python Core Package
"""

__version__ = "1.2.0"

from .config import WorkspaceConfig, ServerPreset, WorkspaceSettings, get_config, parse_cli_args
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
from .orchestrator import LaunchRequest, LaunchResult, WorkspaceOrchestrator
from .engines import (
    BaseEngine,
    EngineStatus,
    LlamaCppEngine,
    IKLlamaEngine,
    TransformersEngine,
    UnslothEngine,
    VLLMEngine,
    ExLlamaV2Engine,
    DiffusersEngine,
    AudioEngine,
    get_engine,
    list_engines,
)
from .converters import (
    BaseConverter,
    ConversionResult,
    LlamaCppConverter,
    IKLlamaConverter,
    UnslothConverter,
    GPTQModelConverter,
    AutoAWQConverter,
    ExLlamaConverter,
    CT2AudioConverter,
    get_converter,
    list_converters,
)

__all__ = [
    "WorkspaceConfig",
    "ServerPreset",
    "WorkspaceSettings",
    "get_config",
    "parse_cli_args",
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
    "LaunchRequest",
    "LaunchResult",
    "WorkspaceOrchestrator",
    "BaseEngine",
    "EngineStatus",
    "LlamaCppEngine",
    "IKLlamaEngine",
    "TransformersEngine",
    "UnslothEngine",
    "VLLMEngine",
    "ExLlamaV2Engine",
    "DiffusersEngine",
    "AudioEngine",
    "get_engine",
    "list_engines",
    "BaseConverter",
    "ConversionResult",
    "LlamaCppConverter",
    "IKLlamaConverter",
    "UnslothConverter",
    "GPTQModelConverter",
    "AutoAWQConverter",
    "ExLlamaConverter",
    "CT2AudioConverter",
    "get_converter",
    "list_converters",
]
