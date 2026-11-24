"""
Multi-Backend Inference Engines
Modular execution backends conforming to BaseEngine ABC with unified plugin registry.
"""

from typing import Dict, List, Optional, Type

from .base import BaseEngine, EngineStatus
from .llama_cpp import LlamaCppEngine
from .ik_llama_engine import IKLlamaEngine
from .transformers_engine import TransformersEngine
from .unsloth_engine import UnslothEngine
from .vllm_engine import VLLMEngine
from .exllamav2_engine import ExLlamaV2Engine
from .diffusers_engine import DiffusersEngine
from .audio_engine import AudioEngine
from ..config import WorkspaceConfig, get_config

ENGINE_REGISTRY: Dict[str, Type[BaseEngine]] = {
    "llama.cpp": LlamaCppEngine,
    "llama_cpp": LlamaCppEngine,
    "ik_llama": IKLlamaEngine,
    "ik_llama.cpp": IKLlamaEngine,
    "transformers": TransformersEngine,
    "unsloth": UnslothEngine,
    "vllm": VLLMEngine,
    "exllamav2": ExLlamaV2Engine,
    "exl2": ExLlamaV2Engine,
    "diffusers": DiffusersEngine,
    "diffusion": DiffusersEngine,
    "faster-whisper": AudioEngine,
    "audio": AudioEngine,
}


def list_engines() -> List[str]:
    """Returns canonical list of supported inference engine names."""
    return [
        "llama.cpp",
        "ik_llama",
        "vllm",
        "transformers",
        "unsloth",
        "exllamav2",
        "diffusers",
        "faster-whisper",
    ]


def normalize_engine_name(name: str) -> str:
    """Normalizes engine aliases to canonical backend ID."""
    clean = (name or "llama.cpp").lower().strip()
    aliases = {
        "llama_cpp": "llama.cpp",
        "ik_llama.cpp": "ik_llama",
        "exl2": "exllamav2",
        "diffusion": "diffusers",
        "audio": "faster-whisper",
    }
    return aliases.get(clean, clean)


def get_engine(name: str, config: Optional[WorkspaceConfig] = None) -> BaseEngine:
    """Instantiates an engine plugin by name."""
    clean = (name or "llama.cpp").lower().strip()
    cls = ENGINE_REGISTRY.get(clean)
    if not cls:
        raise ValueError(f"Unknown engine: '{name}'. Supported engines: {list_engines()}")
    return cls(config or get_config())


__all__ = [
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
    "ENGINE_REGISTRY",
    "list_engines",
    "normalize_engine_name",
    "get_engine",
]
