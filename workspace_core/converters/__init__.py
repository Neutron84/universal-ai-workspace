"""
Modular Converter & Quantizer Ecosystem
Provides unified plugin registry for llama.cpp, ik_llama, unsloth, gptqmodel, autoawq, exllamav2, and ct2.
"""

from typing import Dict, List, Optional, Type

from .base import BaseConverter, ConversionResult
from .llama_cpp_converter import LlamaCppConverter
from .ik_llama_converter import IKLlamaConverter
from .unsloth_converter import UnslothConverter
from .hf_quantizers import AutoAWQConverter, CT2AudioConverter, ExLlamaConverter, GPTQModelConverter
from ..config import WorkspaceConfig, get_config

CONVERTER_REGISTRY: Dict[str, Type[BaseConverter]] = {
    "llama.cpp": LlamaCppConverter,
    "llama_cpp": LlamaCppConverter,
    "ik_llama": IKLlamaConverter,
    "ik_llama.cpp": IKLlamaConverter,
    "unsloth": UnslothConverter,
    "gptqmodel": GPTQModelConverter,
    "gptq": GPTQModelConverter,
    "autoawq": AutoAWQConverter,
    "awq": AutoAWQConverter,
    "exllamav2": ExLlamaConverter,
    "exl2": ExLlamaConverter,
    "ct2": CT2AudioConverter,
    "ctranslate2": CT2AudioConverter,
}


def list_converters() -> List[str]:
    """Returns list of supported converter names."""
    return [
        "llama.cpp",
        "ik_llama",
        "unsloth",
        "gptqmodel",
        "autoawq",
        "exllamav2",
        "ct2"
    ]


def get_converter(name: str, config: Optional[WorkspaceConfig] = None) -> BaseConverter:
    """Retrieves an instantiated converter plugin by name."""
    clean_name = name.lower().strip()
    cls = CONVERTER_REGISTRY.get(clean_name)
    if not cls:
        raise ValueError(f"Unknown converter: '{name}'. Supported converters: {list_converters()}")
    return cls(config or get_config())


__all__ = [
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
    "CONVERTER_REGISTRY",
]
