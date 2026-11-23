"""
Intelligent Auto-Selector & Capability-Driven Analysis Engine
Computes realistic VRAM budgets, routes models to optimal engines and converters,
evaluates non-uniform ik_llama quants, MoE hybrid offloading, and hardware tiers.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import WorkspaceConfig, get_config
from .hardware import HardwareProfile, get_hardware_profile
from .model_inspector import GGUFVariant, InspectionResult, QUANT_ORDER, QUANT_SIZE_FACTORS


@dataclass
class SelectionRecommendation:
    source_model_id: str
    selected_variant: Optional[GGUFVariant]
    recommended_quant: str
    recommended_engine: str = "llama.cpp"
    recommended_converter: Optional[str] = None
    needs_requant: bool = False
    requant_target_quant: Optional[str] = None
    requant_source_file: Optional[str] = None
    vram_budget_mb: float = 0.0
    estimated_model_vram_mb: float = 0.0
    recommended_ctx: int = 4096
    recommended_n_gpu_layers: int = -1  # -1 for all layers on GPU
    flash_attn: bool = True
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    is_cpu_fallback: bool = False
    reasoning: str = ""
    decision_summary: str = ""
    mmproj_file: Optional[str] = None


class AutoSelector:
    """Evaluates hardware and inspected models to determine optimal backends and parameters."""

    def __init__(self, hardware: Optional[HardwareProfile] = None, config: Optional[WorkspaceConfig] = None):
        self.hw = hardware or get_hardware_profile()
        self.config = config or get_config()

    def recommend(
        self,
        inspection: InspectionResult,
        requested_ctx: int = 4096,
        override_quant: Optional[str] = None,
        override_engine: Optional[str] = None,
        override_quantizer: Optional[str] = None,
        task: str = "serve",
        cuda_overhead_mb: float = 1200.0,
        override_n_gpu_layers: Optional[int] = None,
        override_flash_attn: Optional[bool] = None,
        override_cache_type_k: Optional[str] = None,
        override_cache_type_v: Optional[str] = None,
        custom_args: Optional[Dict[str, Any]] = None,
    ) -> SelectionRecommendation:
        """
        Analyzes model architecture, precision, and VRAM budget to recommend
        the optimal inference engine, quantizer, and server parameters.
        Strictly respects explicit user overrides when specified.
        """
        primary_gpu = self.hw.primary_gpu
        has_gpu = self.hw.has_gpu and primary_gpu is not None

        # GPU capability profiling
        supports_fa = primary_gpu.supports_flash_attn if primary_gpu else False
        supports_kvq = primary_gpu.supports_kv_quant if primary_gpu else False
        cache_type_k = "q8_0" if supports_kvq else "f16"
        cache_type_v = "q8_0" if supports_kvq else "f16"

        cc = primary_gpu.compute_capability if primary_gpu else (0, 0)
        gpu_name = (primary_gpu.name or "").lower() if primary_gpu else ""
        is_modern_gpu = (cc >= (8, 9)) or any(k in gpu_name for k in ("l4", "a100", "h100", "rtx 40", "rtx 50", "ada"))

        # Explicit user overrides
        user_engine = override_engine if (override_engine and override_engine.lower() != "auto") else None
        user_quantizer = override_quantizer if (override_quantizer and override_quantizer.lower() != "auto") else None
        user_quant = override_quant if (override_quant and override_quant.lower() != "auto") else None
        user_n_gpu_layers = override_n_gpu_layers
        user_flash_attn = override_flash_attn
        if override_cache_type_k and override_cache_type_k.lower() != "auto":
            cache_type_k = override_cache_type_k
        if override_cache_type_v and override_cache_type_v.lower() != "auto":
            cache_type_v = override_cache_type_v
        if user_flash_attn is not None:
            supports_fa = user_flash_attn

        if not has_gpu:
            return self._recommend_cpu_fallback(
                inspection=inspection,
                requested_ctx=requested_ctx,
                user_engine=user_engine,
                user_quantizer=user_quantizer,
                user_quant=user_quant,
                user_n_gpu_layers=user_n_gpu_layers,
                user_flash_attn=user_flash_attn,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v
            )

        free_vram_mb = primary_gpu.free_vram_mb
        from .hardware import HardwareProfiler
        vram_budget_mb = HardwareProfiler.calculate_vram_budget_mb(
            free_vram_mb=free_vram_mb,
            cuda_overhead_mb=cuda_overhead_mb,
            ctx_size=requested_ctx,
            kv_cache_quant=cache_type_k
        )

        variants = inspection.gguf_variants

        # ---------------------------------------------------------
        # Engine & Converter Smart Routing Logic
        # ---------------------------------------------------------
        recommended_engine = "llama.cpp"
        recommended_converter = "llama.cpp"
        reasoning = ""

        # Case 1: Audio Models (Whisper)
        if inspection.model_type == "audio":
            recommended_engine = user_engine or "faster-whisper"
            recommended_converter = user_quantizer or "ct2"
            reasoning = "Whisper audio model detected. CTranslate2 provides optimized INT8/FP16 audio transcription."

        # Case 2: Diffusion Models
        elif inspection.model_type == "diffusion":
            recommended_engine = user_engine or "diffusers"
            recommended_converter = user_quantizer or None
            reasoning = "Diffusion model detected. Diffusers engine recommended for image generation."

        # Case 3: ik_llama non-uniform quants OR oversized MoE GGUF
        elif inspection.is_ik_quant or (
            inspection.is_moe and variants and (min(v.total_size_bytes for v in variants) / (1024 * 1024) > vram_budget_mb)
        ):
            recommended_engine = user_engine or "ik_llama"
            recommended_converter = user_quantizer or "ik_llama"
            reasoning = (
                "Non-uniform IK quant or oversized MoE architecture detected. "
                "ik_llama.cpp supports IQ non-uniform tensors and tensor offload (-ot) for hybrid CPU+GPU execution."
            )

        # Case 4: Standard GGUF / VLM / Embedding GGUF
        elif variants or inspection.model_type in ("llm-gguf", "vlm-gguf", "embedding-gguf"):
            recommended_engine = user_engine or "llama.cpp"
            recommended_converter = user_quantizer or "llama.cpp"
            reasoning = "GGUF format detected. llama.cpp provides native CUDA offloading with Flash Attention."

        # Case 5: LoRA adapter OR Unsloth repo OR (unquantized fp16 on low-VRAM GPU with conversion goal)
        elif inspection.is_lora or ("unsloth" in inspection.model_id.lower()) or (
            inspection.quant_format == "unquantized_fp16" and not is_modern_gpu and task == "convert_and_quantize"
        ):
            recommended_engine = user_engine or "unsloth"
            recommended_converter = user_quantizer or "unsloth"
            reasoning = (
                "LoRA adapter, unsloth weights, or low-VRAM conversion workflow detected. "
                "unsloth provides memory-efficient 4-bit loading, LoRA merge, and dynamic GGUF export."
            )

        # Case 6: Quantized Safetensors (AWQ, GPTQ, FP8)
        elif inspection.quant_format in ("awq", "gptq", "fp8"):
            if is_modern_gpu:
                recommended_engine = user_engine or "vllm"
                recommended_converter = user_quantizer or ("autoawq" if inspection.quant_format == "awq" else "gptqmodel")
                reasoning = (
                    f"Quantized Safetensors ({inspection.quant_format}) on compute capability {cc[0]}.{cc[1]}. "
                    "vLLM PagedAttention provides high throughput."
                )
            else:
                recommended_engine = user_engine or ("exllamav2" if inspection.quant_format == "gptq" else "transformers")
                recommended_converter = user_quantizer or ("autoawq" if inspection.quant_format == "awq" else "gptqmodel")
                reasoning = (
                    f"Quantized Safetensors ({inspection.quant_format}) on sm_75 architecture. "
                    "Transformers with BitsAndBytes/GPTQ provides reliable memory-safe execution."
                )

        # Case 7: EXL2 format
        elif inspection.quant_format == "exl2":
            recommended_engine = user_engine or "exllamav2"
            recommended_converter = user_quantizer or "exllamav2"
            reasoning = "EXL2 format detected. ExLlamaV2 provides native fractional bpw execution."

        # Case 8: Default unquantized Safetensors
        else:
            if task == "convert_and_quantize":
                recommended_engine = user_engine or "llama.cpp"
                recommended_converter = user_quantizer or "llama.cpp"
                reasoning = "Unquantized Safetensors selected for conversion. llama.cpp convert_hf_to_gguf pipeline recommended."
            else:
                if is_modern_gpu and primary_gpu.total_vram_mb >= 16000:
                    recommended_engine = user_engine or "vllm"
                    recommended_converter = user_quantizer or "llama.cpp"
                    reasoning = "Unquantized Safetensors model on modern GPU. vLLM PagedAttention selected for high throughput."
                else:
                    recommended_engine = user_engine or "transformers"
                    recommended_converter = user_quantizer or "llama.cpp"
                    reasoning = "Unquantized Safetensors model. Transformers with BitsAndBytes 4-bit offload recommended to prevent out-of-memory errors."

        # ---------------------------------------------------------
        # Quantization Variant Selection (for GGUF models)
        # ---------------------------------------------------------
        if not variants:
            arch_name = inspection.architecture_name or inspection.model_type
            summary = (
                f"Architecture: {arch_name} | Format: {inspection.quant_format} | "
                f"Usable VRAM: {vram_budget_mb:.0f} MB | Engine: {recommended_engine} | "
                f"Converter: {recommended_converter or 'none'} | Reason: {reasoning}"
            )
            n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else -1
            return SelectionRecommendation(
                source_model_id=inspection.model_id,
                selected_variant=None,
                recommended_quant=user_quant or "N/A",
                recommended_engine=recommended_engine,
                recommended_converter=recommended_converter,
                vram_budget_mb=vram_budget_mb,
                recommended_ctx=requested_ctx,
                recommended_n_gpu_layers=n_layers,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                reasoning=reasoning,
                decision_summary=summary,
            )

        # If user explicitly requested a quant variant
        if user_quant:
            req_upper = user_quant.upper()
            matched = [v for v in variants if v.quant.upper() == req_upper]
            if matched:
                v = matched[0]
                model_mb = v.total_size_bytes / (1024 * 1024)
                fits = model_mb <= vram_budget_mb
                if user_n_gpu_layers is not None:
                    n_layers = user_n_gpu_layers
                else:
                    n_layers = -1 if fits else self._calculate_partial_layers(model_mb, vram_budget_mb)
                summary = (
                    f"Architecture: {inspection.architecture_name or 'GGUF'} | Quant: {v.quant} (Override) | "
                    f"Usable VRAM: {vram_budget_mb:.0f} MB (Model: {model_mb:.0f} MB) | "
                    f"Engine: {recommended_engine} | Converter: {recommended_converter} | "
                    f"Reason: User override '{user_quant}' selected. {'Fits completely in VRAM.' if fits else 'Exceeds VRAM budget, partial GPU offload applied.'}"
                )
                return SelectionRecommendation(
                    source_model_id=inspection.model_id,
                    selected_variant=v,
                    recommended_quant=v.quant,
                    recommended_engine=recommended_engine,
                    recommended_converter=recommended_converter,
                    vram_budget_mb=vram_budget_mb,
                    estimated_model_vram_mb=model_mb,
                    recommended_ctx=requested_ctx if fits else max(2048, requested_ctx // 2),
                    recommended_n_gpu_layers=n_layers,
                    flash_attn=supports_fa,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                    mmproj_file=v.mmproj_file,
                    reasoning=f"User override '{user_quant}' selected.",
                    decision_summary=summary,
                )
            else:
                # User requested a specific quant variant not present in repo -> trigger re-quantization!
                smallest_existing = min(variants, key=lambda v: v.total_size_bytes)
                n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else -1
                summary = (
                    f"Architecture: {inspection.architecture_name or 'GGUF'} | Re-quant Target: {user_quant} (Override) | "
                    f"Usable VRAM: {vram_budget_mb:.0f} MB | Engine: {recommended_engine} | "
                    f"Converter: {recommended_converter} | "
                    f"Reason: User override '{user_quant}' not found in repo. Scheduled re-quantization from {smallest_existing.quant}."
                )
                return SelectionRecommendation(
                    source_model_id=inspection.model_id,
                    selected_variant=smallest_existing,
                    recommended_quant=user_quant,
                    recommended_engine=recommended_engine,
                    recommended_converter=recommended_converter,
                    needs_requant=True,
                    requant_target_quant=user_quant,
                    requant_source_file=smallest_existing.shard_files[0],
                    vram_budget_mb=vram_budget_mb,
                    estimated_model_vram_mb=smallest_existing.total_size_bytes / (1024 * 1024),
                    recommended_ctx=requested_ctx,
                    recommended_n_gpu_layers=n_layers,
                    flash_attn=supports_fa,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                    mmproj_file=smallest_existing.mmproj_file,
                    reasoning=f"User requested quant '{user_quant}' not present in repo. Re-quantization scheduled.",
                    decision_summary=summary,
                )

        # Find highest quality variant fitting in VRAM
        fitting_variants = [v for v in variants if (v.total_size_bytes / (1024 * 1024)) <= vram_budget_mb]
        if fitting_variants:
            best = fitting_variants[0]
            model_mb = best.total_size_bytes / (1024 * 1024)
            n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else -1
            summary = (
                f"Architecture: {inspection.architecture_name or 'GGUF'} | Quant: {best.quant} | "
                f"Usable VRAM: {vram_budget_mb:.0f} MB (Model: {model_mb:.0f} MB) | "
                f"Engine: {recommended_engine} | Converter: {recommended_converter} | "
                f"Reason: Highest quality fitting quant {best.quant} fits within usable VRAM."
            )
            return SelectionRecommendation(
                source_model_id=inspection.model_id,
                selected_variant=best,
                recommended_quant=best.quant,
                recommended_engine=recommended_engine,
                recommended_converter=recommended_converter,
                vram_budget_mb=vram_budget_mb,
                estimated_model_vram_mb=model_mb,
                recommended_ctx=requested_ctx,
                recommended_n_gpu_layers=n_layers,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                mmproj_file=best.mmproj_file,
                reasoning=f"Highest quality fitting quant: {best.quant} ({best.total_size_gb:.1f}GB <= {vram_budget_mb/1024:.1f}GB usable VRAM).",
                decision_summary=summary,
            )

        # If no variant fits: check re-quantization
        smallest_existing = min(variants, key=lambda v: v.total_size_bytes)
        smallest_mb = smallest_existing.total_size_bytes / (1024 * 1024)
        requant_target = self._calculate_highest_fitting_quant(
            smallest_existing.total_size_bytes,
            smallest_existing.quant,
            vram_budget_mb
        )

        if requant_target:
            n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else -1
            summary = (
                f"Architecture: {inspection.architecture_name or 'GGUF'} | Re-quant Target: {requant_target} | "
                f"Usable VRAM: {vram_budget_mb:.0f} MB | Engine: {recommended_engine} | "
                f"Converter: {recommended_converter} | "
                f"Reason: Existing quants exceed VRAM. Re-quantizing to {requant_target} will fit inside usable VRAM."
            )
            return SelectionRecommendation(
                source_model_id=inspection.model_id,
                selected_variant=smallest_existing,
                recommended_quant=requant_target,
                recommended_engine=recommended_engine,
                recommended_converter=recommended_converter,
                needs_requant=True,
                requant_target_quant=requant_target,
                requant_source_file=smallest_existing.shard_files[0],
                vram_budget_mb=vram_budget_mb,
                estimated_model_vram_mb=smallest_mb,
                recommended_ctx=requested_ctx,
                recommended_n_gpu_layers=n_layers,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                mmproj_file=smallest_existing.mmproj_file,
                reasoning=f"All available quants exceed VRAM. Re-quantizing to {requant_target} will fit 100% inside usable VRAM.",
                decision_summary=summary,
            )

        # Fallback: Partial layer offload
        partial_layers = user_n_gpu_layers if user_n_gpu_layers is not None else self._calculate_partial_layers(smallest_mb, vram_budget_mb)
        safe_ctx = 2048 if requested_ctx > 2048 else requested_ctx
        summary = (
            f"Architecture: {inspection.architecture_name or 'GGUF'} | Quant: {smallest_existing.quant} | "
            f"Usable VRAM: {vram_budget_mb:.0f} MB (Model: {smallest_mb:.0f} MB) | "
            f"Engine: {recommended_engine} | Converter: {recommended_converter} | "
            f"Reason: Model exceeds VRAM budget. Partial GPU offload ({partial_layers} layers) applied to prevent out-of-memory errors."
        )
        return SelectionRecommendation(
            source_model_id=inspection.model_id,
            selected_variant=smallest_existing,
            recommended_quant=smallest_existing.quant,
            recommended_engine=recommended_engine,
            recommended_converter=recommended_converter,
            vram_budget_mb=vram_budget_mb,
            estimated_model_vram_mb=smallest_mb,
            recommended_ctx=safe_ctx,
            recommended_n_gpu_layers=partial_layers,
            flash_attn=supports_fa,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            mmproj_file=smallest_existing.mmproj_file,
            reasoning=f"Model exceeds VRAM. Partial offload ({partial_layers} GPU layers) and safe context ({safe_ctx}) applied.",
            decision_summary=summary,
        )

    def _recommend_cpu_fallback(
        self,
        inspection: InspectionResult,
        requested_ctx: int,
        user_engine: Optional[str] = None,
        user_quantizer: Optional[str] = None,
        user_quant: Optional[str] = None,
        user_n_gpu_layers: Optional[int] = None,
        user_flash_attn: Optional[bool] = None,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
    ) -> SelectionRecommendation:
        """Generates CPU-only recommendation."""
        variants = inspection.gguf_variants
        selected = None
        for q in ["Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"]:
            matched = [v for v in variants if v.quant.upper() == q]
            if matched:
                selected = matched[0]
                break
        if not selected and variants:
            selected = variants[-1]

        quant = user_quant or (selected.quant if selected else "Q4_0")
        engine = user_engine or ("faster-whisper" if inspection.model_type == "audio" else "llama.cpp")
        converter = user_quantizer or ("ct2" if inspection.model_type == "audio" else "llama.cpp")
        n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else 0
        fa = user_flash_attn if user_flash_attn is not None else False
        summary = (
            f"Architecture: {inspection.architecture_name or inspection.model_type} | Quant: {quant} | "
            f"Hardware: CPU Mode | Engine: {engine} | Converter: {converter} | Reason: No GPU detected."
        )

        return SelectionRecommendation(
            source_model_id=inspection.model_id,
            selected_variant=selected,
            recommended_quant=quant,
            recommended_engine=engine,
            recommended_converter=converter,
            vram_budget_mb=0.0,
            recommended_ctx=min(requested_ctx, 2048),
            recommended_n_gpu_layers=n_layers,
            flash_attn=fa,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            is_cpu_fallback=True,
            reasoning="No GPU detected. Running in CPU fallback mode (0 GPU layers).",
            decision_summary=summary,
        )

    @staticmethod
    def _calculate_highest_fitting_quant(source_bytes: int, source_quant: str, vram_budget_mb: float) -> Optional[str]:
        source_quant_upper = source_quant.upper()
        if source_quant_upper not in ["F32", "F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M"]:
            return None

        source_factor = QUANT_SIZE_FACTORS.get(source_quant_upper, 1.0)
        approx_f16_mb = (source_bytes / (1024 * 1024)) / source_factor

        candidate_quants = ["Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "IQ4_XS", "Q3_K_M", "Q2_K"]
        try:
            source_idx = candidate_quants.index(source_quant_upper)
            candidates = candidate_quants[source_idx + 1:]
        except ValueError:
            candidates = candidate_quants

        for q in candidates:
            est_mb = approx_f16_mb * QUANT_SIZE_FACTORS.get(q, 0.3)
            if est_mb <= vram_budget_mb:
                return q
        return None

    @staticmethod
    def _calculate_partial_layers(model_mb: float, vram_budget_mb: float, total_layers_est: int = 32) -> int:
        if model_mb <= 0:
            return -1
        fraction = min(1.0, max(0.0, vram_budget_mb / model_mb))
        safe_layers = int(total_layers_est * fraction * 0.90)
        return max(1, safe_layers)

    @classmethod
    def execute_requantize(
        cls,
        source_path: Path,
        target_quant: str,
        output_path: Optional[Path] = None,
        backup_to_drive: bool = True,
        cleanup_source: bool = True
    ) -> Path:
        """Executes quantization via LlamaCppConverter and handles persistence."""
        from .converters import get_converter
        cfg = get_config()
        converter = get_converter("llama.cpp", cfg)

        out_file = converter.convert_or_quantize(
            input_path=source_path,
            output_path=output_path,
            target_quant=target_quant,
            cleanup_source=cleanup_source
        )

        if backup_to_drive and cfg.models_gguf_dir.is_dir():
            drive_dest = cfg.models_gguf_dir / out_file.name
            print(f"[AutoSelector] Backing up re-quantized model to Drive: {drive_dest}...")
            shutil.copy2(str(out_file), str(drive_dest))

        return out_file
