"""
Intelligent Auto-Selector & Capability-Driven Analysis Engine
Computes architecture-aware VRAM budgets, supports selection strategies (balanced, max_quality,
max_speed, max_context, low_vram_safe), routes models to optimal engines and converters,
evaluates non-uniform ik_llama quants, MoE hybrid offloading, and auto convert-and-serve.
"""

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from .config import WorkspaceConfig, get_config
from .hardware import HardwareProfile, HardwareProfiler, get_hardware_profile
from .model_inspector import GGUFVariant, InspectionResult, QUANT_ORDER, QUANT_SIZE_FACTORS


@dataclass
class SelectionRecommendation:
    source_model_id: str
    selected_variant: Optional[GGUFVariant]
    recommended_quant: str
    recommended_engine: str = "llama.cpp"
    recommended_converter: Optional[str] = None
    needs_requant: bool = False
    needs_convert_to_gguf: bool = False
    requant_target_quant: Optional[str] = None
    requant_source_file: Optional[str] = None
    vram_budget_mb: float = 0.0
    estimated_model_vram_mb: float = 0.0
    recommended_ctx: int = 4096
    recommended_n_gpu_layers: int = -1  # -1 for all layers on GPU
    flash_attn: bool = True
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    parallel: int = 1
    quantization_mode: str = "auto"
    is_cpu_fallback: bool = False
    reasoning: str = ""
    decision_summary: str = ""
    mmproj_file: Optional[str] = None
    suggested_extra_args: Dict[str, Any] = field(default_factory=dict)
    budget_breakdown: Dict[str, Any] = field(default_factory=dict)


class AutoSelector:
    """Evaluates hardware and inspected models to determine optimal backends and parameters."""

    VALID_STRATEGIES = {"balanced", "max_quality", "max_speed", "max_context", "low_vram_safe"}

    def __init__(self, hardware: Optional[HardwareProfile] = None, config: Optional[WorkspaceConfig] = None):
        self.hw = hardware or get_hardware_profile()
        self.config = config or get_config()

    def _resolve_auto_context(
        self,
        requested_ctx: Union[int, str, None],
        inspection: InspectionResult,
        free_vram_mb: float,
        strategy: str,
    ) -> int:
        """Resolves requested context size, handling 'auto' or 0 intelligently."""
        if isinstance(requested_ctx, int) and requested_ctx > 0:
            return requested_ctx
        if isinstance(requested_ctx, str) and requested_ctx.strip().lower() != "auto":
            try:
                val = int(requested_ctx.strip())
                if val > 0:
                    return val
            except ValueError:
                pass

        # Auto context selection based on VRAM, model max_context_length, and strategy
        model_max = inspection.max_context_length or 32768
        if strategy == "max_context":
            target = 16384 if free_vram_mb >= 14000 else 8192
        elif strategy == "low_vram_safe":
            target = 2048
        elif free_vram_mb >= 20000:
            target = 8192
        elif free_vram_mb >= 10000:
            target = 4096
        else:
            target = 2048
        return min(target, model_max)

    def recommend(
        self,
        inspection: InspectionResult,
        requested_ctx: Union[int, str, None] = 4096,
        override_quant: Optional[str] = None,
        override_engine: Optional[str] = None,
        override_quantizer: Optional[str] = None,
        task: str = "serve",
        cuda_overhead_mb: Optional[float] = None,
        override_n_gpu_layers: Optional[int] = None,
        override_flash_attn: Optional[bool] = None,
        override_cache_type_k: Optional[str] = None,
        override_cache_type_v: Optional[str] = None,
        override_gpu_offload_mode: Optional[str] = None,
        override_parallel: Optional[int] = None,
        override_quantization_mode: Optional[str] = None,
        override_mmproj_file: Optional[str] = None,
        strategy: Optional[str] = None,
        custom_args: Optional[Dict[str, Any]] = None,
    ) -> SelectionRecommendation:
        """
        Analyzes model architecture, precision, strategy, and VRAM budget to recommend
        the optimal inference engine, quantizer, and server parameters.
        Strictly respects explicit user overrides when specified.
        """
        active_strategy = (strategy or self.config.settings.selection_strategy or "balanced").lower().strip()
        if active_strategy not in self.VALID_STRATEGIES:
            active_strategy = "balanced"

        overhead_mb = (
            cuda_overhead_mb
            if cuda_overhead_mb is not None
            else float(self.config.settings.cuda_overhead_mb)
        )
        safety_margin = float(self.config.settings.vram_safety_margin_pct or 0.0)
        if active_strategy == "low_vram_safe":
            safety_margin = max(safety_margin, 0.12)

        primary_gpu = self.hw.primary_gpu
        has_gpu = self.hw.has_gpu and primary_gpu is not None
        parallel_slots = max(1, int(override_parallel or 1))

        # GPU capability profiling
        supports_fa = primary_gpu.supports_flash_attn if primary_gpu else False
        supports_kvq = primary_gpu.supports_kv_quant if primary_gpu else False

        if supports_kvq:
            if active_strategy in ("max_context", "low_vram_safe"):
                cache_type_k, cache_type_v = "q4_0", "q4_0"
            elif active_strategy == "max_quality" and primary_gpu and primary_gpu.total_vram_mb >= 24000:
                cache_type_k, cache_type_v = "f16", "f16"
            else:
                cache_type_k, cache_type_v = "q8_0", "q8_0"
        else:
            cache_type_k, cache_type_v = "f16", "f16"

        cc = primary_gpu.compute_capability if primary_gpu else (0, 0)
        gpu_name = (primary_gpu.name or "").lower() if primary_gpu else ""
        is_modern_gpu = (cc >= (8, 9)) or any(k in gpu_name for k in ("l4", "a100", "h100", "h200", "rtx 40", "rtx 50", "ada"))

        # Explicit user overrides
        user_engine = override_engine.strip() if (override_engine and override_engine.strip().lower() != "auto") else None
        user_quantizer = override_quantizer.strip() if (override_quantizer and override_quantizer.strip().lower() != "auto") else None
        user_quant = override_quant.strip() if (override_quant and override_quant.strip().lower() != "auto") else None

        offload_mode = (override_gpu_offload_mode or "auto").lower().strip()
        if offload_mode == "all":
            user_n_gpu_layers: Optional[int] = -1
        elif offload_mode == "cpu":
            user_n_gpu_layers = 0
        else:
            user_n_gpu_layers = override_n_gpu_layers

        user_flash_attn = override_flash_attn
        if override_cache_type_k and override_cache_type_k.lower() != "auto":
            cache_type_k = override_cache_type_k
        if override_cache_type_v and override_cache_type_v.lower() != "auto":
            cache_type_v = override_cache_type_v
        if user_flash_attn is not None:
            supports_fa = user_flash_attn

        free_vram_mb = self.hw.total_free_vram_mb if has_gpu else 0.0
        resolved_ctx = self._resolve_auto_context(requested_ctx, inspection, free_vram_mb, active_strategy)

        if not has_gpu:
            return self._recommend_cpu_fallback(
                inspection=inspection,
                requested_ctx=resolved_ctx,
                user_engine=user_engine,
                user_quantizer=user_quantizer,
                user_quant=user_quant,
                user_n_gpu_layers=user_n_gpu_layers,
                user_flash_attn=user_flash_attn,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                parallel=parallel_slots,
                task=task,
            )

        kv_est_mb = HardwareProfiler.calculate_kv_cache_mb(
            ctx_size=resolved_ctx,
            kv_cache_quant=cache_type_k,
            config_json=inspection.config_json,
            estimated_params_b=inspection.estimated_params_b,
            parallel=parallel_slots,
        )
        vram_budget_mb = HardwareProfiler.calculate_vram_budget_mb(
            free_vram_mb=free_vram_mb,
            cuda_overhead_mb=overhead_mb,
            ctx_size=resolved_ctx,
            kv_cache_quant=cache_type_k,
            config_json=inspection.config_json,
            estimated_params_b=inspection.estimated_params_b,
            parallel=parallel_slots,
            safety_margin_pct=safety_margin,
        )

        budget_info = {
            "strategy": active_strategy,
            "free_vram_mb": round(free_vram_mb, 1),
            "cuda_overhead_mb": round(overhead_mb, 1),
            "estimated_kv_cache_mb": round(kv_est_mb, 1),
            "safety_margin_pct": safety_margin,
            "usable_vram_mb": round(vram_budget_mb, 1),
            "ctx_size": resolved_ctx,
            "parallel": parallel_slots,
        }

        variants = inspection.gguf_variants

        # ---------------------------------------------------------
        # Engine & Converter Smart Routing Logic
        # ---------------------------------------------------------
        recommended_engine = "llama.cpp"
        recommended_converter: Optional[str] = "llama.cpp"
        quant_mode = override_quantization_mode or "auto"
        needs_convert_to_gguf = False
        suggested_extra: Dict[str, Any] = {}
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
            self.config.settings.prefer_ik_llama_for_moe
            and inspection.is_moe
            and variants
            and (min(v.total_size_bytes for v in variants) / (1024 * 1024) > vram_budget_mb)
        ):
            recommended_engine = user_engine or "ik_llama"
            recommended_converter = user_quantizer or "ik_llama"
            if inspection.is_moe and variants and (min(v.total_size_bytes for v in variants) / (1024 * 1024) > vram_budget_mb):
                suggested_extra["override_tensor"] = "ffn_.*_exps.=CPU"
            reasoning = (
                "Non-uniform IK quant or oversized MoE architecture detected. "
                "ik_llama.cpp supports IQ non-uniform tensors and hybrid MoE CPU+GPU tensor offload (-ot)."
            )

        # Case 4: Standard GGUF / VLM / Embedding GGUF
        elif variants or inspection.model_type in ("llm-gguf", "vlm-gguf", "embedding-gguf"):
            recommended_engine = user_engine or "llama.cpp"
            recommended_converter = user_quantizer or "llama.cpp"
            reasoning = "GGUF format detected. llama.cpp provides native CUDA offloading with Flash Attention."

        # Case 5: LoRA adapter OR Unsloth repo
        elif inspection.is_lora or ("unsloth" in inspection.model_id.lower()):
            recommended_engine = user_engine or "unsloth"
            recommended_converter = user_quantizer or "unsloth"
            reasoning = (
                "LoRA adapter or Unsloth repository detected. "
                "Unsloth provides memory-efficient 4-bit loading, LoRA merge, and Dynamic GGUF export."
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
                    f"Quantized Safetensors ({inspection.quant_format}) on sm_{cc[0]}{cc[1]} architecture. "
                    f"{recommended_engine} provides reliable memory-safe execution."
                )

        # Case 7: EXL2 format
        elif inspection.quant_format == "exl2":
            recommended_engine = user_engine or "exllamav2"
            recommended_converter = user_quantizer or "exllamav2"
            reasoning = "EXL2 format detected. ExLlamaV2 provides native fractional bpw execution."

        # Case 8: Default unquantized Safetensors
        else:
            if task == "convert_and_quantize":
                if user_quant and ("XL" in user_quant.upper() or "UD-" in user_quant.upper()):
                    recommended_converter = user_quantizer or "unsloth"
                    recommended_engine = user_engine or "llama.cpp"
                else:
                    recommended_engine = user_engine or "llama.cpp"
                    recommended_converter = user_quantizer or "llama.cpp"
                reasoning = f"Unquantized Safetensors selected for conversion via {recommended_converter}."
            else:
                if user_engine in ("llama.cpp", "ik_llama", "ik_llama.cpp") or task == "convert_and_serve":
                    recommended_engine = user_engine or "llama.cpp"
                    recommended_converter = user_quantizer or (
                        "ik_llama" if recommended_engine in ("ik_llama", "ik_llama.cpp") else "llama.cpp"
                    )
                    needs_convert_to_gguf = True
                    reasoning = (
                        f"Safetensors model targeted for GGUF engine '{recommended_engine}'. "
                        f"Will convert to GGUF via {recommended_converter} before serving."
                    )
                elif is_modern_gpu and primary_gpu.total_vram_mb >= 16000 and active_strategy != "low_vram_safe":
                    recommended_engine = user_engine or "vllm"
                    recommended_converter = user_quantizer or "llama.cpp"
                    reasoning = "Unquantized Safetensors model on modern GPU. vLLM PagedAttention selected for high throughput."
                else:
                    recommended_engine = user_engine or "transformers"
                    recommended_converter = user_quantizer or "llama.cpp"
                    if quant_mode == "auto":
                        quant_mode = "bnb_4bit"
                    reasoning = "Unquantized Safetensors model. Transformers with BitsAndBytes 4-bit offload recommended to prevent out-of-memory errors."

        # Normalize convert_and_serve flag if user requested conversion + serving
        if task == "convert_and_serve" and not variants and inspection.model_type == "llm-safetensors":
            needs_convert_to_gguf = True

        # ---------------------------------------------------------
        # Non-GGUF Models (Safetensors, Diffusion, Audio, EXL2)
        # ---------------------------------------------------------
        if not variants:
            target_q = user_quant or "N/A"
            if needs_convert_to_gguf and (not user_quant):
                repo_mb = (inspection.total_repo_size_bytes or 0) / (1024 * 1024)
                if repo_mb > 0:
                    target_q = self._calculate_highest_fitting_quant(
                        inspection.total_repo_size_bytes,
                        "F16",
                        vram_budget_mb,
                        prefer_ik_quants=(recommended_converter == "ik_llama"),
                    ) or "Q4_K_M"
                else:
                    target_q = "Q4_K_M"

            arch_name = inspection.architecture_name or inspection.model_type
            summary = (
                f"Strategy: {active_strategy} | Architecture: {arch_name} | Format: {inspection.quant_format} | "
                f"Usable VRAM: {vram_budget_mb:.0f} MB | Engine: {recommended_engine} | "
                f"Converter: {recommended_converter or 'none'} | Quant: {target_q} | Reason: {reasoning}"
            )
            n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else -1
            return SelectionRecommendation(
                source_model_id=inspection.model_id,
                selected_variant=None,
                recommended_quant=target_q,
                recommended_engine=recommended_engine,
                recommended_converter=recommended_converter,
                needs_convert_to_gguf=needs_convert_to_gguf,
                vram_budget_mb=vram_budget_mb,
                recommended_ctx=resolved_ctx,
                recommended_n_gpu_layers=n_layers,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                parallel=parallel_slots,
                quantization_mode=quant_mode,
                reasoning=reasoning,
                decision_summary=summary,
                suggested_extra_args=suggested_extra,
                budget_breakdown=budget_info,
            )

        # ---------------------------------------------------------
        # Quantization Variant Selection (GGUF Models)
        # ---------------------------------------------------------
        def pick_mmproj(v: GGUFVariant) -> Optional[str]:
            return override_mmproj_file or v.mmproj_file

        if user_quant:
            req_upper = user_quant.upper()
            matched = [v for v in variants if v.quant.upper() == req_upper]
            if matched:
                v = matched[0]
                model_mb = v.total_size_bytes / (1024 * 1024)
                fits = model_mb <= vram_budget_mb
                if user_n_gpu_layers is not None:
                    n_layers = user_n_gpu_layers
                elif not self.config.settings.auto_partial_offload:
                    n_layers = -1
                else:
                    n_layers = -1 if fits else self._calculate_partial_layers(model_mb, vram_budget_mb)
                summary = (
                    f"Strategy: {active_strategy} | Architecture: {inspection.architecture_name or 'GGUF'} | Quant: {v.quant} (Override) | "
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
                    recommended_ctx=resolved_ctx if fits else max(2048, resolved_ctx // 2),
                    recommended_n_gpu_layers=n_layers,
                    flash_attn=supports_fa,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                    parallel=parallel_slots,
                    quantization_mode=quant_mode,
                    mmproj_file=pick_mmproj(v),
                    reasoning=f"User override '{user_quant}' selected.",
                    decision_summary=summary,
                    suggested_extra_args=suggested_extra,
                    budget_breakdown=budget_info,
                )
            else:
                smallest_existing = min(variants, key=lambda v: v.total_size_bytes)
                n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else -1
                summary = (
                    f"Strategy: {active_strategy} | Architecture: {inspection.architecture_name or 'GGUF'} | Re-quant Target: {user_quant} (Override) | "
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
                    recommended_ctx=resolved_ctx,
                    recommended_n_gpu_layers=n_layers,
                    flash_attn=supports_fa,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                    parallel=parallel_slots,
                    quantization_mode=quant_mode,
                    mmproj_file=pick_mmproj(smallest_existing),
                    reasoning=f"User requested quant '{user_quant}' not present in repo. Re-quantization scheduled.",
                    decision_summary=summary,
                    suggested_extra_args=suggested_extra,
                    budget_breakdown=budget_info,
                )

        # Find fitting variants ordered by strategy
        fitting_variants = [v for v in variants if (v.total_size_bytes / (1024 * 1024)) <= vram_budget_mb]
        if fitting_variants:
            if active_strategy == "max_speed":
                # Prefer fast Q4_K_M / Q4_0 / Q5_K_M over bulky Q8_0 for speed
                preferred_speed = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q6_K", "Q8_0"]
                best = fitting_variants[0]
                for pq in preferred_speed:
                    match_s = [fv for fv in fitting_variants if fv.quant.upper() == pq]
                    if match_s:
                        best = match_s[0]
                        break
            else:
                # balanced, max_quality, max_context, low_vram_safe -> highest quality that fits cleanly
                best = fitting_variants[0]

            model_mb = best.total_size_bytes / (1024 * 1024)
            n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else -1
            summary = (
                f"Strategy: {active_strategy} | Architecture: {inspection.architecture_name or 'GGUF'} | Quant: {best.quant} | "
                f"Usable VRAM: {vram_budget_mb:.0f} MB (Model: {model_mb:.0f} MB) | "
                f"Engine: {recommended_engine} | Converter: {recommended_converter} | "
                f"Reason: Selected fitting quant {best.quant} within usable VRAM."
            )
            return SelectionRecommendation(
                source_model_id=inspection.model_id,
                selected_variant=best,
                recommended_quant=best.quant,
                recommended_engine=recommended_engine,
                recommended_converter=recommended_converter,
                vram_budget_mb=vram_budget_mb,
                estimated_model_vram_mb=model_mb,
                recommended_ctx=resolved_ctx,
                recommended_n_gpu_layers=n_layers,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                parallel=parallel_slots,
                quantization_mode=quant_mode,
                mmproj_file=pick_mmproj(best),
                reasoning=f"Selected {best.quant} ({best.total_size_gb:.1f}GB <= {vram_budget_mb/1024:.1f}GB usable VRAM) under '{active_strategy}' strategy.",
                decision_summary=summary,
                suggested_extra_args=suggested_extra,
                budget_breakdown=budget_info,
            )

        # If no variant fits: check smart re-quantization
        smallest_existing = min(variants, key=lambda v: v.total_size_bytes)
        smallest_mb = smallest_existing.total_size_bytes / (1024 * 1024)
        requant_target = None
        if self.config.settings.auto_requantize:
            requant_target = self._calculate_highest_fitting_quant(
                smallest_existing.total_size_bytes,
                smallest_existing.quant,
                vram_budget_mb,
                prefer_ik_quants=(recommended_converter == "ik_llama"),
            )

        if requant_target:
            n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else -1
            summary = (
                f"Strategy: {active_strategy} | Architecture: {inspection.architecture_name or 'GGUF'} | Re-quant Target: {requant_target} | "
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
                recommended_ctx=resolved_ctx,
                recommended_n_gpu_layers=n_layers,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                parallel=parallel_slots,
                quantization_mode=quant_mode,
                mmproj_file=pick_mmproj(smallest_existing),
                reasoning=f"All available quants exceed VRAM. Re-quantizing to {requant_target} will fit 100% inside usable VRAM.",
                decision_summary=summary,
                suggested_extra_args=suggested_extra,
                budget_breakdown=budget_info,
            )

        # Fallback: Partial layer offload
        partial_layers = (
            user_n_gpu_layers
            if user_n_gpu_layers is not None
            else (self._calculate_partial_layers(smallest_mb, vram_budget_mb) if self.config.settings.auto_partial_offload else -1)
        )
        safe_ctx = 2048 if resolved_ctx > 2048 else resolved_ctx
        summary = (
            f"Strategy: {active_strategy} | Architecture: {inspection.architecture_name or 'GGUF'} | Quant: {smallest_existing.quant} | "
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
            parallel=parallel_slots,
            quantization_mode=quant_mode,
            mmproj_file=pick_mmproj(smallest_existing),
            reasoning=f"Model exceeds VRAM. Partial offload ({partial_layers} GPU layers) and safe context ({safe_ctx}) applied.",
            decision_summary=summary,
            suggested_extra_args=suggested_extra,
            budget_breakdown=budget_info,
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
        parallel: int = 1,
        task: str = "serve",
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
        engine = user_engine or ("faster-whisper" if inspection.model_type == "audio" else ("diffusers" if inspection.model_type == "diffusion" else "llama.cpp"))
        converter = user_quantizer or ("ct2" if inspection.model_type == "audio" else "llama.cpp")
        n_layers = user_n_gpu_layers if user_n_gpu_layers is not None else 0
        fa = user_flash_attn if user_flash_attn is not None else False
        needs_convert = (not variants) and (inspection.model_type == "llm-safetensors") and (
            engine in ("llama.cpp", "ik_llama") or task == "convert_and_serve"
        )
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
            needs_convert_to_gguf=needs_convert,
            vram_budget_mb=0.0,
            recommended_ctx=min(requested_ctx, 2048),
            recommended_n_gpu_layers=n_layers,
            flash_attn=fa,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            parallel=parallel,
            quantization_mode="fp32",
            is_cpu_fallback=True,
            mmproj_file=selected.mmproj_file if selected else None,
            reasoning="No GPU detected. Running in CPU fallback mode (0 GPU layers).",
            decision_summary=summary,
        )

    @staticmethod
    def _calculate_highest_fitting_quant(
        source_bytes: int,
        source_quant: str,
        vram_budget_mb: float,
        prefer_ik_quants: bool = False,
    ) -> Optional[str]:
        source_quant_upper = source_quant.upper()
        if source_quant_upper not in [
            "F32", "F16", "BF16", "Q8_0", "Q6_K_L", "Q6_K", "Q5_K_L", "Q5_K_M", "Q5_K_S"
        ]:
            return None

        source_factor = QUANT_SIZE_FACTORS.get(source_quant_upper, 1.0)
        approx_f16_mb = (source_bytes / (1024 * 1024)) / source_factor

        if prefer_ik_quants:
            candidate_quants = ["IQ6_K", "IQ5_K", "IQ4_K", "IQ4_KS", "IQ4_XS", "IQ3_K", "IQ3_KS", "IQ2_K", "IQ2_KS"]
        else:
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
        cleanup_source: bool = True,
        converter_name: str = "llama.cpp",
        config: Optional[WorkspaceConfig] = None,
        use_imatrix: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Executes quantization via the specified converter plugin and handles persistence."""
        from .converters import get_converter
        cfg = config or get_config()
        converter = get_converter(converter_name or "llama.cpp", cfg)

        out_file = converter.convert_or_quantize(
            input_path=source_path,
            output_path=output_path,
            target_quant=target_quant,
            cleanup_source=cleanup_source,
            use_imatrix=use_imatrix,
            **kwargs,
        )

        if backup_to_drive and cfg.models_gguf_dir.is_dir() and out_file.is_file():
            drive_dest = cfg.models_gguf_dir / out_file.name
            if out_file.resolve() != drive_dest.resolve():
                print(f"[AutoSelector] Backing up re-quantized model to Drive: {drive_dest}...")
                shutil.copy2(str(out_file), str(drive_dest))

        return out_file
