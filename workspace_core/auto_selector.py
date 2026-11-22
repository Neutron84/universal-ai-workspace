"""
Intelligent Auto-Selector & Dynamic Re-quantization Engine
Computes realistic VRAM budgets, picks the highest-quality fitting quantization,
calculates optimal re-quantization targets when only oversized weights exist,
and determines safe partial offloading or context reduction fallbacks.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import WorkspaceConfig, get_config
from .hardware import HardwareProfile, get_hardware_profile
from .model_inspector import GGUFVariant, InspectionResult, QUANT_ORDER, QUANT_SIZE_FACTORS


@dataclass
class SelectionRecommendation:
    source_model_id: str
    selected_variant: Optional[GGUFVariant]
    recommended_quant: str
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
    mmproj_file: Optional[str] = None


class AutoSelector:
    """Selects optimal model quantization and server parameters based on hardware."""

    def __init__(self, hardware: Optional[HardwareProfile] = None, config: Optional[WorkspaceConfig] = None):
        self.hw = hardware or get_hardware_profile()
        self.config = config or get_config()

    def recommend(
        self,
        inspection: InspectionResult,
        requested_ctx: int = 4096,
        override_quant: Optional[str] = None,
        cuda_overhead_mb: float = 1200.0,
    ) -> SelectionRecommendation:
        """
        Analyzes an inspection result against current GPU VRAM and capabilities.
        Selects the best quantization variant or recommends smart re-quantization.
        """
        primary_gpu = self.hw.primary_gpu
        has_gpu = self.hw.has_gpu and primary_gpu is not None

        # GPU features
        supports_fa = primary_gpu.supports_flash_attn if primary_gpu else False
        supports_kvq = primary_gpu.supports_kv_quant if primary_gpu else False

        # Pick KV cache quant types
        cache_type_k = "q8_0" if supports_kvq else "f16"
        cache_type_v = "q8_0" if supports_kvq else "f16"

        if not has_gpu:
            # CPU fallback mode
            return self._recommend_cpu_fallback(inspection, requested_ctx)

        free_vram_mb = primary_gpu.free_vram_mb

        # Calculate realistic VRAM budget
        from .hardware import HardwareProfiler
        vram_budget_mb = HardwareProfiler.calculate_vram_budget_mb(
            free_vram_mb=free_vram_mb,
            cuda_overhead_mb=cuda_overhead_mb,
            ctx_size=requested_ctx,
            kv_cache_quant=cache_type_k
        )

        variants = inspection.gguf_variants

        # If not a GGUF model (e.g. safetensors or unknown)
        if not variants:
            return SelectionRecommendation(
                source_model_id=inspection.model_id,
                selected_variant=None,
                recommended_quant=override_quant or "N/A",
                vram_budget_mb=vram_budget_mb,
                recommended_ctx=requested_ctx,
                recommended_n_gpu_layers=-1,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                reasoning="Non-GGUF model. Standard backend dispatch recommended."
            )

        # 1. User explicit override requested
        if override_quant:
            req_upper = override_quant.upper()
            matched = [v for v in variants if v.quant.upper() == req_upper]
            if matched:
                v = matched[0]
                model_mb = v.total_size_bytes / (1024 * 1024)
                fits = model_mb <= vram_budget_mb
                n_layers = -1 if fits else self._calculate_partial_layers(model_mb, vram_budget_mb)
                return SelectionRecommendation(
                    source_model_id=inspection.model_id,
                    selected_variant=v,
                    recommended_quant=v.quant,
                    vram_budget_mb=vram_budget_mb,
                    estimated_model_vram_mb=model_mb,
                    recommended_ctx=requested_ctx if fits else max(2048, requested_ctx // 2),
                    recommended_n_gpu_layers=n_layers,
                    flash_attn=supports_fa,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                    mmproj_file=v.mmproj_file,
                    reasoning=f"User override '{override_quant}' selected. {'Fits completely in VRAM.' if fits else 'Exceeds VRAM budget, partial GPU offload applied.'}"
                )

        # 2. Find highest quality quantization fitting 100% in usable VRAM
        fitting_variants = [v for v in variants if (v.total_size_bytes / (1024 * 1024)) <= vram_budget_mb]

        if fitting_variants:
            # Already sorted by QUANT_ORDER quality hierarchy in inspector!
            best = fitting_variants[0]
            model_mb = best.total_size_bytes / (1024 * 1024)
            return SelectionRecommendation(
                source_model_id=inspection.model_id,
                selected_variant=best,
                recommended_quant=best.quant,
                vram_budget_mb=vram_budget_mb,
                estimated_model_vram_mb=model_mb,
                recommended_ctx=requested_ctx,
                recommended_n_gpu_layers=-1,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                mmproj_file=best.mmproj_file,
                reasoning=f"Highest quality fitting quant: {best.quant} ({best.total_size_gb:.1f}GB <= {vram_budget_mb/1024:.1f}GB usable VRAM)."
            )

        # 3. No existing variant fits: check if repository only has oversized weights (e.g. F16, BF16, Q8_0)
        # Check if re-quantization to a smaller quant would fit!
        smallest_existing = min(variants, key=lambda v: v.total_size_bytes)
        smallest_mb = smallest_existing.total_size_bytes / (1024 * 1024)

        # Determine highest target quant that WOULD fit if re-quantized
        requant_target = self._calculate_highest_fitting_quant(
            smallest_existing.total_size_bytes,
            smallest_existing.quant,
            vram_budget_mb
        )

        if requant_target:
            # Pick the best source file to re-quantize from (preferably Q8_0 or F16 or smallest available)
            source_variant = smallest_existing
            return SelectionRecommendation(
                source_model_id=inspection.model_id,
                selected_variant=source_variant,
                recommended_quant=requant_target,
                needs_requant=True,
                requant_target_quant=requant_target,
                requant_source_file=source_variant.shard_files[0],
                vram_budget_mb=vram_budget_mb,
                estimated_model_vram_mb=smallest_mb,
                recommended_ctx=requested_ctx,
                recommended_n_gpu_layers=-1,
                flash_attn=supports_fa,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                mmproj_file=source_variant.mmproj_file,
                reasoning=f"All available quants exceed VRAM ({smallest_existing.quant} is {smallest_existing.total_size_gb:.1f}GB). Re-quantizing to {requant_target} will fit 100% inside usable VRAM ({vram_budget_mb/1024:.1f}GB)."
            )

        # 4. Auto-Fallback: Even smallest quant or re-quant exceeds full VRAM offload
        # Calculate safe partial GPU offload layers & reduced context instead of crashing with OOM
        partial_layers = self._calculate_partial_layers(smallest_mb, vram_budget_mb)
        safe_ctx = 2048 if requested_ctx > 2048 else requested_ctx
        return SelectionRecommendation(
            source_model_id=inspection.model_id,
            selected_variant=smallest_existing,
            recommended_quant=smallest_existing.quant,
            vram_budget_mb=vram_budget_mb,
            estimated_model_vram_mb=smallest_mb,
            recommended_ctx=safe_ctx,
            recommended_n_gpu_layers=partial_layers,
            flash_attn=supports_fa,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            mmproj_file=smallest_existing.mmproj_file,
            reasoning=f"Model exceeds VRAM. Auto-fallback activated: partial offload ({partial_layers} GPU layers) and safe context ({safe_ctx}) to prevent OOM."
        )

    def _recommend_cpu_fallback(self, inspection: InspectionResult, requested_ctx: int) -> SelectionRecommendation:
        """Generates CPU-only recommendation."""
        variants = inspection.gguf_variants
        # On CPU, prefer Q4_K_M or Q4_0 for cache and instruction efficiency
        selected = None
        for q in ["Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"]:
            matched = [v for v in variants if v.quant.upper() == q]
            if matched:
                selected = matched[0]
                break
        if not selected and variants:
            selected = variants[-1]

        return SelectionRecommendation(
            source_model_id=inspection.model_id,
            selected_variant=selected,
            recommended_quant=selected.quant if selected else "Q4_0",
            vram_budget_mb=0.0,
            recommended_ctx=min(requested_ctx, 2048),
            recommended_n_gpu_layers=0,
            flash_attn=False,
            cache_type_k="f16",
            cache_type_v="f16",
            is_cpu_fallback=True,
            reasoning="No GPU detected. Running in CPU fallback mode (0 GPU layers)."
        )

    @staticmethod
    def _calculate_highest_fitting_quant(source_bytes: int, source_quant: str, vram_budget_mb: float) -> Optional[str]:
        """
        Estimates the output size for each quantization target relative to source size
        and finds the highest quality quantization that fits inside vram_budget_mb.
        Only valid if the target quantization is strictly lower than the source quantization.
        """
        source_quant_upper = source_quant.upper()
        # Only re-quantize if source is high precision (F16, BF16, Q8_0, Q6_K, Q5_K_M)
        if source_quant_upper not in ["F32", "F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M"]:
            return None

        source_factor = QUANT_SIZE_FACTORS.get(source_quant_upper, 1.0)
        # Approximate base unquantized F16 size in MB
        approx_f16_mb = (source_bytes / (1024 * 1024)) / source_factor

        # Candidate target quants in descending quality order
        candidate_quants = ["Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "IQ4_XS", "Q3_K_M", "Q2_K"]
        try:
            source_idx = candidate_quants.index(source_quant_upper)
            # Only consider quants strictly lower than source
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
        """Calculates safe partial layer offload count."""
        if model_mb <= 0:
            return -1
        fraction = min(1.0, max(0.0, vram_budget_mb / model_mb))
        # Be slightly conservative (leave 10% safety margin)
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
        """
        Executes llama-quantize --allow-requantize <source> <target> <quant>.
        Verifies output, optionally backs up to Drive, and removes oversized source file.
        """
        source_path = Path(source_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source file for re-quantization not found: {source_path}")

        quant_bin = shutil.which("llama-quantize") or "/usr/local/bin/llama-quantize"
        if not Path(quant_bin).exists():
            raise FileNotFoundError(f"llama-quantize binary not found at {quant_bin}")

        if output_path is None:
            stem = source_path.stem
            # Replace old quant tag or append new
            out_name = f"{stem}_{target_quant}.gguf"
            output_path = source_path.parent / out_name

        print(f"[AutoSelector] Re-quantizing {source_path.name} -> {target_quant} ({output_path.name})...")
        cmd = [
            str(quant_bin),
            "--allow-requantize",
            str(source_path),
            str(output_path),
            target_quant.upper()
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            err_msg = res.stderr or res.stdout
            raise RuntimeError(f"llama-quantize failed with code {res.returncode}:\n{err_msg}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Re-quantized output file is empty or missing.")

        print(f"[AutoSelector] Re-quantization complete! New size: {output_path.stat().st_size / (1024**3):.2f} GB")

        # Optional backup to Google Drive
        cfg = get_config()
        if backup_to_drive and cfg.models_gguf_dir.is_dir():
            drive_dest = cfg.models_gguf_dir / output_path.name
            print(f"[AutoSelector] Backing up re-quantized model to Drive: {drive_dest}...")
            shutil.copy2(str(output_path), str(drive_dest))

        # Cleanup source file to protect Colab SSD space
        if cleanup_source and source_path != output_path:
            print(f"[AutoSelector] Cleaning up oversized source file {source_path.name} to free Colab disk space...")
            try:
                source_path.unlink()
            except Exception as e:
                print(f"[AutoSelector] Warning deleting source: {e}")

        return output_path
