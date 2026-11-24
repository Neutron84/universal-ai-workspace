"""
ik_llama.cpp Converter & Quantizer Plugin
Supports non-uniform quantization formats (IQ4_K, IQ4_KS, IQ3_K, IQ3_KS, IQ2_K, IQ2_KS, IQ5_K, IQ6_K),
Safetensors-to-GGUF pre-conversion, optional imatrix calibration, and MoE weight repacking.
"""

from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union

from .base import BaseConverter
from .llama_cpp_converter import LlamaCppConverter
from ..config import WorkspaceConfig
from ..env_manager import EnvironmentManager

IK_SUPPORTED_QUANTS = {
    "IQ4_K", "IQ4_KS", "IQ3_K", "IQ3_KS", "IQ2_K", "IQ2_KS",
    "IQ5_K", "IQ6_K", "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S",
    "IQ2_M", "IQ2_S", "IQ1_S", "IQ1_M", "Q4_K_M", "Q5_K_M", "Q8_0",
}


class IKLlamaConverter(BaseConverter):
    """Converts and quantizes models for ik_llama.cpp with non-uniform IQ quant support."""

    name = "ik_llama"
    supported_inputs = ["safetensors", "hf_repo", "gguf"]
    supported_outputs = ["gguf"]

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.bin_dir = self.config.runtime_root / "bin" / "ik_llama_cpp"

    def check_installed(self) -> bool:
        bin_candidate = self.bin_dir / "llama-quantize"
        return bin_candidate.exists() or shutil.which("llama-quantize") is not None

    def install(self) -> bool:
        em = EnvironmentManager(self.config)
        em.fetch_ik_llama_cpp()
        return self.check_installed()

    def estimate_requirements(self, input_info: Any) -> Dict[str, Any]:
        size_bytes = getattr(input_info, "total_repo_size_bytes", 0) or 0
        size_gb = size_bytes / (1024**3)
        return {
            "estimated_ram_gb": max(8.0, size_gb * 1.3),
            "estimated_vram_gb": 0.0,
            "estimated_disk_gb": size_gb * 2.2,
        }

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "IQ4_KS",
        cleanup_source: bool = False,
        use_imatrix: bool = False,
        calibration_text_path: Optional[Union[str, Path]] = None,
        calibration_text: Optional[str] = None,
        imatrix_chunks: Optional[int] = None,
        outtype: Optional[str] = None,
        threads: int = 0,
        **kwargs: Any,
    ) -> Path:
        """
        Converts Safetensors to base GGUF if needed, optionally computes imatrix,
        and executes non-uniform IQ quantization via ik_llama-quantize.
        """
        src = Path(input_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input path not found: {src}")

        target_q = (target_quant or "IQ4_KS").upper()
        if target_q == "AUTO":
            target_q = "IQ4_KS"
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        base_helper = LlamaCppConverter(self.config)

        # Ensure binaries are ready
        ik_quant_bin = self.bin_dir / "llama-quantize"
        if not ik_quant_bin.exists():
            self.install()

        if ik_quant_bin.exists():
            quant_bin = str(ik_quant_bin)
        else:
            quant_bin = shutil.which("llama-quantize") or "/usr/local/bin/llama-quantize"

        if not Path(quant_bin).exists():
            raise FileNotFoundError(f"ik_llama llama-quantize binary not found at {quant_bin}")

        intermediate_gguf: Optional[Path] = None
        if src.is_dir() or src.suffix.lower() == ".safetensors":
            base_outtype = outtype or self.config.settings.default_outtype or "f16"
            print(f"[IKLlamaConverter] Converting Safetensors {src.name} to base GGUF ({base_outtype}) first...")
            intermediate_gguf = outputs_dir / f"{src.stem}_{base_outtype}.gguf"
            base_helper.convert_to_base_gguf(src, intermediate_gguf, outtype=base_outtype)
            source_for_quant = intermediate_gguf
        else:
            source_for_quant = base_helper._maybe_merge_split_gguf(src, outputs_dir) or src

        imatrix_file: Optional[Path] = None
        if use_imatrix:
            chunks = imatrix_chunks if imatrix_chunks is not None else self.config.settings.imatrix_chunks
            imatrix_file = base_helper.compute_imatrix(
                source_for_quant,
                outputs_dir,
                calibration_text_path=calibration_text_path,
                calibration_text=calibration_text,
                imatrix_chunks=chunks,
            )

        if output_path:
            final_out = Path(output_path).resolve()
        else:
            clean_stem = source_for_quant.stem.replace("_f16", "").replace("_bf16", "").replace("_merged", "")
            final_out = outputs_dir / f"{clean_stem}_{target_q}.gguf"

        final_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"[IKLlamaConverter] Running non-uniform IQ quantization: {source_for_quant.name} -> {target_q} ({final_out.name})...")
        cmd: List[str] = [str(quant_bin)]
        if imatrix_file and imatrix_file.exists():
            cmd.extend(["--imatrix", str(imatrix_file)])
        cmd.extend([
            "--allow-requantize",
            str(source_for_quant),
            str(final_out),
            target_q,
        ])
        if threads and int(threads) > 0:
            cmd.append(str(int(threads)))

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"ik_llama quantization failed with code {res.returncode}:\n{res.stderr or res.stdout}")

        if not final_out.exists() or final_out.stat().st_size == 0:
            raise RuntimeError(f"Quantized output file missing or empty: {final_out}")

        print(f"[IKLlamaConverter] Non-uniform quantization complete: {final_out} ({final_out.stat().st_size / (1024**3):.2f} GB)")

        if cleanup_source:
            self.safe_cleanup(intermediate_gguf, keep_paths=[final_out])
            self.safe_cleanup(imatrix_file, keep_paths=[final_out])
            self.safe_cleanup(src, keep_paths=[final_out])

        return final_out
