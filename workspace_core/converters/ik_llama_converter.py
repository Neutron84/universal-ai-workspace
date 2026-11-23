"""
ik_llama.cpp Converter & Quantizer Plugin
Supports non-uniform quantization formats (IQ4_K, IQ4_KS, IQ3_K, IQ3_KS, IQ2_K, IQ2_KS, IQ5_K, IQ6_K)
and MoE weight repacking for ikawrakow/ik_llama.cpp.
"""

from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union

from .base import BaseConverter
from ..config import WorkspaceConfig, get_config
from ..env_manager import EnvironmentManager

IK_SUPPORTED_QUANTS = {
    "IQ4_K", "IQ4_KS", "IQ3_K", "IQ3_KS", "IQ2_K", "IQ2_KS",
    "IQ5_K", "IQ6_K", "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S",
    "IQ2_M", "IQ2_S", "IQ1_S", "IQ1_M", "Q4_K_M", "Q5_K_M", "Q8_0"
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
            "estimated_disk_gb": size_gb * 2.0,
        }

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "IQ4_KS",
        cleanup_source: bool = False,
        **kwargs
    ) -> Path:
        """Executes non-uniform IQ quantization via ik_llama-quantize or standard llama-quantize."""
        src = Path(input_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input path not found: {src}")

        target_q = (target_quant or "IQ4_KS").upper()
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Locate quantizer binary
        ik_quant_bin = self.bin_dir / "llama-quantize"
        if not ik_quant_bin.exists():
            system_bin = shutil.which("llama-quantize") or "/usr/local/bin/llama-quantize"
            quant_bin = system_bin if Path(system_bin).exists() else None
        else:
            quant_bin = str(ik_quant_bin)

        if not quant_bin:
            self.install()
            quant_bin = str(self.bin_dir / "llama-quantize") if (self.bin_dir / "llama-quantize").exists() else (shutil.which("llama-quantize") or "/usr/local/bin/llama-quantize")

        if output_path:
            final_out = Path(output_path).resolve()
        else:
            clean_stem = src.stem.replace("_f16", "").replace("_bf16", "")
            final_out = outputs_dir / f"{clean_stem}_{target_q}.gguf"

        final_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"[IKLlamaConverter] Running non-uniform IQ quantization: {src.name} -> {target_q} ({final_out.name})...")
        cmd = [
            str(quant_bin),
            "--allow-requantize",
            str(src),
            str(final_out),
            target_q
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"ik_llama quantization failed with code {res.returncode}:\n{res.stderr or res.stdout}")

        if not final_out.exists() or final_out.stat().st_size == 0:
            raise RuntimeError(f"Quantized output file missing or empty: {final_out}")

        print(f"[IKLlamaConverter] Non-uniform quantization complete: {final_out} ({final_out.stat().st_size / (1024**3):.2f} GB)")

        if cleanup_source and src != final_out:
            print(f"[IKLlamaConverter] Cleaning up source file {src.name}...")
            try:
                src.unlink()
            except Exception:
                pass

        return final_out
