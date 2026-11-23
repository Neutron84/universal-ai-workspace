"""
llama.cpp Converter & Quantizer Plugin
Converts Hugging Face Safetensors to GGUF, computes importance matrices with llama-imatrix,
and produces quantized GGUF weights via llama-quantize and llama-gguf-split.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union
import urllib.request

from .base import BaseConverter
from ..config import WorkspaceConfig, get_config
from ..env_manager import EnvironmentManager


DEFAULT_CALIBRATION_TEXT = """Universal artificial intelligence models require accurate numerical representations.
Structured reasoning, mathematical analysis, and linguistic comprehension rely on high-precision tensor representations.
Quantization approximates continuous floating-point weights into low-bit discrete representations while preserving activation patterns.
Evaluation across synthetic and empirical benchmarks demonstrates that importance matrix calibration maintains model quality across tasks.
System operations, memory paging, matrix multiplications, and input-output streaming are optimized for cloud runtime execution."""


class LlamaCppConverter(BaseConverter):
    """Converts HF Safetensors to GGUF and quantizes GGUF using native llama.cpp tooling."""

    name = "llama.cpp"
    supported_inputs = ["safetensors", "hf_repo", "gguf"]
    supported_outputs = ["gguf"]

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.bin_dir = self.config.runtime_root / "bin"
        self.convert_script_path = self.config.bin_dir / "convert_hf_to_gguf.py"

    def check_installed(self) -> bool:
        quant_bin = shutil.which("llama-quantize") or "/usr/local/bin/llama-quantize"
        return Path(quant_bin).exists()

    def install(self) -> bool:
        em = EnvironmentManager(self.config)
        em.fetch_llama_cpp()
        self._ensure_convert_script()
        return self.check_installed()

    def _ensure_convert_script(self) -> Path:
        """Downloads and caches convert_hf_to_gguf.py from llama.cpp repository."""
        if self.convert_script_path.exists() and self.convert_script_path.stat().st_size > 0:
            return self.convert_script_path

        self.convert_script_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
        print(f"[LlamaCppConverter] Fetching convert_hf_to_gguf.py from {url}...")
        try:
            urllib.request.urlretrieve(url, str(self.convert_script_path))
            self.convert_script_path.chmod(0o755)
        except Exception as e:
            print(f"[LlamaCppConverter] Note: could not download convert script: {e}")
        return self.convert_script_path

    def estimate_requirements(self, input_info: Any) -> Dict[str, Any]:
        """Estimates conversion and quantization memory/disk requirements."""
        size_bytes = getattr(input_info, "total_repo_size_bytes", 0) or 0
        size_gb = size_bytes / (1024**3)
        return {
            "estimated_ram_gb": max(8.0, size_gb * 1.2),
            "estimated_vram_gb": 0.0,  # quantization primarily runs on CPU/system RAM
            "estimated_disk_gb": size_gb * 2.2,
        }

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "Q4_K_M",
        cleanup_source: bool = False,
        use_imatrix: bool = False,
        calibration_text_path: Optional[Union[str, Path]] = None,
        split_max_size: Optional[str] = None,
        outtype: str = "f16",
        **kwargs
    ) -> Path:
        """
        Executes conversion from Safetensors to GGUF, optional imatrix calibration,
        and target quantization.
        """
        src = Path(input_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input path not found: {src}")

        target_q = (target_quant or "Q4_K_M").upper()
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)

        intermediate_gguf: Optional[Path] = None

        # Step 1: If input is a directory or safetensors file, convert to base GGUF first
        if src.is_dir() or src.suffix.lower() == ".safetensors":
            print(f"[LlamaCppConverter] Converting Hugging Face model at {src.name} to unquantized GGUF ({outtype})...")
            script = self._ensure_convert_script()
            if not script.exists():
                raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {script}")

            intermediate_name = f"{src.stem}_{outtype}.gguf"
            intermediate_gguf = outputs_dir / intermediate_name

            cmd = [
                sys.executable,
                str(script),
                str(src),
                "--outfile", str(intermediate_gguf),
                "--outtype", outtype
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                raise RuntimeError(f"convert_hf_to_gguf.py failed:\n{res.stderr or res.stdout}")

            source_for_quant = intermediate_gguf
        else:
            source_for_quant = src

        # Step 2: Optional importance matrix calculation
        imatrix_file: Optional[Path] = None
        if use_imatrix:
            imatrix_bin = shutil.which("llama-imatrix") or "/usr/local/bin/llama-imatrix"
            if Path(imatrix_bin).exists():
                calib_file = Path(calibration_text_path).resolve() if calibration_text_path else (outputs_dir / "calibration.txt")
                if not calib_file.exists():
                    calib_file.write_text(DEFAULT_CALIBRATION_TEXT, encoding="utf-8")

                imatrix_file = outputs_dir / f"{source_for_quant.stem}.imatrix.dat"
                print(f"[LlamaCppConverter] Computing importance matrix: {imatrix_file.name}...")
                im_cmd = [
                    str(imatrix_bin),
                    "-m", str(source_for_quant),
                    "-f", str(calib_file),
                    "-o", str(imatrix_file),
                ]
                subprocess.run(im_cmd, check=False)
                if not imatrix_file.exists() or imatrix_file.stat().st_size == 0:
                    print("[LlamaCppConverter] Warning: llama-imatrix did not produce valid output. Proceeding without imatrix.")
                    imatrix_file = None

        # Step 3: Quantize with llama-quantize
        quant_bin = shutil.which("llama-quantize") or "/usr/local/bin/llama-quantize"
        if not Path(quant_bin).exists():
            raise FileNotFoundError(f"llama-quantize binary not found at {quant_bin}")

        if output_path:
            final_out = Path(output_path).resolve()
        else:
            stem = source_for_quant.stem
            # Clean old quant tags from stem if present
            clean_stem = stem.replace("_f16", "").replace("_bf16", "")
            final_out = outputs_dir / f"{clean_stem}_{target_q}.gguf"

        final_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"[LlamaCppConverter] Quantizing {source_for_quant.name} -> {target_q} ({final_out.name})...")
        q_cmd = [str(quant_bin)]
        if imatrix_file and imatrix_file.exists():
            q_cmd.extend(["--imatrix", str(imatrix_file)])
        q_cmd.extend([
            "--allow-requantize",
            str(source_for_quant),
            str(final_out),
            target_q
        ])

        q_res = subprocess.run(q_cmd, capture_output=True, text=True, check=False)
        if q_res.returncode != 0:
            raise RuntimeError(f"llama-quantize failed with code {q_res.returncode}:\n{q_res.stderr or q_res.stdout}")

        if not final_out.exists() or final_out.stat().st_size == 0:
            raise RuntimeError(f"Quantized output file missing or empty: {final_out}")

        print(f"[LlamaCppConverter] Quantization complete: {final_out} ({final_out.stat().st_size / (1024**3):.2f} GB)")

        # Step 4: Optional shard splitting via llama-gguf-split
        if split_max_size:
            split_bin = shutil.which("llama-gguf-split") or "/usr/local/bin/llama-gguf-split"
            if Path(split_bin).exists():
                print(f"[LlamaCppConverter] Splitting GGUF to shards (max size: {split_max_size})...")
                split_prefix = outputs_dir / f"{final_out.stem}_split"
                s_cmd = [
                    str(split_bin),
                    "--split-max-size", str(split_max_size),
                    str(final_out),
                    str(split_prefix)
                ]
                subprocess.run(s_cmd, check=False)

        # Step 5: Clean intermediate files if requested
        if cleanup_source:
            if intermediate_gguf and intermediate_gguf.exists() and intermediate_gguf != final_out:
                print(f"[LlamaCppConverter] Cleaning intermediate unquantized file {intermediate_gguf.name}...")
                try:
                    intermediate_gguf.unlink()
                except Exception:
                    pass
            elif src.exists() and src.resolve() != final_out.resolve():
                print(f"[LlamaCppConverter] Cleaning up source file {src.name}...")
                try:
                    if src.is_file():
                        src.unlink()
                    elif src.is_dir():
                        shutil.rmtree(str(src))
                except Exception:
                    pass
            if imatrix_file and imatrix_file.exists():
                try:
                    imatrix_file.unlink()
                except Exception:
                    pass

        return final_out
