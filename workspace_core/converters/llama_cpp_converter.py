"""
llama.cpp Converter & Quantizer Plugin
Converts Hugging Face Safetensors to GGUF, merges split GGUFs, computes importance matrices
with llama-imatrix, and produces quantized or sharded GGUF weights.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union
import urllib.request

from .base import BaseConverter
from ..config import WorkspaceConfig
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

    def _find_bin(self, name: str) -> Optional[str]:
        found = shutil.which(name)
        if found:
            return found
        usr_local = Path("/usr/local/bin") / name
        if usr_local.exists():
            return str(usr_local)
        for p in self.config.runtime_root.glob(f"bin/**/{name}"):
            if p.is_file():
                return str(p)
        return None

    def check_installed(self) -> bool:
        return self._find_bin("llama-quantize") is not None

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
            "estimated_vram_gb": 0.0,
            "estimated_disk_gb": size_gb * 2.2,
        }

    def convert_to_base_gguf(self, src_dir: Path, out_file: Path, outtype: str = "f16") -> Path:
        """Converts an HF Safetensors directory to an unquantized GGUF file."""
        script = self._ensure_convert_script()
        if not script.exists():
            raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {script}")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script),
            str(src_dir),
            "--outfile", str(out_file),
            "--outtype", str(outtype).lower(),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"convert_hf_to_gguf.py failed:\n{res.stderr or res.stdout}")
        return out_file

    def _maybe_merge_split_gguf(self, src: Path, outputs_dir: Path) -> Optional[Path]:
        """If input is a split GGUF shard, merges shards into a single GGUF prior to quantization."""
        if not src.is_file():
            return None
        if not re.search(r"-\d{5}-of-\d{5}\.gguf$", src.name, re.IGNORECASE):
            return None
        split_bin = self._find_bin("llama-gguf-split")
        if not split_bin:
            return None
        merged = outputs_dir / f"{src.stem}_merged.gguf"
        print(f"[LlamaCppConverter] Merging split GGUF shards starting at {src.name}...")
        res = subprocess.run([split_bin, "--merge", str(src), str(merged)], capture_output=True, text=True, check=False)
        if res.returncode == 0 and merged.exists():
            return merged
        return None

    def compute_imatrix(
        self,
        model_gguf: Path,
        outputs_dir: Path,
        calibration_text_path: Optional[Union[str, Path]] = None,
        calibration_text: Optional[str] = None,
        imatrix_chunks: int = 100,
    ) -> Optional[Path]:
        """Generates an importance matrix (.imatrix.dat) via llama-imatrix."""
        imatrix_bin = self._find_bin("llama-imatrix")
        if not imatrix_bin:
            print("[LlamaCppConverter] llama-imatrix binary not found; skipping imatrix.")
            return None

        calib_file = (
            Path(calibration_text_path).resolve()
            if calibration_text_path
            else (outputs_dir / "calibration.txt")
        )
        if not calib_file.exists() or calibration_text:
            calib_file.write_text(calibration_text or DEFAULT_CALIBRATION_TEXT, encoding="utf-8")

        imatrix_file = outputs_dir / f"{model_gguf.stem}.imatrix.dat"
        print(f"[LlamaCppConverter] Computing importance matrix: {imatrix_file.name}...")
        im_cmd = [
            str(imatrix_bin),
            "-m", str(model_gguf),
            "-f", str(calib_file),
            "-o", str(imatrix_file),
            "--chunks", str(max(1, int(imatrix_chunks))),
        ]
        subprocess.run(im_cmd, check=False)
        if imatrix_file.exists() and imatrix_file.stat().st_size > 0:
            return imatrix_file
        print("[LlamaCppConverter] Warning: llama-imatrix did not produce valid output. Proceeding without imatrix.")
        return None

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "Q4_K_M",
        cleanup_source: bool = False,
        use_imatrix: bool = False,
        calibration_text_path: Optional[Union[str, Path]] = None,
        calibration_text: Optional[str] = None,
        imatrix_chunks: Optional[int] = None,
        split_max_size: Optional[str] = None,
        outtype: Optional[str] = None,
        threads: int = 0,
        output_tensor_type: Optional[str] = None,
        token_embedding_type: Optional[str] = None,
        leave_output_tensor: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        Executes conversion from Safetensors to GGUF, optional imatrix calibration,
        target quantization, and optional shard splitting.
        """
        src = Path(input_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input path not found: {src}")

        if not self.check_installed():
            self.install()

        target_q = (target_quant or self.config.settings.default_target_quant or "Q4_K_M").upper()
        if target_q == "AUTO":
            target_q = "Q4_K_M"
        base_outtype = outtype or self.config.settings.default_outtype or "f16"
        chunks = imatrix_chunks if imatrix_chunks is not None else self.config.settings.imatrix_chunks
        split_limit = split_max_size if split_max_size is not None else (self.config.settings.split_max_size or None)

        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)

        intermediate_gguf: Optional[Path] = None
        merged_gguf: Optional[Path] = None

        # Step 1: Safetensors -> unquantized GGUF or merge split shards
        if src.is_dir() or src.suffix.lower() == ".safetensors":
            print(f"[LlamaCppConverter] Converting {src.name} to unquantized GGUF ({base_outtype})...")
            intermediate_gguf = outputs_dir / f"{src.stem}_{base_outtype}.gguf"
            self.convert_to_base_gguf(src, intermediate_gguf, outtype=base_outtype)
            source_for_quant = intermediate_gguf
        else:
            merged_gguf = self._maybe_merge_split_gguf(src, outputs_dir)
            source_for_quant = merged_gguf or src

        # If target quant is simply the unquantized base outtype (e.g. F16/BF16/F32) and input was safetensors
        if intermediate_gguf and target_q in ("F16", "BF16", "F32") and target_q.lower() == base_outtype.lower():
            final_out = Path(output_path).resolve() if output_path else intermediate_gguf
            if final_out != intermediate_gguf:
                shutil.move(str(intermediate_gguf), str(final_out))
            if cleanup_source:
                self.safe_cleanup(src, keep_paths=[final_out])
            return final_out

        # Step 2: Optional importance matrix calculation
        imatrix_file: Optional[Path] = None
        if use_imatrix:
            imatrix_file = self.compute_imatrix(
                source_for_quant,
                outputs_dir,
                calibration_text_path=calibration_text_path,
                calibration_text=calibration_text,
                imatrix_chunks=chunks,
            )

        # Step 3: Quantize with llama-quantize
        quant_bin = self._find_bin("llama-quantize")
        if not quant_bin:
            raise FileNotFoundError("llama-quantize binary not found after installation attempt.")

        if output_path:
            final_out = Path(output_path).resolve()
        else:
            clean_stem = source_for_quant.stem.replace("_f16", "").replace("_bf16", "").replace("_merged", "")
            final_out = outputs_dir / f"{clean_stem}_{target_q}.gguf"
        final_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"[LlamaCppConverter] Quantizing {source_for_quant.name} -> {target_q} ({final_out.name})...")
        q_cmd: List[str] = [str(quant_bin)]
        if imatrix_file and imatrix_file.exists():
            q_cmd.extend(["--imatrix", str(imatrix_file)])
        if leave_output_tensor:
            q_cmd.append("--leave-output-tensor")
        if output_tensor_type:
            q_cmd.extend(["--output-tensor-type", str(output_tensor_type)])
        if token_embedding_type:
            q_cmd.extend(["--token-embedding-type", str(token_embedding_type)])
        q_cmd.append("--allow-requantize")
        q_cmd.extend([str(source_for_quant), str(final_out), target_q])
        if threads and int(threads) > 0:
            q_cmd.append(str(int(threads)))

        q_res = subprocess.run(q_cmd, capture_output=True, text=True, check=False)
        if q_res.returncode != 0:
            raise RuntimeError(f"llama-quantize failed with code {q_res.returncode}:\n{q_res.stderr or q_res.stdout}")

        if not final_out.exists() or final_out.stat().st_size == 0:
            raise RuntimeError(f"Quantized output file missing or empty: {final_out}")

        print(f"[LlamaCppConverter] Quantization complete: {final_out} ({final_out.stat().st_size / (1024**3):.2f} GB)")

        # Step 4: Optional shard splitting via llama-gguf-split
        result_path = final_out
        if split_limit and str(split_limit).strip():
            split_bin = self._find_bin("llama-gguf-split")
            if split_bin:
                print(f"[LlamaCppConverter] Splitting GGUF into shards (max size: {split_limit})...")
                split_prefix = final_out.parent / f"{final_out.stem}"
                s_cmd = [
                    str(split_bin),
                    "--split-max-size", str(split_limit).strip(),
                    str(final_out),
                    str(split_prefix),
                ]
                s_res = subprocess.run(s_cmd, capture_output=True, text=True, check=False)
                shards = sorted(final_out.parent.glob(f"{final_out.stem}-*-of-*.gguf"))
                if s_res.returncode == 0 and shards:
                    result_path = shards[0]
                    if cleanup_source:
                        self.safe_cleanup(final_out, keep_paths=shards)

        # Step 5: Clean intermediate files
        if cleanup_source:
            self.safe_cleanup(intermediate_gguf, keep_paths=[result_path])
            self.safe_cleanup(merged_gguf, keep_paths=[result_path])
            self.safe_cleanup(imatrix_file, keep_paths=[result_path])
            if not intermediate_gguf and not merged_gguf:
                self.safe_cleanup(src, keep_paths=[result_path])
            elif intermediate_gguf:
                self.safe_cleanup(src, keep_paths=[result_path])

        return result_path
