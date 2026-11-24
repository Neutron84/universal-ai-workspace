"""
Hugging Face & Audio Quantizer Adapters
Modular plugins for GPTQModel, AutoAWQ / LLM-Compressor, ExLlamaV2 (EXL2), and CTranslate2 (CT2).
Executes inside isolated virtualenvs when configured.
"""

from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Optional, Union

from .base import BaseConverter
from ..env_manager import EnvironmentManager


class GPTQModelConverter(BaseConverter):
    """Quantizes Safetensors models using gptqmodel into 4-bit or 8-bit GPTQ."""

    name = "gptqmodel"
    supported_inputs = ["safetensors", "hf_repo"]
    supported_outputs = ["gptq", "safetensors"]

    def check_installed(self) -> bool:
        status = EnvironmentManager(self.config).verify_profile("gptqmodel")
        return status.get("gptqmodel", False)

    def install(self) -> bool:
        print("[GPTQModelConverter] Installing gptqmodel...")
        em = EnvironmentManager(self.config)
        return em.install_profile("gptqmodel")

    def estimate_requirements(self, input_info: Any) -> Dict[str, Any]:
        size_bytes = getattr(input_info, "total_repo_size_bytes", 0) or 0
        size_gb = size_bytes / (1024**3)
        return {
            "estimated_ram_gb": max(12.0, size_gb * 1.5),
            "estimated_vram_gb": max(8.0, size_gb * 0.8),
            "estimated_disk_gb": size_gb * 2.0,
        }

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "4bit",
        cleanup_source: bool = False,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = True,
        sym: bool = True,
        **kwargs: Any,
    ) -> Path:
        src = Path(input_path).resolve()
        bits_val = 8 if str(target_quant or "").lower().startswith("8") else int(bits)
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(output_path).resolve() if output_path else (outputs_dir / f"{src.stem}_gptq_{bits_val}bit")
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.check_installed() and self.config.settings.auto_install_dependencies:
            self.install()
        py_exec = self.get_python_executable("gptqmodel")

        print(f"[GPTQModelConverter] Quantizing {src.name} to {bits_val}-bit GPTQ ({out_dir.name})...")
        code = f"""
import sys
try:
    from gptqmodel import GPTQModel, QuantizeConfig
    quant_config = QuantizeConfig(
        bits={bits_val},
        group_size={int(group_size)},
        desc_act={bool(desc_act)},
        sym={bool(sym)},
    )
    model = GPTQModel.from_pretrained({repr(str(src))}, quantize_config=quant_config)
    model.quantize()
    model.save_quantized({repr(str(out_dir))})
    print("[GPTQModel] Quantization completed successfully.")
except Exception as e:
    print(f"[GPTQModel] Error during quantization: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        res = subprocess.run([py_exec, "-c", code], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"GPTQModel quantization failed:\n{res.stderr or res.stdout}")

        if cleanup_source:
            self.safe_cleanup(src, keep_paths=[out_dir])

        return out_dir


class AutoAWQConverter(BaseConverter):
    """Quantizes Safetensors models using autoawq (AWQ 4-bit & FP8)."""

    name = "autoawq"
    supported_inputs = ["safetensors", "hf_repo"]
    supported_outputs = ["awq", "fp8", "safetensors"]

    def check_installed(self) -> bool:
        em = EnvironmentManager(self.config)
        py_exec = em.get_profile_python("autoawq")
        mod_map = em._check_modules_in_python(py_exec, ["awq"])
        return mod_map.get("awq", False)

    def install(self) -> bool:
        print("[AutoAWQConverter] Installing autoawq...")
        em = EnvironmentManager(self.config)
        return em.install_profile("autoawq")

    def estimate_requirements(self, input_info: Any) -> Dict[str, Any]:
        size_bytes = getattr(input_info, "total_repo_size_bytes", 0) or 0
        size_gb = size_bytes / (1024**3)
        return {
            "estimated_ram_gb": max(12.0, size_gb * 1.5),
            "estimated_vram_gb": max(8.0, size_gb * 0.8),
            "estimated_disk_gb": size_gb * 2.0,
        }

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "4bit",
        cleanup_source: bool = False,
        w_bit: int = 4,
        q_group_size: int = 128,
        zero_point: bool = True,
        version: str = "GEMM",
        **kwargs: Any,
    ) -> Path:
        src = Path(input_path).resolve()
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(output_path).resolve() if output_path else (outputs_dir / f"{src.stem}_awq")
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.check_installed() and self.config.settings.auto_install_dependencies:
            self.install()
        py_exec = self.get_python_executable("autoawq")

        is_fp8 = "fp8" in str(target_quant or "").lower()
        print(f"[AutoAWQConverter] Quantizing {src.name} to {'FP8' if is_fp8 else f'AWQ {w_bit}-bit'}...")

        code = f"""
import sys
try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained({repr(str(src))}, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained({repr(str(src))}, low_cpu_mem_usage=True)
    quant_config = {{
        "zero_point": {bool(zero_point)},
        "q_group_size": {int(q_group_size)},
        "w_bit": {int(w_bit)},
        "version": {repr(version)},
    }}
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized({repr(str(out_dir))})
    tokenizer.save_pretrained({repr(str(out_dir))})
    print("[AutoAWQ] Quantization completed successfully.")
except Exception as e:
    print(f"[AutoAWQ] Error during quantization: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        res = subprocess.run([py_exec, "-c", code], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"AutoAWQ quantization failed:\n{res.stderr or res.stdout}")

        if cleanup_source:
            self.safe_cleanup(src, keep_paths=[out_dir])

        return out_dir


class ExLlamaConverter(BaseConverter):
    """Converts Safetensors models to fractional bpw EXL2 format using exllamav2."""

    name = "exllamav2"
    supported_inputs = ["safetensors", "hf_repo"]
    supported_outputs = ["exl2"]

    def check_installed(self) -> bool:
        status = EnvironmentManager(self.config).verify_profile("exllamav2")
        return status.get("exllamav2", False)

    def install(self) -> bool:
        print("[ExLlamaConverter] Installing exllamav2...")
        em = EnvironmentManager(self.config)
        return em.install_profile("exllamav2")

    def estimate_requirements(self, input_info: Any) -> Dict[str, Any]:
        size_bytes = getattr(input_info, "total_repo_size_bytes", 0) or 0
        size_gb = size_bytes / (1024**3)
        return {
            "estimated_ram_gb": max(12.0, size_gb * 1.5),
            "estimated_vram_gb": max(10.0, size_gb * 0.9),
            "estimated_disk_gb": size_gb * 2.5,
        }

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "4.0",
        cleanup_source: bool = False,
        bits_per_weight: float = 4.0,
        head_bits: Optional[int] = None,
        **kwargs: Any,
    ) -> Path:
        src = Path(input_path).resolve()
        tq = str(target_quant or "").lower().replace("bpw", "").replace("exl2", "").strip("-_ ")
        try:
            bpw = float(tq) if tq else bits_per_weight
        except ValueError:
            bpw = bits_per_weight

        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(output_path).resolve() if output_path else (outputs_dir / f"{src.stem}_{bpw:.2f}bpw_exl2")
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.check_installed() and self.config.settings.auto_install_dependencies:
            self.install()
        py_exec = self.get_python_executable("exllamav2")

        print(f"[ExLlamaConverter] Converting {src.name} to EXL2 ({bpw:.2f} bpw)...")
        cmd = [
            py_exec,
            "-m", "exllamav2.convert",
            "-i", str(src),
            "-o", str(out_dir),
            "-b", str(bpw),
        ]
        if head_bits:
            cmd.extend(["-hb", str(int(head_bits))])

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"ExLlamaV2 conversion failed:\n{res.stderr or res.stdout}")

        if cleanup_source:
            self.safe_cleanup(src, keep_paths=[out_dir])

        return out_dir


class CT2AudioConverter(BaseConverter):
    """Converts Hugging Face Whisper models to faster-whisper CTranslate2 format."""

    name = "ct2"
    supported_inputs = ["safetensors", "hf_repo", "audio"]
    supported_outputs = ["ct2"]

    def check_installed(self) -> bool:
        em = EnvironmentManager(self.config)
        return em.get_profile_bin("audio", "ct2-transformers-converter") is not None

    def install(self) -> bool:
        print("[CT2AudioConverter] Installing ctranslate2...")
        em = EnvironmentManager(self.config)
        return em.install_profile("audio")

    def estimate_requirements(self, input_info: Any) -> Dict[str, Any]:
        size_bytes = getattr(input_info, "total_repo_size_bytes", 0) or 0
        size_gb = size_bytes / (1024**3)
        return {
            "estimated_ram_gb": max(4.0, size_gb * 1.5),
            "estimated_vram_gb": 0.0,
            "estimated_disk_gb": max(2.0, size_gb * 1.5),
        }

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "float16",
        cleanup_source: bool = False,
        **kwargs: Any,
    ) -> Path:
        src = Path(input_path).resolve()
        quant = target_quant if (target_quant and target_quant.lower() != "auto") else "float16"
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(output_path).resolve() if output_path else (outputs_dir / f"{src.stem}_ct2")
        out_dir.mkdir(parents=True, exist_ok=True)

        em = EnvironmentManager(self.config)
        converter_bin = em.get_profile_bin("audio", "ct2-transformers-converter")
        if not converter_bin:
            self.install()
            converter_bin = em.get_profile_bin("audio", "ct2-transformers-converter")

        print(f"[CT2AudioConverter] Converting Whisper model {src.name} to CTranslate2 ({quant})...")
        cmd = [
            converter_bin or "ct2-transformers-converter",
            "--model", str(src),
            "--output_dir", str(out_dir),
            "--quantization", str(quant),
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
            "--force",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"CTranslate2 conversion failed:\n{res.stderr or res.stdout}")

        if cleanup_source:
            self.safe_cleanup(src, keep_paths=[out_dir])

        return out_dir
