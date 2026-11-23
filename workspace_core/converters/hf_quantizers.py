"""
Hugging Face & Audio Quantizer Adapters
Modular plugins for GPTQModel, AutoAWQ / LLM-Compressor, ExLlamaV2 (EXL2), and CTranslate2 (CT2).
"""

from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

from .base import BaseConverter
from ..config import WorkspaceConfig, get_config
from ..env_manager import EnvironmentManager


class GPTQModelConverter(BaseConverter):
    """Quantizes Safetensors models using gptqmodel into 4-bit or 8-bit GPTQ."""

    name = "gptqmodel"
    supported_inputs = ["safetensors", "hf_repo"]
    supported_outputs = ["gptq", "safetensors"]

    def check_installed(self) -> bool:
        try:
            import gptqmodel
            return True
        except ImportError:
            return False

    def install(self) -> bool:
        print("[GPTQModelConverter] Installing gptqmodel...")
        em = EnvironmentManager(self.config)
        em.install_profile("gptqmodel")
        return self.check_installed()

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
        **kwargs
    ) -> Path:
        src = Path(input_path).resolve()
        bits_val = 8 if str(target_quant).lower().startswith("8") else bits
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(output_path).resolve() if output_path else (outputs_dir / f"{src.stem}_gptq_{bits_val}bit")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GPTQModelConverter] Quantizing {src.name} to {bits_val}-bit GPTQ ({out_dir.name})...")
        code = f"""
import sys
try:
    from gptqmodel import GPTQModel, QuantizeConfig
    quant_config = QuantizeConfig(bits={bits_val}, group_size={group_size})
    model = GPTQModel.from_pretrained("{str(src)}", quantize_config=quant_config)
    model.quantize()
    model.save_quantized("{str(out_dir)}")
    print("[GPTQModel] Quantization completed successfully.")
except Exception as e:
    print(f"[GPTQModel] Error during quantization: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"GPTQModel quantization failed:\n{res.stderr or res.stdout}")

        if cleanup_source and src.resolve() != out_dir.resolve():
            try:
                if src.is_file():
                    src.unlink()
                elif src.is_dir():
                    shutil.rmtree(str(src))
            except Exception:
                pass

        return out_dir


class AutoAWQConverter(BaseConverter):
    """Quantizes Safetensors models using autoawq or llm-compressor (AWQ 4-bit & FP8)."""

    name = "autoawq"
    supported_inputs = ["safetensors", "hf_repo"]
    supported_outputs = ["awq", "fp8", "safetensors"]

    def check_installed(self) -> bool:
        try:
            import awq
            return True
        except ImportError:
            return False

    def install(self) -> bool:
        print("[AutoAWQConverter] Installing autoawq...")
        em = EnvironmentManager(self.config)
        em.install_profile("autoawq")
        return self.check_installed()

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
        **kwargs
    ) -> Path:
        src = Path(input_path).resolve()
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(output_path).resolve() if output_path else (outputs_dir / f"{src.stem}_awq")
        out_dir.mkdir(parents=True, exist_ok=True)

        is_fp8 = "fp8" in str(target_quant).lower()
        print(f"[AutoAWQConverter] Quantizing {src.name} to {'FP8' if is_fp8 else 'AWQ 4-bit'}...")

        code = f"""
import sys
try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("{str(src)}", trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained("{str(src)}", low_cpu_mem_usage=True)
    quant_config = {{"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}}
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized("{str(out_dir)}")
    tokenizer.save_pretrained("{str(out_dir)}")
    print("[AutoAWQ] Quantization completed successfully.")
except Exception as e:
    print(f"[AutoAWQ] Error during quantization: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"AutoAWQ quantization failed:\n{res.stderr or res.stdout}")

        if cleanup_source and src.resolve() != out_dir.resolve():
            try:
                if src.is_file():
                    src.unlink()
                elif src.is_dir():
                    shutil.rmtree(str(src))
            except Exception:
                pass

        return out_dir


class ExLlamaConverter(BaseConverter):
    """Converts Safetensors models to fractional bpw EXL2 format using exllamav2."""

    name = "exllamav2"
    supported_inputs = ["safetensors", "hf_repo"]
    supported_outputs = ["exl2"]

    def check_installed(self) -> bool:
        try:
            import exllamav2
            return True
        except ImportError:
            return False

    def install(self) -> bool:
        print("[ExLlamaConverter] Installing exllamav2...")
        em = EnvironmentManager(self.config)
        em.install_profile("exllamav2")
        return self.check_installed()

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
        **kwargs
    ) -> Path:
        src = Path(input_path).resolve()
        bpw = float(target_quant) if (target_quant and target_quant.replace(".", "", 1).isdigit()) else bits_per_weight
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(output_path).resolve() if output_path else (outputs_dir / f"{src.stem}_{bpw:.1f}bpw_exl2")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ExLlamaConverter] Converting {src.name} to EXL2 ({bpw:.1f} bpw)...")
        convert_script = shutil.which("exllamav2_convert") or "exllamav2/convert.py"
        cmd = [
            sys.executable,
            "-m", "exllamav2.convert",
            "-i", str(src),
            "-o", str(out_dir),
            "-b", str(bpw)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"ExLlamaV2 conversion failed:\n{res.stderr or res.stdout}")

        if cleanup_source and src.resolve() != out_dir.resolve():
            try:
                if src.is_file():
                    src.unlink()
                elif src.is_dir():
                    shutil.rmtree(str(src))
            except Exception:
                pass

        return out_dir


class CT2AudioConverter(BaseConverter):
    """Converts Hugging Face Whisper models to faster-whisper CTranslate2 format."""

    name = "ct2"
    supported_inputs = ["safetensors", "hf_repo", "audio"]
    supported_outputs = ["ct2"]

    def check_installed(self) -> bool:
        return shutil.which("ct2-transformers-converter") is not None

    def install(self) -> bool:
        print("[CT2AudioConverter] Installing ctranslate2...")
        em = EnvironmentManager(self.config)
        em.install_profile("audio")
        return self.check_installed()

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
        **kwargs
    ) -> Path:
        src = Path(input_path).resolve()
        quant = target_quant or "float16"
        outputs_dir = self.config.runtime_outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(output_path).resolve() if output_path else (outputs_dir / f"{src.stem}_ct2")
        out_dir.mkdir(parents=True, exist_ok=True)

        converter_bin = shutil.which("ct2-transformers-converter")
        if not converter_bin:
            self.install()
            converter_bin = shutil.which("ct2-transformers-converter")

        print(f"[CT2AudioConverter] Converting Whisper model {src.name} to CTranslate2 ({quant})...")
        cmd = [
            converter_bin or "ct2-transformers-converter",
            "--model", str(src),
            "--output_dir", str(out_dir),
            "--quantization", str(quant),
            "--copy_files", "tokenizer.json", "preprocessor_config.json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"CTranslate2 conversion failed:\n{res.stderr or res.stdout}")

        if cleanup_source and src.resolve() != out_dir.resolve():
            try:
                if src.is_file():
                    src.unlink()
                elif src.is_dir():
                    shutil.rmtree(str(src))
            except Exception:
                pass

        return out_dir
