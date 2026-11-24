"""
Unsloth Converter & Dynamic GGUF Quantizer Plugin
Supports loading Hugging Face models and LoRA adapters, merging LoRA adapter weights,
and exporting to Unsloth Dynamic GGUF formats (q4_k_xl, q4_k_m, q5_k_m, q8_0, f16) or merged Safetensors.
Runs in isolated unsloth virtualenv when available.
"""

from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Union

from .base import BaseConverter
from ..config import WorkspaceConfig
from ..env_manager import EnvironmentManager


class UnslothConverter(BaseConverter):
    """Merges LoRA adapters and exports models to Unsloth Dynamic GGUF or merged Safetensors formats."""

    name = "unsloth"
    supported_inputs = ["safetensors", "hf_repo", "lora"]
    supported_outputs = ["gguf", "safetensors"]

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.outputs_dir = self.config.runtime_outputs_dir

    def check_installed(self) -> bool:
        status = EnvironmentManager(self.config).verify_profile("unsloth")
        return status.get("unsloth", False)

    def install(self) -> bool:
        print("[UnslothConverter] Installing unsloth and dependencies...")
        em = EnvironmentManager(self.config)
        return em.install_profile("unsloth")

    def estimate_requirements(self, input_info: Any) -> Dict[str, Any]:
        size_bytes = getattr(input_info, "total_repo_size_bytes", 0) or 0
        size_gb = size_bytes / (1024**3)
        return {
            "estimated_ram_gb": max(12.0, size_gb * 1.5),
            "estimated_vram_gb": max(6.0, size_gb * 0.7),
            "estimated_disk_gb": size_gb * 2.5,
        }

    def convert_or_quantize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_quant: Optional[str] = "q4_k_m",
        cleanup_source: bool = False,
        base_model_name: Optional[str] = None,
        merge_lora: bool = True,
        maximum_context_length: int = 4096,
        output_format: str = "gguf",
        load_in_4bit: bool = True,
        **kwargs: Any,
    ) -> Path:
        """
        Loads base model or LoRA weights inside isolated unsloth python, merges LoRA weights if needed,
        and saves as Unsloth Dynamic GGUF or merged 16-bit/4-bit Safetensors.
        """
        src = Path(input_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Input path not found: {src}")

        target_q = (target_quant or "q4_k_m").lower()
        if target_q == "auto":
            target_q = "q4_k_m"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        is_safetensors_out = output_format.lower() == "safetensors" or target_q in ("merged_16bit", "merged_4bit", "safetensors")
        if output_path:
            out_target = Path(output_path).resolve()
        elif is_safetensors_out:
            out_target = self.outputs_dir / f"{src.stem}_merged"
        else:
            out_target = self.outputs_dir / f"{src.stem}_{target_q}.gguf"

        out_target.parent.mkdir(parents=True, exist_ok=True)
        py_exec = self.get_python_executable("unsloth")

        if not self.check_installed() and self.config.settings.auto_install_dependencies:
            self.install()
            py_exec = self.get_python_executable("unsloth")

        print(f"[UnslothConverter] Exporting {src.name} via Unsloth ({target_q}, format={output_format})...")

        save_method = "merged_16bit" if target_q in ("f16", "merged_16bit", "safetensors") else "merged_4bit"
        export_code = f"""
import sys
from pathlib import Path

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name={repr(str(src))},
        max_seq_length={int(maximum_context_length)},
        load_in_4bit={bool(load_in_4bit)},
    )
    is_st = {bool(is_safetensors_out)}
    if is_st:
        print("[Unsloth] Saving merged Safetensors model...")
        model.save_pretrained_merged(
            {repr(str(out_target))},
            tokenizer,
            save_method={repr(save_method)},
        )
    else:
        print("[Unsloth] Exporting to Dynamic GGUF {target_q}...")
        model.save_pretrained_gguf(
            {repr(str(out_target.parent / out_target.stem))},
            tokenizer,
            quantization_method={repr(target_q)},
        )
    print("[Unsloth] Export complete.")
except Exception as e:
    print(f"[Unsloth] Unsloth export error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        res = subprocess.run([py_exec, "-c", export_code], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"[UnslothConverter] FastLanguageModel export note: {res.stderr.strip()[:250]}")
            if not is_safetensors_out:
                print("[UnslothConverter] Falling back to llama.cpp conversion pipeline...")
                from .llama_cpp_converter import LlamaCppConverter
                fallback = LlamaCppConverter(self.config)
                return fallback.convert_or_quantize(
                    input_path=src,
                    output_path=out_target,
                    target_quant=target_q.upper(),
                    cleanup_source=cleanup_source,
                    **kwargs,
                )
            raise RuntimeError(f"Unsloth merged export failed:\n{res.stderr or res.stdout}")

        if is_safetensors_out:
            final_path = out_target
        else:
            candidates = list(out_target.parent.glob(f"{out_target.stem}*.gguf"))
            final_path = candidates[0] if candidates else out_target

        print(f"[UnslothConverter] Conversion finished: {final_path}")
        if cleanup_source:
            self.safe_cleanup(src, keep_paths=[final_path])

        return final_path
