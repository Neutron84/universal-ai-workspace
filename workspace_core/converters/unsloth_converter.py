"""
Unsloth Converter & Dynamic GGUF Quantizer Plugin
Supports loading Hugging Face models and LoRA adapters, merging LoRA adapter weights,
and exporting to Unsloth Dynamic GGUF formats (q4_k_xl, q4_k_m, q5_k_m, q8_0, f16).
"""

from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

from .base import BaseConverter
from ..config import WorkspaceConfig, get_config
from ..env_manager import EnvironmentManager


class UnslothConverter(BaseConverter):
    """Merges LoRA adapters and exports models to Unsloth Dynamic GGUF formats."""

    name = "unsloth"
    supported_inputs = ["safetensors", "hf_repo", "lora"]
    supported_outputs = ["gguf", "safetensors"]

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        super().__init__(config)
        self.outputs_dir = self.config.runtime_outputs_dir

    def check_installed(self) -> bool:
        try:
            import unsloth
            return True
        except ImportError:
            return False

    def install(self) -> bool:
        print("[UnslothConverter] Installing unsloth and dependencies...")
        em = EnvironmentManager(self.config)
        em.install_profile("unsloth")
        return self.check_installed()

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
        **kwargs
    ) -> Path:
        """
        Loads base model and LoRA weights, optionally merges them,
        and saves as standard or Dynamic GGUF.
        """
        src = Path(input_path).resolve()
        target_q = (target_quant or "q4_k_m").lower()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        if output_path:
            out_file = Path(output_path).resolve()
        else:
            out_file = self.outputs_dir / f"{src.stem}_{target_q}.gguf"

        out_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"[UnslothConverter] Exporting model {src.name} to GGUF ({target_q})...")

        # Python execution script to run inside isolated or current environment
        export_code = f"""
import sys
from pathlib import Path

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="{str(src)}",
        max_seq_length={maximum_context_length},
        load_in_4bit=True,
    )
    print("[Unsloth] Model loaded. Exporting to GGUF {target_q}...")
    model.save_pretrained_gguf(
        "{str(out_file.parent / out_file.stem)}",
        tokenizer,
        quantization_method="{target_q}",
    )
    print("[Unsloth] Dynamic GGUF export complete.")
except Exception as e:
    print(f"[Unsloth] Unsloth export error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        res = subprocess.run([sys.executable, "-c", export_code], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"[UnslothConverter] FastLanguageModel export note: {res.stderr.strip()[:200]}")
            print("[UnslothConverter] Falling back to llama.cpp conversion pipeline...")
            from .llama_cpp_converter import LlamaCppConverter
            fallback = LlamaCppConverter(self.config)
            return fallback.convert_or_quantize(
                input_path=src,
                output_path=out_file,
                target_quant=target_q.upper(),
                cleanup_source=cleanup_source,
                **kwargs
            )

        # Locate generated file
        candidates = list(out_file.parent.glob(f"{out_file.stem}*.gguf"))
        final_path = candidates[0] if candidates else out_file

        print(f"[UnslothConverter] Conversion finished: {final_path}")
        if cleanup_source and src.resolve() != final_path.resolve():
            try:
                if src.is_file():
                    src.unlink()
                elif src.is_dir():
                    shutil.rmtree(str(src))
            except Exception:
                pass

        return final_path
