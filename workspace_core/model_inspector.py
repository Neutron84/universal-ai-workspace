"""
Universal Model Inspector
Inspects HuggingFace Hub repositories, direct URLs, CivitAI models, and local files.
Auto-classifies model architectures (LLM, VLM, Embeddings, Safetensors, Diffusion, Audio).
Handles split/sharded GGUF detection, non-uniform ik_llama quants, MoE, and LoRA adapters.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download
from .security import SecurityManager
from .io_resolver import IOResolver, InputMode, ResolvedInput


# Known GGUF quantization patterns (including ik_llama non-uniform quants)
QUANT_ORDER = [
    "Q8_0",
    "Q6_K",
    "IQ6_K",
    "Q5_K_M",
    "IQ5_K",
    "Q5_K_S",
    "Q5_0",
    "Q5_1",
    "Q4_K_M",
    "IQ4_K",
    "IQ4_KS",
    "Q4_K_S",
    "Q4_0",
    "Q4_1",
    "IQ4_XS",
    "IQ4_NL",
    "Q3_K_L",
    "Q3_K_M",
    "IQ3_K",
    "IQ3_KS",
    "Q3_K_S",
    "IQ3_M",
    "IQ3_S",
    "Q2_K",
    "IQ2_K",
    "IQ2_KS",
    "IQ2_M",
    "IQ2_S",
    "IQ1_S",
    "IQ1_M",
]

IK_QUANT_TAGS = {
    "IQ4_K", "IQ4_KS", "IQ3_K", "IQ3_KS", "IQ2_K", "IQ2_KS",
    "IQ5_K", "IQ6_K", "_R4", "_R16"
}

# Quantization relative size factor relative to F16 (used for re-quantization size estimates)
QUANT_SIZE_FACTORS = {
    "F32": 2.0,
    "F16": 1.0,
    "BF16": 1.0,
    "Q8_0": 0.54,
    "Q6_K": 0.43,
    "IQ6_K": 0.43,
    "Q5_K_M": 0.38,
    "IQ5_K": 0.38,
    "Q5_K_S": 0.36,
    "Q5_0": 0.37,
    "Q5_1": 0.40,
    "Q4_K_M": 0.31,
    "IQ4_K": 0.31,
    "IQ4_KS": 0.30,
    "Q4_K_S": 0.30,
    "Q4_0": 0.29,
    "Q4_1": 0.33,
    "IQ4_XS": 0.28,
    "IQ4_NL": 0.28,
    "Q3_K_L": 0.27,
    "Q3_K_M": 0.25,
    "IQ3_K": 0.25,
    "IQ3_KS": 0.24,
    "Q3_K_S": 0.24,
    "IQ3_M": 0.24,
    "IQ3_S": 0.23,
    "Q2_K": 0.18,
    "IQ2_K": 0.18,
    "IQ2_KS": 0.17,
    "IQ2_M": 0.17,
    "IQ2_S": 0.16,
    "IQ1_M": 0.12,
    "IQ1_S": 0.11,
}


@dataclass
class GGUFVariant:
    name: str  # e.g., "Meta-Llama-3.1-8B-Instruct-Q4_K_M"
    quant: str  # e.g., "Q4_K_M"
    is_split: bool  # True if composed of multiple shards
    shard_files: List[str]  # e.g. ["model-00001-of-00003.gguf", ...]
    total_size_bytes: int
    total_size_gb: float
    is_oversized: bool = False
    is_ik_quant: bool = False
    mmproj_file: Optional[str] = None


@dataclass
class InspectionResult:
    source_type: str  # "hf_repo", "direct_url", "civitai", "local"
    raw_source: str
    model_id: str
    model_type: str  # "llm-gguf", "vlm-gguf", "embedding-gguf", "llm-safetensors", "diffusion", "audio"
    repo_id: Optional[str] = None
    pipeline_tag: Optional[str] = None
    total_files_count: int = 0
    total_repo_size_bytes: int = 0
    gguf_variants: List[GGUFVariant] = field(default_factory=list)
    mmproj_files: List[str] = field(default_factory=list)
    safetensors_files: List[str] = field(default_factory=list)
    primary_download_files: List[str] = field(default_factory=list)
    config_json: Optional[Dict[str, Any]] = None
    suggested_backend: str = "llama.cpp"
    is_moe: bool = False
    is_lora: bool = False
    is_vlm: bool = False
    is_ik_quant: bool = False
    quant_format: str = "unknown"
    architecture_name: Optional[str] = None
    error: Optional[str] = None


class ModelInspector:
    """Inspects model repositories, URLs, and local files to determine structure and sizing."""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or SecurityManager.get_hf_token()
        self.api = HfApi(token=self.hf_token)

    def inspect(self, source: str, input_mode: Optional[Union[str, InputMode]] = None) -> InspectionResult:
        """Inspects any input source and returns structured metadata with IOResolver integration."""
        source = source.strip()
        if not source:
            return InspectionResult(
                source_type="unknown",
                raw_source=source,
                model_id="empty",
                model_type="unknown",
                error="Source input is empty."
            )

        resolved_input = IOResolver.resolve_input(source, mode=input_mode or InputMode.AUTO)

        # 1. Local path check (direct, google_drive, gdrive_relative)
        if resolved_input.mode in (InputMode.LOCAL_PATH, InputMode.DIRECT, InputMode.GOOGLE_DRIVE, InputMode.GDRIVE_RELATIVE):
            if resolved_input.local_path and resolved_input.local_path.exists():
                return self._inspect_local(resolved_input.local_path)
            elif Path(source).exists():
                return self._inspect_local(Path(source))

        # 2. CivitAI check
        if resolved_input.mode == InputMode.CIVITAI:
            return self._inspect_civitai(source, resolved_input)

        # 3. Google Drive Share check
        if resolved_input.mode == InputMode.GDRIVE_SHARE:
            return InspectionResult(
                source_type="gdrive_share",
                raw_source=source,
                model_id=f"gdrive-{resolved_input.identifier}",
                model_type="llm-gguf",
                suggested_backend="llama.cpp",
                config_json=resolved_input.extra_metadata,
            )

        # 4. Git Repo check
        if resolved_input.mode == InputMode.GIT_REPO:
            return InspectionResult(
                source_type="git_repo",
                raw_source=source,
                model_id=resolved_input.identifier,
                model_type="llm-safetensors",
                suggested_backend="transformers",
                config_json=resolved_input.extra_metadata,
            )

        # 5. Direct URL check (e.g. ends with .gguf or direct http download)
        if resolved_input.mode == InputMode.URL:
            return self._inspect_direct_url(source)

        # 6. Hugging Face check
        if resolved_input.mode == InputMode.HUGGINGFACE:
            if resolved_input.subpath_or_file:
                return self._inspect_hf_file(resolved_input.identifier, resolved_input.subpath_or_file, source)
            return self._inspect_hf_repo(resolved_input.identifier)

        # Fallback to local check
        local_path = Path(source)
        if local_path.exists():
            return self._inspect_local(local_path)

        # Direct URL check
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            if "huggingface.co" in parsed.netloc:
                hf_match = self._parse_hf_url(source)
                if hf_match:
                    repo_id, file_path = hf_match
                    if file_path:
                        return self._inspect_hf_file(repo_id, file_path, source)
                    return self._inspect_hf_repo(repo_id)
            return self._inspect_direct_url(source)

        if "/" in source and not source.startswith("http"):
            return self._inspect_hf_repo(source)

        return InspectionResult(
            source_type="unknown",
            raw_source=source,
            model_id=source,
            model_type="unknown",
            error=f"Unrecognized source format: '{source}'"
        )

    def _parse_hf_url(self, url: str) -> Optional[Tuple[str, Optional[str]]]:
        """Extracts (repo_id, subpath/filename) from Hugging Face URLs."""
        pattern = r"huggingface\.co/([^/]+/[^/]+)(?:/(?:resolve|blob)/[^/]+/(.+))?"
        match = re.search(pattern, url)
        if match:
            repo_id = match.group(1)
            file_path = match.group(2)
            return repo_id, file_path
        return None

    def _extract_architecture_and_quant_flags(
        self,
        config_data: Dict[str, Any],
        file_names: List[str],
        repo_id: str = "",
        tags: Optional[List[str]] = None,
        variants: Optional[List[GGUFVariant]] = None,
    ) -> Dict[str, Any]:
        """Inspects config.json data, file listings, and tags to detect MoE, LoRA, VLM, and Quantization format."""
        tags = tags or []
        lower_tags = [t.lower() for t in tags]
        lower_repo = repo_id.lower()
        lower_files = [f.lower() for f in file_names]

        # 1. MoE Architecture Check
        is_moe = False
        num_local_experts = config_data.get("num_local_experts") or config_data.get("n_routed_experts") or config_data.get("num_experts") or config_data.get("moe_num_experts") or 0
        if int(num_local_experts) > 1:
            is_moe = True

        architectures = config_data.get("architectures", [])
        architecture_name = architectures[0] if architectures else None
        if architecture_name:
            arch_lower = architecture_name.lower()
            if any(k in arch_lower for k in ("moe", "mixtral", "deepseek", "qwen2moe", "dbrx", "jamba")):
                is_moe = True

        if any(k in lower_repo for k in ("mixtral", "deepseek-v", "qwen2-moe", "dbrx")):
            is_moe = True

        # 2. LoRA Adapter Check
        is_lora = False
        if any(f in ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors") for f in file_names):
            is_lora = True
        elif "lora" in lower_repo.split("/")[-1] or any("lora" in t for t in lower_tags):
            is_lora = True
        elif config_data.get("is_lora", False):
            is_lora = True

        # 3. Vision-Language (VLM) Check
        is_vlm = False
        has_mmproj = any("mmproj" in f for f in lower_files)
        if has_mmproj:
            is_vlm = True
        elif config_data.get("image_token_index") is not None or "vision_config" in config_data or "visual" in config_data:
            is_vlm = True
        elif architecture_name and any(k in architecture_name.lower() for k in ("vl", "vision", "llava", "qwen2vl", "paligemma")):
            is_vlm = True

        # 4. ik_llama Quantization Check
        is_ik_quant = False
        for f in file_names:
            upper_f = f.upper()
            if any(ik_tag in upper_f for ik_tag in IK_QUANT_TAGS):
                is_ik_quant = True
                break
        if variants:
            for v in variants:
                if v.quant.upper() in IK_QUANT_TAGS or v.is_ik_quant:
                    is_ik_quant = True
                    break

        # 5. Quantization Format Classification
        has_gguf = any(f.endswith(".gguf") for f in lower_files)
        has_safetensors = any(f.endswith(".safetensors") for f in lower_files)

        quant_cfg = config_data.get("quantization_config", {})
        quant_method = str(quant_cfg.get("quant_method", "")).lower()

        quant_format = "unknown"
        if is_ik_quant:
            quant_format = "gguf_ik"
        elif has_gguf:
            quant_format = "gguf_standard"
        elif quant_method == "gptq" or "gptq" in lower_repo or any("gptq" in f for f in lower_files):
            quant_format = "gptq"
        elif quant_method == "awq" or "awq" in lower_repo or any("awq" in f for f in lower_files):
            quant_format = "awq"
        elif quant_method in ("fp8", "float8") or "fp8" in lower_repo or any("fp8" in f for f in lower_files):
            quant_format = "fp8"
        elif "exl2" in lower_repo or any(f.endswith(".exl2") for f in lower_files):
            quant_format = "exl2"
        elif quant_cfg.get("load_in_4bit", False):
            quant_format = "bnb_4bit"
        elif has_safetensors:
            quant_format = "unquantized_fp16"

        return {
            "is_moe": is_moe,
            "is_lora": is_lora,
            "is_vlm": is_vlm,
            "is_ik_quant": is_ik_quant,
            "quant_format": quant_format,
            "architecture_name": architecture_name,
        }

    def _inspect_hf_repo(self, repo_id: str) -> InspectionResult:
        """Inspects a Hugging Face Hub repository using files_metadata and config inspection."""
        try:
            info = self.api.model_info(repo_id, files_metadata=True)
        except Exception as e:
            return InspectionResult(
                source_type="hf_repo",
                raw_source=repo_id,
                model_id=repo_id.split("/")[-1],
                model_type="unknown",
                repo_id=repo_id,
                error=f"HuggingFace API error: {str(e)}"
            )

        siblings = getattr(info, "siblings", []) or []
        pipeline_tag = getattr(info, "pipeline_tag", None)
        tags = getattr(info, "tags", []) or []

        all_files: Dict[str, int] = {}
        total_size = 0
        file_names = []
        for s in siblings:
            fname = getattr(s, "rfilename", "")
            fsize = getattr(s, "size", 0) or 0
            all_files[fname] = fsize
            total_size += fsize
            file_names.append(fname)

        # Attempt to load config.json if available
        config_data: Dict[str, Any] = {}
        if "config.json" in file_names:
            try:
                local_cfg = hf_hub_download(repo_id=repo_id, filename="config.json", token=self.hf_token)
                with open(local_cfg, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        if "adapter_config.json" in file_names and not config_data:
            try:
                local_cfg = hf_hub_download(repo_id=repo_id, filename="adapter_config.json", token=self.hf_token)
                with open(local_cfg, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        # Classify files
        gguf_files = {f: sz for f, sz in all_files.items() if f.lower().endswith(".gguf")}
        safetensors_files = [f for f in all_files.keys() if f.lower().endswith(".safetensors")]
        mmproj_files = [f for f in gguf_files.keys() if "mmproj" in f.lower()]

        # Parse GGUF variants
        gguf_variants = self._parse_gguf_variants(gguf_files, mmproj_files)

        # Detect architecture flags
        flags = self._extract_architecture_and_quant_flags(
            config_data=config_data,
            file_names=file_names,
            repo_id=repo_id,
            tags=tags,
            variants=gguf_variants,
        )

        # Detect model architecture type
        model_type, suggested_backend = self._classify_architecture(
            repo_id=repo_id,
            pipeline_tag=pipeline_tag,
            tags=tags,
            has_gguf=len(gguf_files) > 0,
            has_safetensors=len(safetensors_files) > 0,
            has_mmproj=len(mmproj_files) > 0 or flags["is_vlm"]
        )

        return InspectionResult(
            source_type="hf_repo",
            raw_source=repo_id,
            model_id=repo_id.split("/")[-1],
            model_type=model_type,
            repo_id=repo_id,
            pipeline_tag=pipeline_tag,
            total_files_count=len(all_files),
            total_repo_size_bytes=total_size,
            gguf_variants=gguf_variants,
            mmproj_files=mmproj_files,
            safetensors_files=safetensors_files,
            primary_download_files=list(gguf_files.keys()) if gguf_files else safetensors_files,
            suggested_backend=suggested_backend,
            config_json=config_data,
            is_moe=flags["is_moe"],
            is_lora=flags["is_lora"],
            is_vlm=flags["is_vlm"],
            is_ik_quant=flags["is_ik_quant"],
            quant_format=flags["quant_format"],
            architecture_name=flags["architecture_name"],
        )

    def _inspect_hf_file(self, repo_id: str, file_path: str, raw_url: str) -> InspectionResult:
        """Inspects a specific file inside a Hugging Face Hub repository."""
        base_res = self._inspect_hf_repo(repo_id)
        if base_res.error:
            return base_res

        base_res.raw_source = raw_url
        target_fname = Path(file_path).name

        filtered_variants = []
        for v in base_res.gguf_variants:
            if target_fname in v.shard_files or target_fname == v.name:
                filtered_variants.append(v)

        if filtered_variants:
            base_res.gguf_variants = filtered_variants
        else:
            base_res.primary_download_files = [file_path]

        return base_res

    def _inspect_direct_url(self, url: str) -> InspectionResult:
        """Inspects a direct HTTP/HTTPS URL download."""
        fname = Path(urlparse(url).path).name or "downloaded_model"
        is_gguf = fname.lower().endswith(".gguf")
        model_type = "llm-gguf" if is_gguf else "llm-safetensors"
        backend = "llama.cpp" if is_gguf else "transformers"

        variants = []
        is_ik = False
        if is_gguf:
            quant = self._extract_quant_tag(fname)
            is_ik = any(ik_tag in fname.upper() for ik_tag in IK_QUANT_TAGS)
            variants.append(
                GGUFVariant(
                    name=fname,
                    quant=quant,
                    is_split=False,
                    shard_files=[fname],
                    total_size_bytes=0,
                    total_size_gb=0.0,
                    is_ik_quant=is_ik
                )
            )

        return InspectionResult(
            source_type="direct_url",
            raw_source=url,
            model_id=fname,
            model_type=model_type,
            primary_download_files=[fname],
            gguf_variants=variants,
            suggested_backend="ik_llama" if is_ik else backend,
            is_ik_quant=is_ik,
            quant_format="gguf_ik" if is_ik else ("gguf_standard" if is_gguf else "unquantized_fp16")
        )

    def _inspect_civitai(self, url: str, resolved_input: Optional[ResolvedInput] = None) -> InspectionResult:
        """Inspects a CivitAI URL using live CivitAI API metadata resolution."""
        try:
            model_id = resolved_input.extra_metadata.get("model_id") if resolved_input else None
            version_id = resolved_input.extra_metadata.get("version_id") if resolved_input else None
            if not model_id:
                m_id, v_id = IOResolver._extract_civitai_ids(url)
                model_id, version_id = m_id, v_id

            civitai_info = IOResolver.resolve_civitai_model(model_id=str(model_id), version_id=version_id)
            total_bytes = int(civitai_info.get("size_kb", 0)) * 1024

            return InspectionResult(
                source_type="civitai",
                raw_source=url,
                model_id=civitai_info.get("name", f"civitai-{model_id}"),
                model_type="diffusion",
                total_files_count=1,
                total_repo_size_bytes=total_bytes,
                primary_download_files=[civitai_info.get("primary_file_name", "model.safetensors")],
                config_json=civitai_info,
                suggested_backend="diffusers"
            )
        except Exception as e:
            print(f"[ModelInspector] CivitAI lookup note ({e}), returning default profile.")
            return InspectionResult(
                source_type="civitai",
                raw_source=url,
                model_id="civitai_model",
                model_type="diffusion",
                suggested_backend="diffusers"
            )

    def _inspect_local(self, path: Path) -> InspectionResult:
        """Inspects a local model file or directory."""
        if path.is_file():
            fname = path.name
            size = path.stat().st_size
            is_gguf = fname.lower().endswith(".gguf")
            model_type = "llm-gguf" if is_gguf else "llm-safetensors"
            quant = self._extract_quant_tag(fname) if is_gguf else "N/A"
            is_ik = any(ik_tag in fname.upper() for ik_tag in IK_QUANT_TAGS)

            variants = [
                GGUFVariant(
                    name=fname,
                    quant=quant,
                    is_split=False,
                    shard_files=[str(path.resolve())],
                    total_size_bytes=size,
                    total_size_gb=size / (1024**3),
                    is_ik_quant=is_ik
                )
            ] if is_gguf else []

            backend = "ik_llama" if is_ik else ("llama.cpp" if is_gguf else "transformers")

            return InspectionResult(
                source_type="local",
                raw_source=str(path),
                model_id=path.stem,
                model_type=model_type,
                total_files_count=1,
                total_repo_size_bytes=size,
                gguf_variants=variants,
                primary_download_files=[str(path.resolve())],
                suggested_backend=backend,
                is_ik_quant=is_ik,
                quant_format="gguf_ik" if is_ik else ("gguf_standard" if is_gguf else "unquantized_fp16")
            )

        # Directory inspection
        files = list(path.glob("**/*"))
        file_names = [f.name for f in files]
        gguf_files = {str(f.relative_to(path)): f.stat().st_size for f in files if f.suffix.lower() == ".gguf"}
        safetensors = [str(f.relative_to(path)) for f in files if f.suffix.lower() == ".safetensors"]
        mmproj = [f for f in gguf_files.keys() if "mmproj" in f.lower()]
        variants = self._parse_gguf_variants(gguf_files, mmproj)

        config_data = {}
        cfg_path = path / "config.json"
        if cfg_path.is_file():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        flags = self._extract_architecture_and_quant_flags(
            config_data=config_data,
            file_names=file_names,
            repo_id=path.name,
            variants=variants,
        )

        model_type, suggested_backend = self._classify_architecture(
            repo_id=path.name,
            pipeline_tag=None,
            tags=[],
            has_gguf=len(gguf_files) > 0,
            has_safetensors=len(safetensors) > 0,
            has_mmproj=len(mmproj) > 0 or flags["is_vlm"]
        )

        if flags["is_ik_quant"]:
            suggested_backend = "ik_llama"

        return InspectionResult(
            source_type="local",
            raw_source=str(path),
            model_id=path.name,
            model_type=model_type,
            total_files_count=len(files),
            total_repo_size_bytes=sum(f.stat().st_size for f in files if f.is_file()),
            gguf_variants=variants,
            mmproj_files=mmproj,
            safetensors_files=safetensors,
            suggested_backend=suggested_backend,
            config_json=config_data,
            is_moe=flags["is_moe"],
            is_lora=flags["is_lora"],
            is_vlm=flags["is_vlm"],
            is_ik_quant=flags["is_ik_quant"],
            quant_format=flags["quant_format"],
            architecture_name=flags["architecture_name"],
        )

    def _parse_gguf_variants(self, gguf_files: Dict[str, int], mmproj_files: List[str]) -> List[GGUFVariant]:
        """
        Parses all GGUF files in a repo, properly grouping split/sharded files
        into single logical variants with summed sizes.
        """
        split_pattern = re.compile(r"^(.*?)[.-](\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
        variants_map: Dict[str, Dict[str, Any]] = {}
        default_mmproj = mmproj_files[0] if mmproj_files else None

        for fname, size in gguf_files.items():
            if fname in mmproj_files:
                continue

            split_match = split_pattern.match(fname)
            if split_match:
                base_name = split_match.group(1)
                total_shards = int(split_match.group(3))
                key = f"{base_name} (Split {total_shards} shards)"

                if key not in variants_map:
                    quant = self._extract_quant_tag(base_name)
                    is_ik = any(ik in base_name.upper() for ik in IK_QUANT_TAGS)
                    variants_map[key] = {
                        "name": base_name,
                        "quant": quant,
                        "is_split": True,
                        "shard_files": [],
                        "total_size": 0,
                        "mmproj": default_mmproj,
                        "is_ik_quant": is_ik,
                    }
                variants_map[key]["shard_files"].append(fname)
                variants_map[key]["total_size"] += size
            else:
                base_name = fname.replace(".gguf", "")
                quant = self._extract_quant_tag(fname)
                is_ik = any(ik in fname.upper() for ik in IK_QUANT_TAGS)
                variants_map[fname] = {
                    "name": base_name,
                    "quant": quant,
                    "is_split": False,
                    "shard_files": [fname],
                    "total_size": size,
                    "mmproj": default_mmproj,
                    "is_ik_quant": is_ik,
                }

        results = []
        for _, v in variants_map.items():
            v["shard_files"].sort()
            total_bytes = v["total_size"]
            total_gb = total_bytes / (1024**3)
            results.append(
                GGUFVariant(
                    name=v["name"],
                    quant=v["quant"],
                    is_split=v["is_split"],
                    shard_files=v["shard_files"],
                    total_size_bytes=total_bytes,
                    total_size_gb=round(total_gb, 2),
                    mmproj_file=v["mmproj"],
                    is_ik_quant=v.get("is_ik_quant", False),
                )
            )

        def sort_key(v: GGUFVariant):
            try:
                return QUANT_ORDER.index(v.quant.upper())
            except ValueError:
                return 999

        results.sort(key=sort_key)
        return results

    @staticmethod
    def _extract_quant_tag(name: str) -> str:
        """Extracts quantization tag from a filename."""
        upper = name.upper()
        for q in QUANT_ORDER:
            if q in upper:
                return q
        for q in ["F16", "BF16", "F32"]:
            if q in upper:
                return q
        m = re.search(r"(Q\d_[A-Z0-9_]+|IQ\d_[A-Z0-9_]+)", upper)
        if m:
            return m.group(1)
        return "UNKNOWN"

    @staticmethod
    def _classify_architecture(
        repo_id: str,
        pipeline_tag: Optional[str],
        tags: List[str],
        has_gguf: bool,
        has_safetensors: bool,
        has_mmproj: bool
    ) -> Tuple[str, str]:
        """Classifies architecture into model_type and default backend engine."""
        lower_repo = repo_id.lower()
        lower_tags = [t.lower() for t in tags]

        # Audio check
        if pipeline_tag in ["automatic-speech-recognition", "text-to-speech", "audio-to-audio"] or "whisper" in lower_repo:
            return "audio", "faster-whisper"

        # Diffusion check
        if pipeline_tag in ["text-to-image", "image-to-image"] or any("diffusers" in t or "stable-diffusion" in t for t in lower_tags):
            return "diffusion", "diffusers"

        # VLM (Vision-Language) check
        if has_mmproj or "vl" in lower_repo or "vision" in lower_repo or pipeline_tag == "image-text-to-text":
            return "vlm-gguf" if has_gguf else "llm-safetensors", "llama.cpp" if has_gguf else "transformers"

        # Embedding check
        if pipeline_tag == "feature-extraction" or any("embedding" in t for t in lower_tags) or "bge" in lower_repo or "gte" in lower_repo:
            return "embedding-gguf" if has_gguf else "llm-safetensors", "llama.cpp" if has_gguf else "transformers"

        # General LLM
        if has_gguf:
            return "llm-gguf", "llama.cpp"
        elif has_safetensors:
            return "llm-safetensors", "vllm"
        return "unknown", "llama.cpp"
