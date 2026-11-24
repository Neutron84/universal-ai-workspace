"""
Workspace Pipeline Orchestrator
Unified controller coordinating input resolution, inspection, AutoSelector planning,
downloading, conversion/quantization, registry registration, server + tunnel launch,
benchmarking, and artifact export for Notebook, Dashboard, and CLI.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .auto_selector import AutoSelector, SelectionRecommendation
from .config import WorkspaceConfig, get_config
from .converters import get_converter
from .downloader import DownloadResult, Downloader
from .env_manager import EnvironmentManager
from .io_resolver import IOResolver, OutputManager, ResolvedInput
from .model_inspector import InspectionResult, ModelInspector
from .process_manager import ProcessManager, SystemState
from .registry import ModelEntry, ModelRegistry
from .security import SecurityManager


@dataclass
class LaunchRequest:
    # Input
    input_type: str = "auto"
    input_path: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    input_file: Optional[str] = None
    hf_revision: Optional[str] = None
    unpack_archive: bool = True
    model_type_override: Optional[str] = None

    # Task & Routing (Auto vs Manual)
    task: str = "serve"  # serve, convert_and_quantize, convert_and_serve, benchmark, inspect_only, dashboard_only, serve_and_dashboard
    strategy: str = "balanced"  # balanced, max_quality, max_speed, max_context, low_vram_safe
    engine: str = "auto"
    quantizer: str = "auto"
    target_quant: str = "auto"
    preset_name: Optional[str] = None

    # Detailed Server Tuning
    context_size: Union[int, str] = 4096
    gpu_offload_mode: str = "auto"  # auto, all, cpu, manual
    n_gpu_layers: int = -1
    flash_attn: Union[bool, str, None] = "auto"  # auto, True, False
    cache_type_k: str = "auto"
    cache_type_v: str = "auto"
    parallel: int = 1
    batch_size: int = 512
    ubatch_size: int = 512
    threads: int = 0
    split_mode: str = "layer"
    tensor_split: Optional[str] = None
    override_tensor: Optional[str] = None
    mlock: bool = False
    mmap: bool = True
    rope_scaling: Optional[str] = None
    rope_freq_base: Optional[float] = None
    chat_template: Optional[str] = None
    reasoning_format: Optional[str] = None
    quantization_mode: str = "auto"
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    mmproj_file: Optional[str] = None
    custom_server_args: str = ""

    # Conversion & Quantization Tuning
    outtype: str = "f16"
    use_imatrix: bool = False
    imatrix_chunks: int = 100
    calibration_text_path: Optional[str] = None
    split_max_size: Optional[str] = None
    merge_lora: bool = True
    base_model_name: Optional[str] = None
    gptq_bits: int = 4
    gptq_group_size: int = 128
    exl2_bpw: float = 4.0
    cleanup_intermediate: bool = True

    # Output & Export
    output_type: str = "none"
    output_path: str = "AI_Workspace/outputs"
    output_name: Optional[str] = None
    zip_output: bool = False
    hf_private: bool = False
    cleanup_after_export: bool = False

    # Network & Auth
    enable_tunnel: bool = True
    tunnel_type: str = "cloudflared"
    tunnel_token: Optional[str] = None
    tunnel_hostname: Optional[str] = None
    enable_api_auth: bool = True
    api_key_override: Optional[str] = None
    port: int = 8000
    dashboard_port: int = 7860

    # Credentials & Environment
    hf_token: Optional[str] = None
    civitai_token: Optional[str] = None
    ngrok_token: Optional[str] = None
    env_isolation_mode: Optional[str] = None
    auto_install_dependencies: Optional[bool] = None


@dataclass
class LaunchResult:
    task: str
    success: bool
    resolved_input: Optional[ResolvedInput] = None
    inspection: Optional[InspectionResult] = None
    recommendation: Optional[SelectionRecommendation] = None
    download_result: Optional[DownloadResult] = None
    model_entry: Optional[ModelEntry] = None
    artifact_path: Optional[Path] = None
    companion_files: List[Path] = field(default_factory=list)
    system_state: Optional[SystemState] = None
    benchmark_results: Optional[Dict[str, Any]] = None
    export_result: Optional[Dict[str, Any]] = None
    snippets: Tuple[str, str] = ("", "")
    message: str = ""


class WorkspaceOrchestrator:
    """Executes the full Universal AI Workspace lifecycle cleanly and consistently."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.env_manager = EnvironmentManager(self.config)
        self.registry = ModelRegistry(self.config)
        self.process_manager = ProcessManager(self.config)

    @staticmethod
    def _normalize_flash_attn(val: Union[bool, str, None]) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        clean = str(val).strip().lower()
        if clean in ("auto", ""):
            return None
        if clean in ("true", "on", "yes", "1"):
            return True
        if clean in ("false", "off", "no", "0"):
            return False
        return None

    def _apply_request_secrets_and_settings(self, req: LaunchRequest) -> None:
        if req.hf_token and req.hf_token.strip():
            SecurityManager.set_cached_secret("HF_TOKEN", req.hf_token.strip())
        if req.civitai_token and req.civitai_token.strip():
            SecurityManager.set_cached_secret("CIVITAI_API_KEY", req.civitai_token.strip())
        if req.ngrok_token and req.ngrok_token.strip():
            SecurityManager.set_cached_secret("NGROK_AUTHTOKEN", req.ngrok_token.strip())
        if req.tunnel_token and req.tunnel_token.strip() and req.tunnel_type == "cloudflared":
            SecurityManager.set_cached_secret("CF_TUNNEL_TOKEN", req.tunnel_token.strip())

        updates: Dict[str, Any] = {}
        if req.env_isolation_mode:
            updates["env_isolation_mode"] = req.env_isolation_mode
        if req.auto_install_dependencies is not None:
            updates["auto_install_dependencies"] = req.auto_install_dependencies
        if req.tunnel_hostname:
            updates["tunnel_hostname"] = req.tunnel_hostname
        if updates:
            self.config.update_settings(**updates)

    def _build_server_overrides(self, req: LaunchRequest, rec: SelectionRecommendation) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {
            "ctx_size": rec.recommended_ctx,
            "n_gpu_layers": rec.recommended_n_gpu_layers,
            "flash_attn": rec.flash_attn,
            "cache_type_k": rec.cache_type_k,
            "cache_type_v": rec.cache_type_v,
            "parallel": max(1, int(req.parallel or 1)),
            "batch_size": int(req.batch_size or 512),
            "ubatch_size": int(req.ubatch_size or 512),
            "threads": int(req.threads or 0),
            "split_mode": req.split_mode or "layer",
            "mlock": bool(req.mlock),
            "mmap": bool(req.mmap),
            "dtype": req.dtype or "auto",
            "gpu_memory_utilization": float(req.gpu_memory_utilization or 0.90),
        }
        if req.gpu_offload_mode:
            overrides["gpu_offload_mode"] = req.gpu_offload_mode
        if req.tensor_split:
            overrides["tensor_split"] = req.tensor_split
        if req.override_tensor:
            overrides["override_tensor"] = req.override_tensor
        elif rec.suggested_extra_args.get("override_tensor"):
            overrides["override_tensor"] = rec.suggested_extra_args["override_tensor"]
        if req.rope_scaling:
            overrides["rope_scaling"] = req.rope_scaling
        if req.rope_freq_base is not None:
            overrides["rope_freq_base"] = req.rope_freq_base
        if req.chat_template:
            overrides["chat_template"] = req.chat_template
        if req.reasoning_format:
            overrides["reasoning_format"] = req.reasoning_format
        if req.quantization_mode and req.quantization_mode != "auto":
            overrides["quantization_mode"] = req.quantization_mode
        elif rec.quantization_mode and rec.quantization_mode != "auto":
            overrides["quantization_mode"] = rec.quantization_mode
        if req.mmproj_file:
            overrides["mmproj_path"] = req.mmproj_file
        if req.custom_server_args and req.custom_server_args.strip():
            overrides["custom_server_args"] = req.custom_server_args.strip()
        return overrides

    def run(
        self,
        req: LaunchRequest,
        upload_handler: Optional[Callable[[], str]] = None,
    ) -> LaunchResult:
        """Executes the full orchestrated workflow described by LaunchRequest."""
        self._apply_request_secrets_and_settings(req)
        task = (req.task or "serve").lower().strip()

        if task == "dashboard_only":
            return LaunchResult(task=task, success=True, message="Dashboard launch requested.")

        hf_token = SecurityManager.get_hf_token()
        civitai_token = SecurityManager.get_civitai_api_key()
        inspector = ModelInspector(hf_token=hf_token)

        raw_input_path = req.input_path
        if req.input_type == "direct_upload" and upload_handler is not None:
            raw_input_path = upload_handler()

        resolved = IOResolver.resolve_input(
            uri=raw_input_path,
            mode=req.input_type,
            download_dir=self.config.runtime_root / "inputs",
            hf_token=hf_token,
            civitai_token=civitai_token,
            unpack_archive=req.unpack_archive,
            input_file=req.input_file,
            hf_revision=req.hf_revision,
        )

        insp_target = str(resolved.local_path) if resolved.local_path else raw_input_path
        insp = inspector.inspect(
            insp_target,
            input_mode=req.input_type,
            hf_revision=req.hf_revision,
            model_type_override=req.model_type_override,
            input_file=req.input_file,
        )
        if insp.error:
            return LaunchResult(
                task=task,
                success=False,
                resolved_input=resolved,
                inspection=insp,
                message=f"Inspection error: {insp.error}",
            )

        selector = AutoSelector(config=self.config)
        n_gpu_override = (
            req.n_gpu_layers
            if (req.gpu_offload_mode == "manual" or (req.gpu_offload_mode == "auto" and req.n_gpu_layers not in (-1, None)))
            else None
        )
        rec = selector.recommend(
            inspection=insp,
            requested_ctx=req.context_size,
            override_quant=req.target_quant if req.target_quant != "auto" else None,
            override_engine=req.engine if req.engine != "auto" else None,
            override_quantizer=req.quantizer if req.quantizer != "auto" else None,
            task=task,
            override_n_gpu_layers=n_gpu_override,
            override_flash_attn=self._normalize_flash_attn(req.flash_attn),
            override_cache_type_k=req.cache_type_k if req.cache_type_k != "auto" else None,
            override_cache_type_v=req.cache_type_v if req.cache_type_v != "auto" else None,
            override_gpu_offload_mode=req.gpu_offload_mode,
            override_parallel=req.parallel,
            override_quantization_mode=req.quantization_mode if req.quantization_mode != "auto" else None,
            override_mmproj_file=req.mmproj_file,
            strategy=req.strategy,
        )

        if task == "inspect_only":
            return LaunchResult(
                task=task,
                success=True,
                resolved_input=resolved,
                inspection=insp,
                recommendation=rec,
                message=rec.decision_summary,
            )

        sync_to_drive = self.config.mode.value in ("hybrid", "google_drive", "gdrive_relative")
        downloader = Downloader(self.config, hf_token=hf_token)
        dl_res = downloader.resolve_or_download(
            inspection=insp,
            variant=rec.selected_variant,
            sync_to_drive=sync_to_drive,
            unpack_archive=req.unpack_archive,
            input_file=req.input_file,
        )

        final_path = dl_res.primary_file_path
        final_quant = rec.recommended_quant
        companion_files: List[Path] = []
        if dl_res.mmproj_file_path and dl_res.mmproj_file_path.exists():
            companion_files.append(dl_res.mmproj_file_path)

        # Conversion / Quantization / Re-quantization step
        should_convert = (
            task in ("convert_and_quantize", "convert_and_serve")
            or rec.needs_requant
            or rec.needs_convert_to_gguf
        )
        if should_convert:
            conv_name = rec.recommended_converter or "llama.cpp"
            if conv_name == "none":
                conv_name = "llama.cpp"
            target_q = rec.requant_target_quant or rec.recommended_quant
            if target_q in ("N/A", "auto", "UNKNOWN"):
                target_q = "Q4_K_M"

            print(f"[Orchestrator] Running conversion/quantization ({conv_name} -> {target_q})...")
            converter = get_converter(conv_name, self.config)
            converted_path = converter.convert_or_quantize(
                input_path=final_path,
                target_quant=target_q,
                cleanup_source=req.cleanup_intermediate,
                use_imatrix=req.use_imatrix,
                calibration_text_path=req.calibration_text_path,
                imatrix_chunks=req.imatrix_chunks,
                split_max_size=req.split_max_size,
                outtype=req.outtype,
                merge_lora=req.merge_lora,
                base_model_name=req.base_model_name or insp.base_model_name,
                bits=req.gptq_bits,
                group_size=req.gptq_group_size,
                bits_per_weight=req.exl2_bpw,
            )
            final_path = converted_path
            final_quant = target_q
            if converted_path.is_file() and "-00001-of-" in converted_path.name:
                prefix = converted_path.name.split("-00001-of-")[0]
                companion_files.extend(
                    p for p in converted_path.parent.glob(f"{prefix}-*-of-*.gguf")
                    if p.resolve() != converted_path.resolve()
                )
        else:
            for p in dl_res.all_downloaded_files:
                if p.exists() and p.resolve() != final_path.resolve() and p not in companion_files:
                    companion_files.append(p)

        # Compute accurate final size & model type
        if final_path.is_file():
            shard_paths = [str(p) for p in companion_files if p.exists() and p != dl_res.mmproj_file_path]
            final_size = final_path.stat().st_size + sum(
                p.stat().st_size for p in companion_files if p.exists() and p.is_file()
            )
        else:
            shard_paths = []
            final_size = sum(f.stat().st_size for f in final_path.glob("**/*") if f.is_file())

        reg_type = insp.model_type
        if should_convert and final_path.suffix.lower() == ".gguf":
            reg_type = "vlm-gguf" if insp.is_vlm else ("embedding-gguf" if insp.model_type == "embedding-gguf" else "llm-gguf")

        cat_map = {
            "llm-gguf": self.config.models_gguf_dir,
            "vlm-gguf": self.config.models_gguf_dir,
            "embedding-gguf": self.config.models_gguf_dir,
            "llm-safetensors": self.config.models_safetensors_dir,
            "diffusion": self.config.models_diffusion_dir,
            "audio": self.config.models_audio_dir,
        }
        dest_dir = cat_map.get(reg_type, self.config.models_gguf_dir)
        drive_path = str(dest_dir / final_path.name) if sync_to_drive else None

        if sync_to_drive and should_convert and final_path.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            if final_path.is_file() and final_path.resolve() != Path(drive_path).resolve():
                shutil.copy2(str(final_path), drive_path)

        entry = self.registry.register_model(
            model_id=final_path.stem if should_convert else dl_res.model_id,
            name=final_path.stem if should_convert else dl_res.model_id,
            type=reg_type,
            source_url=insp.raw_source,
            quantization=final_quant,
            total_size_bytes=final_size,
            local_path=str(final_path),
            drive_path=drive_path,
            is_synced_to_drive=sync_to_drive,
            mmproj_path=str(dl_res.mmproj_file_path) if dl_res.mmproj_file_path else None,
            shard_files=shard_paths,
            set_active=True,
            backend_hint=rec.recommended_engine,
            repo_id=insp.repo_id,
            hf_revision=insp.hf_revision,
            preset_name=req.preset_name,
        )

        state: Optional[SystemState] = None
        bench_res: Optional[Dict[str, Any]] = None
        snippets: Tuple[str, str] = ("", "")

        if task in ("serve", "convert_and_serve", "benchmark", "serve_and_dashboard"):
            server_kwargs = self._build_server_overrides(req, rec)
            started = self.process_manager.start_server(
                model_id=entry.model_id,
                preset_name=req.preset_name,
                port=int(req.port),
                enable_auth=req.enable_api_auth,
                recommendation=rec,
                api_key_override=req.api_key_override,
                **server_kwargs,
            )

            if started and req.enable_tunnel and req.tunnel_type != "none" and task != "benchmark":
                self.process_manager.start_tunnel(
                    port=int(req.port),
                    tunnel_type=req.tunnel_type,
                    tunnel_token=req.tunnel_token,
                    tunnel_hostname=req.tunnel_hostname,
                )

            state = self.process_manager.get_state()
            snippets = ProcessManager.generate_code_snippets(state)

            if started and task == "benchmark" and self.process_manager.current_engine:
                bench_res = self.process_manager.current_engine.benchmark()
                if "error" not in bench_res:
                    self.registry.update_benchmark(entry.model_id, bench_res)

        export_res: Optional[Dict[str, Any]] = None
        if req.output_type and req.output_type.lower().strip() != "none" and final_path.exists():
            out_mgr = OutputManager(paths=self.config.paths, hf_token=hf_token)
            protected = [final_path] if (state and state.is_server_running) else []
            export_res = out_mgr.export_artifact(
                source_path=final_path,
                output_mode=req.output_type,
                destination=req.output_path,
                output_name=req.output_name or None,
                zip_output=req.zip_output,
                cleanup_source=req.cleanup_after_export,
                companion_files=companion_files,
                hf_private=req.hf_private,
                protected_paths=protected,
            )

        ok = True if task == "convert_and_quantize" else bool(state and state.is_server_running)
        return LaunchResult(
            task=task,
            success=ok,
            resolved_input=resolved,
            inspection=insp,
            recommendation=rec,
            download_result=dl_res,
            model_entry=entry,
            artifact_path=final_path,
            companion_files=companion_files,
            system_state=state,
            benchmark_results=bench_res,
            export_result=export_res,
            snippets=snippets,
            message=rec.decision_summary,
        )
