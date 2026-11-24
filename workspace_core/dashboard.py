"""
Interactive Management Dashboard for Google Colab & Linux Workstations
Provides an 8-tab Gradio control interface with full Auto + Manual controls:
Tab 1: Hardware, Doctor & Telemetry
Tab 2: Model Hub, Inspector & Auto/Manual Planner
Tab 3: Model Registry, Scanner & Exporter
Tab 4: Server, Presets & Tunnel Control
Tab 5: Environment Profiles & Cache Manager
Tab 6: Multi-Modal Playground (LLM, VLM, Image Gen, Audio)
Tab 7: Benchmark & Model Comparison
Tab 8: Workspace Settings & Secrets
"""

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore

from .config import ServerPreset, WorkspaceConfig, get_config
from .hardware import get_hardware_profile
from .env_manager import EnvironmentManager
from .model_inspector import InspectionResult, ModelInspector
from .auto_selector import AutoSelector, SelectionRecommendation
from .downloader import Downloader
from .registry import ModelRegistry
from .process_manager import ProcessManager, SystemState
from .orchestrator import LaunchRequest, WorkspaceOrchestrator
from .security import SecurityManager


class WorkspaceDashboard:
    """Encapsulates the 8-tab Gradio management dashboard."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.registry = ModelRegistry(self.config)
        self.inspector = ModelInspector(hf_token=SecurityManager.get_hf_token())
        self.selector = AutoSelector(config=self.config)
        self.downloader = Downloader(self.config)
        self.process_manager = ProcessManager(self.config)
        self.env_manager = EnvironmentManager(self.config)
        self.orchestrator = WorkspaceOrchestrator(self.config)
        self.last_inspection: Optional[InspectionResult] = None
        self.last_recommendation: Optional[SelectionRecommendation] = None

    # --- Tab 1: Telemetry & Doctor ---
    def get_hardware_telemetry_ui(self) -> Tuple[str, str, str, str, str, str]:
        hw = get_hardware_profile()
        gpu_str = "No GPU Detected (CPU Mode)"
        gpu_detail = "N/A"
        if hw.primary_gpu:
            g = hw.primary_gpu
            gpu_str = f"**GPU**: {g.name} ({g.sm_name}) | Devices: {len(hw.gpus)}"
            temp = f"{g.temperature_c:.0f}C" if g.temperature_c is not None else "N/A"
            util = f"{g.utilization_pct:.0f}%" if g.utilization_pct is not None else "N/A"
            gpu_detail = (
                f"**VRAM Total**: {hw.total_vram_mb:.0f} MB\n\n"
                f"**VRAM Free**: {hw.total_free_vram_mb:.0f} MB\n\n"
                f"**VRAM Used**: {g.used_vram_mb:.0f} MB\n\n"
                f"**Utilization**: {util} | **Temp**: {temp}\n\n"
                f"**Flash Attention**: {'Supported' if g.supports_flash_attn else 'Not Supported'} | "
                f"**BF16**: {'Yes' if g.supports_bf16 else 'No'} | "
                f"**FP8**: {'Yes' if g.supports_fp8 else 'No'}\n\n"
                f"**KV Quantization**: {'Supported' if g.supports_kv_quant else 'Not Supported'} | "
                f"**vLLM Capable**: {'Yes' if g.supports_vllm else 'No'}"
            )

        ram_str = (
            f"**Total RAM**: {hw.ram_total_gb:.1f} GB\n\n"
            f"**Free RAM**: {hw.ram_available_gb:.1f} GB\n\n"
            f"**Used RAM**: {hw.ram_used_gb:.1f} GB ({hw.ram_percent}% used)\n\n"
            f"**CPU Cores**: {hw.cpu_cores}"
        )

        disk_str = (
            f"**Local Fast SSD**: {hw.ssd_free_gb:.1f} GB free / {hw.ssd_total_gb:.1f} GB total\n\n"
            f"**Persistent/Drive Storage**: {hw.drive_free_gb:.1f} GB free / {hw.drive_total_gb:.1f} GB total"
        )

        env_str = (
            f"**CUDA Version**: {hw.cuda_version}\n\n"
            f"**NVIDIA Driver**: {hw.driver_version}\n\n"
            f"**Python Runtime**: {hw.python_version} (`{hw.python_executable}`)"
        )

        state = self.process_manager.get_state()
        proc_str = (
            f"**Inference Server**: {'Running (PID ' + str(state.server_pid) + ')' if state.is_server_running else 'Stopped'}\n\n"
            f"**Backend**: {state.engine_name}\n\n"
            f"**Active Model**: {state.active_model or 'None'}\n\n"
            f"**Public Tunnel**: {'Active (' + state.tunnel_type + ')' if state.is_tunnel_running else 'Stopped'}\n\n"
            f"**Public URL**: {state.public_url or 'None'}"
        )

        return gpu_str, gpu_detail, ram_str, disk_str, env_str, proc_str

    def run_doctor_ui(self) -> Tuple[List[List[str]], str]:
        report = self.env_manager.run_doctor()
        rows = [[c["check"], c["status"], c["detail"]] for c in report["checks"]]
        cache = report["cache_stats"]
        summary = (
            f"**Cache Usage** — UV: {cache['uv_cache_gb']} GB | Binaries: {cache['bin_cache_gb']} GB | "
            f"Venvs: {cache['envs_gb']} GB | Models Cache: {cache['models_cache_gb']} GB | "
            f"Outputs: {cache['outputs_gb']} GB | Logs: {cache['logs_mb']} MB"
        )
        return rows, summary

    # --- Tab 2: Model Hub & Auto/Manual Planner ---
    def inspect_model_ui(
        self,
        source_url: str,
        requested_ctx: str,
        input_mode: str = "auto",
        input_file: str = "",
        hf_revision: str = "",
        model_type_override: str = "auto",
        strategy: str = "balanced",
        engine_override: str = "auto",
        quantizer_override: str = "auto",
        target_quant_override: str = "auto",
    ) -> Tuple[str, List[List[str]], Any, str]:
        if not source_url.strip():
            empty_dd = gr.Dropdown(choices=[]) if gr else None
            return "Please provide a model repository, URL, or local path.", [], empty_dd, ""

        self.inspector = ModelInspector(hf_token=SecurityManager.get_hf_token())
        inspection = self.inspector.inspect(
            source_url.strip(),
            input_mode=input_mode,
            hf_revision=hf_revision.strip() or None,
            model_type_override=model_type_override if model_type_override != "auto" else None,
            input_file=input_file.strip() or None,
        )
        self.last_inspection = inspection

        if inspection.error:
            empty_dd = gr.Dropdown(choices=[]) if gr else None
            return f"Inspection Error: {inspection.error}", [], empty_dd, ""

        ctx_val: Any = requested_ctx.strip() if isinstance(requested_ctx, str) else requested_ctx
        if isinstance(ctx_val, str) and ctx_val.isdigit():
            ctx_val = int(ctx_val)

        recommendation = self.selector.recommend(
            inspection,
            requested_ctx=ctx_val or 4096,
            override_quant=target_quant_override if target_quant_override != "auto" else None,
            override_engine=engine_override,
            override_quantizer=quantizer_override,
            strategy=strategy,
        )
        self.last_recommendation = recommendation

        bd = recommendation.budget_breakdown or {}
        summary_text = (
            f"### Model: `{inspection.model_id}`\n\n"
            f"- **Type**: `{inspection.model_type}` | **Params**: `{inspection.estimated_params_b or 'Unknown'}B` | **Max Ctx**: `{inspection.max_context_length or 'Unknown'}`\n"
            f"- **Architecture**: `{inspection.architecture_name or 'N/A'}` (MoE: {'Yes' if inspection.is_moe else 'No'}, LoRA: {'Yes' if inspection.is_lora else 'No'}, VLM: {'Yes' if inspection.is_vlm else 'No'})\n"
            f"- **Quant Format**: `{inspection.quant_format}` (IK Quant: {'Yes' if inspection.is_ik_quant else 'No'})\n"
            f"- **Strategy**: `{strategy}` | **Engine**: `{recommendation.recommended_engine}` | **Converter**: `{recommendation.recommended_converter or 'none'}`\n"
            f"- **Recommended Quant**: **`{recommendation.recommended_quant}`** | **Context**: `{recommendation.recommended_ctx}` | **GPU Layers**: `{recommendation.recommended_n_gpu_layers}`\n"
            f"- **VRAM Budget**: {recommendation.vram_budget_mb / 1024:.2f} GB usable (Free: {bd.get('free_vram_mb', 0):.0f} MB - Overhead: {bd.get('cuda_overhead_mb', 1200):.0f} MB - Est KV: {bd.get('estimated_kv_cache_mb', 0):.0f} MB)\n"
            f"- **Plan**: {recommendation.reasoning}\n"
        )

        table_data = []
        quant_choices = []
        for v in inspection.gguf_variants:
            quant_choices.append(v.quant)
            fits = (v.total_size_bytes / (1024 * 1024)) <= recommendation.vram_budget_mb
            table_data.append([
                v.quant,
                f"{v.total_size_gb:.2f} GB",
                "Yes" if v.is_split else "No",
                str(len(v.shard_files)),
                "Yes" if v.is_ik_quant else "No",
                v.mmproj_file or "None",
                "Fits" if fits else "Exceeds VRAM",
            ])

        if not quant_choices:
            quant_choices = ["auto", "Q4_K_M", "Q5_K_M", "Q8_0", "IQ4_KS", "N/A (Safetensors / Direct)"]

        rec_quant = recommendation.recommended_quant
        default_val = rec_quant if rec_quant in quant_choices else quant_choices[0]
        dd = gr.Dropdown(choices=quant_choices, value=default_val) if gr else default_val

        return (
            summary_text,
            table_data,
            dd,
            recommendation.decision_summary or recommendation.reasoning,
        )

    def execute_hub_task_ui(
        self,
        task: str,
        input_mode: str,
        source_url: str,
        input_file: str,
        hf_revision: str,
        unpack_archive: bool,
        model_type_override: str,
        strategy: str,
        engine_override: str,
        quantizer_override: str,
        selected_quant: str,
        requested_ctx: str,
        use_imatrix: bool,
        split_max_size: str,
        outtype: str,
        cleanup_intermediate: bool,
        sync_to_drive: bool,
    ) -> Tuple[str, List[List[str]]]:
        if not source_url.strip():
            return "Please enter a model source URI.", self.get_registry_table_ui()

        ctx_val: Any = requested_ctx.strip() if isinstance(requested_ctx, str) else requested_ctx
        if isinstance(ctx_val, str) and ctx_val.isdigit():
            ctx_val = int(ctx_val)

        target_q = selected_quant
        if not target_q or target_q.startswith("N/A"):
            target_q = "auto"

        req = LaunchRequest(
            input_type=input_mode,
            input_path=source_url.strip(),
            input_file=input_file.strip() or None,
            hf_revision=hf_revision.strip() or None,
            unpack_archive=unpack_archive,
            model_type_override=model_type_override if model_type_override != "auto" else None,
            task=task,
            strategy=strategy,
            engine=engine_override,
            quantizer=quantizer_override,
            target_quant=target_q,
            context_size=ctx_val or 4096,
            use_imatrix=use_imatrix,
            split_max_size=split_max_size.strip() or None,
            outtype=outtype,
            cleanup_intermediate=cleanup_intermediate,
            enable_tunnel=False,
        )
        # Honor explicit Drive sync preference if workspace mode allows it
        if sync_to_drive and self.config.mode.value == "local_ssd":
            pass
        try:
            if task == "download_only":
                insp = self.inspector.inspect(
                    source_url.strip(),
                    input_mode=input_mode,
                    hf_revision=hf_revision.strip() or None,
                    model_type_override=model_type_override if model_type_override != "auto" else None,
                    input_file=input_file.strip() or None,
                )
                if insp.error:
                    return f"Inspection error: {insp.error}", self.get_registry_table_ui()
                rec = self.selector.recommend(
                    insp,
                    requested_ctx=ctx_val or 4096,
                    override_quant=target_q if target_q != "auto" else None,
                    override_engine=engine_override,
                    override_quantizer=quantizer_override,
                    strategy=strategy,
                )
                dl_res = Downloader(self.config, hf_token=SecurityManager.get_hf_token()).resolve_or_download(
                    inspection=insp,
                    variant=rec.selected_variant,
                    sync_to_drive=sync_to_drive,
                    unpack_archive=unpack_archive,
                    input_file=input_file.strip() or None,
                )
                final_path = dl_res.primary_file_path
                if rec.needs_requant and rec.requant_target_quant:
                    final_path = AutoSelector.execute_requantize(
                        source_path=final_path,
                        target_quant=rec.requant_target_quant,
                        backup_to_drive=sync_to_drive,
                        cleanup_source=cleanup_intermediate,
                        converter_name=rec.recommended_converter or "llama.cpp",
                        config=self.config,
                        use_imatrix=use_imatrix,
                    )
                entry = self.registry.register_model(
                    model_id=dl_res.model_id,
                    name=dl_res.model_id,
                    type=insp.model_type,
                    source_url=insp.raw_source,
                    quantization=rec.recommended_quant,
                    total_size_bytes=dl_res.total_size_bytes,
                    local_path=str(final_path),
                    is_synced_to_drive=sync_to_drive,
                    mmproj_path=str(dl_res.mmproj_file_path) if dl_res.mmproj_file_path else None,
                    shard_files=[str(p) for p in dl_res.all_downloaded_files if p.resolve() != final_path.resolve()],
                    backend_hint=rec.recommended_engine,
                    set_active=True,
                )
                msg = (
                    f"**Model Ready and Registered Active**\n\n"
                    f"- **Model ID**: `{entry.model_id}` | **Quant**: `{entry.quantization}` | **Engine Hint**: `{entry.backend_hint}`\n"
                    f"- **Path**: `{entry.local_path}`"
                )
                return msg, self.get_registry_table_ui()

            res = self.orchestrator.run(req)
            if not res.success and not res.model_entry:
                return f"Task failed: {res.message}", self.get_registry_table_ui()
            lines = [f"**Task `{task}` Completed**", f"- **Model ID**: `{res.model_entry.model_id if res.model_entry else 'N/A'}`"]
            if res.artifact_path:
                lines.append(f"- **Artifact Path**: `{res.artifact_path}`")
            if res.system_state and res.system_state.is_server_running:
                lines.append(f"- **Server**: Running ({res.system_state.engine_name}) on port `{res.system_state.port}`")
            return "\n\n".join(lines), self.get_registry_table_ui()
        except Exception as e:
            return f"Execution error: {str(e)}", self.get_registry_table_ui()

    # --- Tab 3: Registry, Scanner & Exporter ---
    def get_registry_table_ui(self) -> List[List[str]]:
        self.registry.load()
        models = self.registry.list_models()
        data = []
        for m in models:
            tps = m.benchmark_metrics.get("tokens_per_sec")
            bench_str = f"{tps} tok/s" if tps else "-"
            data.append([
                "Active" if m.is_active else "Inactive",
                m.model_id,
                m.type,
                m.quantization,
                m.backend_hint or "auto",
                f"{m.total_size_gb:.2f} GB",
                "Yes" if m.is_synced_to_drive else "No",
                bench_str,
                m.local_path,
            ])
        return data

    def scan_models_ui(self) -> Tuple[str, List[List[str]]]:
        added = self.registry.scan_workspace_models()
        return f"Scan complete. Imported {len(added)} unregistered model(s).", self.get_registry_table_ui()

    def hot_swap_model_ui(self, model_id: str) -> Tuple[str, List[List[str]]]:
        if not model_id.strip():
            return "Please enter a Model ID to switch.", self.get_registry_table_ui()
        res = self.registry.set_active_model(model_id.strip())
        if res:
            state = self.process_manager.get_state()
            if state.is_server_running:
                self.process_manager.start_server(
                    model_id=res.model_id,
                    port=state.port,
                    enable_auth=bool(state.api_key),
                )
            return f"Switched active model to `{res.model_id}`.", self.get_registry_table_ui()
        return f"Model `{model_id}` not found in registry.", self.get_registry_table_ui()

    def sync_model_drive_ui(self, model_id: str) -> Tuple[str, List[List[str]]]:
        if not model_id.strip():
            return "Please enter a Model ID to sync.", self.get_registry_table_ui()
        ok = self.registry.sync_to_drive(model_id.strip())
        msg = f"Synced `{model_id}` to persistent storage." if ok else f"Failed to sync `{model_id}`."
        return msg, self.get_registry_table_ui()

    def delete_local_cache_ui(self, model_id: str) -> Tuple[str, List[List[str]]]:
        if not model_id.strip():
            return "Please enter a Model ID.", self.get_registry_table_ui()
        ok = self.registry.delete_local_cache(model_id.strip())
        msg = f"Deleted local SSD cache for `{model_id}`." if ok else f"Failed to delete local cache for `{model_id}`."
        return msg, self.get_registry_table_ui()

    def remove_model_entry_ui(self, model_id: str, delete_files: bool) -> Tuple[str, List[List[str]]]:
        if not model_id.strip():
            return "Please enter a Model ID.", self.get_registry_table_ui()
        ok = self.registry.remove_model(model_id.strip(), delete_files=delete_files)
        msg = f"Removed `{model_id}` from registry." if ok else f"Model `{model_id}` not found."
        return msg, self.get_registry_table_ui()

    def purge_temp_files_ui(self) -> str:
        freed = self.registry.purge_temp_files()
        return f"Purged temporary files. Freed {freed / (1024*1024):.2f} MB."

    def export_artifact_ui(
        self,
        model_id: str,
        output_mode: str,
        destination: str,
        output_name: str,
        zip_output: bool,
        hf_private: bool,
        cleanup_source: bool,
    ) -> str:
        if not model_id.strip():
            return "Please enter a Model ID."
        entry = self.registry.get_model(model_id.strip())
        if not entry:
            return f"Model '{model_id}' not found in registry."

        src_path = entry.local_path if Path(entry.local_path).exists() else entry.drive_path
        if not src_path or not Path(src_path).exists():
            return f"File not found for model '{model_id}' on local SSD or Drive."

        from .io_resolver import OutputManager
        out_mgr = OutputManager(paths=self.config.paths, hf_token=SecurityManager.get_hf_token())
        companions = [s for s in entry.shard_files if Path(s).exists()]
        if entry.mmproj_path and Path(entry.mmproj_path).exists():
            companions.append(entry.mmproj_path)

        state = self.process_manager.get_state()
        protected = [src_path] if (state.is_server_running and state.active_model == entry.model_id) else []

        try:
            res = out_mgr.export_artifact(
                source_path=src_path,
                output_mode=output_mode,
                destination=destination.strip() or None,
                output_name=output_name.strip() or None,
                zip_output=zip_output,
                cleanup_source=cleanup_source,
                companion_files=companions,
                hf_private=hf_private,
                protected_paths=protected,
            )
            return f"**Export Complete.** {res.get('message', '')}"
        except Exception as e:
            return f"Export error: {str(e)}"

    # --- Tab 4: Server, Presets & Tunnel Control ---
    def start_server_ui(
        self,
        preset_name: str,
        port: int,
        enable_auth: bool,
        api_key_override: str,
        engine_override: str,
        ctx_override: str,
        offload_mode: str,
        gpu_layers_override: int,
        flash_attn_mode: str,
        cache_k_override: str,
        cache_v_override: str,
        parallel_override: int,
        batch_override: int,
        ubatch_override: int,
        threads_override: int,
        split_mode: str,
        tensor_split: str,
        override_tensor: str,
        quant_mode: str,
        dtype: str,
        gpu_mem_util: float,
        rope_scaling: str,
        chat_template: str,
        reasoning_format: str,
        mlock: bool,
        mmap: bool,
        custom_cli_args: str,
    ) -> Tuple[str, str, str, str, str]:
        try:
            overrides: Dict[str, Any] = {
                "parallel": int(parallel_override or 1),
                "batch_size": int(batch_override or 512),
                "ubatch_size": int(ubatch_override or 512),
                "threads": int(threads_override or 0),
                "split_mode": split_mode or "layer",
                "mlock": bool(mlock),
                "mmap": bool(mmap),
                "dtype": dtype or "auto",
                "gpu_memory_utilization": float(gpu_mem_util or 0.90),
            }
            if engine_override and engine_override.lower() != "auto":
                overrides["backend"] = engine_override
            if ctx_override and str(ctx_override).strip().lower() != "auto":
                overrides["ctx_size"] = int(ctx_override)
            else:
                overrides["ctx_size"] = 0
            if offload_mode:
                overrides["gpu_offload_mode"] = offload_mode
            if gpu_layers_override is not None:
                overrides["n_gpu_layers"] = int(gpu_layers_override)
            if flash_attn_mode == "on":
                overrides["flash_attn"] = True
            elif flash_attn_mode == "off":
                overrides["flash_attn"] = False
            if cache_k_override and cache_k_override.lower() != "auto":
                overrides["cache_type_k"] = cache_k_override
            if cache_v_override and cache_v_override.lower() != "auto":
                overrides["cache_type_v"] = cache_v_override
            if tensor_split.strip():
                overrides["tensor_split"] = tensor_split.strip()
            if override_tensor.strip():
                overrides["override_tensor"] = override_tensor.strip()
            if quant_mode and quant_mode.lower() != "auto":
                overrides["quantization_mode"] = quant_mode
            if rope_scaling.strip():
                overrides["rope_scaling"] = rope_scaling.strip()
            if chat_template.strip():
                overrides["chat_template"] = chat_template.strip()
            if reasoning_format.strip():
                overrides["reasoning_format"] = reasoning_format.strip()
            if custom_cli_args.strip():
                overrides["custom_server_args"] = custom_cli_args.strip()

            self.process_manager.start_server(
                preset_name=preset_name if preset_name != "auto" else None,
                port=int(port),
                enable_auth=enable_auth,
                api_key_override=api_key_override.strip() or None,
                **overrides,
            )
            state = self.process_manager.get_state()
            cmd_preview = (
                self.process_manager.current_engine.get_command_preview("0.0.0.0", int(port), state.api_key)
                if self.process_manager.current_engine else ""
            )
            status_text = (
                f"**Server Running** (`{state.engine_name}`, PID `{state.server_pid}`)"
                + (f" | API Key: `{state.api_key}`" if state.api_key else " | Auth: Disabled")
                if state.is_server_running
                else "**Server Failed to Start** (check logs below)"
            )
            py_snippet, curl_snippet = ProcessManager.generate_code_snippets(state)
            return status_text, json.dumps(state.local_endpoints, indent=2), cmd_preview, py_snippet, curl_snippet
        except Exception as e:
            return f"Server start error: {str(e)}", "{}", "", "", ""

    def stop_server_ui(self) -> Tuple[str, str]:
        self.process_manager.stop_server()
        return "Server Stopped", "{}"

    def save_preset_ui(
        self,
        preset_key: str,
        backend: str,
        ctx_val: str,
        ngl: int,
        flash_attn_mode: str,
        cache_k: str,
        cache_v: str,
        parallel: int,
        custom_args: str,
        desc: str,
    ) -> Tuple[str, Any]:
        key = preset_key.strip().replace(" ", "_")
        if not key:
            dd = gr.Dropdown(choices=list(self.config.presets.keys())) if gr else None
            return "Please enter a preset key name.", dd
        ctx_int = int(ctx_val) if str(ctx_val).isdigit() else 4096
        fa_bool = None if flash_attn_mode == "auto" else (flash_attn_mode == "on")
        preset = ServerPreset(
            name=key,
            backend=backend if backend != "auto" else "llama.cpp",
            ctx_size=ctx_int,
            n_gpu_layers=int(ngl),
            flash_attn=fa_bool,
            cache_type_k=cache_k,
            cache_type_v=cache_v,
            parallel=int(parallel or 1),
            custom_args=custom_args,
            description=desc or "Custom user preset",
        )
        self.config.upsert_preset(key, preset)
        dd = gr.Dropdown(choices=["auto"] + list(self.config.presets.keys()), value=key) if gr else key
        return f"Saved preset `{key}` to presets.yaml.", dd

    def start_tunnel_ui(
        self,
        port: int,
        tunnel_type: str,
        tunnel_token: str = "",
        tunnel_hostname: str = "",
    ) -> Tuple[str, str, str]:
        pub_url = self.process_manager.start_tunnel(
            port=int(port),
            tunnel_type=tunnel_type,
            tunnel_token=tunnel_token.strip() or None,
            tunnel_hostname=tunnel_hostname.strip() or None,
        )
        state = self.process_manager.get_state()
        status_text = (
            f"Tunnel Active ({state.tunnel_type}): {pub_url or 'Named Tunnel Connected'}"
            if state.is_tunnel_running
            else "Tunnel Inactive"
        )
        py_snippet, curl_snippet = ProcessManager.generate_code_snippets(state)
        return status_text, py_snippet, curl_snippet

    def stop_tunnel_ui(self) -> Tuple[str, str, str]:
        self.process_manager.stop_tunnel()
        state = self.process_manager.get_state()
        py_snippet, curl_snippet = ProcessManager.generate_code_snippets(state)
        return "Tunnel Stopped", py_snippet, curl_snippet

    def generate_code_snippets(self, state: SystemState) -> Tuple[str, str]:
        return ProcessManager.generate_code_snippets(state)

    def view_logs_ui(self, log_type: str, lines: int = 120) -> str:
        return self.process_manager.get_log_tail(log_type=log_type, lines=int(lines or 120))

    # --- Tab 5: Environment Profiles & Cache ---
    def get_profiles_table_ui(self) -> List[List[str]]:
        status = self.env_manager.get_all_profiles_status()
        rows = []
        for prof, data in status.items():
            comp_str = ", ".join([f"{k}: {'Installed' if v else 'Missing'}" for k, v in data["components"].items()])
            rows.append([
                prof,
                "Installed" if data["installed"] else "Incomplete",
                "Isolated Venv" if data.get("isolated") else "Host Python",
                comp_str,
            ])
        return rows

    def install_profile_ui(self, profile_name: str, isolate: bool) -> Tuple[str, List[List[str]]]:
        ok = self.env_manager.install_profile(profile_name, isolate=isolate)
        msg = f"Profile '{profile_name}' installed (isolate={isolate})." if ok else f"Profile '{profile_name}' installation failed."
        return msg, self.get_profiles_table_ui()

    def clear_cache_ui(self, scope: str) -> str:
        res = self.env_manager.clear_cache(scope)
        return f"Cleared `{res['scope']}` cache. Freed {res['freed_mb']} MB ({res['freed_gb']} GB)."

    # --- Tab 6: Multi-Modal Playground ---
    def chat_interact(
        self,
        message: str,
        image_path: Optional[str],
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        history = list(history or [])
        if not message.strip() and not image_path:
            return "", history

        state = self.process_manager.get_state()
        if not state.is_server_running:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "Server is not running. Please start the server in Tab 4 first."})
            return "", history

        chat_url = f"http://127.0.0.1:{state.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if state.api_key:
            headers["Authorization"] = f"Bearer {state.api_key}"

        messages: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.extend(history)

        if image_path and Path(image_path).is_file():
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
            user_content: Any = [
                {"type": "text", "text": message or "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        else:
            user_content = message

        messages.append({"role": "user", "content": user_content})

        try:
            resp = requests.post(
                chat_url,
                json={
                    "model": state.active_model or "default",
                    "messages": messages,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_tokens": int(max_tokens),
                    "stream": False,
                },
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"]
            else:
                reply = f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            reply = f"Request exception: {str(e)}"

        history.append({"role": "user", "content": message or "[Image uploaded]"})
        history.append({"role": "assistant", "content": reply})
        return "", history

    def generate_image_ui(
        self,
        prompt: str,
        negative_prompt: str,
        size: str,
        steps: int,
        guidance: float,
    ) -> Tuple[Any, str]:
        state = self.process_manager.get_state()
        if not state.is_server_running:
            return None, "Server is not running."
        headers = {"Content-Type": "application/json"}
        if state.api_key:
            headers["Authorization"] = f"Bearer {state.api_key}"
        try:
            resp = requests.post(
                f"http://127.0.0.1:{state.port}/v1/images/generations",
                json={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "size": size,
                    "num_inference_steps": int(steps),
                    "guidance_scale": float(guidance),
                },
                headers=headers,
                timeout=120,
            )
            if resp.status_code != 200:
                return None, f"Image generation error ({resp.status_code}): {resp.text}"
            b64 = resp.json()["data"][0]["b64_json"]
            out_img = self.config.runtime_outputs_dir / "playground_image.png"
            out_img.write_bytes(base64.b64decode(b64))
            return str(out_img), f"Generated image saved to {out_img}"
        except Exception as e:
            return None, f"Exception: {e}"

    def transcribe_audio_ui(self, audio_path: Optional[str], language: str, task_mode: str) -> str:
        if not audio_path or not Path(audio_path).is_file():
            return "Please upload an audio file."
        state = self.process_manager.get_state()
        if not state.is_server_running:
            return "Server is not running."
        headers = {}
        if state.api_key:
            headers["Authorization"] = f"Bearer {state.api_key}"
        endpoint = "translations" if task_mode == "translate" else "transcriptions"
        try:
            with open(audio_path, "rb") as f:
                data = {"response_format": "verbose_json"}
                if language.strip():
                    data["language"] = language.strip()
                resp = requests.post(
                    f"http://127.0.0.1:{state.port}/v1/audio/{endpoint}",
                    headers=headers,
                    files={"file": (Path(audio_path).name, f)},
                    data=data,
                    timeout=120,
                )
            if resp.status_code == 200:
                return json.dumps(resp.json(), indent=2, ensure_ascii=False)
            return f"Audio API error ({resp.status_code}): {resp.text}"
        except Exception as e:
            return f"Exception: {e}"

    # --- Tab 7: Benchmark & Comparison ---
    def run_benchmark_ui(self, prompt: str, max_tokens: int, runs: int) -> Tuple[str, List[List[str]]]:
        state = self.process_manager.get_state()
        if not state.is_server_running or not self.process_manager.current_engine:
            return "Server is not running. Start the server first in Tab 4.", self.get_benchmark_history_ui()

        res = self.process_manager.current_engine.benchmark(prompt=prompt, max_tokens=int(max_tokens), runs=int(runs))
        if "error" in res:
            return f"Benchmark Error: {res['error']}", self.get_benchmark_history_ui()

        if state.active_model:
            self.registry.update_benchmark(state.active_model, res)

        output = (
            f"### Benchmark Results ({res.get('runs', 1)} run(s))\n\n"
            f"- **Model**: `{state.active_model}` | **Engine**: `{state.engine_name}`\n"
            f"- **Time To First Token (TTFT)**: **{res.get('ttft_ms')} ms**\n"
            f"- **Generation Speed**: **{res.get('tokens_per_sec')} tokens/sec**\n"
            f"- **Total Tokens**: {res.get('total_tokens')}\n"
            f"- **Total Latency**: {res.get('total_time_sec')} seconds\n\n"
            f"**Response Preview**:\n> {res.get('response_preview')}\n"
        )
        return output, self.get_benchmark_history_ui()

    def get_benchmark_history_ui(self) -> List[List[str]]:
        self.registry.load()
        rows = []
        for m in self.registry.list_models():
            bm = m.benchmark_metrics
            if bm:
                rows.append([
                    m.model_id,
                    m.quantization,
                    m.backend_hint or "-",
                    str(bm.get("ttft_ms", "-")),
                    str(bm.get("tokens_per_sec", "-")),
                    str(bm.get("total_tokens", "-")),
                    str(bm.get("tested_at", "-")),
                ])
        return rows

    # --- Tab 8: Settings & Secrets ---
    def save_settings_ui(
        self,
        strategy: str,
        iso_mode: str,
        venv_inherit: bool,
        auto_install: bool,
        cuda_target: str,
        llama_tag: str,
        dl_backend: str,
        aria_conns: int,
        cuda_overhead: float,
        safety_margin: float,
        auto_requant: bool,
        auto_partial: bool,
        prefer_ik_moe: bool,
        use_imatrix: bool,
        cleanup_inter: bool,
    ) -> str:
        self.config.update_settings(
            selection_strategy=strategy,
            env_isolation_mode=iso_mode,
            venv_system_site_packages=bool(venv_inherit),
            auto_install_dependencies=bool(auto_install),
            cuda_target=cuda_target,
            llama_cpp_release_tag=llama_tag.strip() or "latest",
            download_backend=dl_backend,
            aria2_connections=int(aria_conns),
            cuda_overhead_mb=float(cuda_overhead),
            vram_safety_margin_pct=float(safety_margin),
            auto_requantize=bool(auto_requant),
            auto_partial_offload=bool(auto_partial),
            prefer_ik_llama_for_moe=bool(prefer_ik_moe),
            use_imatrix=bool(use_imatrix),
            cleanup_intermediate_files=bool(cleanup_inter),
        )
        return f"Settings saved to `{self.config.settings_file}`."

    def update_secrets_ui(
        self,
        hf_tok: str,
        civitai_tok: str,
        cf_tok: str,
        ngrok_tok: str,
        ws_api_key: str,
    ) -> str:
        if hf_tok.strip():
            SecurityManager.set_cached_secret("HF_TOKEN", hf_tok.strip())
        if civitai_tok.strip():
            SecurityManager.set_cached_secret("CIVITAI_API_KEY", civitai_tok.strip())
        if cf_tok.strip():
            SecurityManager.set_cached_secret("CF_TUNNEL_TOKEN", cf_tok.strip())
        if ngrok_tok.strip():
            SecurityManager.set_cached_secret("NGROK_AUTHTOKEN", ngrok_tok.strip())
        if ws_api_key.strip():
            SecurityManager.set_cached_secret("WORKSPACE_API_KEY", ws_api_key.strip())
        status = SecurityManager.list_secrets_status()
        return "\n".join([f"- **{k}**: `{v}`" for k, v in status.items()])

    # --- UI Builder ---
    def build_ui(self) -> Any:
        if gr is None:
            raise ImportError("Gradio is not installed. Run `env_manager.install_profile('core')` first.")

        s = self.config.settings
        preset_choices = ["auto"] + list(self.config.presets.keys())
        default_preset = "gguf_balanced" if "gguf_balanced" in self.config.presets else preset_choices[0]

        with gr.Blocks(title="Universal AI Workspace") as demo:
            gr.Markdown("# Universal AI Workspace")
            gr.Markdown("Modular Auto + Full Manual control center for running, converting, quantizing, and benchmarking AI models.")

            with gr.Tabs():
                # Tab 1: Hardware, Doctor & Telemetry
                with gr.Tab("Hardware & Doctor"):
                    with gr.Row():
                        refresh_hw_btn = gr.Button("Refresh Telemetry", variant="primary")
                        run_doc_btn = gr.Button("Run Pre-Flight Doctor", variant="secondary")
                    with gr.Row():
                        with gr.Column(scale=1):
                            gpu_title_md = gr.Markdown("### GPU")
                            gpu_detail_md = gr.Markdown("Loading...")
                        with gr.Column(scale=1):
                            gr.Markdown("### System RAM & CPU")
                            ram_md = gr.Markdown("Loading...")
                        with gr.Column(scale=1):
                            gr.Markdown("### Storage")
                            disk_md = gr.Markdown("Loading...")
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Runtime Environment")
                            env_md = gr.Markdown("Loading...")
                        with gr.Column(scale=1):
                            gr.Markdown("### Active Processes")
                            proc_md = gr.Markdown("Loading...")

                    doc_summary_md = gr.Markdown("")
                    doc_table = gr.Dataframe(headers=["Check", "Status", "Detail"], label="System Diagnostics")

                    refresh_hw_btn.click(
                        fn=self.get_hardware_telemetry_ui,
                        outputs=[gpu_title_md, gpu_detail_md, ram_md, disk_md, env_md, proc_md],
                    )
                    run_doc_btn.click(fn=self.run_doctor_ui, outputs=[doc_table, doc_summary_md])

                # Tab 2: Model Hub & Auto/Manual Planner
                with gr.Tab("Model Hub & Planner"):
                    ws_mode_str = self.config.mode.value if hasattr(self.config.mode, "value") else str(self.config.mode)
                    gr.Markdown(f"**Workspace Storage Mode**: `{ws_mode_str}` | **Runtime Root**: `{self.config.runtime_root}`")
                    with gr.Row():
                        input_mode_dropdown = gr.Dropdown(
                            label="Input Source Mode",
                            choices=[
                                "auto", "huggingface", "local_path",
                                "google_drive", "gdrive_relative", "gdrive_share",
                                "url", "civitai", "git_repo",
                            ],
                            value="auto",
                            scale=1,
                        )
                        source_input = gr.Textbox(
                            label="Model Source URI",
                            placeholder="Qwen/Qwen2.5-0.5B-Instruct-GGUF, hf://..., gdrive://..., civitai://..., or direct URL",
                            scale=3,
                        )
                        ctx_input = gr.Textbox(label="Context Size (int or 'auto')", value="4096", scale=1)

                    with gr.Row():
                        input_file_box = gr.Textbox(label="Specific File / Subpath (Optional)", placeholder="Leave blank for auto", scale=2)
                        hf_rev_box = gr.Textbox(label="HF Revision / Branch (Optional)", placeholder="main", scale=1)
                        model_type_dd = gr.Dropdown(
                            label="Model Type Override",
                            choices=["auto", "llm-gguf", "vlm-gguf", "embedding-gguf", "llm-safetensors", "diffusion", "audio"],
                            value="auto",
                            scale=1,
                        )
                        unpack_cb = gr.Checkbox(label="Unpack Archives", value=True, scale=1)

                    with gr.Row():
                        strategy_dd = gr.Dropdown(
                            label="Auto-Selector Strategy",
                            choices=["balanced", "max_quality", "max_speed", "max_context", "low_vram_safe"],
                            value=s.selection_strategy,
                        )
                        engine_override_dropdown = gr.Dropdown(
                            label="Engine Override",
                            choices=["auto", "llama.cpp", "ik_llama", "vllm", "transformers", "unsloth", "exllamav2", "diffusers", "faster-whisper"],
                            value="auto",
                        )
                        quantizer_override_dropdown = gr.Dropdown(
                            label="Quantizer Override",
                            choices=["auto", "llama.cpp", "ik_llama", "unsloth", "gptqmodel", "autoawq", "exllamav2", "ct2"],
                            value="auto",
                        )
                        target_quant_box = gr.Textbox(label="Target Quant (auto or e.g. Q4_K_M / IQ4_KS)", value="auto")

                    inspect_btn = gr.Button("Inspect Source & Compute VRAM Plan", variant="primary")
                    inspect_summary_md = gr.Markdown("Enter a source URI and click Inspect Source to analyze.")
                    variants_table = gr.Dataframe(
                        headers=["Quant", "Size", "Split Shards", "Shard Count", "IK Quant", "Vision mmproj", "VRAM Fit"],
                        label="Available Variants",
                    )

                    with gr.Accordion("Conversion, Quantization & Execution Options", open=True):
                        with gr.Row():
                            hub_task_dd = gr.Dropdown(
                                label="Action to Execute",
                                choices=["download_only", "serve", "convert_and_quantize", "convert_and_serve"],
                                value="download_only",
                            )
                            quant_dropdown = gr.Dropdown(label="Selected Quantization Variant", choices=["auto"], allow_custom_value=True)
                            sync_drive_cb = gr.Checkbox(label="Sync to Persistent Storage / Drive", value=False)
                        with gr.Row():
                            hub_imatrix_cb = gr.Checkbox(label="Use imatrix Calibration", value=False)
                            hub_split_box = gr.Textbox(label="Split Max Size (e.g. 2G)", placeholder="Leave blank for single file")
                            hub_outtype_dd = gr.Dropdown(label="Base Outtype (Safetensors->GGUF)", choices=["f16", "bf16", "f32", "q8_0"], value="f16")
                            hub_cleanup_cb = gr.Checkbox(label="Cleanup Intermediate Files", value=True)

                    download_btn = gr.Button("Execute Action (Download / Convert / Launch)", variant="secondary")
                    download_status_md = gr.Markdown("")

                # Tab 3: Model Registry, Scanner & Exporter
                with gr.Tab("Registry & Exporter"):
                    with gr.Row():
                        refresh_reg_btn = gr.Button("Refresh Registry", variant="secondary")
                        scan_reg_btn = gr.Button("Scan Disk & Drive for Models", variant="primary")
                    reg_table = gr.Dataframe(
                        headers=["Status", "Model ID", "Type", "Quant", "Engine Hint", "Size", "Synced", "Benchmark", "Local Path"],
                        value=self.get_registry_table_ui(),
                    )
                    with gr.Row():
                        selected_mid = gr.Textbox(label="Target Model ID", placeholder="Enter Model ID from table above", scale=2)
                        swap_btn = gr.Button("Switch Active Model", variant="primary")
                        sync_btn = gr.Button("Sync to Drive", variant="secondary")
                        del_local_btn = gr.Button("Clear Local Cache", variant="stop")
                        rm_entry_btn = gr.Button("Remove Entry", variant="stop")
                    with gr.Row():
                        rm_files_cb = gr.Checkbox(label="Also Delete Files on Remove", value=False)
                        purge_temp_btn = gr.Button("Purge Runtime Temp Files", variant="secondary")
                    storage_action_md = gr.Markdown("")

                    gr.Markdown("### Artifact & Model Exporter")
                    with gr.Row():
                        export_mid_input = gr.Textbox(label="Model ID to Export", placeholder="Enter Model ID", scale=2)
                        export_mode_dropdown = gr.Dropdown(
                            label="Output Export Mode",
                            choices=["none", "local_ssd", "direct_download", "google_drive", "gdrive_relative", "gdrive_direct", "huggingface_hub", "custom_path"],
                            value="local_ssd",
                            scale=2,
                        )
                        export_dest_input = gr.Textbox(label="Destination Path or HF Repo ('org/repo')", placeholder="Optional", scale=3)
                        export_name_input = gr.Textbox(label="Output Name Override", placeholder="Optional", scale=2)
                    with gr.Row():
                        zip_output_cb = gr.Checkbox(label="Zip Output", value=False)
                        hf_priv_cb = gr.Checkbox(label="Private HF Repo", value=False)
                        cleanup_source_cb = gr.Checkbox(label="Clean Up Source After Export", value=False)
                        export_btn = gr.Button("Export Artifact", variant="primary")
                    export_status_md = gr.Markdown("")

                    refresh_reg_btn.click(fn=self.get_registry_table_ui, outputs=[reg_table])
                    scan_reg_btn.click(fn=self.scan_models_ui, outputs=[storage_action_md, reg_table])
                    swap_btn.click(fn=self.hot_swap_model_ui, inputs=[selected_mid], outputs=[storage_action_md, reg_table])
                    sync_btn.click(fn=self.sync_model_drive_ui, inputs=[selected_mid], outputs=[storage_action_md, reg_table])
                    del_local_btn.click(fn=self.delete_local_cache_ui, inputs=[selected_mid], outputs=[storage_action_md, reg_table])
                    rm_entry_btn.click(fn=self.remove_model_entry_ui, inputs=[selected_mid, rm_files_cb], outputs=[storage_action_md, reg_table])
                    purge_temp_btn.click(fn=self.purge_temp_files_ui, outputs=[storage_action_md])
                    export_btn.click(
                        fn=self.export_artifact_ui,
                        inputs=[export_mid_input, export_mode_dropdown, export_dest_input, export_name_input, zip_output_cb, hf_priv_cb, cleanup_source_cb],
                        outputs=[export_status_md],
                    )

                    inspect_btn.click(
                        fn=self.inspect_model_ui,
                        inputs=[source_input, ctx_input, input_mode_dropdown, input_file_box, hf_rev_box, model_type_dd, strategy_dd, engine_override_dropdown, quantizer_override_dropdown, target_quant_box],
                        outputs=[inspect_summary_md, variants_table, quant_dropdown, download_status_md],
                    )
                    download_btn.click(
                        fn=self.execute_hub_task_ui,
                        inputs=[hub_task_dd, input_mode_dropdown, source_input, input_file_box, hf_rev_box, unpack_cb, model_type_dd, strategy_dd, engine_override_dropdown, quantizer_override_dropdown, quant_dropdown, ctx_input, hub_imatrix_cb, hub_split_box, hub_outtype_dd, hub_cleanup_cb, sync_drive_cb],
                        outputs=[download_status_md, reg_table],
                    )

                # Tab 4: Server, Presets & Tunnel Control
                with gr.Tab("Server & Tunnel"):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(label="Inference Preset", choices=preset_choices, value=default_preset)
                        port_input = gr.Number(label="Server Port", value=s.port, precision=0)
                        auth_checkbox = gr.Checkbox(label="Enable API Key Auth", value=s.enforce_auth)
                        api_key_box = gr.Textbox(label="Custom API Key (Blank = Auto-Generate)", type="password")

                    with gr.Accordion("Full Manual Engine & Hardware Overrides", open=False):
                        with gr.Row():
                            srv_engine_override = gr.Dropdown(
                                label="Engine Override",
                                choices=["auto", "llama.cpp", "ik_llama", "vllm", "transformers", "unsloth", "exllamav2", "diffusers", "faster-whisper"],
                                value="auto",
                            )
                            srv_ctx_override = gr.Textbox(label="Context Size (int or 'auto')", value="4096")
                            srv_offload_mode = gr.Dropdown(label="GPU Offload Mode", choices=["auto", "all", "cpu", "manual"], value="auto")
                            srv_layers_override = gr.Number(label="GPU Layers (-1=all, 0=CPU)", value=-1, precision=0)
                        with gr.Row():
                            srv_fa_override = gr.Dropdown(label="Flash Attention", choices=["auto", "on", "off"], value="auto")
                            srv_cache_k = gr.Dropdown(label="KV Cache K", choices=["auto", "q8_0", "q4_0", "f16"], value="auto")
                            srv_cache_v = gr.Dropdown(label="KV Cache V", choices=["auto", "q8_0", "q4_0", "f16"], value="auto")
                            srv_parallel = gr.Number(label="Parallel Slots", value=1, precision=0)
                        with gr.Row():
                            srv_batch = gr.Number(label="Batch Size (-b)", value=512, precision=0)
                            srv_ubatch = gr.Number(label="UBatch Size (-ub)", value=512, precision=0)
                            srv_threads = gr.Number(label="CPU Threads (0=auto)", value=0, precision=0)
                            srv_split_mode = gr.Dropdown(label="Multi-GPU Split", choices=["layer", "row", "none"], value="layer")
                        with gr.Row():
                            srv_tensor_split = gr.Textbox(label="Tensor Split (e.g. 0.5,0.5)", value="")
                            srv_override_tensor = gr.Textbox(label="Tensor Offload (-ot, e.g. ffn_.*_exps.=CPU)", value="")
                            srv_quant_mode = gr.Dropdown(label="HF/vLLM Quant Mode", choices=["auto", "bnb_4bit", "bnb_8bit", "fp16", "bf16", "fp8", "awq", "gptq"], value="auto")
                            srv_dtype = gr.Dropdown(label="Compute Dtype", choices=["auto", "fp16", "bf16", "fp32"], value="auto")
                        with gr.Row():
                            srv_gpu_util = gr.Slider(minimum=0.3, maximum=0.98, step=0.02, value=0.90, label="vLLM GPU Memory Util")
                            srv_rope_scaling = gr.Textbox(label="RoPE Scaling (none/linear/yarn)", value="")
                            srv_chat_tpl = gr.Textbox(label="Chat Template Override", value="")
                            srv_reasoning_fmt = gr.Textbox(label="Reasoning Format (e.g. deepseek)", value="")
                        with gr.Row():
                            srv_mlock = gr.Checkbox(label="mlock", value=False)
                            srv_mmap = gr.Checkbox(label="mmap", value=True)
                            srv_custom_cli = gr.Textbox(label="Raw Extra CLI Flags", placeholder="e.g. --seed 42 --no-webui", scale=3)

                    with gr.Accordion("Save Custom Preset", open=False):
                        with gr.Row():
                            new_preset_key = gr.Textbox(label="Preset Key", placeholder="my_custom_preset")
                            new_preset_desc = gr.Textbox(label="Description", placeholder="Notes for this preset", scale=2)
                            save_preset_btn = gr.Button("Save Current Controls as Preset", variant="secondary")
                        preset_save_md = gr.Markdown("")

                    with gr.Row():
                        start_srv_btn = gr.Button("Start Server", variant="primary")
                        stop_srv_btn = gr.Button("Stop Server", variant="stop")

                    server_status_md = gr.Markdown("Server Stopped")
                    cmd_preview_code = gr.Code(label="Resolved Server Command", language="shell", value="")
                    endpoints_json = gr.Code(label="Local API Endpoints", language="json", value="{}")

                    gr.Markdown("### Public Tunneling (Cloudflare & Ngrok)")
                    with gr.Row():
                        tunnel_dropdown = gr.Dropdown(label="Tunnel Provider", choices=["cloudflared", "ngrok"], value="cloudflared", scale=1)
                        tunnel_token_input = gr.Textbox(label="Tunnel Token (Optional)", type="password", scale=2)
                        tunnel_host_input = gr.Textbox(label="Named Tunnel Hostname (Optional)", placeholder="https://llm.example.com", scale=2)
                        start_tun_btn = gr.Button("Start Tunnel", variant="primary")
                        stop_tun_btn = gr.Button("Stop Tunnel", variant="stop")

                    tunnel_status_md = gr.Markdown("Tunnel Stopped")

                    gr.Markdown("### Client Integration Snippets")
                    with gr.Row():
                        py_code = gr.Code(label="Python Client", language="python")
                        curl_code = gr.Code(label="cURL Command", language="shell")

                    gr.Markdown("### Server & Tunnel Logs")
                    with gr.Row():
                        log_type_dropdown = gr.Dropdown(
                            label="Select Log",
                            choices=["llama_server", "ik_llama_server", "cloudflared", "transformers", "unsloth", "vllm", "exllamav2", "diffusers", "audio"],
                            value="llama_server",
                        )
                        log_lines_num = gr.Number(label="Lines", value=120, precision=0)
                        refresh_log_btn = gr.Button("Refresh Logs")
                    log_viewer = gr.TextArea(label="Recent Log Output", lines=12, max_lines=25)

                    start_srv_btn.click(
                        fn=self.start_server_ui,
                        inputs=[
                            preset_dropdown, port_input, auth_checkbox, api_key_box,
                            srv_engine_override, srv_ctx_override, srv_offload_mode, srv_layers_override,
                            srv_fa_override, srv_cache_k, srv_cache_v, srv_parallel,
                            srv_batch, srv_ubatch, srv_threads, srv_split_mode,
                            srv_tensor_split, srv_override_tensor, srv_quant_mode, srv_dtype,
                            srv_gpu_util, srv_rope_scaling, srv_chat_tpl, srv_reasoning_fmt,
                            srv_mlock, srv_mmap, srv_custom_cli,
                        ],
                        outputs=[server_status_md, endpoints_json, cmd_preview_code, py_code, curl_code],
                    )
                    stop_srv_btn.click(fn=self.stop_server_ui, outputs=[server_status_md, endpoints_json])
                    save_preset_btn.click(
                        fn=self.save_preset_ui,
                        inputs=[
                            new_preset_key, srv_engine_override, srv_ctx_override, srv_layers_override,
                            srv_fa_override, srv_cache_k, srv_cache_v, srv_parallel,
                            srv_custom_cli, new_preset_desc,
                        ],
                        outputs=[preset_save_md, preset_dropdown],
                    )
                    start_tun_btn.click(
                        fn=self.start_tunnel_ui,
                        inputs=[port_input, tunnel_dropdown, tunnel_token_input, tunnel_host_input],
                        outputs=[tunnel_status_md, py_code, curl_code],
                    )
                    stop_tun_btn.click(fn=self.stop_tunnel_ui, outputs=[tunnel_status_md, py_code, curl_code])
                    refresh_log_btn.click(fn=self.view_logs_ui, inputs=[log_type_dropdown, log_lines_num], outputs=[log_viewer])

                # Tab 5: Environment Profiles & Cache
                with gr.Tab("Environments & Cache"):
                    refresh_prof_btn = gr.Button("Refresh Status", variant="secondary")
                    profiles_table = gr.Dataframe(
                        headers=["Profile", "Status", "Isolation", "Components"],
                        value=self.get_profiles_table_ui(),
                    )
                    with gr.Row():
                        prof_select = gr.Dropdown(
                            label="Profile to Install / Repair",
                            choices=["core", "gguf", "ik_llama", "unsloth", "transformers", "vllm", "diffusion", "audio", "gptqmodel", "autoawq", "exllamav2", "ct2"],
                            value="gguf",
                        )
                        prof_isolate_cb = gr.Checkbox(label="Use Isolated Venv", value=True)
                        install_prof_btn = gr.Button("Install / Repair Profile", variant="primary")
                    prof_msg_md = gr.Markdown("")

                    gr.Markdown("### Cache Cleanup")
                    with gr.Row():
                        cache_scope_dd = gr.Dropdown(label="Cache Scope", choices=["temp", "models_cache", "envs", "bin", "uv", "all"], value="temp")
                        clear_cache_btn = gr.Button("Clear Selected Cache", variant="stop")
                    cache_clear_md = gr.Markdown("")

                    refresh_prof_btn.click(fn=self.get_profiles_table_ui, outputs=[profiles_table])
                    install_prof_btn.click(fn=self.install_profile_ui, inputs=[prof_select, prof_isolate_cb], outputs=[prof_msg_md, profiles_table])
                    clear_cache_btn.click(fn=self.clear_cache_ui, inputs=[cache_scope_dd], outputs=[cache_clear_md])

                # Tab 6: Multi-Modal Playground
                with gr.Tab("Playground"):
                    with gr.Tabs():
                        with gr.Tab("Chat & Vision (LLM / VLM)"):
                            chatbot = gr.Chatbot(label="Chat Window")
                            with gr.Row():
                                chat_msg = gr.Textbox(label="Message", placeholder="Type prompt and press Enter...", lines=2, scale=3)
                                chat_img = gr.Image(label="Optional VLM Image", type="filepath", scale=1)
                            with gr.Accordion("Generation Parameters", open=False):
                                chat_sys = gr.Textbox(label="System Prompt", value="You are a helpful assistant.")
                                with gr.Row():
                                    chat_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                                    chat_top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-P")
                                    chat_max_tok = gr.Slider(32, 4096, value=512, step=32, label="Max Tokens")
                            with gr.Row():
                                chat_send_btn = gr.Button("Send", variant="primary")
                                chat_clear_btn = gr.Button("Clear Chat")

                            chat_msg.submit(
                                fn=self.chat_interact,
                                inputs=[chat_msg, chat_img, chat_sys, chat_temp, chat_top_p, chat_max_tok, chatbot],
                                outputs=[chat_msg, chatbot],
                            )
                            chat_send_btn.click(
                                fn=self.chat_interact,
                                inputs=[chat_msg, chat_img, chat_sys, chat_temp, chat_top_p, chat_max_tok, chatbot],
                                outputs=[chat_msg, chatbot],
                            )
                            chat_clear_btn.click(fn=lambda: [], outputs=[chatbot])

                        with gr.Tab("Image Generation (Diffusers)"):
                            with gr.Row():
                                img_prompt = gr.Textbox(label="Prompt", placeholder="Describe the image...", scale=2)
                                img_neg = gr.Textbox(label="Negative Prompt", value="", scale=1)
                            with gr.Row():
                                img_size = gr.Dropdown(label="Size", choices=["512x512", "768x768", "1024x1024"], value="512x512")
                                img_steps = gr.Slider(5, 60, value=25, step=1, label="Steps")
                                img_guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
                                img_gen_btn = gr.Button("Generate Image", variant="primary")
                            img_out = gr.Image(label="Generated Output")
                            img_status = gr.Markdown("")
                            img_gen_btn.click(
                                fn=self.generate_image_ui,
                                inputs=[img_prompt, img_neg, img_size, img_steps, img_guidance],
                                outputs=[img_out, img_status],
                            )

                        with gr.Tab("Audio Transcription (Whisper)"):
                            with gr.Row():
                                aud_file = gr.Audio(label="Upload Audio", type="filepath")
                                aud_lang = gr.Textbox(label="Language Code (Optional, e.g. en, fa)", value="")
                                aud_task = gr.Dropdown(label="Task", choices=["transcribe", "translate"], value="transcribe")
                            aud_btn = gr.Button("Run Whisper", variant="primary")
                            aud_out = gr.Code(label="Transcription Result", language="json")
                            aud_btn.click(fn=self.transcribe_audio_ui, inputs=[aud_file, aud_lang, aud_task], outputs=[aud_out])

                # Tab 7: Benchmark & Comparison
                with gr.Tab("Benchmark & Compare"):
                    with gr.Row():
                        bench_prompt = gr.Textbox(
                            label="Benchmark Prompt",
                            value="Explain the theory of general relativity and its historical experimental verifications in three concise paragraphs.",
                            lines=2,
                            scale=3,
                        )
                        bench_tokens = gr.Slider(minimum=32, maximum=1024, step=32, value=128, label="Max Tokens", scale=1)
                        bench_runs = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Averaged Runs", scale=1)
                    bench_run_btn = gr.Button("Run Benchmark", variant="primary")
                    bench_results_md = gr.Markdown("Benchmark results will appear here.")
                    bench_history_table = gr.Dataframe(
                        headers=["Model ID", "Quant", "Engine", "TTFT (ms)", "Speed (tok/s)", "Tokens", "Tested At"],
                        value=self.get_benchmark_history_ui(),
                        label="Model Benchmark Comparison",
                    )
                    bench_run_btn.click(
                        fn=self.run_benchmark_ui,
                        inputs=[bench_prompt, bench_tokens, bench_runs],
                        outputs=[bench_results_md, bench_history_table],
                    )

                # Tab 8: Settings & Secrets
                with gr.Tab("Settings & Secrets"):
                    gr.Markdown("### Workspace Auto & Manual Policies (`settings.yaml`)")
                    with gr.Row():
                        set_strategy = gr.Dropdown(label="Default Auto Strategy", choices=["balanced", "max_quality", "max_speed", "max_context", "low_vram_safe"], value=s.selection_strategy)
                        set_iso = gr.Dropdown(label="Env Isolation Mode", choices=["auto", "isolated_venv", "system"], value=s.env_isolation_mode)
                        set_cuda = gr.Dropdown(label="CUDA Target", choices=["auto", "cuda-12", "cuda-13", "cpu"], value=s.cuda_target)
                        set_llama_tag = gr.Textbox(label="llama.cpp Release Tag", value=s.llama_cpp_release_tag)
                    with gr.Row():
                        set_dl = gr.Dropdown(label="Download Backend", choices=["auto", "aria2c", "hf_transfer", "urllib"], value=s.download_backend)
                        set_conns = gr.Slider(1, 16, value=s.aria2_connections, step=1, label="aria2c Connections")
                        set_overhead = gr.Number(label="CUDA Overhead (MB)", value=s.cuda_overhead_mb)
                        set_margin = gr.Slider(0.0, 0.25, value=s.vram_safety_margin_pct, step=0.01, label="VRAM Safety Margin")
                    with gr.Row():
                        set_inherit = gr.Checkbox(label="Venvs Inherit System Site Packages", value=s.venv_system_site_packages)
                        set_autoinstall = gr.Checkbox(label="Auto-Install Missing Engines", value=s.auto_install_dependencies)
                        set_autorequant = gr.Checkbox(label="Auto Re-quantize Oversized GGUFs", value=s.auto_requantize)
                        set_partial = gr.Checkbox(label="Auto Partial GPU Offload", value=s.auto_partial_offload)
                        set_ik_moe = gr.Checkbox(label="Prefer ik_llama for Oversized MoE", value=s.prefer_ik_llama_for_moe)
                        set_imatrix = gr.Checkbox(label="Default Use imatrix", value=s.use_imatrix)
                        set_cleanup = gr.Checkbox(label="Cleanup Intermediate Files", value=s.cleanup_intermediate_files)
                    save_settings_btn = gr.Button("Save Settings", variant="primary")
                    settings_status_md = gr.Markdown("")

                    save_settings_btn.click(
                        fn=self.save_settings_ui,
                        inputs=[
                            set_strategy, set_iso, set_inherit, set_autoinstall, set_cuda,
                            set_llama_tag, set_dl, set_conns, set_overhead, set_margin,
                            set_autorequant, set_partial, set_ik_moe, set_imatrix, set_cleanup,
                        ],
                        outputs=[settings_status_md],
                    )

                    gr.Markdown("### Session Secrets & Tokens (In-Memory / Environment)")
                    with gr.Row():
                        sec_hf = gr.Textbox(label="HF_TOKEN", type="password")
                        sec_civitai = gr.Textbox(label="CIVITAI_API_KEY", type="password")
                        sec_cf = gr.Textbox(label="CF_TUNNEL_TOKEN", type="password")
                        sec_ngrok = gr.Textbox(label="NGROK_AUTHTOKEN", type="password")
                        sec_ws = gr.Textbox(label="WORKSPACE_API_KEY", type="password")
                    save_sec_btn = gr.Button("Apply Session Secrets", variant="secondary")
                    sec_status_md = gr.Markdown(
                        "\n".join([f"- **{k}**: `{v}`" for k, v in SecurityManager.list_secrets_status().items()])
                    )
                    save_sec_btn.click(
                        fn=self.update_secrets_ui,
                        inputs=[sec_hf, sec_civitai, sec_cf, sec_ngrok, sec_ws],
                        outputs=[sec_status_md],
                    )

            demo.load(
                fn=self.get_hardware_telemetry_ui,
                outputs=[gpu_title_md, gpu_detail_md, ram_md, disk_md, env_md, proc_md],
            )

        return demo


def launch_dashboard(
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
    config: Optional[WorkspaceConfig] = None,
) -> Any:
    """Launches the workspace Gradio management dashboard."""
    dashboard = WorkspaceDashboard(config=config)
    app = dashboard.build_ui()
    app.queue().launch(server_name="0.0.0.0", server_port=port, share=share, inbrowser=inbrowser)
    return app
