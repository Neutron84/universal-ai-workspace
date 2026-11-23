"""
Interactive Management Dashboard for Google Colab
Provides a 6-tab Gradio control interface:
Tab 1: Hardware & Telemetry
Tab 2: Model Hub & Downloader
Tab 3: Model Registry & Storage
Tab 4: Server & Tunnel Control
Tab 5: Environment Profiles
Tab 6: Chat & Benchmark
"""

import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import requests

from .config import ServerPreset, WorkspaceConfig, get_config
from .hardware import HardwareProfiler, get_hardware_profile
from .env_manager import EnvironmentManager
from .model_inspector import GGUFVariant, InspectionResult, ModelInspector
from .auto_selector import AutoSelector, SelectionRecommendation
from .downloader import Downloader
from .registry import ModelEntry, ModelRegistry
from .process_manager import ProcessManager, SystemState
from .security import SecurityManager


class WorkspaceDashboard:
    """Encapsulates the Gradio management dashboard."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.registry = ModelRegistry(self.config)
        self.inspector = ModelInspector()
        self.selector = AutoSelector(config=self.config)
        self.downloader = Downloader(self.config)
        self.process_manager = ProcessManager(self.config)
        self.env_manager = EnvironmentManager(self.config)
        self.last_inspection: Optional[InspectionResult] = None
        self.last_recommendation: Optional[SelectionRecommendation] = None

    # --- Tab 1 Callbacks ---
    def get_hardware_telemetry_ui(self) -> Tuple[str, str, str, str, str, str]:
        hw = get_hardware_profile()
        gpu_str = "No GPU Detected (CPU Mode)"
        gpu_detail = "N/A"
        if hw.primary_gpu:
            g = hw.primary_gpu
            gpu_str = f"**GPU**: {g.name} ({g.sm_name})"
            temp = f"{g.temperature_c:.0f}°C" if g.temperature_c is not None else "N/A"
            util = f"{g.utilization_pct:.0f}%" if g.utilization_pct is not None else "N/A"
            gpu_detail = (
                f"**VRAM Total**: {g.total_vram_mb:.0f} MB\n\n"
                f"**VRAM Free**: {g.free_vram_mb:.0f} MB\n\n"
                f"**VRAM Used**: {g.used_vram_mb:.0f} MB\n\n"
                f"**Utilization**: {util} | **Temp**: {temp}\n\n"
                f"**Flash Attention**: {'Supported' if g.supports_flash_attn else 'Not Supported'}\n\n"
                f"**KV Quantization**: {'Supported' if g.supports_kv_quant else 'Not Supported'}\n\n"
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
            f"**Google Drive Storage**: {hw.drive_free_gb:.1f} GB free / {hw.drive_total_gb:.1f} GB total"
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

    # --- Tab 2 Callbacks ---
    def inspect_model_ui(
        self,
        source_url: str,
        requested_ctx: int,
        input_mode: str = "auto",
        engine_override: str = "auto",
        quantizer_override: str = "auto",
    ) -> Tuple[str, List[List[str]], gr.Dropdown, str]:
        if not source_url.strip():
            return "Please provide a model repository, URL, or local path.", [], gr.Dropdown(choices=[]), ""

        inspection = self.inspector.inspect(source_url.strip(), input_mode=input_mode)
        self.last_inspection = inspection

        if inspection.error:
            return f"Inspection Error: {inspection.error}", [], gr.Dropdown(choices=[]), ""

        recommendation = self.selector.recommend(
            inspection,
            requested_ctx=requested_ctx,
            override_engine=engine_override,
            override_quantizer=quantizer_override
        )
        self.last_recommendation = recommendation

        summary_text = (
            f"### Model: `{inspection.model_id}`\n\n"
            f"- **Type**: `{inspection.model_type}`\n"
            f"- **Architecture**: `{inspection.architecture_name or 'N/A'}` (MoE: {'Yes' if inspection.is_moe else 'No'}, LoRA: {'Yes' if inspection.is_lora else 'No'}, VLM: {'Yes' if inspection.is_vlm else 'No'})\n"
            f"- **Quant Format**: `{inspection.quant_format}` (IK Quant: {'Yes' if inspection.is_ik_quant else 'No'})\n"
            f"- **Input Source**: `{inspection.source_type}` (Mode: `{input_mode}`)\n"
            f"- **Selected Engine**: `{recommendation.recommended_engine}`\n"
            f"- **Selected Converter**: `{recommendation.recommended_converter or 'none'}`\n"
            f"- **Total Files**: {inspection.total_files_count} ({inspection.total_repo_size_bytes / (1024**3):.2f} GB)\n"
            f"- **VRAM Budget**: {recommendation.vram_budget_mb / 1024:.2f} GB\n"
            f"- **Recommended Quant**: **`{recommendation.recommended_quant}`**\n"
            f"- **Strategy**: {recommendation.reasoning}\n"
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
            quant_choices = ["N/A (Safetensors / Direct)"]

        rec_quant = recommendation.recommended_quant
        default_val = rec_quant if rec_quant in quant_choices else quant_choices[0]

        return (
            summary_text,
            table_data,
            gr.Dropdown(choices=quant_choices, value=default_val),
            recommendation.decision_summary or recommendation.reasoning
        )

    def download_model_ui(
        self,
        source_url: str,
        selected_quant: str,
        requested_ctx: int,
        sync_to_drive: bool
    ) -> str:
        if not self.last_inspection:
            return "Please click 'Inspect Source' first before downloading."

        insp = self.last_inspection

        chosen_variant = None
        for v in insp.gguf_variants:
            if v.quant == selected_quant:
                chosen_variant = v
                break
        if not chosen_variant and insp.gguf_variants:
            chosen_variant = insp.gguf_variants[0]

        rec = self.selector.recommend(insp, requested_ctx=requested_ctx, override_quant=selected_quant)

        try:
            dl_res = self.downloader.resolve_or_download(
                inspection=insp,
                variant=chosen_variant,
                sync_to_drive=sync_to_drive
            )

            final_path = dl_res.primary_file_path

            if rec.needs_requant and rec.requant_target_quant:
                print(f"[Dashboard] Performing re-quantization to {rec.requant_target_quant}...")
                final_path = AutoSelector.execute_requantize(
                    source_path=final_path,
                    target_quant=rec.requant_target_quant,
                    backup_to_drive=sync_to_drive,
                    cleanup_source=True
                )

            cat_map = {
                "llm-gguf": self.config.models_gguf_dir,
                "vlm-gguf": self.config.models_gguf_dir,
                "embedding-gguf": self.config.models_gguf_dir,
                "llm-safetensors": self.config.models_safetensors_dir,
                "diffusion": self.config.models_diffusion_dir,
                "audio": self.config.models_audio_dir,
            }
            dest_dir = cat_map.get(insp.model_type, self.config.models_gguf_dir)
            drive_path = str(dest_dir / final_path.name) if sync_to_drive else None
            shard_paths = [str(p) for p in dl_res.all_downloaded_files if p.resolve() != final_path.resolve()]

            mid = dl_res.model_id
            entry = self.registry.register_model(
                model_id=mid,
                name=mid,
                type=insp.model_type,
                source_url=insp.raw_source,
                quantization=rec.recommended_quant,
                total_size_bytes=dl_res.total_size_bytes,
                local_path=str(final_path),
                drive_path=drive_path,
                is_synced_to_drive=sync_to_drive,
                mmproj_path=str(dl_res.mmproj_file_path) if dl_res.mmproj_file_path else None,
                shard_files=shard_paths,
                set_active=True
            )

            status_msg = (
                f"**Model Ready and Registered Active**\n\n"
                f"- **Model ID**: `{entry.model_id}`\n"
                f"- **Quant**: `{entry.quantization}`\n"
                f"- **Path**: `{entry.local_path}`\n"
                f"- **Drive Sync**: {'Yes' if entry.is_synced_to_drive else 'No'}\n"
                f"- **Vision Projector**: `{entry.mmproj_path or 'None'}`"
            )
            return status_msg
        except Exception as e:
            return f"Download and registration error: {str(e)}"

    def export_artifact_ui(
        self,
        model_id: str,
        output_mode: str,
        destination: str,
        output_name: str,
        zip_output: bool,
        cleanup_source: bool
    ) -> str:
        """Exports a registered model or artifact to the selected sink."""
        if not model_id:
            return "Please select or enter a Model ID."
        entry = self.registry.get_model(model_id.strip())
        if not entry:
            return f"Model '{model_id}' not found in registry."

        src_path = entry.local_path if Path(entry.local_path).exists() else entry.drive_path
        if not src_path or not Path(src_path).exists():
            return f"File not found for model '{model_id}' on local SSD or Drive."

        from .io_resolver import OutputManager
        out_mgr = OutputManager(paths=self.config.paths)
        try:
            res = out_mgr.export_artifact(
                source_path=src_path,
                output_mode=output_mode,
                destination=destination.strip() or None,
                output_name=output_name.strip() or None,
                zip_output=zip_output,
                cleanup_source=cleanup_source,
            )
            return f"**Export Complete.** {res.get('message', '')}"
        except Exception as e:
            return f"Export error: {str(e)}"

    # --- Tab 3 Callbacks ---
    def get_registry_table_ui(self) -> List[List[str]]:
        self.registry.load()
        models = self.registry.list_models()
        data = []
        for m in models:
            data.append([
                "Active" if m.is_active else "Inactive",
                m.model_id,
                m.type,
                m.quantization,
                f"{m.total_size_gb:.2f} GB",
                "Yes" if m.is_synced_to_drive else "No",
                m.local_path,
                m.last_used_at or "Never",
            ])
        return data

    def hot_swap_model_ui(self, model_id: str) -> Tuple[str, List[List[str]]]:
        if not model_id:
            return "Please enter a Model ID to switch.", self.get_registry_table_ui()
        res = self.registry.set_active_model(model_id.strip())
        if res:
            state = self.process_manager.get_state()
            if state.is_server_running:
                self.process_manager.start_server(model_id=res.model_id, port=state.port)
            return f"Switched active model to `{res.model_id}`.", self.get_registry_table_ui()
        return f"Model `{model_id}` not found in registry.", self.get_registry_table_ui()

    def sync_model_drive_ui(self, model_id: str) -> Tuple[str, List[List[str]]]:
        if not model_id:
            return "Please enter a Model ID to sync.", self.get_registry_table_ui()
        ok = self.registry.sync_to_drive(model_id.strip())
        if ok:
            return f"Synced `{model_id}` to Google Drive.", self.get_registry_table_ui()
        return f"Failed to sync `{model_id}` to Drive.", self.get_registry_table_ui()

    def delete_local_cache_ui(self, model_id: str) -> Tuple[str, List[List[str]]]:
        if not model_id:
            return "Please enter a Model ID.", self.get_registry_table_ui()
        ok = self.registry.delete_local_cache(model_id.strip())
        if ok:
            return f"Deleted local SSD cache for `{model_id}`. Drive archive preserved.", self.get_registry_table_ui()
        return f"Failed to delete local cache for `{model_id}`.", self.get_registry_table_ui()

    def purge_temp_files_ui(self) -> str:
        freed = self.registry.purge_temp_files()
        return f"Purged temporary files. Freed {freed / (1024*1024):.2f} MB."

    # --- Tab 4 Callbacks ---
    def start_server_ui(
        self,
        preset_name: str,
        port: int,
        enable_auth: bool,
        engine_override: str = "auto",
        ctx_override: int = 4096,
        gpu_layers_override: int = -1,
        flash_attn_override: bool = True,
        cache_k_override: str = "auto",
        cache_v_override: str = "auto",
    ) -> Tuple[str, str, str, str]:
        try:
            overrides = {}
            if engine_override and str(engine_override).lower() != "auto":
                overrides["backend"] = engine_override
            if ctx_override:
                overrides["ctx_size"] = int(ctx_override)
            if gpu_layers_override is not None:
                overrides["n_gpu_layers"] = int(gpu_layers_override)
            if flash_attn_override is not None:
                overrides["flash_attn"] = bool(flash_attn_override)
            if cache_k_override and str(cache_k_override).lower() != "auto":
                overrides["cache_type_k"] = cache_k_override
            if cache_v_override and str(cache_v_override).lower() != "auto":
                overrides["cache_type_v"] = cache_v_override

            success = self.process_manager.start_server(
                preset_name=preset_name,
                port=int(port),
                enable_auth=enable_auth,
                **overrides
            )
            state = self.process_manager.get_state()
            status_text = "Server Running" if state.is_server_running else "Server Failed to Start"
            py_snippet, curl_snippet = self.generate_code_snippets(state)
            return status_text, json.dumps(state.local_endpoints, indent=2), py_snippet, curl_snippet
        except Exception as e:
            return f"Server start error: {str(e)}", "{}", "", ""

    def stop_server_ui(self) -> Tuple[str, str]:
        self.process_manager.stop_server()
        return "Server Stopped", "{}"

    def start_tunnel_ui(self, port: int, tunnel_type: str, tunnel_token: str = "") -> Tuple[str, str, str]:
        token_val = tunnel_token.strip() if tunnel_token and tunnel_token.strip() else None
        pub_url = self.process_manager.start_tunnel(port=int(port), tunnel_type=tunnel_type, tunnel_token=token_val)
        state = self.process_manager.get_state()
        status_text = f"Tunnel Active: {pub_url}" if pub_url else "Tunnel Inactive"
        py_snippet, curl_snippet = self.generate_code_snippets(state)
        return status_text, py_snippet, curl_snippet

    def stop_tunnel_ui(self) -> Tuple[str, str, str]:
        self.process_manager.stop_tunnel()
        state = self.process_manager.get_state()
        py_snippet, curl_snippet = self.generate_code_snippets(state)
        return "Tunnel Stopped", py_snippet, curl_snippet

    def generate_code_snippets(self, state: SystemState) -> Tuple[str, str]:
        base_url = state.public_url or f"http://127.0.0.1:{state.port}"
        model_name = state.active_model or "active-model"
        api_key = state.api_key or "your-api-key"

        py_snippet = f"""from openai import OpenAI

# Connect to API endpoint
client = OpenAI(
    base_url="{base_url}/v1",
    api_key="{api_key}"
)

# Streaming request
response = client.chat.completions.create(
    model="{model_name}",
    messages=[
        {{"role": "system", "content": "You are a helpful assistant."}},
        {{"role": "user", "content": "Explain quantum computing in two sentences."}}
    ],
    temperature=0.7,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
"""

        curl_snippet = f"""curl -X POST "{base_url}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {api_key}" \\
  -d '{{
    "model": "{model_name}",
    "messages": [
      {{"role": "system", "content": "You are a helpful assistant."}},
      {{"role": "user", "content": "Explain quantum computing in two sentences."}}
    ],
    "temperature": 0.7,
    "stream": false
  }}'
"""
        return py_snippet, curl_snippet

    def view_logs_ui(self, log_type: str) -> str:
        log_map = {
            "llama_server": self.config.runtime_logs_dir / "llama_server.log",
            "ik_llama_server": self.config.runtime_logs_dir / "ik_llama_server.log",
            "cloudflared": self.config.runtime_logs_dir / "cloudflared.log",
            "transformers": self.config.runtime_logs_dir / "transformers_server.log",
            "unsloth": self.config.runtime_logs_dir / "unsloth_server.log",
            "vllm": self.config.runtime_logs_dir / "vllm_server.log",
        }
        target = log_map.get(log_type, self.config.runtime_logs_dir / "llama_server.log")
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                return "".join(lines[-100:])
            except Exception as e:
                return f"Error reading logs: {e}"
        return f"Log file '{target.name}' does not exist yet."

    # --- Tab 5 Callbacks ---
    def get_profiles_table_ui(self) -> List[List[str]]:
        status = self.env_manager.get_all_profiles_status()
        rows = []
        for prof, data in status.items():
            comp_str = ", ".join([f"{k}: {'Installed' if v else 'Missing'}" for k, v in data["components"].items()])
            rows.append([
                prof,
                "Installed" if data["installed"] else "Incomplete",
                comp_str,
            ])
        return rows

    def install_profile_ui(self, profile_name: str) -> Tuple[str, List[List[str]]]:
        ok = self.env_manager.install_profile(profile_name)
        msg = f"Profile '{profile_name}' processed successfully." if ok else f"Profile '{profile_name}' installation failed."
        return msg, self.get_profiles_table_ui()

    # --- Tab 6 Callbacks ---
    def chat_interact(self, message: str, history: Optional[List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
        history = list(history or [])
        if not message.strip():
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

        messages = list(history)
        messages.append({"role": "user", "content": message})

        try:
            resp = requests.post(
                chat_url,
                json={
                    "model": state.active_model or "default",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "stream": False
                },
                headers=headers,
                timeout=30
            )
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"]
            else:
                reply = f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            reply = f"Request exception: {str(e)}"

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return "", history

    def run_benchmark_ui(self, prompt: str, max_tokens: int) -> str:
        state = self.process_manager.get_state()
        if not state.is_server_running or not self.process_manager.current_engine:
            return "Server is not running. Start the server first in Tab 4."

        res = self.process_manager.current_engine.benchmark(prompt=prompt, max_tokens=max_tokens)
        if "error" in res:
            return f"Benchmark Error: {res['error']}"

        if state.active_model:
            self.registry.update_benchmark(state.active_model, res)

        output = (
            f"### Benchmark Results\n\n"
            f"- **Model**: `{state.active_model}`\n"
            f"- **Time To First Token (TTFT)**: **{res.get('ttft_ms')} ms**\n"
            f"- **Generation Speed**: **{res.get('tokens_per_sec')} tokens/sec**\n"
            f"- **Total Tokens Generated**: {res.get('total_tokens')}\n"
            f"- **Total Latency**: {res.get('total_time_sec')} seconds\n\n"
            f"**Response Preview**:\n> {res.get('response_preview')}\n"
        )
        return output

    # --- UI Builder ---
    def build_ui(self) -> gr.Blocks:
        with gr.Blocks(title="Universal AI Workspace") as demo:
            gr.Markdown("# Universal AI Workspace")
            gr.Markdown("Interactive control dashboard for local and cloud model execution.")

            with gr.Tabs():
                # Tab 1: Telemetry
                with gr.Tab("Hardware & Telemetry"):
                    with gr.Row():
                        refresh_hw_btn = gr.Button("Refresh Telemetry", variant="primary")
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

                    refresh_hw_btn.click(
                        fn=self.get_hardware_telemetry_ui,
                        outputs=[gpu_title_md, gpu_detail_md, ram_md, disk_md, env_md, proc_md]
                    )

                # Tab 2: Model Hub & Downloader
                with gr.Tab("Model Hub & Downloader"):
                    ws_mode_str = self.config.mode.value if hasattr(self.config.mode, "value") else str(self.config.mode)
                    gr.Markdown(f"**Workspace Storage Mode**: `{ws_mode_str}` | **Runtime Root**: `{self.config.runtime_root}`")
                    with gr.Row():
                        input_mode_dropdown = gr.Dropdown(
                            label="Input Source Mode",
                            choices=[
                                "auto", "huggingface", "direct_upload", "local_path",
                                "google_drive", "gdrive_relative", "gdrive_share",
                                "url", "civitai", "git_repo"
                            ],
                            value="auto",
                            scale=1
                        )
                        source_input = gr.Textbox(
                            label="Model Source URI",
                            placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct-GGUF, hf://..., gdrive://..., civitai://..., or direct URL",
                            scale=3
                        )
                        ctx_input = gr.Slider(
                            minimum=1024, maximum=32768, step=1024, value=4096, label="Context Size (n_ctx)", scale=1
                        )
                    with gr.Row():
                        engine_override_dropdown = gr.Dropdown(
                            label="Engine Override",
                            choices=["auto", "llama.cpp", "ik_llama", "vllm", "transformers", "unsloth", "exllamav2", "diffusers", "faster-whisper"],
                            value="auto",
                            scale=1
                        )
                        quantizer_override_dropdown = gr.Dropdown(
                            label="Quantizer Override",
                            choices=["auto", "llama.cpp", "ik_llama", "unsloth", "gptqmodel", "autoawq", "exllamav2", "ct2"],
                            value="auto",
                            scale=1
                        )
                    with gr.Row():
                        inspect_btn = gr.Button("Inspect Source & Compute VRAM Fit", variant="primary")

                    inspect_summary_md = gr.Markdown("Enter a source URI and click Inspect Source to analyze.")
                    variants_table = gr.Dataframe(
                        headers=["Quant", "Size", "Split Shards", "Shard Count", "IK Quant", "Vision mmproj", "VRAM Fit"],
                        label="Available Variants"
                    )

                    with gr.Row():
                        quant_dropdown = gr.Dropdown(label="Selected Quantization Variant", choices=[])
                        sync_drive_cb = gr.Checkbox(label="Sync Model to Google Drive", value=False)

                    download_btn = gr.Button("Download & Prepare Model", variant="secondary")
                    download_status_md = gr.Markdown("")

                    inspect_btn.click(
                        fn=self.inspect_model_ui,
                        inputs=[source_input, ctx_input, input_mode_dropdown, engine_override_dropdown, quantizer_override_dropdown],
                        outputs=[inspect_summary_md, variants_table, quant_dropdown, download_status_md]
                    )

                    download_btn.click(
                        fn=self.download_model_ui,
                        inputs=[source_input, quant_dropdown, ctx_input, sync_drive_cb],
                        outputs=[download_status_md]
                    )

                # Tab 3: Model Registry & Storage
                with gr.Tab("Model Registry & Storage"):
                    refresh_reg_btn = gr.Button("Refresh Registry", variant="secondary")
                    reg_table = gr.Dataframe(
                        headers=["Status", "Model ID", "Type", "Quant", "Size", "Drive Synced", "Local Path", "Last Used"],
                        value=self.get_registry_table_ui()
                    )
                    with gr.Row():
                        selected_mid = gr.Textbox(label="Target Model ID", placeholder="Enter Model ID from table above")
                        swap_btn = gr.Button("Switch Active Model", variant="primary")
                        sync_btn = gr.Button("Sync to Drive", variant="secondary")
                        del_local_btn = gr.Button("Clear Local SSD Cache (Preserve Drive)", variant="stop")
                    with gr.Row():
                        purge_temp_btn = gr.Button("Purge Runtime Temp Files", variant="secondary")
                        storage_action_md = gr.Markdown("")

                    gr.Markdown("### Artifact & Model Exporter")
                    with gr.Row():
                        export_mid_input = gr.Textbox(label="Model ID to Export", placeholder="Enter Model ID from table above", scale=2)
                        export_mode_dropdown = gr.Dropdown(
                            label="Output Export Mode",
                            choices=[
                                "none", "local_ssd", "direct_download", "google_drive",
                                "gdrive_relative", "gdrive_direct", "huggingface_hub", "custom_path"
                            ],
                            value="local_ssd",
                            scale=2
                        )
                        export_dest_input = gr.Textbox(
                            label="Export Destination (Drive Path, HF Repo e.g. 'org/repo', or Directory)",
                            placeholder="Optional: leave blank for default output path",
                            scale=3
                        )
                        export_name_input = gr.Textbox(
                            label="Output Name Override",
                            placeholder="Leave blank to use model default",
                            scale=2
                        )
                    with gr.Row():
                        zip_output_cb = gr.Checkbox(label="Zip Output Artifact", value=False)
                        cleanup_source_cb = gr.Checkbox(label="Clean Up Source Files After Export", value=False)
                        export_btn = gr.Button("Export Artifact", variant="primary", scale=1)
                    export_status_md = gr.Markdown("")

                    refresh_reg_btn.click(fn=self.get_registry_table_ui, outputs=[reg_table])
                    swap_btn.click(fn=self.hot_swap_model_ui, inputs=[selected_mid], outputs=[storage_action_md, reg_table])
                    sync_btn.click(fn=self.sync_model_drive_ui, inputs=[selected_mid], outputs=[storage_action_md, reg_table])
                    del_local_btn.click(fn=self.delete_local_cache_ui, inputs=[selected_mid], outputs=[storage_action_md, reg_table])
                    purge_temp_btn.click(fn=self.purge_temp_files_ui, outputs=[storage_action_md])
                    export_btn.click(
                        fn=self.export_artifact_ui,
                        inputs=[export_mid_input, export_mode_dropdown, export_dest_input, export_name_input, zip_output_cb, cleanup_source_cb],
                        outputs=[export_status_md]
                    )

                # Tab 4: Server & Tunnel Control
                with gr.Tab("Server & Tunnel Control"):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            label="Inference Preset",
                            choices=list(self.config.presets.keys()),
                            value="gguf_balanced"
                        )
                        port_input = gr.Number(label="Server Port", value=8000, precision=0)
                        auth_checkbox = gr.Checkbox(label="Enable API Key Auth", value=True)

                    with gr.Accordion("Manual Hardware & Engine Overrides", open=False):
                        with gr.Row():
                            srv_engine_override = gr.Dropdown(
                                label="Engine Override",
                                choices=["auto", "llama.cpp", "ik_llama", "vllm", "transformers", "unsloth", "exllamav2", "diffusers", "faster-whisper"],
                                value="auto",
                                scale=1
                            )
                            srv_ctx_override = gr.Slider(
                                minimum=512, maximum=65536, step=512, value=4096, label="Context Size (n_ctx)", scale=1
                            )
                            srv_layers_override = gr.Number(
                                label="GPU Layers (-1 = all, 0 = CPU)", value=-1, precision=0, scale=1
                            )
                        with gr.Row():
                            srv_fa_override = gr.Checkbox(label="Flash Attention", value=True)
                            srv_cache_k = gr.Dropdown(label="KV Cache K Quant", choices=["auto", "q8_0", "q4_0", "f16"], value="auto")
                            srv_cache_v = gr.Dropdown(label="KV Cache V Quant", choices=["auto", "q8_0", "q4_0", "f16"], value="auto")

                    with gr.Row():
                        start_srv_btn = gr.Button("Start Server", variant="primary")
                        stop_srv_btn = gr.Button("Stop Server", variant="stop")

                    server_status_md = gr.Markdown("Server Stopped")
                    endpoints_json = gr.Code(label="Local API Endpoints", language="json", value="{}")

                    gr.Markdown("### Public Tunneling (Cloudflare & Ngrok)")
                    with gr.Row():
                        tunnel_dropdown = gr.Dropdown(label="Tunnel Provider", choices=["cloudflared", "ngrok"], value="cloudflared", scale=1)
                        tunnel_token_input = gr.Textbox(label="Tunnel Token (Optional: Named Cloudflare / Ngrok)", type="password", placeholder="Enter persistent tunnel token if available", scale=2)
                        start_tun_btn = gr.Button("Start Public Tunnel", variant="primary", scale=1)
                        stop_tun_btn = gr.Button("Stop Tunnel", variant="stop", scale=1)

                    tunnel_status_md = gr.Markdown("Tunnel Stopped")

                    gr.Markdown("### Client Integration Snippets")
                    with gr.Row():
                        py_code = gr.Code(label="Python (OpenAI Client)", language="python")
                        curl_code = gr.Code(label="cURL Command", language="shell")

                    gr.Markdown("### Server Logs")
                    with gr.Row():
                        log_type_dropdown = gr.Dropdown(
                            label="Select Log",
                            choices=["llama_server", "ik_llama_server", "cloudflared", "transformers", "unsloth", "vllm"],
                            value="llama_server"
                        )
                        refresh_log_btn = gr.Button("Refresh Logs")
                    log_viewer = gr.TextArea(label="Recent Log Output", lines=12, max_lines=25)

                    start_srv_btn.click(
                        fn=self.start_server_ui,
                        inputs=[preset_dropdown, port_input, auth_checkbox, srv_engine_override, srv_ctx_override, srv_layers_override, srv_fa_override, srv_cache_k, srv_cache_v],
                        outputs=[server_status_md, endpoints_json, py_code, curl_code]
                    )
                    stop_srv_btn.click(fn=self.stop_server_ui, outputs=[server_status_md, endpoints_json])
                    start_tun_btn.click(
                        fn=self.start_tunnel_ui,
                        inputs=[port_input, tunnel_dropdown, tunnel_token_input],
                        outputs=[tunnel_status_md, py_code, curl_code]
                    )
                    stop_tun_btn.click(fn=self.stop_tunnel_ui, outputs=[tunnel_status_md, py_code, curl_code])
                    refresh_log_btn.click(fn=self.view_logs_ui, inputs=[log_type_dropdown], outputs=[log_viewer])

                # Tab 5: Environment Profiles
                with gr.Tab("Environment Profiles"):
                    refresh_prof_btn = gr.Button("Refresh Status", variant="secondary")
                    profiles_table = gr.Dataframe(
                        headers=["Profile", "Status", "Components"],
                        value=self.get_profiles_table_ui()
                    )
                    with gr.Row():
                        prof_select = gr.Dropdown(
                            label="Profile to Install / Repair",
                            choices=[
                                "core", "gguf", "ik_llama", "unsloth", "transformers",
                                "vllm", "diffusion", "audio", "gptqmodel", "autoawq",
                                "exllamav2", "ct2"
                            ],
                            value="gguf"
                        )
                        install_prof_btn = gr.Button("Install / Repair Profile", variant="primary")
                    prof_msg_md = gr.Markdown("")

                    refresh_prof_btn.click(fn=self.get_profiles_table_ui, outputs=[profiles_table])
                    install_prof_btn.click(fn=self.install_profile_ui, inputs=[prof_select], outputs=[prof_msg_md, profiles_table])

                # Tab 6: Playground & Benchmark
                with gr.Tab("Playground & Benchmark"):
                    gr.Markdown("### Interactive Chat Playground")
                    chatbot = gr.Chatbot(label="Chat Window")
                    chat_msg = gr.Textbox(label="Message", placeholder="Type prompt and press Enter...", lines=1)
                    chat_clear_btn = gr.Button("Clear Chat")

                    chat_msg.submit(fn=self.chat_interact, inputs=[chat_msg, chatbot], outputs=[chat_msg, chatbot])
                    chat_clear_btn.click(fn=lambda: [], outputs=[chatbot])

                    gr.Markdown("### Real Benchmark (Tokens/sec & TTFT)")
                    with gr.Row():
                        bench_prompt = gr.Textbox(
                            label="Benchmark Prompt",
                            value="Explain the theory of general relativity and its historical experimental verifications in three concise paragraphs.",
                            lines=2
                        )
                        bench_tokens = gr.Slider(minimum=32, maximum=1024, step=32, value=128, label="Max Tokens")
                    bench_run_btn = gr.Button("Run Benchmark", variant="primary")
                    bench_results_md = gr.Markdown("Benchmark results will appear here.")

                    bench_run_btn.click(
                        fn=self.run_benchmark_ui,
                        inputs=[bench_prompt, bench_tokens],
                        outputs=[bench_results_md]
                    )

            # Auto-populate hardware info on launch
            demo.load(
                fn=self.get_hardware_telemetry_ui,
                outputs=[gpu_title_md, gpu_detail_md, ram_md, disk_md, env_md, proc_md]
            )

        return demo


def launch_dashboard(port: int = 7860, share: bool = False, inbrowser: bool = False) -> gr.Blocks:
    """Launches the workspace Gradio management dashboard."""
    dashboard = WorkspaceDashboard()
    app = dashboard.build_ui()
    app.queue().launch(server_name="0.0.0.0", server_port=port, share=share, inbrowser=inbrowser)
    return app
