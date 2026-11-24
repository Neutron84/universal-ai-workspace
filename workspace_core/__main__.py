"""
Command-Line Interface for Universal AI Workspace
Run via: python -m workspace_core <command> [options]
"""

import argparse
import json
import sys

from .config import get_config
from .env_manager import EnvironmentManager
from .orchestrator import LaunchRequest, WorkspaceOrchestrator
from .registry import ModelRegistry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="workspace_core",
        description="Universal AI Workspace CLI (Auto + Full Manual Controls)",
    )
    parser.add_argument("--workspace-mode", default="local_ssd", help="local_ssd, hybrid, google_drive, gdrive_relative, custom_path")
    parser.add_argument("--workspace-path", default=None, help="Workspace directory path")

    sub = parser.add_subparsers(dest="command", required=True)

    # Shared helper to add common model/engine args
    def add_pipeline_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("source", help="Model source URI (HF repo, URL, local path, civitai://, gdrive://)")
        p.add_argument("--input-type", default="auto")
        p.add_argument("--input-file", default=None)
        p.add_argument("--revision", default=None)
        p.add_argument("--strategy", default="balanced", choices=["balanced", "max_quality", "max_speed", "max_context", "low_vram_safe"])
        p.add_argument("--engine", default="auto")
        p.add_argument("--quantizer", default="auto")
        p.add_argument("--target-quant", default="auto")
        p.add_argument("--ctx", default="4096", help="Context size (integer or 'auto')")
        p.add_argument("--ngl", type=int, default=-1, help="GPU layers (-1=all, 0=cpu, N=partial)")
        p.add_argument("--offload-mode", default="auto", choices=["auto", "all", "cpu", "manual"])
        p.add_argument("--flash-attn", default="auto", choices=["auto", "true", "false"])
        p.add_argument("--cache-k", default="auto")
        p.add_argument("--cache-v", default="auto")
        p.add_argument("--parallel", type=int, default=1)
        p.add_argument("--override-tensor", default=None, help="Tensor offload rules (-ot)")
        p.add_argument("--custom-args", default="", help="Raw server CLI flags")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--no-auth", action="store_true")
        p.add_argument("--tunnel", default="none", choices=["none", "cloudflared", "ngrok"])
        p.add_argument("--tunnel-token", default=None)
        p.add_argument("--imatrix", action="store_true")
        p.add_argument("--split-max-size", default=None)
        p.add_argument("--output-type", default="none")
        p.add_argument("--output-path", default="AI_Workspace/outputs")
        p.add_argument("--output-name", default=None)
        p.add_argument("--zip-output", action="store_true")

    p_inspect = sub.add_parser("inspect", help="Inspect a model source and compute AutoSelector plan (dry-run)")
    add_pipeline_args(p_inspect)

    p_serve = sub.add_parser("serve", help="Download/load and serve a model")
    add_pipeline_args(p_serve)

    p_convert = sub.add_parser("convert", help="Convert and/or quantize a model")
    add_pipeline_args(p_convert)

    p_bench = sub.add_parser("benchmark", help="Download/serve and benchmark a model")
    add_pipeline_args(p_bench)

    p_dash = sub.add_parser("dashboard", help="Launch Gradio management dashboard")
    p_dash.add_argument("--port", type=int, default=7860)
    p_dash.add_argument("--share", action="store_true")

    sub.add_parser("doctor", help="Run system, GPU, CUDA, and binary diagnostics")
    sub.add_parser("scan", help="Scan workspace directories for unregistered models")
    sub.add_parser("presets", help="List configured server presets")

    args = parser.parse_args(argv)
    cfg = get_config(workspace_mode=args.workspace_mode, workspace_path=args.workspace_path)

    if args.command == "doctor":
        report = EnvironmentManager(cfg).run_doctor()
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "scan":
        added = ModelRegistry(cfg).scan_workspace_models()
        print(f"Discovered and registered {len(added)} new model(s):")
        for e in added:
            print(f"  - {e.model_id} ({e.type}, {e.quantization}, {e.total_size_gb} GB)")
        return 0

    if args.command == "presets":
        for key, p in cfg.presets.items():
            print(f"{key}: backend={p.backend}, ctx={p.ctx_size}, ngl={p.n_gpu_layers} -- {p.description}")
        return 0

    if args.command == "dashboard":
        from .dashboard import launch_dashboard
        launch_dashboard(port=args.port, share=args.share, config=cfg)
        return 0

    task_map = {
        "inspect": "inspect_only",
        "serve": "serve",
        "convert": "convert_and_quantize",
        "benchmark": "benchmark",
    }
    ctx_val = args.ctx if str(args.ctx).lower() == "auto" else int(args.ctx)
    req = LaunchRequest(
        input_type=args.input_type,
        input_path=args.source,
        input_file=args.input_file,
        hf_revision=args.revision,
        task=task_map[args.command],
        strategy=args.strategy,
        engine=args.engine,
        quantizer=args.quantizer,
        target_quant=args.target_quant,
        context_size=ctx_val,
        gpu_offload_mode=args.offload_mode,
        n_gpu_layers=args.ngl,
        flash_attn=args.flash_attn,
        cache_type_k=args.cache_k,
        cache_type_v=args.cache_v,
        parallel=args.parallel,
        override_tensor=args.override_tensor,
        custom_server_args=args.custom_args,
        port=args.port,
        enable_api_auth=not args.no_auth,
        enable_tunnel=(args.tunnel != "none"),
        tunnel_type=args.tunnel,
        tunnel_token=args.tunnel_token,
        use_imatrix=args.imatrix,
        split_max_size=args.split_max_size,
        output_type=args.output_type,
        output_path=args.output_path,
        output_name=args.output_name,
        zip_output=args.zip_output,
    )

    orch = WorkspaceOrchestrator(cfg)
    result = orch.run(req)
    print(result.message)
    if result.system_state:
        print(json.dumps({
            "running": result.system_state.is_server_running,
            "engine": result.system_state.engine_name,
            "port": result.system_state.port,
            "public_url": result.system_state.public_url,
            "api_key": result.system_state.api_key,
        }, indent=2))
    if result.benchmark_results:
        print("Benchmark:", json.dumps(result.benchmark_results, indent=2))
    if result.export_result:
        print("Export:", json.dumps(result.export_result, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
