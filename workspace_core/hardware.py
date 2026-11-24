"""
Hardware Profiler & Dynamic Environment Diagnostics
Detects GPU, Compute Capability, VRAM, exact CUDA version, CPU, RAM, and Disk space.
Provides exact architecture-aware KV cache estimation and NVIDIA library setup.
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import psutil

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class GPUInfo:
    index: int
    name: str
    compute_capability: Tuple[int, int]
    sm_name: str
    total_vram_mb: float
    free_vram_mb: float
    used_vram_mb: float
    temperature_c: Optional[float] = None
    utilization_pct: Optional[float] = None
    supports_flash_attn: bool = False
    supports_kv_quant: bool = False
    supports_vllm: bool = False
    supports_bf16: bool = False
    supports_fp8: bool = False


@dataclass
class HardwareProfile:
    has_gpu: bool
    gpus: List[GPUInfo] = field(default_factory=list)
    primary_gpu: Optional[GPUInfo] = None
    cuda_version: str = "N/A"
    cuda_major: Optional[int] = None
    cuda_minor: Optional[int] = None
    driver_version: str = "N/A"
    python_version: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    python_executable: str = field(default_factory=lambda: sys.executable)
    cpu_cores: int = field(default_factory=lambda: os.cpu_count() or 1)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    ssd_total_gb: float = 0.0
    ssd_free_gb: float = 0.0
    ssd_used_gb: float = 0.0
    drive_total_gb: float = 0.0
    drive_free_gb: float = 0.0
    drive_used_gb: float = 0.0
    nvidia_lib_paths: List[str] = field(default_factory=list)

    @property
    def total_vram_mb(self) -> float:
        return sum(g.total_vram_mb for g in self.gpus)

    @property
    def total_free_vram_mb(self) -> float:
        return sum(g.free_vram_mb for g in self.gpus)

    def summary(self) -> Dict[str, Any]:
        gpu_summary = "None (CPU/TPU fallback)"
        if self.primary_gpu:
            g = self.primary_gpu
            multi = f" x{len(self.gpus)}" if len(self.gpus) > 1 else ""
            gpu_summary = f"{g.name}{multi} ({g.sm_name}, {self.total_free_vram_mb:.0f}MB / {self.total_vram_mb:.0f}MB free)"

        return {
            "GPU": gpu_summary,
            "CUDA": self.cuda_version,
            "Driver": self.driver_version,
            "Python": f"{self.python_version} ({self.python_executable})",
            "CPU Cores": self.cpu_cores,
            "System RAM": f"{self.ram_available_gb:.1f}GB free / {self.ram_total_gb:.1f}GB total ({self.ram_percent}% used)",
            "Local SSD": f"{self.ssd_free_gb:.1f}GB free / {self.ssd_total_gb:.1f}GB total",
            "Drive Storage": f"{self.drive_free_gb:.1f}GB free / {self.drive_total_gb:.1f}GB total",
            "Flash Attention": self.primary_gpu.supports_flash_attn if self.primary_gpu else False,
            "KV Quantization": self.primary_gpu.supports_kv_quant if self.primary_gpu else False,
            "BF16 Supported": self.primary_gpu.supports_bf16 if self.primary_gpu else False,
            "FP8 Supported": self.primary_gpu.supports_fp8 if self.primary_gpu else False,
            "vLLM Supported": self.primary_gpu.supports_vllm if self.primary_gpu else False,
        }


_NVIDIA_CONFIGURED = False
_CACHED_VALID_PATHS: List[str] = []


class HardwareProfiler:
    """Profiles system hardware and configures shared library environments."""

    @staticmethod
    def configure_nvidia_libraries(force: bool = False) -> List[str]:
        """
        Discovers CUDA/NVIDIA shared library directories and safely adds them
        to LD_LIBRARY_PATH and os.environ.
        Runs ldconfig at most once per process and never writes Google Drive paths to /etc/ld.so.conf.d/.
        """
        global _NVIDIA_CONFIGURED, _CACHED_VALID_PATHS
        if _NVIDIA_CONFIGURED and not force and _CACHED_VALID_PATHS:
            return list(_CACHED_VALID_PATHS)

        candidate_paths = [
            "/usr/lib64-nvidia",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/compat",
            "/usr/local/cuda-13.0/compat",
            "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
            "/usr/local/cuda-12.8/targets/x86_64-linux/lib",
            "/usr/local/cuda-12.6/targets/x86_64-linux/lib",
            "/usr/local/cuda-12.2/targets/x86_64-linux/lib",
            "/usr/local/cuda-12.0/targets/x86_64-linux/lib",
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/opt/bin/.nvidia",
        ]

        valid_paths = [
            p for p in candidate_paths
            if Path(p).is_dir() and not str(p).startswith("/content/drive")
        ]
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        existing_parts = [p for p in current_ld.split(":") if p]

        new_parts = list(existing_parts)
        for p in valid_paths:
            if p not in new_parts:
                new_parts.append(p)

        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts)

        if not _NVIDIA_CONFIGURED or force:
            try:
                if shutil.which("ldconfig") and os.geteuid() == 0:
                    ld_conf = Path("/etc/ld.so.conf.d/nvidia_workspace.conf")
                    ld_conf.write_text("\n".join(valid_paths) + "\n")
                    subprocess.run(["ldconfig"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            except Exception:
                pass
            _NVIDIA_CONFIGURED = True

        _CACHED_VALID_PATHS = valid_paths
        return valid_paths

    @staticmethod
    def detect_cuda_version() -> Tuple[str, Optional[int], Optional[int]]:
        """
        Detects exact CUDA version supporting both CUDA 12.x and CUDA 13.x.
        Returns: (version_str, major, minor)
        """
        if HAS_TORCH and torch.cuda.is_available() and torch.version.cuda:
            v_str = torch.version.cuda
            parts = v_str.split(".")
            try:
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                return v_str, major, minor
            except (ValueError, IndexError):
                pass

        try:
            nvcc_res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
            if nvcc_res.returncode == 0:
                match = re.search(r"release\s+([0-9]+)\.([0-9]+)", nvcc_res.stdout)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    return f"{major}.{minor}", major, minor
        except Exception:
            pass

        try:
            smi_res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            if smi_res.returncode == 0:
                match = re.search(r"CUDA\s+Version:\s*([0-9]+)\.([0-9]+)", smi_res.stdout)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    return f"{major}.{minor}", major, minor
        except Exception:
            pass

        cuda_paths = list(Path("/usr/local").glob("cuda-*.*"))
        if cuda_paths:
            latest = sorted(cuda_paths)[-1].name
            match = re.search(r"cuda-([0-9]+)\.([0-9]+)", latest)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2))
                return f"{major}.{minor}", major, minor

        return "N/A", None, None

    @staticmethod
    def _build_gpu_flags(major: int, minor: int, total_mb: float) -> Dict[str, bool]:
        supports_flash_attn = (major > 7) or (major == 7 and minor >= 5)
        supports_kv_quant = (major > 7) or (major == 7 and minor >= 5)
        supports_vllm = supports_flash_attn and (total_mb >= 12000)
        supports_bf16 = major >= 8
        supports_fp8 = (major > 8) or (major == 8 and minor >= 9)
        return {
            "supports_flash_attn": supports_flash_attn,
            "supports_kv_quant": supports_kv_quant,
            "supports_vllm": supports_vllm,
            "supports_bf16": supports_bf16,
            "supports_fp8": supports_fp8,
        }

    @classmethod
    def detect_gpus(cls) -> List[GPUInfo]:
        """Detects all GPUs with compute capability, memory, temperature, and utilization."""
        gpus: List[GPUInfo] = []

        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for idx in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    raw_name = pynvml.nvmlDeviceGetName(handle)
                    name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mb = mem_info.total / (1024 * 1024)
                    free_mb = mem_info.free / (1024 * 1024)
                    used_mb = mem_info.used / (1024 * 1024)

                    try:
                        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    except Exception:
                        major, minor = (7, 5) if "T4" in name else (8, 0)

                    sm_name = f"sm_{major}{minor}"

                    try:
                        temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        temp = None

                    try:
                        util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    except Exception:
                        util = None

                    flags = cls._build_gpu_flags(major, minor, total_mb)
                    gpus.append(
                        GPUInfo(
                            index=idx,
                            name=name,
                            compute_capability=(major, minor),
                            sm_name=sm_name,
                            total_vram_mb=total_mb,
                            free_vram_mb=free_mb,
                            used_vram_mb=used_mb,
                            temperature_c=temp,
                            utilization_pct=util,
                            **flags,
                        )
                    )
                return gpus
            except Exception:
                pass

        if HAS_TORCH and torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(idx)
                cc = torch.cuda.get_device_capability(idx)
                major, minor = cc
                sm_name = f"sm_{major}{minor}"
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                total_mb = total_bytes / (1024 * 1024)
                free_mb = free_bytes / (1024 * 1024)
                used_mb = total_mb - free_mb

                flags = cls._build_gpu_flags(major, minor, total_mb)
                gpus.append(
                    GPUInfo(
                        index=idx,
                        name=name,
                        compute_capability=cc,
                        sm_name=sm_name,
                        total_vram_mb=total_mb,
                        free_vram_mb=free_mb,
                        used_vram_mb=used_mb,
                        **flags,
                    )
                )
            return gpus

        # Fallback: parse nvidia-smi when neither pynvml nor torch is present
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,compute_cap",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode != 0 or not res.stdout.strip():
                    res = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                if res.returncode == 0 and res.stdout.strip():
                    for line in res.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 5:
                            continue
                        idx = int(parts[0])
                        name = parts[1]
                        total_mb = float(parts[2])
                        free_mb = float(parts[3])
                        used_mb = float(parts[4])
                        temp = float(parts[5]) if len(parts) > 5 and parts[5].replace(".", "", 1).isdigit() else None
                        util = float(parts[6]) if len(parts) > 6 and parts[6].replace(".", "", 1).isdigit() else None
                        cc_str = parts[7] if len(parts) > 7 else ""
                        if "." in cc_str:
                            try:
                                c_parts = cc_str.split(".")
                                major, minor = int(c_parts[0]), int(c_parts[1])
                            except ValueError:
                                major, minor = (7, 5)
                        else:
                            upper_name = name.upper()
                            if "H100" in upper_name or "H200" in upper_name:
                                major, minor = (9, 0)
                            elif "L4" in upper_name or "RTX 40" in upper_name or "ADA" in upper_name:
                                major, minor = (8, 9)
                            elif "A100" in upper_name or "A10" in upper_name or "RTX 30" in upper_name:
                                major, minor = (8, 0)
                            elif "V100" in upper_name:
                                major, minor = (7, 0)
                            elif any(p in upper_name for p in ("GTX 10", "P100", "P40", "P4")):
                                major, minor = (6, 1)
                            else:
                                major, minor = (7, 5)
                        flags = cls._build_gpu_flags(major, minor, total_mb)
                        gpus.append(
                            GPUInfo(
                                index=idx,
                                name=name,
                                compute_capability=(major, minor),
                                sm_name=f"sm_{major}{minor}",
                                total_vram_mb=total_mb,
                                free_vram_mb=free_mb,
                                used_vram_mb=used_mb,
                                temperature_c=temp,
                                utilization_pct=util,
                                **flags,
                            )
                        )
            except Exception:
                pass

        return gpus

    @classmethod
    def get_profile(cls, drive_path: str = "/content/drive/MyDrive", ssd_path: str = "/content") -> HardwareProfile:
        """Collects complete hardware profile and returns HardwareProfile dataclass."""
        lib_paths = cls.configure_nvidia_libraries()
        cuda_ver, cuda_maj, cuda_min = cls.detect_cuda_version()
        gpus = cls.detect_gpus()
        has_gpu = len(gpus) > 0
        primary = gpus[0] if has_gpu else None

        driver_ver = "N/A"
        if HAS_PYNVML:
            try:
                driver_ver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver_ver, bytes):
                    driver_ver = driver_ver.decode("utf-8")
            except Exception:
                pass
        elif shutil.which("nvidia-smi"):
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if smi.returncode == 0 and smi.stdout.strip():
                    driver_ver = smi.stdout.strip().splitlines()[0].strip()
            except Exception:
                pass

        vm = psutil.virtual_memory()
        ram_total = vm.total / (1024**3)
        ram_avail = vm.available / (1024**3)
        ram_used = vm.used / (1024**3)
        ram_pct = vm.percent

        ssd_target = ssd_path if Path(ssd_path).is_dir() else "/"
        ssd_usage = shutil.disk_usage(ssd_target)
        ssd_total = ssd_usage.total / (1024**3)
        ssd_free = ssd_usage.free / (1024**3)
        ssd_used = ssd_usage.used / (1024**3)

        drive_target = drive_path if Path(drive_path).is_dir() else ssd_target
        try:
            drive_usage = shutil.disk_usage(drive_target)
            drive_total = drive_usage.total / (1024**3)
            drive_free = drive_usage.free / (1024**3)
            drive_used = drive_usage.used / (1024**3)
        except Exception:
            drive_total, drive_free, drive_used = ssd_total, ssd_free, ssd_used

        return HardwareProfile(
            has_gpu=has_gpu,
            gpus=gpus,
            primary_gpu=primary,
            cuda_version=cuda_ver,
            cuda_major=cuda_maj,
            cuda_minor=cuda_min,
            driver_version=str(driver_ver),
            cpu_cores=os.cpu_count() or 1,
            ram_total_gb=ram_total,
            ram_available_gb=ram_avail,
            ram_used_gb=ram_used,
            ram_percent=ram_pct,
            ssd_total_gb=ssd_total,
            ssd_free_gb=ssd_free,
            ssd_used_gb=ssd_used,
            drive_total_gb=drive_total,
            drive_free_gb=drive_free,
            drive_used_gb=drive_used,
            nvidia_lib_paths=lib_paths,
        )

    @staticmethod
    def calculate_kv_cache_mb(
        ctx_size: int = 4096,
        kv_cache_quant: str = "q8_0",
        config_json: Optional[Dict[str, Any]] = None,
        estimated_params_b: Optional[float] = None,
        parallel: int = 1,
    ) -> float:
        """
        Estimates KV cache VRAM requirement in MB.
        Uses exact transformer dimensions from config.json when available,
        scaling cleanly with GQA (num_key_value_heads) and parallel slots.
        """
        bytes_per_elem = {
            "f32": 4.0,
            "fp32": 4.0,
            "f16": 2.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "q8_0": 1.06,
            "q4_0": 0.56,
            "q4_1": 0.62,
        }.get((kv_cache_quant or "q8_0").lower(), 1.06)

        slots = max(1, int(parallel or 1))
        cfg = config_json or {}

        # Exact calculation if model architecture dimensions are present
        n_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
        n_heads = cfg.get("num_attention_heads") or cfg.get("n_head")
        hidden_size = cfg.get("hidden_size") or cfg.get("n_embd")
        n_kv_heads = cfg.get("num_key_value_heads") or n_heads
        head_dim = cfg.get("head_dim")

        if n_layers and n_kv_heads and (head_dim or (hidden_size and n_heads)):
            try:
                h_dim = int(head_dim) if head_dim else (int(hidden_size) // int(n_heads))
                kv_bytes = 2.0 * int(n_layers) * int(n_kv_heads) * h_dim * int(ctx_size) * slots * bytes_per_elem
                return max(16.0, kv_bytes / (1024 * 1024))
            except Exception:
                pass

        # Fallback heuristic scaled by estimated parameter count
        base_rate_per_k = {
            "f32": 1000.0,
            "fp32": 1000.0,
            "f16": 500.0,
            "fp16": 500.0,
            "bf16": 500.0,
            "q8_0": 256.0,
            "q4_0": 130.0,
        }.get((kv_cache_quant or "q8_0").lower(), 256.0)

        scale = 1.0
        if estimated_params_b and estimated_params_b > 0:
            # 7B-8B GQA models serve as ~1.0x reference; 0.5B uses ~0.15x; 32B uses ~2.0x
            scale = max(0.15, min(3.5, (estimated_params_b / 7.0) ** 0.7))

        return (ctx_size / 1024.0) * base_rate_per_k * scale * slots

    @classmethod
    def calculate_vram_budget_mb(
        cls,
        free_vram_mb: float,
        cuda_overhead_mb: float = 1200.0,
        ctx_size: int = 4096,
        kv_cache_quant: str = "q8_0",
        config_json: Optional[Dict[str, Any]] = None,
        estimated_params_b: Optional[float] = None,
        parallel: int = 1,
        safety_margin_pct: float = 0.0,
    ) -> float:
        """
        Calculates safe usable VRAM budget for model weights:
        Usable = Free_VRAM * (1 - safety_margin_pct) - CUDA_Runtime_Overhead - Estimated_KV_Cache
        """
        estimated_kv_mb = cls.calculate_kv_cache_mb(
            ctx_size=ctx_size,
            kv_cache_quant=kv_cache_quant,
            config_json=config_json,
            estimated_params_b=estimated_params_b,
            parallel=parallel,
        )
        effective_free = free_vram_mb * max(0.5, 1.0 - max(0.0, safety_margin_pct))
        usable = effective_free - cuda_overhead_mb - estimated_kv_mb
        return max(0.0, usable)


def get_hardware_profile() -> HardwareProfile:
    return HardwareProfiler.get_profile()
