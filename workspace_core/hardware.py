"""
Hardware Profiler & Dynamic Environment Diagnostics
Detects GPU, Compute Capability, VRAM, exact CUDA version, CPU, RAM, and Disk space.
Configures dynamic NVIDIA shared library paths safely.
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

    def summary(self) -> Dict[str, Any]:
        gpu_summary = "None (CPU/TPU fallback)"
        if self.primary_gpu:
            g = self.primary_gpu
            gpu_summary = f"{g.name} ({g.sm_name}, {g.free_vram_mb:.0f}MB / {g.total_vram_mb:.0f}MB free)"

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

        # NEVER include Google Drive paths in ld.so.conf or system library search
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

        new_ld = ":".join(new_parts)
        os.environ["LD_LIBRARY_PATH"] = new_ld

        # Attempt to run ldconfig at most once per process
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
        # Method 1: Check torch
        if HAS_TORCH and torch.cuda.is_available() and torch.version.cuda:
            v_str = torch.version.cuda
            parts = v_str.split(".")
            try:
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                return v_str, major, minor
            except (ValueError, IndexError):
                pass

        # Method 2: Check nvcc command
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

        # Method 3: Check nvidia-smi
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

        # Method 4: Check /usr/local/cuda version files or directories
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
    def detect_gpus() -> List[GPUInfo]:
        """
        Detects all GPUs with compute capability, memory, temperature, and utilization.
        """
        gpus: List[GPUInfo] = []

        # Try pynvml first
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

                    # Temperature & utilization
                    try:
                        temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        temp = None

                    try:
                        util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    except Exception:
                        util = None

                    # Features
                    supports_flash_attn = (major > 7) or (major == 7 and minor >= 5)  # Turing sm_75 and above
                    supports_kv_quant = (major > 7) or (major == 7 and minor >= 5)
                    supports_vllm = ((major > 7) or (major == 7 and minor >= 5)) and (total_mb >= 12000)

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
                            supports_flash_attn=supports_flash_attn,
                            supports_kv_quant=supports_kv_quant,
                            supports_vllm=supports_vllm,
                        )
                    )
                return gpus
            except Exception as e:
                pass

        # Fallback to PyTorch
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

                supports_flash_attn = (major > 7) or (major == 7 and minor >= 5)
                supports_kv_quant = (major > 7) or (major == 7 and minor >= 5)
                supports_vllm = ((major > 7) or (major == 7 and minor >= 5)) and (total_mb >= 12000)

                gpus.append(
                    GPUInfo(
                        index=idx,
                        name=name,
                        compute_capability=cc,
                        sm_name=sm_name,
                        total_vram_mb=total_mb,
                        free_vram_mb=free_mb,
                        used_vram_mb=used_mb,
                        supports_flash_attn=supports_flash_attn,
                        supports_kv_quant=supports_kv_quant,
                        supports_vllm=supports_vllm,
                    )
                )

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

        # System RAM
        vm = psutil.virtual_memory()
        ram_total = vm.total / (1024**3)
        ram_avail = vm.available / (1024**3)
        ram_used = vm.used / (1024**3)
        ram_pct = vm.percent

        # SSD Space
        ssd_target = ssd_path if Path(ssd_path).is_dir() else "/"
        ssd_usage = shutil.disk_usage(ssd_target)
        ssd_total = ssd_usage.total / (1024**3)
        ssd_free = ssd_usage.free / (1024**3)
        ssd_used = ssd_usage.used / (1024**3)

        # Drive Space
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
    def calculate_vram_budget_mb(
        free_vram_mb: float,
        cuda_overhead_mb: float = 1200.0,
        ctx_size: int = 4096,
        kv_cache_quant: str = "q8_0"
    ) -> float:
        """
        Calculates safe usable VRAM budget for model weights:
        Usable = Free_VRAM - CUDA_Runtime_Overhead - Estimated_KV_Cache
        """
        # KV cache size estimate: 2 * n_layers * n_heads * head_dim * ctx * bytes_per_elem
        # Approximation for typical 7B-14B models:
        # fp16 (16-bit): ~0.5MB per token context -> ~2048MB for 4k context
        # q8_0 (8-bit): ~0.25MB per token context -> ~1024MB for 4k context
        # q4_0 (4-bit): ~0.13MB per token context -> ~512MB for 4k context
        kv_rate_per_k_ctx = {
            "fp16": 500.0,
            "q8_0": 256.0,
            "q4_0": 130.0,
        }.get(kv_cache_quant.lower(), 256.0)

        estimated_kv_mb = (ctx_size / 1024.0) * kv_rate_per_k_ctx
        usable = free_vram_mb - cuda_overhead_mb - estimated_kv_mb
        return max(0.0, usable)


def get_hardware_profile() -> HardwareProfile:
    return HardwareProfiler.get_profile()
