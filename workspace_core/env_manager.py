"""
Environment & Tooling Manager
Manages uv package caching on Google Drive, dynamic llama.cpp prebuilt CUDA binary fetching,
cloudflared tunneling binaries, and modular environment profiles.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import WorkspaceConfig, get_config
from .hardware import HardwareProfiler, get_hardware_profile


class EnvironmentManager:
    """Manages system dependencies, uv environments, and native binaries."""

    PROFILES: Dict[str, List[str]] = {
        "core": [
            "fastapi",
            "uvicorn",
            "pydantic",
            "pyyaml",
            "huggingface_hub",
            "hf_transfer",
            "pynvml",
            "rich",
            "gradio",
            "psutil",
            "requests",
            "aiohttp",
            "gdown",
        ],
        "gguf": [
            # gguf runtime relies on native llama-server binary
        ],
        "transformers": [
            "transformers",
            "accelerate",
            "bitsandbytes",
            "safetensors",
            "peft",
        ],
        "vllm": [
            "vllm",
        ],
        "diffusion": [
            "diffusers",
            "transformers",
            "accelerate",
            "safetensors",
        ],
        "audio": [
            "faster-whisper",
            "soundfile",
        ],
    }

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or get_config()
        self.hw = get_hardware_profile()
        self.uv_bin = shutil.which("uv") or "/usr/local/bin/uv"
        if not shutil.which("uv") and not Path(self.uv_bin).exists():
            print("[EnvManager] 'uv' not found in system. Auto-installing uv...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=False)
                self.uv_bin = shutil.which("uv") or "/usr/local/bin/uv"
            except Exception as e:
                print(f"[EnvManager] Note: pip install uv returned {e}, attempting standalone install...")
                try:
                    subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=False)
                    self.uv_bin = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
                except Exception:
                    pass
        # Ensure uv cache directory exists
        self.config.uv_cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_system_packages(self) -> bool:
        """Installs aria2 and required system utilities via apt if missing."""
        missing = []
        for pkg in ["aria2", "psmisc", "procps"]:
            if not shutil.which(pkg) and pkg == "aria2":
                missing.append("aria2")
            elif not shutil.which("fuser") and pkg == "psmisc":
                missing.append("psmisc")

        if missing:
            print(f"[EnvManager] Installing missing system tools: {missing}...")
            try:
                subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, check=False)
                subprocess.run(
                    ["apt-get", "install", "-y", "-qq"] + missing,
                    stdout=subprocess.DEVNULL,
                    check=False
                )
            except Exception as e:
                print(f"[EnvManager] Warning: apt install failed ({e})")
        return shutil.which("aria2c") is not None

    def fetch_llama_cpp(self, cuda_major: Optional[int] = None, force: bool = False) -> Dict[str, str]:
        """
        Dynamically finds and installs the matching Ubuntu x64 CUDA 12/13
        (or CPU fallback) prebuilt llama.cpp binaries from ggml-org/llama.cpp.
        Caches tarball on Google Drive, extracts to fast local SSD, and symlinks to /usr/local/bin.
        """
        if cuda_major is None:
            cuda_major = self.hw.cuda_major

        # Query GitHub Releases to find latest tag and matching asset
        api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"
        headers = {"User-Agent": "Mozilla/5.0 (AI-Workspace-Colab)"}
        req = urllib.request.Request(api_url, headers=headers)

        releases_data = []
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                releases_data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"[EnvManager] Note: GitHub Releases API query returned: {e}")

        if not releases_data:
            # Fallback to recent known stable release tag if rate-limited
            releases_data = [{"tag_name": "b12633", "assets": []}]

        target_release = releases_data[0]
        tag_name = target_release.get("tag_name", "b12633")

        # Determine target asset name pattern
        asset_name = ""
        download_url = ""

        # Search assets in release
        assets = target_release.get("assets", [])
        if assets:
            # Note: ggml-org prebuilds include cuda-12.8 and cuda-13.5.
            # Driver 580.x (CUDA 13.0) supports CUDA 12.8 backward-compatibly, whereas CUDA 13.5
            # requires driver >= 590 (otherwise "PTX compiled with unsupported toolchain" error).
            # Thus, for CUDA 12.x and CUDA 13.0, cuda-12.8 provides optimal universal CUDA acceleration.
            if cuda_major and cuda_major >= 12:
                match = [a for a in assets if "cuda-12" in a["name"] and "ubuntu" in a["name"] and "x64" in a["name"] and not a["name"].startswith("cudart-")]
                if match:
                    asset_name = match[0]["name"]
                    download_url = match[0]["browser_download_url"]

            # Fallback to CPU if no GPU match found or CPU requested
            if not download_url:
                cpu_match = [a for a in assets if "ubuntu-x64.tar.gz" in a["name"] and "cuda" not in a["name"] and not a["name"].startswith("cudart-")]
                if cpu_match:
                    asset_name = cpu_match[0]["name"]
                    download_url = cpu_match[0]["browser_download_url"]

        # If download_url wasn't found in API response, construct standard GitHub download URL
        if not download_url:
            if cuda_major and cuda_major >= 12:
                asset_name = f"llama-{tag_name}-bin-ubuntu-cuda-12.8-x64.tar.gz"
            else:
                asset_name = f"llama-{tag_name}-bin-ubuntu-x64.tar.gz"
            download_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag_name}/{asset_name}"

        # Target tarball path on persistent Drive cache
        tar_cache_file = self.config.bin_dir / asset_name
        cuda_tag = "cuda12" if cuda_major and cuda_major >= 12 else "cpu"

        # Local SSD extraction path for fast execution without FUSE overhead
        local_extract_dir = self.config.runtime_root / "bin" / f"llama_cpp_{tag_name}_{cuda_tag}"
        key_binaries = ["llama-server", "llama-quantize", "llama-cli", "llama-bench"]
        found_binaries = {}

        # 1. Ensure main llama.cpp tarball is cached on Drive
        if not tar_cache_file.exists() or tar_cache_file.stat().st_size == 0 or force:
            print(f"[EnvManager] Downloading llama.cpp ({tag_name}, {cuda_tag}) from {download_url} to Drive cache...")
            self.config.bin_dir.mkdir(parents=True, exist_ok=True)
            if shutil.which("aria2c"):
                subprocess.run(
                    [
                        "aria2c",
                        "-x", "16",
                        "-s", "16",
                        "-k", "1M",
                        "-c",
                        "--dir", str(self.config.bin_dir),
                        "-o", asset_name,
                        download_url
                    ],
                    check=False
                )
            if not tar_cache_file.exists() or tar_cache_file.stat().st_size == 0:
                urllib.request.urlretrieve(download_url, str(tar_cache_file))

        # 1b. Ensure matching CUDA runtime (cudart) libraries are downloaded for CUDA 12/13
        tar_cudart_file = None
        if cuda_major and cuda_major >= 12:
            cudart_match = [a for a in assets if "cudart" in a["name"] and "cuda-12" in a["name"] and "ubuntu" in a["name"] and "x64" in a["name"]]
            if cudart_match:
                cudart_asset_name = cudart_match[0]["name"]
                cudart_download_url = cudart_match[0]["browser_download_url"]
            else:
                cudart_asset_name = f"cudart-llama-{tag_name}-bin-ubuntu-cuda-12.8-x64.tar.gz"
                cudart_download_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag_name}/{cudart_asset_name}"

            tar_cudart_file = self.config.bin_dir / cudart_asset_name
            if not tar_cudart_file.exists() or tar_cudart_file.stat().st_size == 0 or force:
                print(f"[EnvManager] Downloading matching CUDA runtime libraries ({cudart_asset_name}) to Drive cache...")
                self.config.bin_dir.mkdir(parents=True, exist_ok=True)
                if shutil.which("aria2c"):
                    subprocess.run(
                        [
                            "aria2c",
                            "-x", "16",
                            "-s", "16",
                            "-k", "1M",
                            "-c",
                            "--dir", str(self.config.bin_dir),
                            "-o", cudart_asset_name,
                            cudart_download_url
                        ],
                        check=False
                    )
                if not tar_cudart_file.exists() or tar_cudart_file.stat().st_size == 0:
                    urllib.request.urlretrieve(cudart_download_url, str(tar_cudart_file))

        # 2. Extract into fast Local SSD runtime
        needs_extract = force or not local_extract_dir.exists()
        if not needs_extract:
            for b in key_binaries:
                if not any(local_extract_dir.glob(f"**/{b}")):
                    needs_extract = True
                    break

        if needs_extract:
            print(f"[EnvManager] Extracting {tar_cache_file.name} to local SSD {local_extract_dir}...")
            local_extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                # Fast extraction via tar
                subprocess.run(
                    ["tar", "-xzf", str(tar_cache_file), "-C", str(local_extract_dir), "--strip-components=1"],
                    check=False
                )
            except Exception:
                with tarfile.open(str(tar_cache_file), "r:gz") as tar:
                    tar.extractall(path=str(local_extract_dir))

        # Extract cudart .so libraries into the same local SSD bin directory
        if tar_cudart_file and tar_cudart_file.exists():
            has_cudart_so = any(local_extract_dir.glob("**/libcudart.so*"))
            if needs_extract or not has_cudart_so or force:
                print(f"[EnvManager] Extracting CUDA runtime libraries from {tar_cudart_file.name} to {local_extract_dir}...")
                try:
                    subprocess.run(
                        ["tar", "-xzf", str(tar_cudart_file), "-C", str(local_extract_dir), "--strip-components=1"],
                        check=False
                    )
                except Exception:
                    with tarfile.open(str(tar_cudart_file), "r:gz") as tar:
                        tar.extractall(path=str(local_extract_dir))

        # Ensure all .so* libraries found in subfolders are linked or present in local_extract_dir top level
        for so_file in list(local_extract_dir.glob("**/*.so*")):
            if so_file.parent != local_extract_dir:
                dest = local_extract_dir / so_file.name
                if not dest.exists():
                    try:
                        dest.symlink_to(so_file)
                    except Exception:
                        shutil.copy2(str(so_file), str(dest))

        # 3. Locate binaries and symlink to /usr/local/bin
        usr_local_bin = Path("/usr/local/bin")
        for b in key_binaries:
            candidates = list(local_extract_dir.glob(f"**/{b}"))
            if candidates:
                b_file = candidates[0]
                b_file.chmod(0o755)
                found_binaries[b] = str(b_file)

                symlink_path = usr_local_bin / b
                try:
                    if symlink_path.is_symlink() or symlink_path.exists():
                        symlink_path.unlink()
                    symlink_path.symlink_to(b_file)
                except Exception as e:
                    print(f"[EnvManager] Warning symlinking {b}: {e}")

        # 4. Configure shared libraries on local SSD
        lib_dirs = set()
        for p in local_extract_dir.glob("**/*.so*"):
            lib_dirs.add(str(p.parent.resolve()))
        lib_dirs.add(str(local_extract_dir.resolve()))

        # Register with ldconfig and LD_LIBRARY_PATH
        ld_conf = Path("/etc/ld.so.conf.d/llama_cpp.conf")
        try:
            ld_conf.write_text("\n".join(lib_dirs) + "\n")
            if shutil.which("ldconfig") and os.geteuid() == 0:
                subprocess.run(["ldconfig"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current_ld.split(":") if p]
        for ld in lib_dirs:
            if ld not in parts:
                parts.insert(0, ld)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

        # 5. Verify llama-server binary
        if "llama-server" in found_binaries:
            try:
                res = subprocess.run([found_binaries["llama-server"], "--version"], capture_output=True, text=True, check=False)
                ver_output = (res.stdout or res.stderr).strip().splitlines()
                first_line = ver_output[0] if ver_output else "llama-server ready"
                print(f"[EnvManager] llama-server verified: {first_line}")
            except Exception as e:
                print(f"[EnvManager] llama-server verification error: {e}")

        return found_binaries

    def fetch_cloudflared(self, force: bool = False) -> str:
        """
        Installs cloudflared binary, caching to Google Drive and symlinking to /usr/local/bin.
        """
        drive_bin = self.config.bin_dir / "cloudflared"
        usr_bin = Path("/usr/local/bin/cloudflared")

        if not drive_bin.exists() or drive_bin.stat().st_size == 0 or force:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            print(f"[EnvManager] Downloading cloudflared from {url} to {drive_bin}...")
            self.config.bin_dir.mkdir(parents=True, exist_ok=True)
            if shutil.which("aria2c"):
                subprocess.run(
                    [
                        "aria2c",
                        "-x", "8",
                        "-s", "8",
                        "-c",
                        "--dir", str(self.config.bin_dir),
                        "-o", "cloudflared",
                        url
                    ],
                    check=False
                )
            if not drive_bin.exists() or drive_bin.stat().st_size == 0:
                urllib.request.urlretrieve(url, str(drive_bin))

        drive_bin.chmod(0o755)

        # Copy to /usr/local/bin for native execution
        try:
            shutil.copy2(str(drive_bin), str(usr_bin))
            usr_bin.chmod(0o755)
        except Exception:
            try:
                if usr_bin.is_symlink() or usr_bin.exists():
                    usr_bin.unlink()
                usr_bin.symlink_to(drive_bin)
            except Exception as e:
                print(f"[EnvManager] Warning linking cloudflared: {e}")

        # Verify
        try:
            res = subprocess.run([str(usr_bin), "--version"], capture_output=True, text=True, check=False)
            print(f"[EnvManager] cloudflared verified: {res.stdout.strip()}")
        except Exception as e:
            print(f"[EnvManager] cloudflared verification failed: {e}")

        return str(usr_bin)

    def install_profile(self, profile_name: str, isolate: bool = False) -> bool:
        """
        Installs an environment profile using uv with persistent cache on Google Drive.
        """
        if profile_name not in self.PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(self.PROFILES.keys())}")

        if profile_name == "gguf":
            self.fetch_llama_cpp()
            return True

        packages = self.PROFILES[profile_name]
        if not packages:
            return True

        # Check vLLM hardware prerequisites
        if profile_name == "vllm":
            if not self.hw.primary_gpu or not self.hw.primary_gpu.supports_vllm:
                print(f"[EnvManager] Warning: vLLM requires Compute Capability >= 7.5 and >= 12GB VRAM. (Detected GPU: {self.hw.primary_gpu.name if self.hw.primary_gpu else 'None'})")

        cmd = [self.uv_bin, "pip", "install", "--cache-dir", str(self.config.uv_cache_dir)]

        if isolate:
            venv_dir = self.config.runtime_envs_dir / profile_name
            if not venv_dir.exists():
                print(f"[EnvManager] Creating isolated virtualenv at {venv_dir}...")
                subprocess.run([self.uv_bin, "venv", str(venv_dir)], check=True)
            python_bin = venv_dir / "bin" / "python"
            cmd.extend(["--python", str(python_bin)])
        else:
            cmd.extend(["--python", sys.executable])

        cmd.extend(packages)
        print(f"[EnvManager] Installing profile '{profile_name}' packages: {packages}...")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"[EnvManager] Installation output:\n{res.stderr or res.stdout}")
            return False

        print(f"[EnvManager] Profile '{profile_name}' successfully installed and cached to Drive.")
        return True

    PKG_TO_MODULE = {
        "pyyaml": "yaml",
        "hf-transfer": "hf_transfer",
        "faster-whisper": "faster_whisper",
        "pynvml": "pynvml",
    }

    def verify_profile(self, profile_name: str) -> Dict[str, bool]:
        """Verifies whether required packages/binaries of a profile are accessible."""
        results = {}
        if profile_name == "gguf":
            for b in ["llama-server", "llama-quantize", "llama-cli"]:
                results[b] = shutil.which(b) is not None
            return results

        import importlib.util
        packages = self.PROFILES.get(profile_name, [])
        for pkg in packages:
            mod_name = self.PKG_TO_MODULE.get(pkg, pkg.replace("-", "_"))
            if mod_name == "pynvml":
                spec = importlib.util.find_spec("pynvml") or importlib.util.find_spec("nvidia_smi")
                results[pkg] = spec is not None
                continue
            spec = importlib.util.find_spec(mod_name)
            results[pkg] = spec is not None
        return results

    def get_all_profiles_status(self) -> Dict[str, Dict[str, Any]]:
        """Returns overall status of all environment profiles."""
        summary = {}
        for prof in self.PROFILES.keys():
            verif = self.verify_profile(prof)
            all_ok = all(verif.values()) if verif else True
            summary[prof] = {
                "installed": all_ok,
                "components": verif
            }
        return summary
