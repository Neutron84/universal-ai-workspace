"""
Environment & Tooling Manager
Manages uv isolated virtualenvs and caching on Drive/SSD, dynamic llama.cpp prebuilt CUDA binaries,
ik_llama.cpp builds, cloudflared tunneling binaries, cache cleanup, and pre-flight diagnostics.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import WorkspaceConfig, get_config
from .hardware import HardwareProfiler, get_hardware_profile


class EnvironmentManager:
    """Manages system dependencies, isolated uv environments, native binaries, and diagnostics."""

    RUNTIME_SERVER_DEPS: List[str] = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
    ]

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
        "gguf": [],
        "ik_llama": [],
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
            "pillow",
        ],
        "audio": [
            "faster-whisper",
            "soundfile",
            "ctranslate2",
        ],
        "unsloth": [
            "unsloth",
            "torch",
            "transformers",
            "accelerate",
            "bitsandbytes",
            "peft",
            "trl",
        ],
        "gptqmodel": [
            "gptqmodel",
            "optimum",
            "accelerate",
        ],
        "autoawq": [
            "autoawq",
            "accelerate",
        ],
        "exllamav2": [
            "exllamav2",
            "torch",
        ],
        "ct2": [
            "ctranslate2",
            "faster-whisper",
        ],
    }

    ENGINE_TO_PROFILE: Dict[str, str] = {
        "llama.cpp": "gguf",
        "llama_cpp": "gguf",
        "ik_llama": "ik_llama",
        "ik_llama.cpp": "ik_llama",
        "transformers": "transformers",
        "vllm": "vllm",
        "unsloth": "unsloth",
        "exllamav2": "exllamav2",
        "exl2": "exllamav2",
        "diffusers": "diffusion",
        "diffusion": "diffusion",
        "faster-whisper": "audio",
        "audio": "audio",
        "ct2": "ct2",
        "gptqmodel": "gptqmodel",
        "autoawq": "autoawq",
    }

    PKG_TO_MODULE: Dict[str, str] = {
        "pyyaml": "yaml",
        "hf-transfer": "hf_transfer",
        "faster-whisper": "faster_whisper",
        "pynvml": "pynvml",
        "pillow": "PIL",
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
        self.config.uv_cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_system_packages(self) -> bool:
        """Installs aria2 and required system utilities via apt if missing."""
        missing = []
        if not shutil.which("aria2c"):
            missing.append("aria2")
        if not shutil.which("fuser"):
            missing.append("psmisc")

        if missing and shutil.which("apt-get") and os.geteuid() == 0:
            print(f"[EnvManager] Installing missing system tools: {missing}...")
            try:
                subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, check=False)
                subprocess.run(
                    ["apt-get", "install", "-y", "-qq"] + missing,
                    stdout=subprocess.DEVNULL,
                    check=False,
                )
            except Exception as e:
                print(f"[EnvManager] Warning: apt install failed ({e})")
        return shutil.which("aria2c") is not None

    def get_profile_python(self, profile_name: str) -> str:
        """
        Returns the isolated virtualenv python executable for a profile if present,
        otherwise falls back to the current Python interpreter.
        """
        canon = self.ENGINE_TO_PROFILE.get(profile_name.lower().strip(), profile_name.lower().strip())
        venv_py = self.config.runtime_envs_dir / canon / "bin" / "python"
        if venv_py.exists() and os.access(str(venv_py), os.X_OK):
            return str(venv_py)
        return sys.executable

    def get_profile_bin(self, profile_name: str, bin_name: str) -> Optional[str]:
        """Finds an executable inside an isolated profile venv or system PATH."""
        canon = self.ENGINE_TO_PROFILE.get(profile_name.lower().strip(), profile_name.lower().strip())
        venv_bin = self.config.runtime_envs_dir / canon / "bin" / bin_name
        if venv_bin.exists() and os.access(str(venv_bin), os.X_OK):
            return str(venv_bin)
        return shutil.which(bin_name)

    def _should_isolate(self, profile_name: str, isolate: Optional[bool] = None) -> bool:
        if isolate is not None:
            return isolate
        mode = (self.config.settings.env_isolation_mode or "auto").lower().strip()
        if mode == "isolated_venv":
            return profile_name != "core"
        if mode == "system":
            return False
        # In "auto" mode: isolate heavy/conflict-prone profiles (vllm, unsloth, exllamav2, gptqmodel, autoawq)
        return profile_name in {"vllm", "unsloth", "exllamav2", "gptqmodel", "autoawq"}

    def fetch_llama_cpp(
        self,
        cuda_major: Optional[int] = None,
        force: bool = False,
        release_tag: Optional[str] = None,
        cuda_target: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Dynamically finds and installs the matching Ubuntu x64 CUDA 12/13
        (or CPU fallback) prebuilt llama.cpp binaries from ggml-org/llama.cpp.
        Caches tarball persistently, extracts to fast local SSD, and symlinks to /usr/local/bin.
        """
        if cuda_major is None:
            cuda_major = self.hw.cuda_major

        target_override = (cuda_target or self.config.settings.cuda_target or "auto").lower().strip()
        desired_tag = (release_tag or self.config.settings.llama_cpp_release_tag or "latest").strip()

        if desired_tag and desired_tag.lower() != "latest":
            api_url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{desired_tag}"
        else:
            api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"

        headers = {"User-Agent": "Mozilla/5.0 (AI-Workspace-Colab)"}
        req = urllib.request.Request(api_url, headers=headers)

        releases_data = []
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                parsed = json.loads(resp.read().decode())
                releases_data = parsed if isinstance(parsed, list) else [parsed]
        except Exception as e:
            print(f"[EnvManager] Note: GitHub Releases API query returned: {e}")

        if not releases_data:
            fallback_tag = desired_tag if (desired_tag and desired_tag.lower() != "latest") else "b12633"
            releases_data = [{"tag_name": fallback_tag, "assets": []}]

        target_release = releases_data[0]
        tag_name = target_release.get("tag_name", "b12633")

        use_cpu = target_override == "cpu" or not (cuda_major and cuda_major >= 12)
        cuda_pattern = "cuda-13" if target_override == "cuda-13" else "cuda-12"

        asset_name = ""
        download_url = ""
        assets = target_release.get("assets", [])

        if assets:
            if not use_cpu:
                match = [
                    a for a in assets
                    if cuda_pattern in a["name"] and "ubuntu" in a["name"] and "x64" in a["name"]
                    and not a["name"].startswith("cudart-")
                ]
                if not match and cuda_pattern == "cuda-13":
                    match = [
                        a for a in assets
                        if "cuda-12" in a["name"] and "ubuntu" in a["name"] and "x64" in a["name"]
                        and not a["name"].startswith("cudart-")
                    ]
                    cuda_pattern = "cuda-12"
                if match:
                    asset_name = match[0]["name"]
                    download_url = match[0]["browser_download_url"]

            if not download_url:
                cpu_match = [
                    a for a in assets
                    if "ubuntu-x64.tar.gz" in a["name"] and "cuda" not in a["name"]
                    and not a["name"].startswith("cudart-")
                ]
                if cpu_match:
                    asset_name = cpu_match[0]["name"]
                    download_url = cpu_match[0]["browser_download_url"]

        if not download_url:
            if not use_cpu:
                cuda_ver_label = "cuda-13.5" if cuda_pattern == "cuda-13" else "cuda-12.8"
                asset_name = f"llama-{tag_name}-bin-ubuntu-{cuda_ver_label}-x64.tar.gz"
            else:
                asset_name = f"llama-{tag_name}-bin-ubuntu-x64.tar.gz"
            download_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag_name}/{asset_name}"

        tar_cache_file = self.config.bin_dir / asset_name
        cuda_tag = "cpu" if use_cpu else cuda_pattern.replace("-", "")
        local_extract_dir = self.config.runtime_root / "bin" / f"llama_cpp_{tag_name}_{cuda_tag}"
        key_binaries = ["llama-server", "llama-quantize", "llama-cli", "llama-bench", "llama-imatrix", "llama-gguf-split"]
        found_binaries: Dict[str, str] = {}

        if not tar_cache_file.exists() or tar_cache_file.stat().st_size == 0 or force:
            print(f"[EnvManager] Downloading llama.cpp ({tag_name}, {cuda_tag}) from {download_url}...")
            self.config.bin_dir.mkdir(parents=True, exist_ok=True)
            if shutil.which("aria2c"):
                subprocess.run(
                    [
                        "aria2c", "-x", "16", "-s", "16", "-k", "1M", "-c",
                        "--dir", str(self.config.bin_dir), "-o", asset_name, download_url,
                    ],
                    check=False,
                )
            if not tar_cache_file.exists() or tar_cache_file.stat().st_size == 0:
                urllib.request.urlretrieve(download_url, str(tar_cache_file))

        tar_cudart_file = None
        if not use_cpu:
            cudart_match = [
                a for a in assets
                if "cudart" in a["name"] and cuda_pattern in a["name"]
                and "ubuntu" in a["name"] and "x64" in a["name"]
            ]
            if cudart_match:
                cudart_asset_name = cudart_match[0]["name"]
                cudart_download_url = cudart_match[0]["browser_download_url"]
            else:
                cuda_ver_label = "cuda-13.5" if cuda_pattern == "cuda-13" else "cuda-12.8"
                cudart_asset_name = f"cudart-llama-{tag_name}-bin-ubuntu-{cuda_ver_label}-x64.tar.gz"
                cudart_download_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag_name}/{cudart_asset_name}"

            tar_cudart_file = self.config.bin_dir / cudart_asset_name
            if not tar_cudart_file.exists() or tar_cudart_file.stat().st_size == 0 or force:
                print(f"[EnvManager] Downloading CUDA runtime libraries ({cudart_asset_name})...")
                self.config.bin_dir.mkdir(parents=True, exist_ok=True)
                if shutil.which("aria2c"):
                    subprocess.run(
                        [
                            "aria2c", "-x", "16", "-s", "16", "-k", "1M", "-c",
                            "--dir", str(self.config.bin_dir), "-o", cudart_asset_name, cudart_download_url,
                        ],
                        check=False,
                    )
                if not tar_cudart_file.exists() or tar_cudart_file.stat().st_size == 0:
                    try:
                        urllib.request.urlretrieve(cudart_download_url, str(tar_cudart_file))
                    except Exception as e:
                        print(f"[EnvManager] Note downloading cudart ({e})")

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
                subprocess.run(
                    ["tar", "-xzf", str(tar_cache_file), "-C", str(local_extract_dir), "--strip-components=1"],
                    check=False,
                )
            except Exception:
                with tarfile.open(str(tar_cache_file), "r:gz") as tar:
                    tar.extractall(path=str(local_extract_dir))

        if tar_cudart_file and tar_cudart_file.exists() and tar_cudart_file.stat().st_size > 0:
            has_cudart_so = any(local_extract_dir.glob("**/libcudart.so*"))
            if needs_extract or not has_cudart_so or force:
                print(f"[EnvManager] Extracting CUDA runtime libraries from {tar_cudart_file.name}...")
                try:
                    subprocess.run(
                        ["tar", "-xzf", str(tar_cudart_file), "-C", str(local_extract_dir), "--strip-components=1"],
                        check=False,
                    )
                except Exception:
                    with tarfile.open(str(tar_cudart_file), "r:gz") as tar:
                        tar.extractall(path=str(local_extract_dir))

        for so_file in list(local_extract_dir.glob("**/*.so*")):
            if so_file.parent != local_extract_dir:
                dest = local_extract_dir / so_file.name
                if not dest.exists():
                    try:
                        dest.symlink_to(so_file)
                    except Exception:
                        shutil.copy2(str(so_file), str(dest))

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
                except Exception:
                    pass

        lib_dirs = {str(p.parent.resolve()) for p in local_extract_dir.glob("**/*.so*")}
        lib_dirs.add(str(local_extract_dir.resolve()))

        try:
            if shutil.which("ldconfig") and os.geteuid() == 0:
                Path("/etc/ld.so.conf.d/llama_cpp.conf").write_text("\n".join(lib_dirs) + "\n")
                subprocess.run(["ldconfig"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current_ld.split(":") if p]
        for ld in lib_dirs:
            if ld not in parts:
                parts.insert(0, ld)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

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
        """Installs cloudflared binary, caching in bin_dir and copying to /usr/local/bin."""
        drive_bin = self.config.bin_dir / "cloudflared"
        usr_bin = Path("/usr/local/bin/cloudflared")

        if not drive_bin.exists() or drive_bin.stat().st_size == 0 or force:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            print(f"[EnvManager] Downloading cloudflared from {url} to {drive_bin}...")
            self.config.bin_dir.mkdir(parents=True, exist_ok=True)
            if shutil.which("aria2c"):
                subprocess.run(
                    ["aria2c", "-x", "8", "-s", "8", "-c", "--dir", str(self.config.bin_dir), "-o", "cloudflared", url],
                    check=False,
                )
            if not drive_bin.exists() or drive_bin.stat().st_size == 0:
                urllib.request.urlretrieve(url, str(drive_bin))

        drive_bin.chmod(0o755)
        target_bin = str(drive_bin)
        try:
            shutil.copy2(str(drive_bin), str(usr_bin))
            usr_bin.chmod(0o755)
            target_bin = str(usr_bin)
        except Exception:
            try:
                if usr_bin.is_symlink() or usr_bin.exists():
                    usr_bin.unlink()
                usr_bin.symlink_to(drive_bin)
                target_bin = str(usr_bin)
            except Exception:
                pass

        try:
            res = subprocess.run([target_bin, "--version"], capture_output=True, text=True, check=False)
            print(f"[EnvManager] cloudflared verified: {res.stdout.strip()}")
        except Exception as e:
            print(f"[EnvManager] cloudflared verification failed: {e}")

        return target_bin

    def fetch_ik_llama_cpp(
        self,
        prebuilt_url: Optional[str] = None,
        force: bool = False,
        cmake_extra_flags: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Locates or builds ikawrakow/ik_llama.cpp binaries with caching and fallback."""
        local_dir = self.config.runtime_root / "bin" / "ik_llama_cpp"
        drive_cache_dir = self.config.bin_dir / "ik_llama_cpp"
        key_bins = ["llama-server", "llama-quantize", "llama-cli", "llama-imatrix", "llama-gguf-split"]
        found: Dict[str, str] = {}

        if not force and local_dir.is_dir():
            if all((local_dir / b).exists() for b in ["llama-server", "llama-quantize"]):
                for b in key_bins:
                    p = local_dir / b
                    if p.exists():
                        found[b] = str(p)
                return found

        if not force and drive_cache_dir.is_dir():
            if all((drive_cache_dir / b).exists() for b in ["llama-server", "llama-quantize"]):
                print(f"[EnvManager] Restoring ik_llama.cpp binaries from cache to {local_dir}...")
                local_dir.mkdir(parents=True, exist_ok=True)
                for b in key_bins:
                    dp = drive_cache_dir / b
                    lp = local_dir / b
                    if dp.exists():
                        shutil.copy2(str(dp), str(lp))
                        lp.chmod(0o755)
                        found[b] = str(lp)
                return found

        local_dir.mkdir(parents=True, exist_ok=True)

        if prebuilt_url:
            print(f"[EnvManager] Downloading prebuilt ik_llama.cpp from {prebuilt_url}...")
            tar_tmp = self.config.runtime_root / "ik_llama_prebuilt.tar.gz"
            try:
                urllib.request.urlretrieve(prebuilt_url, str(tar_tmp))
                subprocess.run(["tar", "-xzf", str(tar_tmp), "-C", str(local_dir)], check=False)
                for b in key_bins:
                    for candidate in local_dir.glob(f"**/{b}"):
                        candidate.chmod(0o755)
                        found[b] = str(candidate)
                if found:
                    return found
            except Exception as e:
                print(f"[EnvManager] Prebuilt download note ({e}), attempting source build...")

        has_cmake = shutil.which("cmake") is not None
        has_git = shutil.which("git") is not None

        if has_cmake and has_git and self.hw.has_gpu:
            print("[EnvManager] Compiling ikawrakow/ik_llama.cpp with CUDA and Flash Attention...")
            src_dir = self.config.runtime_root / "src" / "ik_llama_cpp"
            src_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not (src_dir / ".git").is_dir():
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "https://github.com/ikawrakow/ik_llama.cpp.git", str(src_dir)],
                        check=True,
                    )
                build_dir = src_dir / "build"
                cmake_config_cmd = [
                    "cmake",
                    "-B", str(build_dir),
                    "-DGGML_CUDA=ON",
                    "-DGGML_CUDA_FA_ALL_QUANTS=ON",
                ]
                if cmake_extra_flags:
                    cmake_config_cmd.extend(cmake_extra_flags)
                subprocess.run(cmake_config_cmd, cwd=str(src_dir), check=True)
                cmake_build_cmd = [
                    "cmake",
                    "--build", str(build_dir),
                    "--config", "Release",
                    "-j", str(max(2, os.cpu_count() or 2)),
                    "--target", "llama-server", "llama-quantize", "llama-cli",
                ]
                subprocess.run(cmake_build_cmd, cwd=str(src_dir), check=True)

                for b in key_bins:
                    built_file = build_dir / "bin" / b
                    if not built_file.exists():
                        candidates = list(build_dir.glob(f"**/{b}"))
                        built_file = candidates[0] if candidates else None
                    if built_file and built_file.exists():
                        target = local_dir / b
                        shutil.copy2(str(built_file), str(target))
                        target.chmod(0o755)
                        found[b] = str(target)

                if found and self.config.bin_dir.is_dir():
                    try:
                        drive_cache_dir.mkdir(parents=True, exist_ok=True)
                        for b, p in found.items():
                            shutil.copy2(p, str(drive_cache_dir / b))
                        print(f"[EnvManager] Cached ik_llama binaries to: {drive_cache_dir}")
                    except Exception:
                        pass

                if found:
                    return found
            except Exception as e:
                print(f"[EnvManager] Compilation note ({e}), falling back to standard llama.cpp...")

        print("[EnvManager] Using standard llama.cpp binary suite for ik_llama execution.")
        std_bins = self.fetch_llama_cpp()
        for b in key_bins:
            if b in std_bins:
                found[b] = std_bins[b]
                local_b = local_dir / b
                if not local_b.exists():
                    try:
                        shutil.copy2(std_bins[b], str(local_b))
                        local_b.chmod(0o755)
                    except Exception:
                        pass
        return found

    def install_profile(
        self,
        profile_name: str,
        isolate: Optional[bool] = None,
        system_site_packages: Optional[bool] = None,
    ) -> bool:
        """Installs an environment profile using uv with persistent cache and optional venv isolation."""
        canon = self.ENGINE_TO_PROFILE.get(profile_name.lower().strip(), profile_name.lower().strip())
        if canon not in self.PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(self.PROFILES.keys())}")

        if canon == "gguf":
            self.fetch_llama_cpp()
            return True

        if canon == "ik_llama":
            self.fetch_ik_llama_cpp()
            return True

        packages = list(self.PROFILES[canon])
        if not packages:
            return True

        if canon == "vllm":
            if not self.hw.primary_gpu or not self.hw.primary_gpu.supports_vllm:
                print(
                    f"[EnvManager] Warning: vLLM requires Compute Capability >= 7.5 and >= 12GB VRAM. "
                    f"(Detected GPU: {self.hw.primary_gpu.name if self.hw.primary_gpu else 'None'})"
                )

        use_isolate = self._should_isolate(canon, isolate)
        inherit_sys = (
            self.config.settings.venv_system_site_packages
            if system_site_packages is None
            else system_site_packages
        )

        cmd = [self.uv_bin, "pip", "install", "--cache-dir", str(self.config.uv_cache_dir)]

        if use_isolate:
            venv_dir = self.config.runtime_envs_dir / canon
            if not venv_dir.exists():
                print(f"[EnvManager] Creating isolated virtualenv at {venv_dir} (inherit_system={inherit_sys})...")
                venv_cmd = [self.uv_bin, "venv", str(venv_dir)]
                if inherit_sys and canon != "vllm":
                    venv_cmd.append("--system-site-packages")
                subprocess.run(venv_cmd, check=True)
            python_bin = venv_dir / "bin" / "python"
            cmd.extend(["--python", str(python_bin)])
            # Guarantee server subprocess dependencies inside isolated venv
            for srv_dep in self.RUNTIME_SERVER_DEPS:
                if srv_dep not in packages:
                    packages.append(srv_dep)
        else:
            cmd.extend(["--python", sys.executable])

        cmd.extend(packages)
        print(f"[EnvManager] Installing profile '{canon}' packages (isolate={use_isolate}): {packages}...")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"[EnvManager] Installation output:\n{res.stderr or res.stdout}")
            return False

        print(f"[EnvManager] Profile '{canon}' successfully installed.")
        return True

    def ensure_ready(self, component_name: str, isolate: Optional[bool] = None) -> bool:
        """Ensures required binaries or packages for an engine/converter are ready."""
        canon = self.ENGINE_TO_PROFILE.get(component_name.lower().strip(), component_name.lower().strip())
        if canon not in self.PROFILES:
            return True
        status = self.verify_profile(canon)
        if status and all(status.values()):
            return True
        if not self.config.settings.auto_install_dependencies:
            return False
        return self.install_profile(canon, isolate=isolate)

    def _check_modules_in_python(self, py_exec: str, modules: List[str]) -> Dict[str, bool]:
        """Checks module availability inside an external python executable."""
        script = (
            "import importlib.util, json, sys;"
            "mods = json.loads(sys.argv[1]);"
            "res = {m: (importlib.util.find_spec(m) is not None) for m in mods};"
            "print(json.dumps(res))"
        )
        try:
            proc = subprocess.run(
                [py_exec, "-c", script, json.dumps(modules)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            pass
        return {m: False for m in modules}

    def verify_profile(self, profile_name: str) -> Dict[str, bool]:
        """Verifies whether required packages/binaries of a profile are accessible (host or isolated venv)."""
        results: Dict[str, bool] = {}
        canon = self.ENGINE_TO_PROFILE.get(profile_name.lower().strip(), profile_name.lower().strip())

        if canon == "gguf":
            for b in ["llama-server", "llama-quantize", "llama-cli"]:
                results[b] = shutil.which(b) is not None or Path(f"/usr/local/bin/{b}").exists()
            return results

        if canon == "ik_llama":
            ik_dir = self.config.runtime_root / "bin" / "ik_llama_cpp"
            for b in ["llama-server", "llama-quantize", "llama-cli"]:
                results[b] = (ik_dir / b).exists() or shutil.which(b) is not None
            return results

        packages = self.PROFILES.get(canon, [])
        venv_py = self.config.runtime_envs_dir / canon / "bin" / "python"

        if venv_py.exists():
            mod_names = [self.PKG_TO_MODULE.get(pkg, pkg.replace("-", "_")) for pkg in packages]
            mod_map = self._check_modules_in_python(str(venv_py), mod_names)
            for pkg, mod_name in zip(packages, mod_names):
                results[pkg] = mod_map.get(mod_name, False)
            return results

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
        summary: Dict[str, Dict[str, Any]] = {}
        for prof in self.PROFILES.keys():
            verif = self.verify_profile(prof)
            all_ok = all(verif.values()) if verif else True
            venv_exists = (self.config.runtime_envs_dir / prof / "bin" / "python").exists()
            summary[prof] = {
                "installed": all_ok,
                "isolated": venv_exists,
                "python": str(self.config.runtime_envs_dir / prof / "bin" / "python") if venv_exists else sys.executable,
                "components": verif,
            }
        return summary

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
        return total

    def get_cache_stats(self) -> Dict[str, float]:
        """Returns disk usage in GB across workspace cache and runtime directories."""
        return {
            "uv_cache_gb": round(self._dir_size_bytes(self.config.uv_cache_dir) / (1024**3), 3),
            "bin_cache_gb": round(self._dir_size_bytes(self.config.bin_dir) / (1024**3), 3),
            "envs_gb": round(self._dir_size_bytes(self.config.runtime_envs_dir) / (1024**3), 3),
            "models_cache_gb": round(self._dir_size_bytes(self.config.runtime_models_cache_dir) / (1024**3), 3),
            "outputs_gb": round(self._dir_size_bytes(self.config.runtime_outputs_dir) / (1024**3), 3),
            "logs_mb": round(self._dir_size_bytes(self.config.runtime_logs_dir) / (1024**2), 2),
        }

    def clear_cache(self, scope: str = "temp") -> Dict[str, Any]:
        """Clears targeted cache directories ('temp', 'uv', 'bin', 'envs', 'models_cache', 'all')."""
        freed_bytes = 0
        target_map = {
            "uv": [self.config.uv_cache_dir],
            "bin": [self.config.bin_dir],
            "envs": [self.config.runtime_envs_dir],
            "models_cache": [self.config.runtime_models_cache_dir],
        }

        clean_scope = scope.lower().strip()
        if clean_scope == "temp":
            for pattern in ["*.tmp*", "*.aria2", "*.log"]:
                for p in self.config.runtime_root.glob(f"**/{pattern}"):
                    try:
                        sz = p.stat().st_size
                        p.unlink()
                        freed_bytes += sz
                    except Exception:
                        pass
        elif clean_scope == "all":
            for dirs in target_map.values():
                for d in dirs:
                    if d.exists():
                        freed_bytes += self._dir_size_bytes(d)
                        shutil.rmtree(str(d), ignore_errors=True)
                        d.mkdir(parents=True, exist_ok=True)
        elif clean_scope in target_map:
            for d in target_map[clean_scope]:
                if d.exists():
                    freed_bytes += self._dir_size_bytes(d)
                    shutil.rmtree(str(d), ignore_errors=True)
                    d.mkdir(parents=True, exist_ok=True)

        return {
            "scope": clean_scope,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
            "freed_gb": round(freed_bytes / (1024**3), 3),
        }

    def run_doctor(self) -> Dict[str, Any]:
        """Runs comprehensive environment, hardware, binary, and storage pre-flight checks."""
        hw = get_hardware_profile()
        checks: List[Dict[str, Any]] = []

        # 1. GPU / CUDA check
        if hw.has_gpu and hw.primary_gpu:
            g = hw.primary_gpu
            checks.append({
                "check": "GPU & CUDA",
                "status": "OK",
                "detail": f"{g.name} ({g.sm_name}), CUDA {hw.cuda_version}, {g.free_vram_mb:.0f}/{g.total_vram_mb:.0f} MB free",
            })
        else:
            checks.append({
                "check": "GPU & CUDA",
                "status": "WARN",
                "detail": "No CUDA GPU detected; CPU execution mode active.",
            })

        # 2. System tools check
        for tool in ["uv", "aria2c", "cloudflared", "git", "llama-server", "llama-quantize"]:
            path = shutil.which(tool) or (f"/usr/local/bin/{tool}" if Path(f"/usr/local/bin/{tool}").exists() else None)
            checks.append({
                "check": f"Binary: {tool}",
                "status": "OK" if path else "MISSING",
                "detail": path or "Not installed (auto-fetched on demand)",
            })

        # 3. Disk check
        ssd_warn = hw.ssd_free_gb < self.config.settings.ssd_low_space_warning_gb
        checks.append({
            "check": "Local SSD Space",
            "status": "WARN" if ssd_warn else "OK",
            "detail": f"{hw.ssd_free_gb:.1f} GB free / {hw.ssd_total_gb:.1f} GB total",
        })

        # 4. Shared CUDA runtime libs check for llama.cpp
        llama_bin_dir = self.config.runtime_root / "bin"
        has_cudart = any(llama_bin_dir.glob("**/libcudart.so*")) if llama_bin_dir.exists() else False
        checks.append({
            "check": "llama.cpp CUDA Runtime (libcudart)",
            "status": "OK" if (has_cudart or not hw.has_gpu) else "INFO",
            "detail": "Found bundled libcudart" if has_cudart else "Will download matching cudart on llama.cpp fetch",
        })

        return {
            "hardware": hw.summary(),
            "cache_stats": self.get_cache_stats(),
            "checks": checks,
        }
