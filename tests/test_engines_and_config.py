"""
Unit tests for Config, ServerPreset CLI coercion, Hardware KV math, and all 8 Engines.
"""

import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from workspace_core.config import ServerPreset, WorkspaceConfig, parse_cli_args
from workspace_core.engines import (
    BaseEngine,
    ExLlamaV2Engine,
    LlamaCppEngine,
    IKLlamaEngine,
    TransformersEngine,
    UnslothEngine,
    VLLMEngine,
    DiffusersEngine,
    AudioEngine,
    get_engine,
    list_engines,
)
from workspace_core.hardware import HardwareProfiler
from workspace_core.registry import ModelEntry


class TestEnginesAndConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_engines_"))
        self.config = WorkspaceConfig(
            workspace_mode="local_ssd",
            workspace_path=str(self.temp_dir / "ws"),
            runtime_root=str(self.temp_dir / "rt"),
        )
        self.dummy_gguf = self.temp_dir / "dummy.gguf"
        self.dummy_gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
        self.dummy_dir = self.temp_dir / "dummy_hf"
        self.dummy_dir.mkdir(parents=True, exist_ok=True)
        (self.dummy_dir / "config.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def test_parse_cli_args_and_preset_coercion(self):
        """ServerPreset must cleanly coerce string custom_args into extra_cli_args."""
        d_args, c_tokens = parse_cli_args("--threads 4 --mlock")
        self.assertEqual(d_args, {})
        self.assertEqual(c_tokens, ["--threads", "4", "--mlock"])

        preset = ServerPreset(name="custom", custom_args="--seed 42 --no-webui")
        self.assertEqual(preset.extra_cli_args, ["--seed", "42", "--no-webui"])
        self.assertEqual(preset.custom_args, {})

    def test_settings_and_presets_crud(self):
        """WorkspaceConfig should persist updated settings and custom presets."""
        self.config.update_settings(selection_strategy="max_quality", aria2_connections=8)
        reloaded = WorkspaceConfig(
            workspace_mode="local_ssd",
            workspace_path=str(self.temp_dir / "ws"),
            runtime_root=str(self.temp_dir / "rt"),
        )
        self.assertEqual(reloaded.settings.selection_strategy, "max_quality")
        self.assertEqual(reloaded.settings.aria2_connections, 8)

        reloaded.upsert_preset("my_test_preset", ServerPreset(name="my_test_preset", ctx_size=8192))
        self.assertIn("my_test_preset", reloaded.presets)
        self.assertTrue(reloaded.delete_preset("my_test_preset"))
        self.assertNotIn("my_test_preset", reloaded.presets)

    def test_exact_kv_cache_math(self):
        """HardwareProfiler should compute architecture-aware KV cache using config.json."""
        cfg_8b_gqa = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        }
        kv_fp16 = HardwareProfiler.calculate_kv_cache_mb(4096, "f16", config_json=cfg_8b_gqa)
        kv_q8 = HardwareProfiler.calculate_kv_cache_mb(4096, "q8_0", config_json=cfg_8b_gqa)
        kv_q4 = HardwareProfiler.calculate_kv_cache_mb(4096, "q4_0", config_json=cfg_8b_gqa)
        # 2 * 32 * 8 * 128 * 4096 * 2 bytes / 1024^2 = 512 MB for fp16
        self.assertAlmostEqual(kv_fp16, 512.0, delta=1.0)
        self.assertLess(kv_q8, kv_fp16)
        self.assertLess(kv_q4, kv_q8)

    def test_engine_registry_and_prepare_overrides_contract(self):
        """All 8 registered engines must instantiate and accept dynamic prepare(**overrides)."""
        expected = ["llama.cpp", "ik_llama", "vllm", "transformers", "unsloth", "exllamav2", "diffusers", "faster-whisper"]
        self.assertEqual(list_engines(), expected)

        # Create dummy binary so LlamaCpp/IKLlama prepare doesn't try downloading from GitHub
        dummy_bin = self.config.runtime_root / "bin" / "ik_llama_cpp" / "llama-server"
        dummy_bin.parent.mkdir(parents=True, exist_ok=True)
        dummy_bin.write_text("#!/bin/sh\necho ok\n")
        dummy_bin.chmod(0o755)

        gguf_entry = ModelEntry(
            model_id="dummy-gguf",
            name="dummy-gguf",
            type="llm-gguf",
            source_url=str(self.dummy_gguf),
            quantization="Q4_K_M",
            total_size_bytes=100,
            local_path=str(self.dummy_gguf),
        )
        st_entry = ModelEntry(
            model_id="dummy-st",
            name="dummy-st",
            type="llm-safetensors",
            source_url=str(self.dummy_dir),
            quantization="unquantized_fp16",
            total_size_bytes=100,
            local_path=str(self.dummy_dir),
        )

        for eng_name in expected:
            engine = get_engine(eng_name, self.config)
            self.assertIsInstance(engine, BaseEngine)
            if eng_name in ("llama.cpp", "ik_llama"):
                if eng_name == "llama.cpp":
                    engine.bin_path = dummy_bin
                ok = engine.prepare(
                    gguf_entry,
                    ctx_size=8192,
                    n_gpu_layers=12,
                    flash_attn=False,
                    override_tensor="ffn_.*_exps.=CPU",
                    custom_server_args="--seed 123",
                )
            else:
                ok = engine.prepare(
                    st_entry,
                    ctx_size=8192,
                    n_gpu_layers=12,
                    flash_attn=True,
                    custom_server_args="--seed 123",
                )
            self.assertTrue(ok)
            cmd = engine.build_command("127.0.0.1", 8010, api_key="test-key")
            self.assertIn("--seed", cmd)
            self.assertIn("123", cmd)
            if eng_name in ("llama.cpp", "ik_llama"):
                self.assertIn("-ot", cmd)
                self.assertIn("ffn_.*_exps.=CPU", cmd)
                self.assertIn("-c", cmd)
                self.assertIn("8192", cmd)
            if eng_name == "vllm":
                self.assertIn("--max-model-len", cmd)
                self.assertIn("8192", cmd)
            if eng_name == "exllamav2":
                self.assertIn("--ctx-size", cmd)
                self.assertIn("8192", cmd)


if __name__ == "__main__":
    unittest.main()
