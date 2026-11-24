"""
Unit tests for WorkspaceOrchestrator, ModelRegistry scanner, ProcessManager persistence,
and EnvironmentManager diagnostics.
"""

import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from workspace_core.config import WorkspaceConfig
from workspace_core.env_manager import EnvironmentManager
from workspace_core.orchestrator import LaunchRequest, WorkspaceOrchestrator
from workspace_core.process_manager import ProcessManager
from workspace_core.registry import ModelRegistry


class TestOrchestratorAndProcessManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_orch_"))
        self.config = WorkspaceConfig(
            workspace_mode="local_ssd",
            workspace_path=str(self.temp_dir / "ws"),
            runtime_root=str(self.temp_dir / "rt"),
        )
        # Create valid minimal GGUF file with header metadata
        self.gguf_path = self.config.models_gguf_dir / "TinyTest-Q4_K_M.gguf"
        self.gguf_path.parent.mkdir(parents=True, exist_ok=True)
        # Magic GGUF, version 3, tensor_count=0, kv_count=1: general.architecture = "llama"
        key = b"general.architecture"
        val = b"llama"
        buf = bytearray()
        buf.extend(b"GGUF")
        buf.extend(struct.pack("<IQQ", 3, 0, 1))
        buf.extend(struct.pack("<Q", len(key)))
        buf.extend(key)
        buf.extend(struct.pack("<I", 8))  # string type
        buf.extend(struct.pack("<Q", len(val)))
        buf.extend(val)
        self.gguf_path.write_bytes(bytes(buf))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def test_orchestrator_inspect_only_local_gguf(self):
        """inspect_only dry-run should inspect local GGUF and return AutoSelector plan without starting server."""
        orch = WorkspaceOrchestrator(self.config)
        req = LaunchRequest(
            input_type="local_path",
            input_path=str(self.gguf_path),
            task="inspect_only",
            strategy="max_speed",
            context_size="auto",
        )
        res = orch.run(req)
        self.assertTrue(res.success)
        self.assertIsNotNone(res.inspection)
        self.assertEqual(res.inspection.architecture_name, "llama")
        self.assertIsNotNone(res.recommendation)
        self.assertIsNone(res.system_state)

    def test_registry_scan_discovers_unregistered_models(self):
        """ModelRegistry.scan_workspace_models should discover and register models on disk."""
        reg = ModelRegistry(self.config)
        self.assertEqual(len(reg.list_models()), 0)
        added = reg.scan_workspace_models()
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].quantization, "Q4_K_M")
        self.assertEqual(added[0].type, "llm-gguf")

    def test_process_manager_dead_pid_cleanup(self):
        """ProcessManager should clear stale state if persisted PID is dead."""
        state_file = self.config.runtime_root / "runtime_state.json"
        state_file.write_text(
            json.dumps({
                "server_pid": 999999,
                "backend_name": "llama.cpp",
                "port": 8000,
                "active_model": "ghost",
                "api_key": "sk-test",
                "tunnel_pid": 999998,
                "tunnel_type": "cloudflared",
                "public_url": "https://dead.trycloudflare.com",
            }),
            encoding="utf-8",
        )
        pm = ProcessManager(self.config)
        st = pm.get_state()
        self.assertFalse(st.is_server_running)
        self.assertFalse(st.is_tunnel_running)
        self.assertIsNone(pm.current_engine)

    def test_snippets_for_all_modalities(self):
        """ProcessManager.generate_code_snippets should adapt to LLM, Diffusers, and Audio."""
        from workspace_core.process_manager import SystemState
        for eng, needle in [
            ("llama.cpp", "/v1"),
            ("diffusers", "/v1/images/generations"),
            ("faster-whisper", "/v1/audio/transcriptions"),
        ]:
            st = SystemState(
                engine_name=eng,
                is_server_running=True,
                server_pid=123,
                port=8000,
                active_model="test-model",
                api_key="sk-abc",
                is_tunnel_running=False,
                tunnel_type="none",
                public_url=None,
                local_endpoints={},
                public_endpoints={},
            )
            py_code, curl_code = ProcessManager.generate_code_snippets(st)
            self.assertIn(needle, py_code)
            self.assertIn(needle, curl_code)

    def test_env_manager_doctor_and_cache_stats(self):
        """EnvironmentManager.run_doctor should return structured diagnostics and cache stats."""
        em = EnvironmentManager(self.config)
        doc = em.run_doctor()
        self.assertIn("hardware", doc)
        self.assertIn("cache_stats", doc)
        self.assertIn("checks", doc)
        self.assertTrue(len(doc["checks"]) >= 4)


if __name__ == "__main__":
    unittest.main()
