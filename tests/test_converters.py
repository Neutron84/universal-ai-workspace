"""
Unit tests for modular converters and quantizers.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from workspace_core.converters import (
    BaseConverter,
    LlamaCppConverter,
    IKLlamaConverter,
    UnslothConverter,
    GPTQModelConverter,
    AutoAWQConverter,
    ExLlamaConverter,
    CT2AudioConverter,
    get_converter,
    list_converters,
)
from workspace_core.config import WorkspaceConfig


class TestConverters(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_converters_"))
        self.config = WorkspaceConfig(
            workspace_mode="local_ssd",
            workspace_path=str(self.temp_dir / "ws"),
            runtime_root=str(self.temp_dir / "rt")
        )

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def test_list_and_get_converters(self):
        """Verifies that all required converter plugins are registered and instantiable."""
        converters = list_converters()
        required = ["llama.cpp", "ik_llama", "unsloth", "gptqmodel", "autoawq", "exllamav2", "ct2"]
        for req in required:
            self.assertIn(req, converters)

        for name in required:
            conv = get_converter(name, self.config)
            self.assertIsInstance(conv, BaseConverter)
            self.assertTrue(len(conv.supported_inputs) > 0)
            self.assertTrue(len(conv.supported_outputs) > 0)

    def test_converter_estimate_requirements(self):
        """Verifies requirement estimation across converters."""
        class MockInfo:
            total_repo_size_bytes = 4 * 1024 * 1024 * 1024  # 4 GB

        info = MockInfo()
        conv = get_converter("llama.cpp", self.config)
        reqs = conv.estimate_requirements(info)
        self.assertIn("estimated_ram_gb", reqs)
        self.assertIn("estimated_disk_gb", reqs)
        self.assertGreater(reqs["estimated_ram_gb"], 0)


if __name__ == "__main__":
    unittest.main()
