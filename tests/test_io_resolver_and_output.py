"""
Unit tests for IOResolver and OutputManager.
Verifies drive-free local_ssd operation, path resolution, and artifact export sinks.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
import zipfile
import tarfile

from workspace_core.io_resolver import IOResolver, OutputManager, WorkspaceMode, InputMode, OutputMode


class TestIOResolverAndOutputManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_ai_workspace_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def test_needs_drive_logic(self):
        """Verifies needs_drive returns False for local_ssd and True when Drive is required."""
        # 100% Drive-free local execution
        self.assertFalse(
            IOResolver.needs_drive(
                code_source="github",
                workspace_mode="local_ssd",
                input_type="huggingface",
                output_type="local_ssd"
            )
        )
        self.assertFalse(
            IOResolver.needs_drive(
                code_source="local_dir",
                workspace_mode="local_ssd",
                input_type="url",
                output_type="none"
            )
        )

        # Drive required modes
        self.assertTrue(IOResolver.needs_drive(workspace_mode="hybrid"))
        self.assertTrue(IOResolver.needs_drive(workspace_mode="google_drive"))
        self.assertTrue(IOResolver.needs_drive(code_source="gdrive_relative"))
        self.assertTrue(IOResolver.needs_drive(input_type="google_drive"))
        self.assertTrue(IOResolver.needs_drive(input_type="gdrive_relative"))
        self.assertTrue(IOResolver.needs_drive(output_type="gdrive_direct"))
        self.assertTrue(IOResolver.needs_drive(output_type="google_drive"))

    def test_local_ssd_paths_resolution(self):
        """Verifies local_ssd resolves paths without mounting Google Drive."""
        custom_local = self.temp_dir / "my_local_workspace"
        rt_root = self.temp_dir / "my_runtime"
        paths = IOResolver.resolve_workspace_paths(
            mode=WorkspaceMode.LOCAL_SSD,
            workspace_path=str(custom_local),
            runtime_root=str(rt_root)
        )

        self.assertEqual(paths.mode, WorkspaceMode.LOCAL_SSD)
        self.assertEqual(paths.workspace_root.resolve(), custom_local.resolve())
        self.assertEqual(paths.runtime_root.resolve(), rt_root.resolve())
        self.assertFalse(paths.drive_mounted)
        self.assertTrue(paths.models_gguf_dir.exists())

    def test_archive_unpacking(self):
        """Verifies automatic extraction of zip and tar archives."""
        test_file = self.temp_dir / "sample_model.gguf"
        test_file.write_bytes(b"dummy gguf content")

        # Create zip archive
        zip_path = self.temp_dir / "archive.zip"
        with zipfile.ZipFile(str(zip_path), "w") as z:
            z.write(str(test_file), arcname="sample_model.gguf")

        extracted_dir = IOResolver.unpack_if_archive(zip_path)
        self.assertTrue(extracted_dir.is_dir())
        self.assertTrue((extracted_dir / "sample_model.gguf").exists())

    def test_output_manager_export_and_cleanup(self):
        """Verifies export to local_ssd, zip packaging, and cleanup_source functionality."""
        paths = IOResolver.resolve_workspace_paths(
            mode=WorkspaceMode.LOCAL_SSD,
            workspace_path=str(self.temp_dir / "ws"),
            runtime_root=str(self.temp_dir / "rt")
        )
        out_mgr = OutputManager(paths=paths)

        # Create source artifact
        src_artifact = self.temp_dir / "intermediate_model.gguf"
        src_artifact.write_bytes(b"intermediate weight bytes")

        target_dir = self.temp_dir / "exported_outputs"

        # Export with cleanup_source=True
        res = out_mgr.export_artifact(
            source_path=src_artifact,
            output_mode=OutputMode.LOCAL_SSD,
            destination=str(target_dir),
            output_name="final_model.gguf",
            cleanup_source=True
        )

        self.assertTrue(res["success"])
        exported_file = Path(res["exported_path"])
        self.assertTrue(exported_file.exists())
        self.assertEqual(exported_file.name, "final_model.gguf")

        # Source file should be deleted to save disk space
        self.assertFalse(src_artifact.exists())


if __name__ == "__main__":
    unittest.main()
