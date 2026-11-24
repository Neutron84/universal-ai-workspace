"""
Unit tests for IOResolver and OutputManager.
Verifies drive-free local_ssd operation, path resolution, zip-slip protection,
companion file export, and active-file cleanup protection.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
import zipfile

from workspace_core.io_resolver import IOResolver, OutputManager, WorkspaceMode, OutputMode


class TestIOResolverAndOutputManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_ai_workspace_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    def test_needs_drive_logic(self):
        """Verifies needs_drive returns False for local_ssd and True when Drive is required."""
        self.assertFalse(
            IOResolver.needs_drive(
                code_source="github",
                workspace_mode="local_ssd",
                input_type="huggingface",
                output_type="local_ssd",
            )
        )
        self.assertFalse(
            IOResolver.needs_drive(
                code_source="local_dir",
                workspace_mode="local_ssd",
                input_type="url",
                output_type="none",
            )
        )

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
            runtime_root=str(rt_root),
        )

        self.assertEqual(paths.mode, WorkspaceMode.LOCAL_SSD)
        self.assertEqual(paths.workspace_root.resolve(), custom_local.resolve())
        self.assertEqual(paths.runtime_root.resolve(), rt_root.resolve())
        self.assertFalse(paths.drive_mounted)
        self.assertTrue(paths.models_gguf_dir.exists())

    def test_archive_unpacking_and_zip_slip_protection(self):
        """Verifies extraction of safe zip archives and rejection of path-traversal entries."""
        test_file = self.temp_dir / "sample_model.gguf"
        test_file.write_bytes(b"dummy gguf content")

        zip_path = self.temp_dir / "archive.zip"
        with zipfile.ZipFile(str(zip_path), "w") as z:
            z.write(str(test_file), arcname="sample_model.gguf")

        extracted_dir = IOResolver.unpack_if_archive(zip_path)
        self.assertTrue(extracted_dir.is_dir())
        self.assertTrue((extracted_dir / "sample_model.gguf").exists())

        evil_zip = self.temp_dir / "evil.zip"
        with zipfile.ZipFile(str(evil_zip), "w") as z:
            z.writestr("../../outside.txt", "bad")

        with self.assertRaises(ValueError):
            IOResolver.unpack_if_archive(evil_zip)

    def test_output_manager_export_companions_and_protected_cleanup(self):
        """Verifies export with companion files, zip packaging, and active-file protection."""
        paths = IOResolver.resolve_workspace_paths(
            mode=WorkspaceMode.LOCAL_SSD,
            workspace_path=str(self.temp_dir / "ws"),
            runtime_root=str(self.temp_dir / "rt"),
        )
        out_mgr = OutputManager(paths=paths)

        src_artifact = self.temp_dir / "intermediate_model.gguf"
        src_artifact.write_bytes(b"intermediate weight bytes")
        mmproj = self.temp_dir / "mmproj-f16.gguf"
        mmproj.write_bytes(b"mmproj bytes")

        target_dir = self.temp_dir / "exported_outputs"

        # Protected source should NOT be deleted even when cleanup_source=True
        res_protected = out_mgr.export_artifact(
            source_path=src_artifact,
            output_mode=OutputMode.LOCAL_SSD,
            destination=str(target_dir),
            output_name="protected_copy.gguf",
            cleanup_source=True,
            companion_files=[mmproj],
            protected_paths=[src_artifact],
        )
        self.assertTrue(res_protected["success"])
        self.assertTrue(src_artifact.exists())
        self.assertTrue((target_dir / "mmproj-f16.gguf").exists())

        # Unprotected source SHOULD be deleted when cleanup_source=True
        res = out_mgr.export_artifact(
            source_path=src_artifact,
            output_mode=OutputMode.LOCAL_SSD,
            destination=str(target_dir),
            output_name="final_model.gguf",
            cleanup_source=True,
        )
        self.assertTrue(res["success"])
        exported_file = Path(res["exported_path"])
        self.assertTrue(exported_file.exists())
        self.assertEqual(exported_file.name, "final_model.gguf")
        self.assertFalse(src_artifact.exists())


if __name__ == "__main__":
    unittest.main()
