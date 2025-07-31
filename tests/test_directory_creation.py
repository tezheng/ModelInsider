"""
Test cases for directory creation in HTP exporter.

Tests ensure that the HTP exporter correctly creates output directories
when they don't exist, covering various scenarios.
"""

import os
import tempfile
from pathlib import Path

import pytest

from modelexport.strategies.htp.htp_exporter import HTPExporter


class TestDirectoryCreation:
    """Test suite for directory creation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def simple_model(self):
        """Create a simple PyTorch model for testing."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        return SimpleModel()

    def test_single_level_directory_creation(self, temp_dir, simple_model):
        """Test creating a single-level directory that doesn't exist."""
        # Create path with non-existent directory
        output_path = Path(temp_dir) / "new_dir" / "model.onnx"
        assert not output_path.parent.exists()

        # Export should create the directory
        exporter = HTPExporter()
        input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
        result = exporter.export(
            model=simple_model,
            output_path=str(output_path),
            input_specs=input_specs,
        )

        # Verify directory was created and file exists
        assert output_path.parent.exists()
        assert output_path.exists()
        assert result["export_time"] > 0

    def test_multi_level_directory_creation(self, temp_dir, simple_model):
        """Test creating multiple nested directories that don't exist."""
        # Create path with multiple non-existent directories
        output_path = Path(temp_dir) / "level1" / "level2" / "level3" / "model.onnx"
        assert not output_path.parent.exists()

        # Export should create all necessary directories
        exporter = HTPExporter()
        input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
        result = exporter.export(
            model=simple_model,
            output_path=str(output_path),
            input_specs=input_specs,
        )

        # Verify all directories were created
        assert output_path.parent.exists()
        assert output_path.exists()
        assert (Path(temp_dir) / "level1").exists()
        assert (Path(temp_dir) / "level1" / "level2").exists()
        assert (Path(temp_dir) / "level1" / "level2" / "level3").exists()

    def test_existing_directory_scenario(self, temp_dir, simple_model):
        """Test exporting to an existing directory (should work without issues)."""
        # Create directory first
        output_dir = Path(temp_dir) / "existing_dir"
        output_dir.mkdir()
        output_path = output_dir / "model.onnx"
        
        assert output_dir.exists()
        assert not output_path.exists()

        # Export should work normally
        exporter = HTPExporter()
        input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
        result = exporter.export(
            model=simple_model,
            output_path=str(output_path),
            input_specs=input_specs,
        )

        # Verify file was created
        assert output_path.exists()
        assert result["export_time"] > 0

    def test_directory_creation_with_metadata(self, temp_dir, simple_model):
        """Test that metadata file directory is also created."""
        # Create path with non-existent directory
        output_path = Path(temp_dir) / "metadata_test" / "model.onnx"
        metadata_path = output_path.with_suffix("").with_name(f"{output_path.stem}_htp_metadata.json")
        
        assert not output_path.parent.exists()

        # Export with metadata enabled
        exporter = HTPExporter(enable_reporting=True)
        input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
        result = exporter.export(
            model=simple_model,
            output_path=str(output_path),
            input_specs=input_specs,
        )

        # Verify both files were created
        assert output_path.exists()
        assert metadata_path.exists()

    def test_directory_creation_with_report(self, temp_dir, simple_model):
        """Test that report file directory is also created."""
        # Create path with non-existent directory
        output_path = Path(temp_dir) / "report_test" / "model.onnx"
        report_path = output_path.with_suffix("").with_name(f"{output_path.stem}_htp_export_report.md")
        
        assert not output_path.parent.exists()

        # Export with reporting enabled
        exporter = HTPExporter(verbose=True, enable_reporting=True)
        input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
        result = exporter.export(
            model=simple_model,
            output_path=str(output_path),
            input_specs=input_specs,
        )

        # Verify all files were created
        assert output_path.exists()
        assert report_path.exists()

    def test_permission_error_handling(self, temp_dir, simple_model):
        """Test handling of permission errors when creating directories."""
        if os.name == 'nt':  # Windows
            pytest.skip("Permission testing is platform-specific")

        # Create a read-only directory
        readonly_dir = Path(temp_dir) / "readonly"
        readonly_dir.mkdir()
        os.chmod(readonly_dir, 0o444)  # Read-only

        # Try to create subdirectory (should fail with permission error)
        output_path = readonly_dir / "subdir" / "model.onnx"

        exporter = HTPExporter()
        input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
        with pytest.raises(PermissionError):
            exporter.export(
                model=simple_model,
                output_path=str(output_path),
                input_specs=input_specs,
            )

        # Cleanup: restore permissions
        os.chmod(readonly_dir, 0o755)

    def test_relative_path_directory_creation(self, simple_model):
        """Test creating directories with relative paths."""
        # Use a relative path
        output_path = "temp/relative_test/model.onnx"
        
        # Clean up any existing directory
        import shutil
        if Path("temp/relative_test").exists():
            shutil.rmtree("temp/relative_test")

        # Export should create the directory
        exporter = HTPExporter()
        input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
        result = exporter.export(
            model=simple_model,
            output_path=output_path,
            input_specs=input_specs,
        )

        # Verify directory and file were created
        assert Path(output_path).exists()
        assert result["export_time"] > 0

        # Cleanup
        shutil.rmtree("temp/relative_test")

    def test_special_characters_in_path(self, temp_dir, simple_model):
        """Test creating directories with special characters in the path."""
        # Create path with spaces and special characters
        output_path = Path(temp_dir) / "my models" / "test-model_v1.0" / "model.onnx"
        assert not output_path.parent.exists()

        # Export should handle special characters
        exporter = HTPExporter()
        input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
        result = exporter.export(
            model=simple_model,
            output_path=str(output_path),
            input_specs=input_specs,
        )

        # Verify directory was created correctly
        assert output_path.parent.exists()
        assert output_path.exists()

    def test_cli_integration_directory_creation(self, temp_dir, monkeypatch):
        """Test directory creation through CLI interface."""
        from click.testing import CliRunner

        from modelexport.cli import cli

        # Use a non-existent directory path
        output_path = Path(temp_dir) / "cli_test" / "nested" / "model.onnx"
        assert not output_path.parent.exists()

        runner = CliRunner()
        result = runner.invoke(cli, [
            'export',
            '--model', 'prajjwal1/bert-tiny',
            '--output', str(output_path),
            '--verbose'
        ])

        # Check CLI execution succeeded
        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        
        # Verify directory and file were created
        assert output_path.parent.exists()
        assert output_path.exists()

    @pytest.mark.skip(reason="PyTorch ONNX export is not thread-safe (uses GLOBALS.in_onnx_export)")
    def test_concurrent_directory_creation(self, temp_dir, simple_model):
        """Test that concurrent directory creation doesn't cause race conditions.
        
        Note: This test is skipped because PyTorch's ONNX export uses a global
        state variable (GLOBALS.in_onnx_export) that prevents concurrent exports.
        The directory creation itself works correctly.
        """
        import threading
        
        output_dir = Path(temp_dir) / "concurrent_test"
        results = []
        errors = []

        def export_model(index):
            try:
                output_path = output_dir / f"model_{index}.onnx"
                exporter = HTPExporter()
                input_specs = {"x": {"dtype": "float", "shape": [1, 10]}}
                result = exporter.export(
                    model=simple_model,
                    output_path=str(output_path),
                    input_specs=input_specs,
                )
                results.append((index, result))
            except Exception as e:
                import traceback
                errors.append((index, e, traceback.format_exc()))

        # Create multiple threads trying to create the same directory
        threads = []
        for i in range(5):
            t = threading.Thread(target=export_model, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify no errors occurred
        if errors:
            print("Concurrent test errors:")
            for index, err, tb in errors:
                print(f"Thread {index}: {err}")
                print(tb)
        assert len(errors) == 0, f"Errors occurred in {len(errors)} threads"
        assert len(results) == 5
        
        # Verify all files were created
        for i in range(5):
            assert (output_dir / f"model_{i}.onnx").exists()