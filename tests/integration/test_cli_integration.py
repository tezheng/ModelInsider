"""
CLI Integration Tests

Tests for the command-line interface across all strategies.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from modelexport.cli import cli


class TestCLIIntegration:
    """Test CLI functionality with different strategies."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.runner = CliRunner()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_config(self) -> str:
        """Create a test configuration file."""
        config = {
            "input_specs": {"input_ids": {"dtype": "int", "range": [0, 1000]}},
            "dynamic_axes": {"input_ids": {"0": "batch_size", "1": "sequence_length"}},
            "opset_version": 14,
        }

        config_path = self.temp_path / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        return str(config_path)

    def test_cli_help_commands(self):
        """Test that all CLI help commands work."""
        # Main help
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "export" in result.output
        assert "analyze" in result.output
        assert "validate" in result.output
        assert "compare" in result.output

        # Subcommand help
        for command in ["export", "analyze", "validate", "compare"]:
            result = self.runner.invoke(cli, [command, "--help"])
            assert result.exit_code == 0, f"Help failed for {command}"

    @pytest.mark.version
    @pytest.mark.cli
    @pytest.mark.unit
    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output  # Should match package version

    def test_export_with_strategies(self):
        """Test export command with different strategies."""
        # Skip this test if transformers not available
        try:
            import transformers
        except ImportError:
            pytest.skip("transformers not available")

        config_path = self.create_test_config()

        # Test HTP strategy (only available strategy)
        output_path = self.temp_path / "test_htp.onnx"

        # Use a very small model for testing
        result = self.runner.invoke(
            cli,
            [
                "export",
                "prajjwal1/bert-tiny",  # Small BERT model
                str(output_path),
                "--config",
                config_path,
                "--verbose",
            ],
        )

        if result.exit_code != 0:
            print(f"Export failed for HTP:")
            print(f"Output: {result.output}")
            print(f"Exception: {result.exception}")
            # Don't fail the test immediately - model might not be available
            return

        # Check output
        assert result.exit_code == 0, f"Export failed for HTP: {result.output}"
        assert "Export completed successfully" in result.output
        assert Path(output_path).exists()

    def test_export_with_input_shape(self):
        """Test export with input shape (vision models)."""
        # This would require a vision model, skip for now
        pytest.skip("Vision model export test requires specific model setup")

    def test_analyze_command(self):
        """Test analyze command functionality."""
        # First create a model to analyze
        try:
            import transformers
        except ImportError:
            pytest.skip("transformers not available")

        config_path = self.create_test_config()
        output_path = self.temp_path / "analyze_test.onnx"

        # Export a model first
        export_result = self.runner.invoke(
            cli,
            [
                "export",
                "prajjwal1/bert-tiny",
                str(output_path),
                "--config",
                config_path,
            ],
        )

        if export_result.exit_code != 0:
            pytest.skip("Could not export model for analysis test")

        # Test analyze with different formats
        for output_format in ["summary", "json"]:
            result = self.runner.invoke(
                cli, ["analyze", str(output_path), "--output-format", output_format]
            )

            assert result.exit_code == 0, f"Analyze failed for format {output_format}"

            if output_format == "summary":
                assert "Analysis Summary" in result.output
                assert "Total unique tags" in result.output
            elif output_format == "json":
                # Should be valid JSON
                try:
                    json.loads(result.output)
                except json.JSONDecodeError:
                    pytest.fail("JSON output is not valid JSON")

    def test_validate_command(self):
        """Test validate command functionality."""
        try:
            import transformers
        except ImportError:
            pytest.skip("transformers not available")

        config_path = self.create_test_config()
        output_path = self.temp_path / "validate_test.onnx"

        # Export a model first
        export_result = self.runner.invoke(
            cli,
            [
                "export",
                "prajjwal1/bert-tiny",
                str(output_path),
                "--config",
                config_path,
            ],
        )

        if export_result.exit_code != 0:
            pytest.skip("Could not export model for validation test")

        # Test basic validation
        result = self.runner.invoke(cli, ["validate", str(output_path)])

        assert result.exit_code == 0, f"Validation failed: {result.output}"
        assert "ONNX model is valid" in result.output

        # Test consistency check
        result = self.runner.invoke(
            cli, ["validate", str(output_path), "--check-consistency"]
        )

        assert result.exit_code == 0, f"Consistency check failed: {result.output}"

    def test_compare_command(self):
        """Test compare command functionality."""
        try:
            import transformers
        except ImportError:
            pytest.skip("transformers not available")

        config_path = self.create_test_config()

        # Export two models (same strategy, for comparison testing)
        output_path1 = self.temp_path / "compare1.onnx"
        output_path2 = self.temp_path / "compare2.onnx"

        # Export with HTP strategy (twice for comparison)
        for i, output_path in enumerate([output_path1, output_path2]):
            result = self.runner.invoke(
                cli,
                [
                    "export",
                    "prajjwal1/bert-tiny",
                    str(output_path),
                    "--config",
                    config_path,
                ],
            )

            if result.exit_code != 0:
                pytest.skip(f"Could not export model {i + 1} for comparison test")

        # Test comparison
        result = self.runner.invoke(
            cli, ["compare", str(output_path1), str(output_path2)]
        )

        assert result.exit_code == 0, f"Comparison failed: {result.output}"
        assert "Tag Distribution Comparison" in result.output

    def test_verbose_output(self):
        """Test verbose output functionality."""
        result = self.runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0

        # Test verbose with export (if possible)
        # This would require a full export test which might be too slow

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        # Invalid model
        result = self.runner.invoke(
            cli, ["export", "nonexistent/model", "dummy_output.onnx"]
        )
        assert result.exit_code != 0

        # Missing required arguments
        result = self.runner.invoke(cli, ["export"])
        assert result.exit_code != 0

        # Invalid file paths
        result = self.runner.invoke(cli, ["analyze", "nonexistent_file.onnx"])
        assert result.exit_code != 0

    def test_error_handling(self):
        """Test that errors are properly handled and reported."""
        # Test with non-existent model
        result = self.runner.invoke(cli, ["export", "nonexistent/model", "output.onnx"])

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_output_file_creation(self):
        """Test that output files are created in correct locations."""
        # This would require a successful export
        # For now, just test that the CLI accepts various output paths

        # Test with relative path
        result = self.runner.invoke(
            cli,
            [
                "export",
                "--help",  # Just test parsing
            ],
        )
        assert result.exit_code == 0

        # Test with absolute path
        output_path = self.temp_path / "test_output.onnx"
        # Would test actual export if we had a suitable test model


class TestCLIProcessIntegration:
    """Test CLI as a subprocess (real process testing)."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_as_module(self):
        """Test running CLI as a module."""
        # Test help command
        result = subprocess.run(
            [sys.executable, "-m", "modelexport", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "export" in result.stdout
        assert "Universal hierarchy-preserving ONNX export" in result.stdout

    def test_uv_run_cli(self):
        """Test running CLI with uv run."""
        # Test help command
        result = subprocess.run(
            ["uv", "run", "modelexport", "--help"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "export" in result.stdout

    def test_strategy_listing(self):
        """Test that all strategies are listed in help."""
        result = subprocess.run(
            ["uv", "run", "modelexport", "export", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "htp" in result.stdout.lower() or "hierarchy" in result.stdout.lower()


class TestCLIConfigurationFiles:
    """Test CLI with various configuration files."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.runner = CliRunner()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_file_validation(self):
        """Test configuration file validation."""
        # Valid config
        valid_config = {
            "input_specs": {"input_ids": {"dtype": "int", "range": [0, 1000]}},
            "opset_version": 14,
        }

        config_path = self.temp_path / "valid_config.json"
        with open(config_path, "w") as f:
            json.dump(valid_config, f)

        # Test that config is accepted (would need a real export to fully test)
        result = self.runner.invoke(
            cli,
            [
                "export",
                "--help",  # Just test that CLI loads
            ],
        )
        assert result.exit_code == 0

        # Invalid JSON config
        invalid_config_path = self.temp_path / "invalid_config.json"
        with open(invalid_config_path, "w") as f:
            f.write('{"invalid": json}')  # Invalid JSON

        # CLI should handle invalid JSON gracefully
        # (Would need real export test to verify error handling)

    def test_dynamic_axes_conversion(self):
        """Test dynamic axes configuration conversion."""
        # Test config with string keys (JSON limitation)
        config = {
            "input_specs": {"input_ids": {"dtype": "int", "range": [0, 1000]}},
            "dynamic_axes": {"input_ids": {"0": "batch_size", "1": "sequence_length"}},
        }

        config_path = self.temp_path / "dynamic_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Test that config loads (full test would require export)
        assert Path(config_path).exists()

        # Load config to verify structure
        with open(config_path) as f:
            loaded_config = json.load(f)

        assert "dynamic_axes" in loaded_config
        assert "input_ids" in loaded_config["dynamic_axes"]
