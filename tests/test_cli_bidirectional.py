"""
Test suite for bidirectional conversion CLI commands.

Tests the CLI interface for:
- export-graphml: ONNX → GraphML
- import-onnx: GraphML → ONNX
- validate-roundtrip: Complete validation
"""

import json
from pathlib import Path

import onnx
import pytest
from click.testing import CliRunner

from modelexport.cli import cli
from modelexport.graphml.constants import GRAPHML_FORMAT_VERSION as GRAPHML_VERSION


@pytest.fixture
def bert_tiny_artifacts(tmp_path):
    """Create BERT-tiny-like artifacts for testing."""
    # Create sample ONNX model (simplified BERT structure)
    from onnx import TensorProto, helper

    # Model inputs
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [2, 16])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, [2, 16]
    )
    token_type_ids = helper.make_tensor_value_info(
        "token_type_ids", TensorProto.INT64, [2, 16]
    )

    # Model outputs
    last_hidden_state = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, [2, 16, 128]
    )
    pooler_output = helper.make_tensor_value_info(
        "pooler_output", TensorProto.FLOAT, [2, 128]
    )

    # Create embedding weight
    embedding_weight = helper.make_tensor(
        name="embeddings.word_embeddings.weight",
        data_type=TensorProto.FLOAT,
        dims=[30522, 128],  # vocab_size, hidden_size
        vals=[0.0] * (30522 * 128),  # Zeros for simplicity
    )

    # Create a simple operation (Gather for word embeddings)
    gather_node = helper.make_node(
        "Gather",
        inputs=["embeddings.word_embeddings.weight", "input_ids"],
        outputs=["word_embeddings"],
        name="embeddings/word_embeddings/Gather",
    )

    # Create constant for adding
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        name="embeddings/Constant",
        value=helper.make_tensor("const_value", TensorProto.FLOAT, [1], [1.0]),
    )

    # Create add operation
    add_node = helper.make_node(
        "Add",
        inputs=["word_embeddings", "const_output"],
        outputs=["last_hidden_state"],
        name="embeddings/Add",
    )

    # Create simple pooler (just take first token)
    gather_pooler = helper.make_node(
        "Gather",
        inputs=["last_hidden_state", "token_type_ids"],  # Simplified
        outputs=["pooler_output"],
        name="pooler/Gather",
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[constant_node, gather_node, add_node, gather_pooler],
        name="BertModel",
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=[last_hidden_state, pooler_output],
        initializer=[embedding_weight],
    )

    # Create model
    model = helper.make_model(graph)
    model.opset_import[0].version = 17
    model.producer_name = "pytorch"
    model.producer_version = "2.7.1"
    model.model_version = 1

    # Save model
    model_path = tmp_path / "bert_tiny.onnx"
    onnx.save(model, str(model_path))

    # Create HTP metadata
    metadata = {
        "export_context": {
            "timestamp": "2025-01-01T00:00:00.000Z",
            "strategy": "htp",
            "version": "1.0",
            "exporter": "HTPExporter",
        },
        "model": {
            "name_or_path": "prajjwal1/bert-tiny",
            "class_name": "BertModel",
            "framework": "transformers",
            "total_modules": 48,
            "total_parameters": 4385920,
        },
        "tracing": {
            "builder": "TracingHierarchyBuilder",
            "modules_traced": 18,
            "execution_steps": 36,
        },
        "modules": {
            "class_name": "BertModel",
            "traced_tag": "/BertModel",
            "scope": "",
            "execution_order": 0,
            "children": {
                "BertEmbeddings": {
                    "class_name": "BertEmbeddings",
                    "traced_tag": "/BertModel/BertEmbeddings",
                    "scope": "embeddings",
                    "execution_order": 1,
                },
                "BertPooler": {
                    "class_name": "BertPooler",
                    "traced_tag": "/BertModel/BertPooler",
                    "scope": "pooler",
                    "execution_order": 2,
                },
            },
        },
        "nodes": {
            "embeddings/word_embeddings/Gather": "/BertModel/BertEmbeddings",
            "embeddings/Constant": "/BertModel/BertEmbeddings",
            "embeddings/Add": "/BertModel/BertEmbeddings",
            "pooler/Gather": "/BertModel/BertPooler",
        },
    }

    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path, metadata_path


class TestExportGraphMLV2CLI:
    """Test export-graphml CLI command."""

    def test_export_graphml_v2_basic(self, bert_tiny_artifacts, tmp_path):
        """Test basic GraphML v1.3 export."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()
        output_base = str(tmp_path / "export_basic")

        result = runner.invoke(
            cli,
            [
                "export-graphml",
                str(model_path),
                str(metadata_path),
                "--output",
                output_base,
                "--strategy",
                "sidecar",
            ],
        )

        assert result.exit_code == 0
        assert (
            f"GraphML v{GRAPHML_VERSION} export completed successfully" in result.output
        )
        assert f"Format Version: {GRAPHML_VERSION}" in result.output

        # Verify files created
        graphml_file = Path(f"{output_base}.graphml")
        param_file = Path(f"{output_base}.onnxdata")

        assert graphml_file.exists()
        assert param_file.exists()
        assert graphml_file.stat().st_size > 0
        assert param_file.stat().st_size > 0

    def test_export_graphml_v2_verbose(self, bert_tiny_artifacts, tmp_path):
        """Test verbose output for GraphML v1.3 export."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()
        output_base = str(tmp_path / "export_verbose")

        result = runner.invoke(
            cli,
            [
                "export-graphml",
                str(model_path),
                str(metadata_path),
                "--output",
                output_base,
                "--strategy",
                "sidecar",
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (
            f"Converting ONNX to GraphML v{GRAPHML_VERSION} (Schema-Driven Model Interchange Format)"
            in result.output
        )
        assert f"GraphML v{GRAPHML_VERSION} Features:" in result.output
        assert "Complete ONNX node attributes" in result.output
        assert "Bidirectional conversion ready" in result.output

    def test_export_graphml_v2_embedded_strategy(self, bert_tiny_artifacts, tmp_path):
        """Test GraphML v1.3 export with embedded strategy."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()
        output_base = str(tmp_path / "export_embedded")

        result = runner.invoke(
            cli,
            [
                "export-graphml",
                str(model_path),
                str(metadata_path),
                "--output",
                output_base,
                "--strategy",
                "embedded",
            ],
        )

        assert result.exit_code == 0

        # With embedded strategy, only GraphML file should exist
        graphml_file = Path(f"{output_base}.graphml")
        param_file = Path(f"{output_base}.onnxdata")

        assert graphml_file.exists()
        assert not param_file.exists()  # No separate parameter file

    def test_export_graphml_v2_default_output(self, bert_tiny_artifacts):
        """Test default output naming."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Copy files to current directory
            import shutil

            shutil.copy(str(model_path), "model.onnx")
            shutil.copy(str(metadata_path), "metadata.json")

            result = runner.invoke(
                cli,
                [
                    "export-graphml",
                    "model.onnx",
                    "metadata.json",
                    "--strategy",
                    "sidecar",
                ],
            )

            assert result.exit_code == 0
            assert Path("model.graphml").exists()
            assert Path("model.onnxdata").exists()

    def test_export_graphml_v2_missing_files(self):
        """Test error handling for missing files."""
        runner = CliRunner()

        # Missing ONNX file
        result = runner.invoke(cli, ["export-graphml", "missing.onnx", "missing.json"])
        assert result.exit_code != 0

        # Should show usage or error message
        assert "Error" in result.output or "Usage:" in result.output


class TestImportONNXCLI:
    """Test import-onnx CLI command."""

    def test_import_onnx_basic(self, bert_tiny_artifacts, tmp_path):
        """Test basic ONNX import from GraphML v1.3."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()

        # First export to GraphML v1.3
        export_base = str(tmp_path / "for_import")
        export_result = runner.invoke(
            cli,
            [
                "export-graphml",
                str(model_path),
                str(metadata_path),
                "--output",
                export_base,
                "--strategy",
                "sidecar",
            ],
        )
        assert export_result.exit_code == 0

        # Then import back to ONNX
        import_path = str(tmp_path / "imported.onnx")
        import_result = runner.invoke(
            cli, ["import-onnx", f"{export_base}.graphml", import_path]
        )

        assert import_result.exit_code == 0
        assert "ONNX reconstruction completed successfully" in import_result.output
        assert Path(import_path).exists()

        # Verify file size is reasonable
        imported_size = Path(import_path).stat().st_size
        original_size = Path(model_path).stat().st_size

        # Should be within 20% of original size
        size_ratio = imported_size / original_size
        assert 0.8 <= size_ratio <= 1.2

    def test_import_onnx_verbose(self, bert_tiny_artifacts, tmp_path):
        """Test verbose output for ONNX import."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()

        # Export first
        export_base = str(tmp_path / "for_verbose_import")
        runner.invoke(
            cli,
            [
                "export-graphml",
                str(model_path),
                str(metadata_path),
                "--output",
                export_base,
                "--strategy",
                "sidecar",
            ],
        )

        # Import with verbose
        import_path = str(tmp_path / "verbose_imported.onnx")
        result = runner.invoke(
            cli, ["import-onnx", f"{export_base}.graphml", import_path, "--verbose"]
        )

        assert result.exit_code == 0
        assert f"Converting GraphML v{GRAPHML_VERSION} to ONNX" in result.output
        assert "GraphML Analysis:" in result.output
        assert f"Format Version: {GRAPHML_VERSION}" in result.output
        assert "Parameter Strategy:" in result.output

    def test_import_onnx_with_validation_success(self, bert_tiny_artifacts, tmp_path):
        """Test import with validation (should now succeed after fixes)."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()

        # Export first
        export_base = str(tmp_path / "for_validation")
        runner.invoke(
            cli,
            [
                "export-graphml",
                str(model_path),
                str(metadata_path),
                "--output",
                export_base,
                "--strategy",
                "sidecar",
            ],
        )

        # Import with validation (should now succeed after tensor fixes)
        import_path = str(tmp_path / "validated_import.onnx")
        result = runner.invoke(
            cli, ["import-onnx", f"{export_base}.graphml", import_path, "--validate"]
        )

        # Should succeed now that tensor attribute reconstruction is fixed
        assert result.exit_code == 0
        assert "reconstruction completed successfully" in result.output

    def test_import_onnx_missing_graphml(self):
        """Test error handling for missing GraphML file."""
        runner = CliRunner()

        result = runner.invoke(cli, ["import-onnx", "missing.graphml", "output.onnx"])

        assert result.exit_code != 0


@pytest.mark.skip(reason="validate-roundtrip command was removed from CLI")
class TestValidateRoundtripCLI:
    """Test validate-roundtrip CLI command."""

    def test_validate_roundtrip_help(self):
        """Test validate-roundtrip help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate-roundtrip", "--help"])

        assert result.exit_code == 0
        assert "Validate bidirectional ONNX" in result.output
        assert "ONNX_PATH" in result.output
        assert "HTP_METADATA" in result.output
        assert "--strategy" in result.output
        assert "--tolerance" in result.output

    def test_validate_roundtrip_basic(self, bert_tiny_artifacts, tmp_path):
        """Test basic round-trip validation (should now succeed after fixes)."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate-roundtrip",
                str(model_path),
                str(metadata_path),
                "--strategy",
                "sidecar",
                "--temp-dir",
                str(tmp_path),
            ],
        )

        # Should now succeed after tensor attribute and topological sorting fixes
        # May have warnings but should complete validation framework
        assert result.exit_code == 0 or "Round-trip validation" in result.output
        assert "Round-trip validation" in result.output

    def test_validate_roundtrip_verbose(self, bert_tiny_artifacts, tmp_path):
        """Test verbose round-trip validation."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate-roundtrip",
                str(model_path),
                str(metadata_path),
                "--strategy",
                "sidecar",
                "--tolerance",
                "1e-5",
                "--verbose",
            ],
        )

        assert "Starting Round-Trip Validation" in result.output
        assert "Strategy: sidecar" in result.output
        assert "Tolerance: 1.00e-05" in result.output

    def test_validate_roundtrip_custom_tolerance(self, bert_tiny_artifacts, tmp_path):
        """Test round-trip validation with custom tolerance."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "validate-roundtrip",
                str(model_path),
                str(metadata_path),
                "--tolerance",
                "1e-4",
            ],
        )

        assert "Tolerance: 1.00e-04" in result.output

    def test_validate_roundtrip_missing_files(self):
        """Test error handling for missing files."""
        runner = CliRunner()

        result = runner.invoke(
            cli, ["validate-roundtrip", "missing.onnx", "missing.json"]
        )

        assert result.exit_code != 0


class TestCLIIntegration:
    """Test complete CLI workflow integration."""

    def test_complete_workflow_without_validation(self, bert_tiny_artifacts, tmp_path):
        """Test complete export → import workflow without validation."""
        model_path, metadata_path = bert_tiny_artifacts

        runner = CliRunner()

        # Step 1: Export to GraphML v1.3
        export_base = str(tmp_path / "workflow_export")
        export_result = runner.invoke(
            cli,
            [
                "export-graphml",
                str(model_path),
                str(metadata_path),
                "--output",
                export_base,
                "--strategy",
                "sidecar",
                "--verbose",
            ],
        )

        assert export_result.exit_code == 0
        assert f"Format Version: {GRAPHML_VERSION}" in export_result.output

        # Step 2: Import back to ONNX
        import_path = str(tmp_path / "workflow_import.onnx")
        import_result = runner.invoke(
            cli, ["import-onnx", f"{export_base}.graphml", import_path, "--verbose"]
        )

        assert import_result.exit_code == 0
        assert "reconstruction completed successfully" in import_result.output

        # Step 3: Verify files exist and have reasonable sizes
        assert Path(f"{export_base}.graphml").exists()
        assert Path(f"{export_base}.onnxdata").exists()
        assert Path(import_path).exists()

        # Check size preservation
        original_size = Path(model_path).stat().st_size
        reconstructed_size = Path(import_path).stat().st_size

        size_ratio = reconstructed_size / original_size
        assert 0.7 <= size_ratio <= 1.3  # Allow 30% variation

    def test_cli_commands_exist(self):
        """Test that all bidirectional CLI commands are available."""
        runner = CliRunner()

        # Test main help shows new commands
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "export-graphml" in result.output
        assert "import-onnx" in result.output
        # validate-roundtrip was removed from CLI

    def test_error_consistency(self):
        """Test that error messages are consistent across commands."""
        runner = CliRunner()

        # Test consistent error handling for missing files
        commands_to_test = [
            ["export-graphml", "missing1.onnx", "missing2.json"],
            ["import-onnx", "missing.graphml", "output.onnx"],
            ["validate-roundtrip", "missing1.onnx", "missing2.json"],
        ]

        for cmd in commands_to_test:
            result = runner.invoke(cli, cmd)
            assert result.exit_code != 0
            # All should show some form of error or usage message
            assert len(result.output) > 0


class TestParameterStrategiesCLI:
    """Test different parameter strategies via CLI."""

    def test_all_parameter_strategies(self, bert_tiny_artifacts, tmp_path):
        """Test all parameter storage strategies via CLI."""
        model_path, metadata_path = bert_tiny_artifacts
        runner = CliRunner()

        strategies = ["sidecar", "embedded"]  # 'reference' may not be fully implemented

        for strategy in strategies:
            output_base = str(tmp_path / f"strategy_{strategy}")

            result = runner.invoke(
                cli,
                [
                    "export-graphml",
                    str(model_path),
                    str(metadata_path),
                    "--output",
                    output_base,
                    "--strategy",
                    strategy,
                ],
            )

            if result.exit_code != 0:
                print(f"Error with {strategy} strategy:")
                print(result.output)
            assert result.exit_code == 0
            assert (
                f"Parameter Strategy: {strategy}" in result.output
                or "export completed successfully" in result.output
            )

            # Verify appropriate files are created
            graphml_file = Path(f"{output_base}.graphml")
            param_file = Path(f"{output_base}.onnxdata")

            assert graphml_file.exists()

            if strategy == "sidecar":
                assert param_file.exists()
            elif strategy == "embedded":
                # May or may not create separate parameter file depending on implementation
                pass

    def test_invalid_strategy_error(self, bert_tiny_artifacts):
        """Test error handling for invalid parameter strategy."""
        model_path, metadata_path = bert_tiny_artifacts
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "export-graphml",
                str(model_path),
                str(metadata_path),
                "--strategy",
                "invalid_strategy",
            ],
        )

        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()
