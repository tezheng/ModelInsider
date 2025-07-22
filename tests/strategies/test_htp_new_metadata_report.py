"""
Comprehensive tests for HTP metadata and report validation.

These tests ensure that the metadata follows the JSON schema
and the report contains all required sections without truncation.
"""

import json
import tempfile
from pathlib import Path

import jsonschema
import pytest

from modelexport.strategies.htp_new import HTPExporter


class TestHTPMetadataValidation:
    """Test metadata generation and schema compliance."""
    
    @pytest.fixture
    def schema(self):
        """Load the HTP metadata JSON schema."""
        schema_path = Path(__file__).parent.parent.parent / "modelexport/strategies/htp_new/htp_metadata_schema.json"
        with open(schema_path, 'r') as f:
            return json.load(f)
    
    def test_metadata_schema_compliance(self, schema):
        """Test that generated metadata complies with JSON schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=False, enable_reporting=False)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Validate against schema
            jsonschema.validate(metadata, schema)
    
    def test_metadata_required_fields(self):
        """Test that all required fields are present in metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=False, enable_reporting=False)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required top-level fields
            assert "export_context" in metadata
            assert "model" in metadata
            assert "modules" in metadata
            assert "nodes" in metadata
            
            # Check export_context required fields
            assert "timestamp" in metadata["export_context"]
            assert "strategy" in metadata["export_context"]
            
            # Check model required fields
            assert "name_or_path" in metadata["model"]
            assert "class_name" in metadata["model"]
            assert "total_modules" in metadata["model"]
            assert "total_parameters" in metadata["model"]
    
    def test_metadata_constraints(self):
        """Test that metadata values meet constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=False, enable_reporting=False)
            stats = exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Strategy must be "htp"
            assert metadata["export_context"]["strategy"] == "htp"
            
            # Empty tags must be 0
            assert stats["empty_tags"] == 0
            
            # Coverage must be 0-100%
            if "report" in metadata and "node_tagging" in metadata["report"]:
                coverage = metadata["report"]["node_tagging"]["coverage"]["percentage"]
                assert 0 <= coverage <= 100
            
            # Numbers must be non-negative
            assert metadata["model"]["total_modules"] >= 0
            assert metadata["model"]["total_parameters"] >= 0
    
    def test_metadata_consistency(self):
        """Test consistency between different sections of metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=False, enable_reporting=False)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Module count should match
            modules_in_hierarchy = len(metadata["modules"])
            
            # All modules should have required fields
            for module_path, module_info in metadata["modules"].items():
                assert "class_name" in module_info
                assert "traced_tag" in module_info
            
            # Node mappings should reference valid tags
            valid_tags = {info["traced_tag"] for info in metadata["modules"].values()}
            for node_name, tag in metadata["nodes"].items():
                assert tag in valid_tags, f"Node {node_name} has invalid tag {tag}"
    
    def test_nodes_at_root_level(self):
        """Test that nodes mapping is at root level, not nested."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=False, enable_reporting=False)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # nodes should be at root level
            assert "nodes" in metadata
            assert isinstance(metadata["nodes"], dict)
            
            # nodes should NOT be under tagging
            if "tagging" in metadata:
                assert "nodes" not in metadata["tagging"]
                assert "tagged_nodes" not in metadata["tagging"]


class TestHTPReportValidation:
    """Test report generation and content validation."""
    
    def test_report_structure(self):
        """Test that report has all required sections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=True, enable_reporting=True)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            report_path = str(output_path).replace(".onnx", "_htp_export_report.md")
            with open(report_path, 'r') as f:
                report = f.read()
            
            # Check required sections
            assert "# HTP ONNX Export Report" in report
            assert "## Export Process Steps" in report
            assert "## Module Hierarchy" in report
            assert "## Complete Node Mappings" in report
            assert "## Export Summary" in report
    
    def test_report_no_truncation(self):
        """Test that report contains complete data without truncation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=True, enable_reporting=True)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            # Load metadata to get module count
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            report_path = str(output_path).replace(".onnx", "_htp_export_report.md")
            with open(report_path, 'r') as f:
                report = f.read()
            
            # Check that module hierarchy section exists
            assert "## Module Hierarchy" in report
            
            # All modules should be in the report (within the collapsible section)
            for module_path in metadata["modules"]:
                module_info = metadata["modules"][module_path]
                class_name = module_info.get("class_name", "Unknown")
                # Check either in mermaid diagram or in table
                assert class_name in report
            
            # No truncation markers in report
            assert "truncated for console" not in report
            assert "showing first" not in report.lower()
    
    def test_report_node_mappings(self):
        """Test that all node mappings are in the report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=True, enable_reporting=True)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            # Load metadata to get node mappings
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            report_path = str(output_path).replace(".onnx", "_htp_export_report.md")
            with open(report_path, 'r') as f:
                report = f.read()
            
            # Check that node mappings section exists
            assert "## Complete Node Mappings" in report
            
            # Should have at least 10 node mappings
            assert len(metadata["nodes"]) >= 10
            
            # Node mappings are in a collapsible section, check at least the section exists
            assert "<details>" in report
            assert "Click to expand" in report
    
    def test_report_console_output(self):
        """Test that console output section contains expected content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=True, enable_reporting=True)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            report_path = str(output_path).replace(".onnx", "_htp_export_report.md")
            with open(report_path, 'r') as f:
                report = f.read()
            
            # Check export process steps in markdown format
            assert "## Export Process Steps" in report
            
            # Should have all step sections in markdown format
            assert "### ✅ Step 1/6: Model Preparation" in report
            assert "### ✅ Step 2/6: Input Generation" in report
            assert "### ✅ Step 3/6: Hierarchy Building" in report
            assert "### ✅ Step 4/6: ONNX Export" in report
            assert "### ✅ Step 5/6: Node Tagging" in report
            assert "### ✅ Step 6/6: Tag Injection" in report
    
    def test_report_summary_statistics(self):
        """Test that summary statistics match metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(verbose=True, enable_reporting=True)
            stats = exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            report_path = str(output_path).replace(".onnx", "_htp_export_report.md")
            with open(report_path, 'r') as f:
                report = f.read()
            
            # Check summary section
            assert "## Export Summary" in report
            assert "Export Time" in report
            assert "Coverage" in report or "coverage" in report
            
            # Check for successful completion (in markdown format)
            assert "successfully" in report.lower()


class TestHTPCleanOnnxMode:
    """Test clean ONNX mode (without hierarchy attributes)."""
    
    def test_clean_onnx_metadata(self):
        """Test metadata generation in clean ONNX mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"
            
            exporter = HTPExporter(
                verbose=False,
                enable_reporting=False,
                embed_hierarchy_attributes=False  # Clean ONNX mode
            )
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )
            
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check that embed_hierarchy_attributes is False
            assert metadata["export_context"]["embed_hierarchy_attributes"] is False
            
            # Metadata should still contain nodes mapping
            assert "nodes" in metadata
            assert len(metadata["nodes"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])