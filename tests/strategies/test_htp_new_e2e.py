"""End-to-end tests for the new HTP implementation with bert-tiny."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from modelexport.strategies.htp import HTPExporter


class TestHTPE2E:
    """End-to-end tests for HTP exporter with real models."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available() and os.getenv("CI") == "true",
        reason="Skip E2E tests in CI without GPU"
    )
    def test_bert_tiny_export(self):
        """Test exporting bert-tiny model with new HTP implementation."""
        model_name = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "bert-tiny.onnx")
            
            # Create exporter with reporting enabled
            exporter = HTPExporter(
                verbose=True,
                enable_reporting=True,
                embed_hierarchy_attributes=True,
            )
            
            # Export the model
            stats = exporter.export(
                model_name_or_path=model_name,
                output_path=output_path,
            )
            
            # Check that export was successful
            assert Path(output_path).exists()
            assert stats["strategy"] == "htp"
            assert stats["empty_tags"] == 0
            assert stats["coverage_percentage"] == 100.0
            
            # Check metadata file
            metadata_path = output_path.replace(".onnx", "_htp_metadata.json")
            assert Path(metadata_path).exists()
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Verify metadata structure (new format)
            assert "nodes" in metadata  # Tagged nodes at root level
            assert "report" in metadata
            assert "export_context" in metadata
            assert "model" in metadata
            assert "modules" in metadata
            
            # Check report file
            report_path = output_path.replace(".onnx", "_htp_export_report.md")
            assert Path(report_path).exists()
            
            with open(report_path) as f:
                report = f.read()
            
            # Verify report content in markdown format
            assert "# HTP ONNX Export Report" in report
            assert "## Export Process Steps" in report
            assert "## Module Hierarchy" in report
            assert "## Complete Node Mappings" in report
            
            # Verify some expected modules for bert-tiny
            assert "BertModel" in report
            assert "BertEmbeddings" in report
            assert "BertEncoder" in report
    
    def test_clean_onnx_mode(self):
        """Test exporting without hierarchy attributes (clean ONNX mode)."""
        model_name = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "bert-tiny-clean.onnx")
            
            # Create exporter with clean ONNX mode
            exporter = HTPExporter(
                verbose=False,
                enable_reporting=False,
                embed_hierarchy_attributes=False,  # Clean ONNX mode
            )
            
            # Export the model
            stats = exporter.export(
                model_name_or_path=model_name,
                output_path=output_path,
            )
            
            # Check that export was successful
            assert Path(output_path).exists()
            assert stats["strategy"] == "htp"
            
            # Metadata should still be created
            metadata_path = output_path.replace(".onnx", "_htp_metadata.json")
            assert Path(metadata_path).exists()
            
            # Report should NOT be created (reporting disabled)
            report_path = output_path.replace(".onnx", "_htp_export_report.md")
            assert not Path(report_path).exists()
    
    def test_verbose_console_output(self):
        """Test that verbose mode produces expected console output."""
        import io
        from contextlib import redirect_stdout
        
        model_name = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "bert-tiny-verbose.onnx")
            
            # Capture console output
            output_buffer = io.StringIO()
            
            # Create exporter with verbose mode
            exporter = HTPExporter(
                verbose=True,
                enable_reporting=False,
            )
            
            # Export with output capture
            with redirect_stdout(output_buffer):
                stats = exporter.export(
                    model_name_or_path=model_name,
                    output_path=output_path,
                )
            
            console_output = output_buffer.getvalue()
            
            # Verify console output contains expected sections
            assert "HTP ONNX EXPORT PROCESS" in console_output
            assert "STEP 1/6: MODEL PREPARATION" in console_output
            assert "STEP 2/6: INPUT GENERATION" in console_output
            assert "STEP 3/6: HIERARCHY BUILDING" in console_output
            assert "STEP 4/6: ONNX EXPORT" in console_output
            assert "STEP 5/6: ONNX NODE TAGGING" in console_output
            assert "STEP 6/6: TAG INJECTION" in console_output
            assert "EXPORT COMPLETE" in console_output
    
    def test_backward_compatibility_api(self):
        """Test that the old API functions still work."""
        from modelexport.strategies.htp import (
            export_with_htp,
            export_with_htp_reporting,
        )
        
        model_name = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test export_with_htp
            output_path1 = os.path.join(temp_dir, "bert-tiny-compat1.onnx")
            stats1 = export_with_htp(
                model=None,
                model_name_or_path=model_name,
                output_path=output_path1,
                verbose=False,
            )
            assert Path(output_path1).exists()
            assert stats1["strategy"] == "htp"
            
            # Test export_with_htp_reporting
            output_path2 = os.path.join(temp_dir, "bert-tiny-compat2.onnx")
            stats2 = export_with_htp_reporting(
                model=None,
                model_name_or_path=model_name,
                output_path=output_path2,
                verbose=False,
            )
            assert Path(output_path2).exists()
            assert stats2["strategy"] == "htp"
            
            # Report should be created for the second one
            report_path = output_path2.replace(".onnx", "_htp_export_report.md")
            assert Path(report_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])