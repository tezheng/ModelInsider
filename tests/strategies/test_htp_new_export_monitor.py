"""Integration tests for the HTP Export Monitor."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from modelexport.strategies.htp import ExportStep, HTPExportMonitor
from modelexport.strategies.htp.step_data import (
    TensorInfo,
)


class TestHTPExportMonitor:
    """Test the HTPExportMonitor orchestrator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = f"{self.temp_dir.name}/test_model.onnx"
    
    def teardown_method(self):
        """Clean up temp files."""
        self.temp_dir.cleanup()
    
    def test_monitor_initialization(self):
        """Test monitor initialization with different configurations."""
        # Test with all features enabled
        monitor = HTPExportMonitor(
            output_path=self.output_path,
            model_name="test-model",
            verbose=True,
            enable_report=True,
            embed_hierarchy=True,
        )
        
        assert monitor.output_path == self.output_path
        assert monitor.model_name == "test-model"
        assert monitor.verbose is True
        assert monitor.enable_report is True
        assert monitor.embed_hierarchy is True
        assert len(monitor.writers) == 3  # Console, Metadata, Report
        
        # Test with minimal config
        monitor_min = HTPExportMonitor(
            output_path=self.output_path,
            verbose=False,
            enable_report=False,
        )
        
        assert len(monitor_min.writers) == 1  # Only Metadata
    
    def test_step_data_conversion(self):
        """Test that kwargs are properly converted to typed step data."""
        monitor = HTPExportMonitor(self.output_path)
        
        # Test MODEL_PREP conversion
        monitor.update(
            ExportStep.MODEL_PREP,
            model_class="BertModel",
            total_modules=123,
            total_parameters=4400000,
        )
        
        assert monitor.data.model_prep is not None
        assert monitor.data.model_prep.model_class == "BertModel"
        assert monitor.data.model_prep.total_modules == 123
        assert monitor.data.model_prep.total_parameters == 4400000
        
        # Test INPUT_GEN conversion
        monitor.update(
            ExportStep.INPUT_GEN,
            method="auto_generated",
            model_type="bert",
            task="text-classification",
            inputs={
                "input_ids": {"shape": [1, 128], "dtype": "int64"},
                "attention_mask": {"shape": [1, 128], "dtype": "int64"},
            }
        )
        
        assert monitor.data.input_gen is not None
        assert monitor.data.input_gen.method == "auto_generated"
        assert len(monitor.data.input_gen.inputs) == 2
        assert isinstance(monitor.data.input_gen.inputs["input_ids"], TensorInfo)
    
    def test_full_export_flow(self):
        """Test a complete export flow."""
        with HTPExportMonitor(
            output_path=self.output_path,
            model_name="bert-tiny",
            verbose=True,
            enable_report=True,
        ) as monitor:
            # Step 1: Model Prep
            monitor.update(
                ExportStep.MODEL_PREP,
                model_class="BertModel",
                total_modules=73,
                total_parameters=4385916,
            )
            
            # Step 2: Input Gen
            monitor.update(
                ExportStep.INPUT_GEN,
                method="auto_generated",
                model_type="bert",
                task="fill-mask",
                inputs={
                    "input_ids": {"shape": [1, 7], "dtype": "int64"},
                    "attention_mask": {"shape": [1, 7], "dtype": "int64"},
                }
            )
            
            # Step 3: Hierarchy
            monitor.update(
                ExportStep.HIERARCHY,
                hierarchy={
                    "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
                    "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/embeddings"},
                    "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/encoder"},
                },
                execution_steps=245,
            )
            
            # Step 4: ONNX Export
            monitor.update(
                ExportStep.ONNX_EXPORT,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=None,
                onnx_size_mb=17.6,
            )
            
            # Step 5: Node Tagging
            monitor.update(
                ExportStep.NODE_TAGGING,
                total_nodes=2,
                tagged_nodes={
                    "node1": "/BertModel/embeddings",
                    "node2": "/BertModel/encoder",
                },
                tagging_stats={
                    "direct_matches": 2,
                    "parent_matches": 0,
                    "root_fallbacks": 0,
                },
                coverage=100.0,
                op_counts={"Add": 1, "MatMul": 1},
            )
            
            # Step 6: Tag Injection
            monitor.update(
                ExportStep.TAG_INJECTION,
            )
            
            # Monitor finalizes automatically via context manager
        
        # Verify files were created
        base_path = Path(self.output_path).with_suffix("")
        metadata_path = f"{base_path}_htp_metadata.json"
        report_path = f"{base_path}_htp_export_report.md"
        
        # Check metadata
        assert Path(metadata_path).exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Verify new metadata structure
        assert "nodes" in metadata  # Tagged nodes at root level
        assert "report" in metadata
        assert "steps" in metadata["report"]
        assert "node_tagging" in metadata["report"]["steps"]
        assert "modules" in metadata  # Hierarchy is now at root level as modules
        
        # Check report
        assert Path(report_path).exists()
        with open(report_path) as f:
            report = f.read()
        
        # Report should contain markdown sections
        assert "# HTP ONNX Export Report" in report
        assert "## Export Process Steps" in report
        assert "## Module Hierarchy" in report
        assert "## Complete Node Mappings" in report
    
    def test_console_output_capture(self):
        """Test that report is generated independently from console."""
        with HTPExportMonitor(
            output_path=self.output_path,
            model_name="test-model",
            verbose=False,  # No console output
            enable_report=True,  # But still generate report
        ) as monitor:
            # Simulate all steps to trigger report generation
            monitor.update(
                ExportStep.MODEL_PREP,
                model_class="TestModel",
                total_modules=10,
                total_parameters=1000,
            )
            monitor.update(
                ExportStep.INPUT_GEN,
                method="auto",
                inputs={"input_ids": {"shape": [1, 10], "dtype": "int64"}},
            )
            monitor.update(
                ExportStep.HIERARCHY,
                hierarchy={"": {"class_name": "TestModel", "traced_tag": "/TestModel"}},
                execution_steps=5,
            )
            monitor.update(
                ExportStep.ONNX_EXPORT,
                opset_version=17,
                onnx_size_mb=0.1,
            )
            monitor.update(
                ExportStep.NODE_TAGGING,
                total_nodes=10,
                tagged_nodes={"node1": "/TestModel"},
                tagging_stats={"direct_matches": 10},
                coverage=100.0,
            )
            monitor.update(ExportStep.TAG_INJECTION)
        
        # Report should still be generated even without verbose
        base_path = Path(self.output_path).with_suffix("")
        report_path = f"{base_path}_htp_export_report.md"
        assert Path(report_path).exists()
        
        with open(report_path) as f:
            report = f.read()
        assert "TestModel" in report
        assert "10" in report  # Total modules
    
    def test_error_handling(self):
        """Test that errors in writers don't crash the monitor."""
        # Create a monitor with a mocked writer that raises an error
        monitor = HTPExportMonitor(self.output_path)
        
        # Add a broken writer
        class BrokenWriter:
            def write(self, step, data):
                raise ValueError("Test error")
            
            def close(self):
                raise ValueError("Close error")
        
        monitor.writers.append(BrokenWriter())
        
        # Should not raise - errors are caught and printed
        monitor.update(ExportStep.MODEL_PREP, model_class="Test")
        
        # Should handle errors in close too
        monitor.__exit__(None, None, None)
    
    def test_backward_compatibility(self):
        """Test that the API is backward compatible."""
        # Old API uses kwargs extensively
        monitor = HTPExportMonitor(
            output_path=self.output_path,
            model_name="test",
            verbose=True,
            enable_report=True,
            embed_hierarchy=True,
        )
        
        # Should handle arbitrary kwargs without errors
        monitor.update(
            ExportStep.MODEL_PREP,
            model_class="BertModel",
            total_modules=73,
            total_parameters=4385916,
            extra_field="ignored",  # Should be ignored
        )
        
        # The monitor now uses context manager pattern for finalization
        # No need to call finalize_export manually
    
    def test_traced_modules_in_export_summary(self):
        """Test that 'Traced modules: X/Y' appears in Export Summary output."""
        # Capture console output
        from io import StringIO
        captured_output = StringIO()
        
        with patch('rich.console.Console.print') as mock_print:
            # Store all print calls
            printed_lines = []
            mock_print.side_effect = lambda *args, **kwargs: printed_lines.append(str(args[0]) if args else "")
            
            with HTPExportMonitor(
                output_path=self.output_path,
                model_name="bert-tiny",
                verbose=True,
                enable_report=False,
            ) as monitor:
                # Step 1: Model Prep
                monitor.update(
                    ExportStep.MODEL_PREP,
                    model_class="BertModel",
                    total_modules=73,
                    total_parameters=4385916,
                )
                
                # Step 2: Input Gen
                monitor.update(
                    ExportStep.INPUT_GEN,
                    method="auto_generated",
                    inputs={
                        "input_ids": {"shape": [1, 7], "dtype": "int64"},
                    }
                )
                
                # Step 3: Hierarchy
                monitor.update(
                    ExportStep.HIERARCHY,
                    hierarchy={
                        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
                        "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/embeddings"},
                        "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/encoder"},
                    },
                    execution_steps=245,
                )
                
                # Step 4: ONNX Export
                monitor.update(
                    ExportStep.ONNX_EXPORT,
                    opset_version=17,
                    onnx_size_mb=17.6,
                )
                
                # Step 5: Node Tagging
                monitor.update(
                    ExportStep.NODE_TAGGING,
                    total_nodes=591,
                    tagged_nodes={
                        "node1": "/BertModel/embeddings",
                        "node2": "/BertModel/encoder",
                    },
                    tagging_stats={
                        "direct_matches": 524,
                    },
                    coverage=100.0,
                )
                
                # Step 6: Tag Injection
                monitor.update(
                    ExportStep.TAG_INJECTION,
                )
                
                # Monitor finalizes automatically via context manager
            
            # Check that Export Summary was printed
            full_output = "\n".join(printed_lines)
            
            # Should contain Export Summary section
            assert "📊 Export Summary:" in full_output or "Export Summary:" in full_output
            
            # Should contain the new format "Traced modules: X/Y"
            import re
            traced_pattern = re.compile(r'Traced modules:\s*\[?.*?\]?(\d+)/(\d+)')
            match = traced_pattern.search(full_output)
            
            assert match is not None, f"'Traced modules: X/Y' format not found in output:\n{full_output}"
            
            # Verify the numbers make sense
            traced = int(match.group(1))
            total = int(match.group(2))
            assert traced == 3, f"Expected 3 traced modules, got {traced}"
            assert total == 73, f"Expected 73 total modules, got {total}"
            assert traced <= total, f"Traced modules ({traced}) should be <= total modules ({total})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])