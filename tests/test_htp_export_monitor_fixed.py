"""
Fixed test cases for HTP Export Monitor.

This module contains corrected tests that match the current implementation
of the HTP Export Monitor.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from modelexport.strategies.htp_new.export_monitor import (
    HTPExportMonitor,
    ExportStep as HTPExportStep,
    ExportData as HTPExportData,
    TextStyler,
)


class SimpleModel(nn.Module):
    """Simple test model for export testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, input=None, **kwargs):
        """Forward method that accepts keyword arguments."""
        x = input if input is not None else kwargs.get('x')
        return self.linear(x)


class TestTextStyler:
    """Test the ANSI text styling utilities."""
    
    def test_bold_cyan(self):
        """Test bold cyan formatting."""
        assert TextStyler.bold_cyan(42) == "\033[1;36m42\033[0m"
        assert TextStyler.bold_cyan("100") == "\033[1;36m100\033[0m"
    
    def test_bold_parentheses(self):
        """Test bold parentheses formatting."""
        result = TextStyler.bold_parens("content")
        assert result == "\033[1m(\033[0mcontent\033[1m)\033[0m"
    
    def test_boolean_formatting(self):
        """Test boolean value formatting."""
        assert TextStyler.format_boolean(True) == "\033[3;92mTrue\033[0m"
        assert TextStyler.format_boolean(False) == "\033[3;91mFalse\033[0m"
    
    def test_string_formatting(self):
        """Test string formatting."""
        assert TextStyler.green("'test'") == "\033[32m'test'\033[0m"
    
    def test_path_formatting(self):
        """Test path and class name formatting."""
        assert TextStyler.magenta("/path/") == "\033[35m/path/\033[0m"
        assert TextStyler.bright_magenta("ClassName") == "\033[95mClassName\033[0m"


class TestHTPExportMonitor:
    """Test cases for HTP Export Monitor functionality."""
    
    def test_monitor_creation(self, tmp_path):
        """Test basic monitor creation."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        )
        
        # Check basic attributes
        assert monitor.output_path == output_path
        assert monitor.model_name == "test-model"
        assert monitor.verbose is True
    
    def test_console_output_no_duplicates(self, tmp_path):
        """Test that console output doesn't have duplicate messages."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        )
        
        # Create test data
        data = HTPExportData(
            model_name="test-model",
            model_class="TestModel",
            total_modules=10,
            total_parameters=1000
        )
        
        # Log a step
        monitor.log_step(HTPExportStep.MODEL_PREP, data)
        
        # Check console output
        console_output = monitor.get_console_output()
        
        # Count occurrences of loading message
        loading_count = console_output.count("Loading model and exporting: test-model")
        assert loading_count == 1, f"Loading message appeared {loading_count} times"
    
    def test_ansi_codes_present(self, tmp_path):
        """Test that ANSI codes are present in console output."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        )
        
        data = HTPExportData(
            model_name="test-model",
            model_class="TestModel",
            total_modules=10,
            total_parameters=1000000  # 1M to test formatting
        )
        
        monitor.log_step(HTPExportStep.MODEL_PREP, data)
        
        console_output = monitor.get_console_output()
        
        # Check for ANSI codes
        assert "\033[1;36m10\033[0m" in console_output  # Bold cyan for module count
        assert "\033[1;36m1.0\033[0m" in console_output  # Bold cyan for 1.0M parameters
        assert "\033[1m(\033[0m" in console_output  # Bold parentheses
    
    def test_text_report_no_ansi(self, tmp_path):
        """Test that text report has no ANSI codes."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        ) as monitor:
            data = HTPExportData(
                model_name="test-model",
                model_class="TestModel",
                total_modules=10,
                total_parameters=1000
            )
            
            monitor.log_step(HTPExportStep.MODEL_PREP, data)
        
        # Read text report
        report_path = output_path.replace('.onnx', '_htp_export_report.md')
        report_content = Path(report_path).read_text()
        
        # Check no ANSI codes
        assert "\033[" not in report_content
        assert "test-model" in report_content
        assert "10 modules" in report_content
    
    def test_metadata_structure(self, tmp_path):
        """Test metadata file structure."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(output_path=output_path) as monitor:
            data = HTPExportData(
                model_name="test-model",
                model_class="TestModel",
                total_modules=5,
                total_parameters=2000,
                hierarchy={
                    "": {"class_name": "TestModel", "traced_tag": "/TestModel"},
                    "layer1": {"class_name": "Linear", "traced_tag": "/TestModel/Linear"}
                }
            )
            
            monitor.log_step(HTPExportStep.MODEL_PREP, data)
            monitor.log_step(HTPExportStep.HIERARCHY, data)
        
        # Read metadata
        metadata_path = output_path.replace('.onnx', '_htp_metadata.json')
        metadata = json.loads(Path(metadata_path).read_text())
        
        # Check structure
        assert "export_context" in metadata
        assert "model" in metadata
        assert "modules" in metadata
        assert "report" in metadata
        assert "steps" in metadata["report"]
    
    def test_backward_compatibility_update(self, tmp_path):
        """Test backward compatibility with update() method."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(output_path=output_path, verbose=True)
        
        # Use update() method
        monitor.update(
            HTPExportStep.MODEL_PREP,
            model_name="test-model",
            model_class="TestModel",
            total_modules=10,
            total_parameters=1000
        )
        
        # Check data was stored
        assert hasattr(monitor, '_step_data')
        assert "model_preparation" in monitor._step_data
        assert monitor._step_data["model_preparation"]["model_name"] == "test-model"
        
        # Check console output was generated
        console_output = monitor.get_console_output()
        assert "test-model" in console_output
        assert "MODEL PREPARATION" in console_output
    
    def test_all_steps(self, tmp_path):
        """Test logging all export steps."""
        output_path = str(tmp_path / "test.onnx")
        
        data = HTPExportData(
            model_name="test-model",
            model_class="TestModel",
            total_modules=10,
            total_parameters=1000,
            hierarchy={
                "": {"class_name": "TestModel", "traced_tag": "/TestModel"}
            },
            total_nodes=50,
            tagged_nodes={f"node_{i}": "/TestModel" for i in range(50)},
            tagging_stats={
                "direct_matches": 30,
                "parent_matches": 15,
                "root_fallbacks": 5,
                "empty_tags": 0
            },
            coverage=100.0
        )
        
        with HTPExportMonitor(
            output_path=output_path,
            verbose=True
        ) as monitor:
            # Log all steps
            for step in HTPExportStep:
                monitor.log_step(step, data)
        
        # Check all files created
        assert Path(output_path.replace('.onnx', '_htp_metadata.json')).exists()
        assert Path(output_path.replace('.onnx', '_htp_export_report.md')).exists()
        assert Path(output_path.replace('.onnx', '_console.log')).exists()
        
        # Check console log has ANSI codes
        console_log = Path(output_path.replace('.onnx', '_console.log')).read_text()
        assert "\033[" in console_log
        
        # Check report has no ANSI codes
        report = Path(output_path.replace('.onnx', '_htp_export_report.md')).read_text()
        assert "\033[" not in report
        assert "# HTP ONNX Export Report" in report
        assert "Step 1/6: Model Preparation" in report
        assert "## Export Summary" in report
    
    def test_non_verbose_mode(self, tmp_path):
        """Test non-verbose mode produces minimal output."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=False  # Non-verbose
        )
        
        data = HTPExportData(
            model_name="test-model",
            model_class="TestModel"
        )
        
        monitor.log_step(HTPExportStep.MODEL_PREP, data)
        
        # Console should be empty in non-verbose mode
        console_output = monitor.get_console_output()
        assert len(console_output) == 0
    
    def test_empty_hierarchy_handling(self, tmp_path):
        """Test handling of empty hierarchy data."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(output_path=output_path) as monitor:
            data = HTPExportData(
                model_name="test-model",
                hierarchy={}  # Empty hierarchy
            )
            
            # Should not raise ValueError
            monitor.log_step(HTPExportStep.HIERARCHY, data)
        
        # Check metadata was created
        metadata_path = output_path.replace('.onnx', '_htp_metadata.json')
        assert Path(metadata_path).exists()


class TestIntegration:
    """Integration tests with HTPExporter."""
    
    @patch('torch.onnx.export')
    @patch('onnx.load')
    @patch('onnx.save')
    def test_exporter_integration(self, mock_save, mock_load, mock_export, tmp_path):
        """Test integration with HTPExporter."""
        from modelexport.strategies.htp.htp_exporter import HTPExporter
        
        # Configure mocks
        mock_onnx_model = MagicMock()
        mock_onnx_model.graph.node = []
        mock_load.return_value = mock_onnx_model
        
        # Create exporter
        exporter = HTPExporter(verbose=True, enable_reporting=True)
        
        # Set minimal data with required fields
        exporter._hierarchy_data = {"": {"class_name": "SimpleModel", "traced_tag": "/SimpleModel"}}
        exporter._tagged_nodes = {}
        exporter._tagging_stats = {}
        exporter._export_stats = {"hierarchy_modules": 1}
        
        output_path = str(tmp_path / "test.onnx")
        
        # Mock input generation
        with patch('modelexport.core.model_input_generator.generate_dummy_inputs') as mock_inputs:
            mock_inputs.return_value = {"input": torch.randn(1, 10)}
            
            # Mock hierarchy builder
            with patch.object(exporter, '_trace_model_hierarchy'):
                with patch.object(exporter, '_apply_hierarchy_tags'):
                    # Run export
                    result = exporter.export(
                        model=SimpleModel(),
                        output_path=output_path,
                        model_name_or_path="test-model"
                    )
        
        # Check files were created
        assert Path(output_path.replace('.onnx', '_htp_metadata.json')).exists()
        assert Path(output_path.replace('.onnx', '_console.log')).exists()
        
        # Check console log has content
        console_log = Path(output_path.replace('.onnx', '_console.log')).read_text()
        assert "Loading model and exporting: test-model" in console_log
        assert "MODEL PREPARATION" in console_log


if __name__ == "__main__":
    pytest.main([__file__, "-v"])