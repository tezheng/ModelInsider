"""
Test suite specifically for verifying no duplicate console output messages.

This module tests the fix for the duplicate console output issue where
messages were being printed multiple times.
"""

import io
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from modelexport.strategies.htp.export_monitor import (
    HTPExportMonitor,
    HTPExportStep,
    HTPExportData,
)
from modelexport.strategies.htp.htp_exporter import HTPExporter


class TestNoDuplicateOutput:
    """Test cases to ensure no duplicate console output."""
    
    def test_no_duplicate_initial_messages(self, tmp_path):
        """Test that initial messages appear only once."""
        output_path = str(tmp_path / "test.onnx")
        
        # Create monitor
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="microsoft/resnet-50",
            verbose=True
        )
        
        # Create export data
        export_data = HTPExportData(
            model_name="microsoft/resnet-50",
            model_class="ResNet",
            total_modules=50,
            total_parameters=25000000,
            output_path=output_path
        )
        
        # Log model prep step (where initial messages are printed)
        monitor.log_step(HTPExportStep.MODEL_PREP, export_data)
        
        # Get console output
        console_output = monitor.console_buffer.getvalue()
        
        # Count occurrences of key messages
        loading_count = console_output.count("Loading model and exporting: microsoft/resnet-50")
        strategy_count = console_output.count("Using HTP")
        
        # Should appear exactly once
        assert loading_count == 1, f"Loading message appeared {loading_count} times"
        assert strategy_count == 1, f"Strategy message appeared {strategy_count} times"
    
    def test_no_duplicate_with_multiple_steps(self, tmp_path):
        """Test no duplicates when logging multiple steps."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        )
        
        export_data = HTPExportData(
            model_name="test-model",
            model_class="TestModel",
            total_modules=10,
            total_parameters=1000
        )
        
        # Log multiple steps
        monitor.log_step(HTPExportStep.MODEL_PREP, export_data)
        monitor.log_step(HTPExportStep.INPUT_GEN, export_data)
        monitor.log_step(HTPExportStep.HIERARCHY, export_data)
        
        console_output = monitor.console_buffer.getvalue()
        
        # Initial messages should still appear only once
        loading_count = console_output.count("Loading model and exporting:")
        strategy_count = console_output.count("Using HTP")
        
        assert loading_count == 1, f"Loading message appeared {loading_count} times"
        assert strategy_count == 1, f"Strategy message appeared {strategy_count} times"
        
        # Step headers should appear once each
        assert console_output.count("STEP 1/8: MODEL PREPARATION") == 1
        assert console_output.count("STEP 2/8: INPUT GENERATION") == 1
        assert console_output.count("STEP 3/8: HIERARCHY BUILDING") == 1
    
    def test_cli_does_not_print_messages(self, tmp_path, monkeypatch):
        """Test that CLI doesn't print messages when verbose is True."""
        # Import CLI module
        from modelexport import cli
        
        # Mock click.echo to capture output
        echo_calls = []
        def mock_echo(message):
            echo_calls.append(message)
        
        monkeypatch.setattr("click.echo", mock_echo)
        
        # The CLI should not print these messages anymore
        # (They're handled by the monitor)
        # So we check the actual CLI code doesn't have these prints
        
        # Read CLI source
        cli_path = Path(cli.__file__)
        cli_content = cli_path.read_text()
        
        # These lines should have been removed
        assert "🔄 Loading model and exporting:" not in cli_content
        assert '🧠 Using {strategy.upper()} (Hierarchical Trace-and-Project) strategy' not in cli_content
    
    @patch('modelexport.strategies.htp.htp_exporter.HTPExporter._convert_model_to_onnx')
    @patch('modelexport.strategies.htp.htp_exporter.HTPExporter._trace_model_hierarchy')
    @patch('modelexport.strategies.htp.htp_exporter.HTPExporter._apply_hierarchy_tags')
    @patch('modelexport.strategies.htp.htp_exporter.HTPExporter._embed_tags_in_onnx')
    @patch('modelexport.strategies.htp.htp_exporter.HTPExporter._generate_metadata_file')
    @patch('onnx.load')
    def test_full_export_no_duplicates(
        self,
        mock_onnx_load,
        mock_generate_metadata,
        mock_embed_tags,
        mock_apply_tags,
        mock_trace,
        mock_convert,
        tmp_path
    ):
        """Test full export flow has no duplicate messages."""
        from modelexport.strategies.htp.htp_exporter import HTPExporter
        import torch.nn as nn
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        # Mock returns
        mock_onnx_load.return_value = MagicMock()
        mock_generate_metadata.return_value = "metadata.json"
        
        # Set up exporter data
        exporter = HTPExporter(verbose=True, enable_reporting=True)
        exporter._hierarchy_data = {"": {"class_name": "SimpleModel"}}
        exporter._tagged_nodes = {}
        exporter._tagging_stats = {}
        exporter._export_stats = {"hierarchy_modules": 1}
        
        # Capture all console output
        output_buffer = io.StringIO()
        
        # Create output path
        output_path = str(tmp_path / "test.onnx")
        
        # Patch console to capture output
        with patch('sys.stdout', output_buffer):
            # Also capture from monitor's console buffer
            original_init = HTPExportMonitor.__init__
            captured_monitors = []
            
            def capture_monitor_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                captured_monitors.append(self)
            
            with patch.object(HTPExportMonitor, '__init__', capture_monitor_init):
                # Run export
                result = exporter.export(
                    model=SimpleModel(),
                    output_path=output_path,
                    model_name_or_path="test-model"
                )
        
        # Get all output
        stdout_output = output_buffer.getvalue()
        monitor_output = ""
        if captured_monitors:
            monitor_output = captured_monitors[0].console_buffer.getvalue()
        
        combined_output = stdout_output + monitor_output
        
        # Check for duplicates
        loading_lines = [
            line for line in combined_output.split('\n')
            if 'Loading model and exporting' in line and line.strip()
        ]
        strategy_lines = [
            line for line in combined_output.split('\n')
            if 'Using HTP' in line and line.strip()
        ]
        
        # Should have at most one of each
        assert len(loading_lines) <= 1, f"Found duplicate loading messages: {loading_lines}"
        assert len(strategy_lines) <= 1, f"Found duplicate strategy messages: {strategy_lines}"
    
    def test_console_writer_singleton_pattern(self, tmp_path):
        """Test that console writer doesn't create duplicate messages."""
        output_path = str(tmp_path / "test.onnx")
        
        # Create two monitors
        monitor1 = HTPExportMonitor(
            output_path=output_path,
            model_name="model1",
            verbose=True
        )
        
        monitor2 = HTPExportMonitor(
            output_path=output_path.replace("test", "test2"),
            model_name="model2",
            verbose=True
        )
        
        # Each should have its own console buffer
        assert monitor1.console_buffer is not monitor2.console_buffer
        
        # Each should have printed its own model name
        output1 = monitor1.console_buffer.getvalue()
        output2 = monitor2.console_buffer.getvalue()
        
        # No cross-contamination
        assert "model1" not in output2
        assert "model2" not in output1
    
    def test_step_handler_called_once(self, tmp_path):
        """Test that step handlers are called only once per step."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(output_path=output_path, verbose=True)
        
        # Track calls to console writer
        call_count = 0
        original_write = monitor.console_writer.write_model_prep
        
        def counting_write(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_write(*args, **kwargs)
        
        monitor.console_writer.write_model_prep = counting_write
        
        # Log step once
        export_data = HTPExportData(model_name="test", model_class="Test")
        monitor.log_step(HTPExportStep.MODEL_PREP, export_data)
        
        # Should be called exactly once
        assert call_count == 1
        
        # Log same step again
        monitor.log_step(HTPExportStep.MODEL_PREP, export_data)
        
        # Should be called again (no caching of steps)
        assert call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])