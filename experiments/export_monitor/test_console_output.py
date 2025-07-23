#!/usr/bin/env python3
"""
Test cases for HTP Export Monitor console output functionality.

This tests the specific issue with duplicate console messages and ANSI formatting.
"""

import io
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from modelexport.strategies.htp.export_monitor import (
    HTPExportMonitor,
    HTPExportStep,
    HTPExportData,
    HTPConsoleWriter,
    TextStyler,
)


class TestConsoleOutput:
    """Test console output formatting and behavior."""
    
    def test_no_duplicate_messages(self, tmp_path):
        """Test that messages are not duplicated in console output."""
        output_path = str(tmp_path / "test.onnx")
        
        # Create monitor
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        )
        
        # Get console output
        console_output = monitor.console_buffer.getvalue()
        
        # Count occurrences of the loading message
        loading_count = console_output.count("Loading model and exporting")
        assert loading_count == 1, f"Loading message appeared {loading_count} times, expected 1"
        
        # Count occurrences of the strategy message
        strategy_count = console_output.count("Using HTP")
        assert strategy_count == 1, f"Strategy message appeared {strategy_count} times, expected 1"
    
    def test_ansi_codes_in_console(self, tmp_path):
        """Test that ANSI codes are properly included in console output."""
        output_path = str(tmp_path / "test.onnx")
        
        # Create monitor with verbose output
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
            total_parameters=1000,
            output_path=output_path,
            embed_hierarchy_attributes=True
        )
        
        # Log a step
        monitor.log_step(HTPExportStep.MODEL_PREP, data)
        
        # Get console output
        console_output = monitor.console_buffer.getvalue()
        
        # Check for ANSI codes
        ansi_patterns = {
            r'\033\[1;36m': "Bold cyan (numbers)",
            r'\033\[1m\(\033\[0m': "Bold parentheses",
            r'\033\[3;92m': "Italic green (True)",
            r'\033\[3;91m': "Italic red (False)",
        }
        
        found_patterns = []
        for pattern, description in ansi_patterns.items():
            if re.search(pattern, console_output):
                found_patterns.append(description)
        
        assert len(found_patterns) > 0, f"No ANSI codes found in console output: {repr(console_output[:200])}"
    
    def test_text_styler_methods(self):
        """Test TextStyler produces correct ANSI codes."""
        # Test bold cyan
        assert TextStyler.bold_cyan(42) == "\033[1;36m42\033[0m"
        
        # Test bold
        assert TextStyler.bold("(") == "\033[1m(\033[0m"
        
        # Test italic green
        assert TextStyler.italic_green("True") == "\033[3;92mTrue\033[0m"
        
        # Test italic red
        assert TextStyler.italic_red("False") == "\033[3;91mFalse\033[0m"
        
        # Test green
        assert TextStyler.green("'test'") == "\033[32m'test'\033[0m"
        
        # Test magenta
        assert TextStyler.magenta("/path/") == "\033[35m/path/\033[0m"
        
        # Test bright magenta
        assert TextStyler.bright_magenta("ClassName") == "\033[95mClassName\033[0m"
    
    def test_console_writer_isolation(self, tmp_path):
        """Test that console writer doesn't interfere with normal console output."""
        # Create a mock console
        mock_console = Mock()
        mock_file = io.StringIO()
        mock_console.file = mock_file
        
        # Create console writer
        capture_buffer = io.StringIO()
        writer = HTPConsoleWriter(
            console=mock_console,
            verbose=True,
            capture_buffer=capture_buffer
        )
        
        # Write some text
        writer._print("Test message")
        
        # Check both outputs
        assert mock_file.getvalue() == "Test message\n"
        assert capture_buffer.getvalue() == "Test message\n"
    
    def test_step_logging_output(self, tmp_path):
        """Test that step logging produces expected output format."""
        output_path = str(tmp_path / "test.onnx")
        
        # Create monitor
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        )
        
        # Create test data
        data = HTPExportData(
            model_name="test-model",
            model_class="TestModel",
            total_modules=48,
            total_parameters=4385536,
            output_path=output_path,
            embed_hierarchy_attributes=True,
            hierarchy={
                "": {"class_name": "TestModel", "traced_tag": "/TestModel"},
                "layer1": {"class_name": "Layer", "traced_tag": "/TestModel/Layer"}
            }
        )
        
        # Clear initial output
        monitor.console_buffer = io.StringIO()
        monitor.console_writer.capture_buffer = monitor.console_buffer
        
        # Log steps
        monitor.log_step(HTPExportStep.MODEL_PREP, data)
        monitor.log_step(HTPExportStep.HIERARCHY, data)
        
        # Get output
        output = monitor.console_buffer.getvalue()
        
        # Check for expected content
        assert "STEP 1/8: MODEL PREPARATION" in output
        assert "STEP 3/8: HIERARCHY BUILDING" in output
        assert "48 modules" in output
        assert "4.4M parameters" in output
        assert "Module Hierarchy:" in output
    
    @pytest.mark.parametrize("verbose,expected_output", [
        (True, True),   # Verbose mode should produce output
        (False, False), # Non-verbose mode should not produce output
    ])
    def test_verbose_flag_behavior(self, tmp_path, verbose, expected_output):
        """Test that verbose flag controls console output."""
        output_path = str(tmp_path / "test.onnx")
        
        # Create monitor
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=verbose
        )
        
        # Clear initial output
        monitor.console_buffer = io.StringIO()
        monitor.console_writer.capture_buffer = monitor.console_buffer
        
        # Create test data
        data = HTPExportData(
            model_name="test-model",
            model_class="TestModel",
            total_modules=10,
            total_parameters=1000,
            output_path=output_path
        )
        
        # Log a step
        monitor.log_step(HTPExportStep.MODEL_PREP, data)
        
        # Check output
        output = monitor.console_buffer.getvalue()
        
        if expected_output:
            assert len(output) > 0, "Expected console output in verbose mode"
            assert "MODEL PREPARATION" in output
        else:
            assert len(output) == 0, f"Expected no console output in non-verbose mode, got: {output}"


class TestDuplicateOutputIssue:
    """Specific tests for the duplicate output issue."""
    
    @patch('modelexport.strategies.htp.htp_exporter.HTPExporter.export')
    def test_no_duplicate_initial_messages(self, mock_export, tmp_path):
        """Test that initial messages are not duplicated when using HTPExporter."""
        from modelexport.strategies.htp.htp_exporter import HTPExporter
        
        output_path = str(tmp_path / "test.onnx")
        
        # Create exporter with verbose mode
        exporter = HTPExporter(verbose=True)
        
        # Mock the console to capture output
        console_output = io.StringIO()
        with patch('sys.stdout', console_output):
            # Initialize monitor (this happens inside export)
            monitor = HTPExportMonitor(
                output_path=output_path,
                model_name="microsoft/resnet-50",
                verbose=True
            )
        
        # Get the output
        output = console_output.getvalue() + monitor.console_buffer.getvalue()
        
        # Count occurrences
        loading_lines = [line for line in output.split('\n') if 'Loading model and exporting' in line]
        strategy_lines = [line for line in output.split('\n') if 'Using HTP' in line]
        
        # Should only appear once each
        assert len(loading_lines) <= 1, f"Loading message duplicated: {loading_lines}"
        assert len(strategy_lines) <= 1, f"Strategy message duplicated: {strategy_lines}"
    
    def test_console_buffer_not_shared(self):
        """Test that console buffers are not shared between instances."""
        # Create two monitors
        monitor1 = HTPExportMonitor(
            output_path="test1.onnx",
            model_name="model1",
            verbose=True
        )
        
        monitor2 = HTPExportMonitor(
            output_path="test2.onnx",
            model_name="model2",
            verbose=True
        )
        
        # Check buffers are different
        assert monitor1.console_buffer is not monitor2.console_buffer
        assert monitor1.console_writer.capture_buffer is not monitor2.console_writer.capture_buffer
        
        # Check content is different
        output1 = monitor1.console_buffer.getvalue()
        output2 = monitor2.console_buffer.getvalue()
        
        assert "model1" in output1
        assert "model2" in output2
        assert "model1" not in output2
        assert "model2" not in output1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])