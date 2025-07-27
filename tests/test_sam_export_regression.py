#!/usr/bin/env python3
"""
Test cases for SAM export regression fix (TEZ-29).

This tests the specific issue where dict-to-tuple conversion breaks
SAM model exports that require keyword arguments.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from modelexport.strategies.htp import HTPExporter


class TestSAMExportRegression:
    """Test SAM export regression fix."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = f"{self.temp_dir.name}/sam_model.onnx"
    
    def teardown_method(self):
        """Clean up temp files."""
        self.temp_dir.cleanup()
    
    def test_dict_inputs_preserved_for_export(self):
        """Test that dict inputs are passed directly to torch.onnx.export, not converted to tuple."""
        # Create mock exporter
        exporter = HTPExporter(verbose=False)
        
        # Mock example inputs (simulating SAM-style inputs)
        mock_dict_inputs = {
            "pixel_values": Mock(),
            "input_points": Mock(), 
            "input_labels": Mock()
        }
        exporter.example_inputs = mock_dict_inputs
        
        # Mock model
        mock_model = Mock()
        mock_model.eval.return_value = None
        
        # Mock torch.onnx.export to capture what inputs it receives
        captured_inputs = None
        def capture_export(model, inputs, output_path, **kwargs):
            nonlocal captured_inputs
            captured_inputs = inputs
            
        with patch('torch.onnx.export', side_effect=capture_export):
            with patch('modelexport.core.onnx_utils.infer_output_names', return_value=None):
                try:
                    # This is the core method that should preserve dict structure
                    exporter._convert_model_to_onnx(mock_model, self.output_path, {})
                except Exception:
                    # We expect this to fail in testing due to mocks, but we care about the inputs
                    pass
        
        # CRITICAL TEST: inputs should still be dict, not tuple
        assert isinstance(captured_inputs, dict), f"Expected dict inputs, got {type(captured_inputs)}"
        assert captured_inputs is mock_dict_inputs, "Dict inputs should be passed directly, not converted"
    
    def test_tuple_inputs_preserved(self):
        """Test that tuple inputs are still handled correctly."""
        exporter = HTPExporter(verbose=False)
        
        # Mock tuple inputs (traditional format)
        mock_tuple_inputs = (Mock(), Mock())
        exporter.example_inputs = mock_tuple_inputs
        
        mock_model = Mock()
        mock_model.eval.return_value = None
        
        captured_inputs = None
        def capture_export(model, inputs, output_path, **kwargs):
            nonlocal captured_inputs
            captured_inputs = inputs
            
        with patch('torch.onnx.export', side_effect=capture_export):
            with patch('modelexport.core.onnx_utils.infer_output_names', return_value=None):
                try:
                    exporter._convert_model_to_onnx(mock_model, self.output_path, {})
                except Exception:
                    pass
        
        # Tuple inputs should remain as tuple
        assert isinstance(captured_inputs, tuple), f"Expected tuple inputs, got {type(captured_inputs)}"
        assert captured_inputs is mock_tuple_inputs, "Tuple inputs should be passed directly"
    
    def test_list_inputs_preserved(self):
        """Test that list inputs are handled correctly."""
        exporter = HTPExporter(verbose=False)
        
        # Mock list inputs
        mock_list_inputs = [Mock(), Mock()]
        exporter.example_inputs = mock_list_inputs
        
        mock_model = Mock()
        mock_model.eval.return_value = None
        
        captured_inputs = None
        def capture_export(model, inputs, output_path, **kwargs):
            nonlocal captured_inputs
            captured_inputs = inputs
            
        with patch('torch.onnx.export', side_effect=capture_export):
            with patch('modelexport.core.onnx_utils.infer_output_names', return_value=None):
                try:
                    exporter._convert_model_to_onnx(mock_model, self.output_path, {})
                except Exception:
                    pass
        
        # List inputs should remain as list
        assert isinstance(captured_inputs, list), f"Expected list inputs, got {type(captured_inputs)}"
        assert captured_inputs is mock_list_inputs, "List inputs should be passed directly"

    def test_regression_scenario_dict_to_tuple_conversion_removed(self):
        """
        Test the specific regression scenario where dict->tuple conversion breaks SAM.
        
        This test validates that we don't convert dict inputs to tuple anymore.
        """
        exporter = HTPExporter(verbose=False)
        
        # Simulate SAM-like model inputs (the exact scenario that was failing)
        sam_inputs = {
            "pixel_values": Mock(shape=(1, 3, 1024, 1024)),  # Image tensor
            "input_points": Mock(shape=(1, 1, 2)),           # Point coordinates  
            "input_labels": Mock(shape=(1, 1))               # Point labels
        }
        exporter.example_inputs = sam_inputs
        
        mock_model = Mock()
        mock_model.eval.return_value = None
        
        # Track what gets passed to torch.onnx.export
        export_calls = []
        def track_export_call(model, inputs, output_path, **kwargs):
            export_calls.append({
                'model': model,
                'inputs': inputs,
                'inputs_type': type(inputs),
                'output_path': output_path,
                'kwargs': kwargs
            })
            
        with patch('torch.onnx.export', side_effect=track_export_call):
            with patch('modelexport.core.onnx_utils.infer_output_names', return_value=None):
                try:
                    exporter._convert_model_to_onnx(mock_model, self.output_path, {})
                except Exception:
                    pass
        
        # Verify export was called
        assert len(export_calls) == 1, "torch.onnx.export should be called exactly once"
        
        call = export_calls[0]
        
        # CRITICAL: The inputs should still be a dict, not converted to tuple
        assert isinstance(call['inputs'], dict), (
            f"REGRESSION: Dict inputs were converted to {type(call['inputs'])}, "
            f"this breaks SAM models which need keyword arguments!"
        )
        
        # The dict should be the exact same object (not copied/modified) 
        assert call['inputs'] is sam_inputs, (
            "Dict inputs should be passed directly without modification"
        )
        
        # Verify all keys are preserved (critical for SAM)
        expected_keys = {'pixel_values', 'input_points', 'input_labels'}
        actual_keys = set(call['inputs'].keys())
        assert actual_keys == expected_keys, (
            f"Expected keys {expected_keys}, got {actual_keys}. "
            f"Missing keys would cause SAM export to fail!"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])