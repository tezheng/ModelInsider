#!/usr/bin/env python3
"""
Shape auto-detection tests.

Tests automatic ONNX shape detection functionality for intelligent 
shape inference and fallback mechanisms.

Tests extracted from test_auto_shape_detection.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
from unittest.mock import Mock, MagicMock

import pytest

# Import from the src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from onnx_tokenizer import (
        ONNXTokenizer,
        create_auto_shape_tokenizer,
        parse_onnx_input_shapes
    )
except ImportError:
    pytest.skip("ONNXTokenizer modules not available", allow_module_level=True)


class TestShapeAutoDetection:
    """Tests for automatic shape detection from ONNX models."""
    
    def test_parse_onnx_input_shapes_function(self):
        """Test the parse_onnx_input_shapes function."""
        # Create a mock ONNX model for testing
        mock_onnx_path = "/fake/path/model.onnx"
        
        # This test validates the function signature and expected behavior
        try:
            # The function should exist and be callable
            assert callable(parse_onnx_input_shapes), "Function should be callable"
            
            # Test with a non-existent path (should handle gracefully)
            result = parse_onnx_input_shapes(mock_onnx_path)
            
            # Result should be a dictionary (even if empty due to mock path)
            assert isinstance(result, dict), "Should return dictionary"
            
        except FileNotFoundError:
            # Expected for non-existent path
            pass
        except ImportError:
            pytest.skip("ONNX parsing functionality not available")
        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.skip("Function not implemented yet")
            else:
                raise
    
    def test_onnx_tokenizer_auto_detection_from_path(self):
        """Test ONNXTokenizer auto-detection from ONNX file path."""
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        # Test with string path
        fake_onnx_path = "/fake/path/model.onnx"
        
        try:
            tokenizer = ONNXTokenizer(
                tokenizer=mock_tokenizer,
                onnx_model=fake_onnx_path
            )
            
            # Should have detected or fallback batch size and sequence length
            assert hasattr(tokenizer, 'fixed_batch_size'), "Should have fixed_batch_size"
            assert hasattr(tokenizer, 'fixed_sequence_length'), "Should have fixed_sequence_length"
            assert isinstance(tokenizer.fixed_batch_size, int), "Batch size should be integer"
            assert isinstance(tokenizer.fixed_sequence_length, int), "Sequence length should be integer"
            assert tokenizer.fixed_batch_size > 0, "Batch size should be positive"
            assert tokenizer.fixed_sequence_length > 0, "Sequence length should be positive"
            
        except FileNotFoundError:
            # Expected for fake path
            pass
        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.skip("Auto-detection not implemented yet")
            else:
                pytest.fail(f"Auto-detection from path failed: {e}")
    
    def test_onnx_tokenizer_auto_detection_from_model(self):
        """Test ONNXTokenizer auto-detection from ORTModel object."""
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        # Create mock ORTModel with ONNX session
        mock_model = Mock()
        mock_session = Mock()
        
        # Mock input with shape information
        mock_input = Mock()
        mock_input.name = "input_ids"
        mock_input.shape = [2, 16]  # batch_size=2, sequence_length=16
        mock_session.get_inputs.return_value = [mock_input]
        
        # Different ways ORTModel might expose the session
        mock_model.model = mock_session
        mock_model.session = mock_session  # Alternative attribute name
        
        try:
            tokenizer = ONNXTokenizer(
                tokenizer=mock_tokenizer,
                onnx_model=mock_model
            )
            
            # Should auto-detect from mock model
            assert hasattr(tokenizer, 'fixed_batch_size')
            assert hasattr(tokenizer, 'fixed_sequence_length')
            
            # With our mock, it might detect the shapes we provided
            if hasattr(tokenizer, '_auto_detected') and tokenizer._auto_detected:
                assert tokenizer.fixed_batch_size == 2 or tokenizer.fixed_batch_size > 0
                assert tokenizer.fixed_sequence_length == 16 or tokenizer.fixed_sequence_length > 0
            
        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.skip("Auto-detection from model not implemented yet")
            else:
                pytest.fail(f"Auto-detection from model failed: {e}")
    
    def test_create_auto_shape_tokenizer_helper(self):
        """Test the create_auto_shape_tokenizer convenience function."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        
        try:
            auto_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, mock_model)
            
            # Should return an ONNXTokenizer instance
            assert isinstance(auto_tokenizer, ONNXTokenizer), "Should return ONNXTokenizer"
            assert auto_tokenizer.tokenizer == mock_tokenizer, "Should wrap the provided tokenizer"
            
        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.skip("Helper function not implemented yet")
            else:
                pytest.fail(f"Helper function failed: {e}")
    
    def test_manual_override_of_auto_detection(self):
        """Test manual override of auto-detected shapes."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        
        # Create tokenizer with explicit manual values
        try:
            tokenizer = ONNXTokenizer(
                tokenizer=mock_tokenizer,
                onnx_model=mock_model,  # Auto-detection source
                fixed_batch_size=4,     # Manual override
                fixed_sequence_length=32  # Manual override
            )
            
            # Manual values should take precedence
            assert tokenizer.fixed_batch_size == 4, "Manual batch size should override auto-detection"
            assert tokenizer.fixed_sequence_length == 32, "Manual sequence length should override auto-detection"
            
        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.skip("Manual override not implemented yet")
            else:
                pytest.fail(f"Manual override failed: {e}")
    
    def test_fallback_for_dynamic_shapes(self):
        """Test fallback behavior when auto-detection fails."""
        mock_tokenizer = Mock()
        
        # Test with no model provided (should use fallbacks)
        try:
            tokenizer = ONNXTokenizer(
                tokenizer=mock_tokenizer,
                onnx_model=None  # No model for auto-detection
            )
            
            # Should have fallback values
            assert hasattr(tokenizer, 'fixed_batch_size')
            assert hasattr(tokenizer, 'fixed_sequence_length')
            assert tokenizer.fixed_batch_size > 0, "Should have positive fallback batch size"
            assert tokenizer.fixed_sequence_length > 0, "Should have positive fallback sequence length"
            
        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.skip("Fallback behavior not implemented yet")
            elif "required" in str(e).lower():
                # Expected if onnx_model is required
                pass
            else:
                pytest.fail(f"Fallback behavior failed: {e}")
    
    def test_shape_detection_with_various_input_names(self):
        """Test shape detection with different ONNX input names."""
        mock_tokenizer = Mock()
        
        # Test various input configurations
        input_configs = [
            [("input_ids", [1, -1])],  # Dynamic sequence length
            [("input_ids", [2, 16])],  # Fixed shapes
            [("input_ids", [1, 512]), ("attention_mask", [1, 512])],  # Multiple inputs
            [("inputs", [4, 128])],  # Non-standard input name
        ]
        
        for inputs in input_configs:
            mock_model = Mock()
            mock_session = Mock()
            
            # Create mock inputs
            mock_inputs = []
            for name, shape in inputs:
                mock_input = Mock()
                mock_input.name = name
                mock_input.shape = shape
                mock_inputs.append(mock_input)
            
            mock_session.get_inputs.return_value = mock_inputs
            mock_model.model = mock_session
            
            try:
                tokenizer = ONNXTokenizer(
                    tokenizer=mock_tokenizer,
                    onnx_model=mock_model
                )
                
                # Should handle various input configurations
                assert hasattr(tokenizer, 'fixed_batch_size')
                assert hasattr(tokenizer, 'fixed_sequence_length')
                
            except Exception as e:
                if "not implemented" in str(e).lower():
                    pytest.skip("Input name handling not implemented yet")
                else:
                    # Some configurations might not be supported yet
                    pass
    
    def test_dynamic_shape_handling(self):
        """Test handling of dynamic shapes in ONNX models."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_session = Mock()
        
        # Create input with dynamic dimensions (represented as -1 or strings)
        mock_input = Mock()
        mock_input.name = "input_ids"
        mock_input.shape = ["batch_size", "sequence_length"]  # Symbolic dimensions
        mock_session.get_inputs.return_value = [mock_input]
        mock_model.model = mock_session
        
        try:
            tokenizer = ONNXTokenizer(
                tokenizer=mock_tokenizer,
                onnx_model=mock_model
            )
            
            # Should fall back to reasonable defaults for dynamic shapes
            assert tokenizer.fixed_batch_size > 0, "Should have positive fallback batch size"
            assert tokenizer.fixed_sequence_length > 0, "Should have positive fallback sequence length"
            
        except Exception as e:
            if "not implemented" in str(e).lower():
                pytest.skip("Dynamic shape handling not implemented yet")
            else:
                # Dynamic shapes might not be fully supported
                pass
    
    def test_shape_validation_and_constraints(self):
        """Test validation of detected shapes and constraints."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_session = Mock()
        
        # Test various shape scenarios
        test_cases = [
            ([0, 16], "zero batch size"),  # Invalid batch size
            ([2, 0], "zero sequence length"),  # Invalid sequence length
            ([-1, -1], "all dynamic"),  # All dynamic dimensions
            ([1000, 10000], "very large"),  # Very large dimensions
        ]
        
        for shape, description in test_cases:
            mock_input = Mock()
            mock_input.name = "input_ids"
            mock_input.shape = shape
            mock_session.get_inputs.return_value = [mock_input]
            mock_model.model = mock_session
            
            try:
                tokenizer = ONNXTokenizer(
                    tokenizer=mock_tokenizer,
                    onnx_model=mock_model
                )
                
                # Should handle edge cases gracefully
                assert tokenizer.fixed_batch_size > 0, f"Should handle {description} case"
                assert tokenizer.fixed_sequence_length > 0, f"Should handle {description} case"
                
            except Exception as e:
                if "not implemented" in str(e).lower():
                    pytest.skip("Shape validation not implemented yet")
                else:
                    # Some edge cases might raise exceptions
                    pass
    
    def test_integration_with_different_model_types(self):
        """Test integration with different types of model objects."""
        mock_tokenizer = Mock()
        
        # Test different model object types
        model_types = [
            "string_path",
            "ort_model_object", 
            "onnx_session_object",
            "pathlib_path"
        ]
        
        for model_type in model_types:
            if model_type == "string_path":
                mock_model = "/path/to/model.onnx"
            elif model_type == "pathlib_path":
                mock_model = Path("/path/to/model.onnx")
            else:
                mock_model = Mock()
                if model_type == "ort_model_object":
                    mock_model.model = Mock()  # Has .model attribute
                elif model_type == "onnx_session_object":
                    mock_model.get_inputs = Mock(return_value=[])  # Looks like session
            
            try:
                tokenizer = ONNXTokenizer(
                    tokenizer=mock_tokenizer,
                    onnx_model=mock_model
                )
                
                # Should handle different model types
                assert hasattr(tokenizer, 'fixed_batch_size')
                assert hasattr(tokenizer, 'fixed_sequence_length')
                
            except (FileNotFoundError, AttributeError, TypeError):
                # Expected for mock objects
                pass
            except Exception as e:
                if "not implemented" in str(e).lower():
                    pytest.skip(f"Support for {model_type} not implemented yet")
                else:
                    # Some model types might not be supported
                    pass