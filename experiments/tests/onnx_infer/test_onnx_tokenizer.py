"""
Unit tests for ONNXTokenizer module.

Tests the ONNX tokenizer wrapper functionality including:
- Auto-detection of input shapes from ONNX models
- Fixed-shape enforcement for batch size and sequence length
- Proper handling of various input formats
- Edge cases and error conditions
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import onnx

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tez-144_onnx_automodel_infer" / "src"))

from onnx_tokenizer import (
    ONNXTokenizer,
    parse_onnx_input_shapes,
    create_auto_shape_tokenizer
)


@pytest.mark.unit
class TestParseOnnxInputShapes:
    """Test the parse_onnx_input_shapes function."""
    
    def test_parse_from_onnx_model_proto(self, sample_onnx_model):
        """Test parsing shapes from ONNX ModelProto."""
        model = onnx.load(str(sample_onnx_model))
        shapes = parse_onnx_input_shapes(model)
        
        assert "input_ids" in shapes
        assert "attention_mask" in shapes
        assert shapes["input_ids"] == [1, 128]
        assert shapes["attention_mask"] == [1, 128]
    
    def test_parse_from_onnx_file_path(self, sample_onnx_model):
        """Test parsing shapes from ONNX file path."""
        shapes = parse_onnx_input_shapes(sample_onnx_model)
        
        assert "input_ids" in shapes
        assert "attention_mask" in shapes
        assert shapes["input_ids"] == [1, 128]
        assert shapes["attention_mask"] == [1, 128]
    
    def test_parse_from_string_path(self, sample_onnx_model):
        """Test parsing shapes from string file path."""
        shapes = parse_onnx_input_shapes(str(sample_onnx_model))
        
        assert "input_ids" in shapes
        assert "attention_mask" in shapes
        assert shapes["input_ids"] == [1, 128]
        assert shapes["attention_mask"] == [1, 128]
    
    def test_parse_dynamic_dimensions(self, temp_model_dir):
        """Test parsing models with dynamic dimensions."""
        # Create ONNX model with dynamic dimensions
        model_path = temp_model_dir / "dynamic_model.onnx"
        
        # Create input with dynamic batch size
        input_ids = onnx.helper.make_tensor_value_info(
            'input_ids', onnx.TensorProto.INT64, ['batch_size', 128]
        )
        
        # Create output
        logits = onnx.helper.make_tensor_value_info(
            'logits', onnx.TensorProto.FLOAT, ['batch_size', 2]
        )
        
        # Create simple node
        node = onnx.helper.make_node('Identity', inputs=['input_ids'], outputs=['logits'])
        
        # Create graph and model
        graph = onnx.helper.make_graph([node], 'dynamic_model', [input_ids], [logits])
        model = onnx.helper.make_model(graph)
        onnx.save(model, str(model_path))
        
        shapes = parse_onnx_input_shapes(model_path)
        assert shapes["input_ids"] == [-1, 128]  # Dynamic dimension represented as -1


@pytest.mark.unit
class TestONNXTokenizer:
    """Test the ONNXTokenizer class."""
    
    def test_init_with_manual_shapes(self, sample_tokenizer):
        """Test initialization with manually specified shapes."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=64
        )
        
        assert tokenizer.fixed_batch_size == 2
        assert tokenizer.fixed_sequence_length == 64
        assert tokenizer.tokenizer == sample_tokenizer
    
    def test_init_with_onnx_model_auto_detection(self, sample_tokenizer, sample_onnx_model):
        """Test initialization with ONNX model auto-detection."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            onnx_model=sample_onnx_model
        )
        
        assert tokenizer.fixed_batch_size == 1
        assert tokenizer.fixed_sequence_length == 128
    
    def test_init_with_ort_model_mock(self, sample_tokenizer):
        """Test initialization with mock ORTModel."""
        mock_ort_model = Mock()
        mock_ort_model.path = Path("/fake/path/model.onnx")
        
        # Mock the parse_onnx_input_shapes function
        with patch('onnx_tokenizer.parse_onnx_input_shapes') as mock_parse:
            mock_parse.return_value = {"input_ids": [2, 256], "attention_mask": [2, 256]}
            
            tokenizer = ONNXTokenizer(
                tokenizer=sample_tokenizer,
                onnx_model=mock_ort_model
            )
            
            assert tokenizer.fixed_batch_size == 2
            assert tokenizer.fixed_sequence_length == 256
    
    def test_init_with_session_fallback(self, sample_tokenizer):
        """Test initialization with session fallback when path is unavailable."""
        mock_ort_model = Mock()
        mock_ort_model.path = None
        mock_ort_model.model_path = None
        
        # Mock session with get_inputs
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input_ids"
        mock_input.shape = [1, 512]
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_ort_model.model = mock_session
        
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            onnx_model=mock_ort_model
        )
        
        assert tokenizer.fixed_batch_size == 1
        assert tokenizer.fixed_sequence_length == 512
    
    def test_init_invalid_shapes(self, sample_tokenizer):
        """Test initialization with invalid shapes raises ValueError."""
        with pytest.raises(ValueError, match="Invalid shapes"):
            ONNXTokenizer(
                tokenizer=sample_tokenizer,
                fixed_batch_size=0,
                fixed_sequence_length=128
            )
        
        with pytest.raises(ValueError, match="Invalid shapes"):
            ONNXTokenizer(
                tokenizer=sample_tokenizer,
                fixed_batch_size=1,
                fixed_sequence_length=-1
            )
    
    def test_call_single_text(self, sample_tokenizer):
        """Test tokenization of single text input."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=128
        )
        
        result = tokenizer("Hello world!")
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (1, 128)
        assert result["attention_mask"].shape == (1, 128)
    
    def test_call_batch_text(self, sample_tokenizer, sample_batch_texts):
        """Test tokenization of batch text inputs."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=3,
            fixed_sequence_length=64
        )
        
        result = tokenizer(sample_batch_texts)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (3, 64)
        assert result["attention_mask"].shape == (3, 64)
    
    def test_call_batch_padding(self, sample_tokenizer):
        """Test batch padding when input is smaller than fixed batch size."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=5,
            fixed_sequence_length=64
        )
        
        # Input only 2 texts, should be padded to 5
        result = tokenizer(["Text 1", "Text 2"])
        
        assert result["input_ids"].shape == (5, 64)
        assert result["attention_mask"].shape == (5, 64)
    
    def test_call_batch_truncation(self, sample_tokenizer):
        """Test batch truncation when input is larger than fixed batch size."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=64
        )
        
        # Input 4 texts, should be truncated to 2
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        result = tokenizer(texts)
        
        assert result["input_ids"].shape == (2, 64)
        assert result["attention_mask"].shape == (2, 64)
    
    def test_call_return_tensors_pt(self, sample_tokenizer):
        """Test tokenization with PyTorch tensor return type."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=64
        )
        
        result = tokenizer("Hello world!", return_tensors="pt")
        
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
    
    def test_call_return_tensors_np(self, sample_tokenizer):
        """Test tokenization with NumPy array return type."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=64
        )
        
        result = tokenizer("Hello world!", return_tensors="np")
        
        assert isinstance(result["input_ids"], np.ndarray)
        assert isinstance(result["attention_mask"], np.ndarray)
    
    def test_validate_shapes_success(self, sample_tokenizer):
        """Test successful shape validation."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=64
        )
        
        # This should not raise any exception
        result = tokenizer(["Text 1", "Text 2"])
        tokenizer._validate_shapes(result)
    
    def test_validate_shapes_failure(self, sample_tokenizer):
        """Test shape validation failure."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=64
        )
        
        # Create a result with wrong shapes
        wrong_result = {
            "input_ids": torch.zeros((3, 64)),  # Wrong batch size
            "attention_mask": torch.zeros((2, 64))
        }
        
        with pytest.raises(ValueError, match="has shape"):
            tokenizer._validate_shapes(wrong_result)
    
    def test_decode_passthrough(self, sample_tokenizer):
        """Test that decode methods pass through to underlying tokenizer."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=64
        )
        
        # Test batch_decode
        token_ids = [[1, 2, 3], [4, 5, 6]]
        batch_result = tokenizer.batch_decode(token_ids)
        assert isinstance(batch_result, list)
        
        # Test decode
        single_result = tokenizer.decode([1, 2, 3])
        assert isinstance(single_result, str)
    
    def test_getattr_passthrough(self, sample_tokenizer):
        """Test that other attributes pass through to underlying tokenizer."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=64
        )
        
        # Test accessing vocab_size through passthrough
        assert hasattr(tokenizer, 'vocab_size')
        assert tokenizer.vocab_size == sample_tokenizer.vocab_size
    
    def test_auto_detect_shapes_exception_handling(self, sample_tokenizer):
        """Test graceful handling of auto-detection exceptions."""
        # Mock an object that will cause an exception during shape detection
        bad_onnx_model = Mock()
        bad_onnx_model.path = None
        bad_onnx_model.model_path = None
        del bad_onnx_model.model  # Remove model attribute to cause AttributeError
        
        # Should fall back to defaults without raising
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            onnx_model=bad_onnx_model
        )
        
        assert tokenizer.fixed_batch_size == 1
        assert tokenizer.fixed_sequence_length == 128
    
    def test_parse_from_ort_session_various_paths(self, sample_tokenizer):
        """Test parsing from ORTModel session with various attribute paths."""
        # Test with model.get_inputs
        mock_ort_1 = Mock()
        mock_ort_1.path = None
        mock_ort_1.model_path = None
        mock_input = Mock()
        mock_input.name = "input_ids"
        mock_input.shape = [1, 256]
        mock_ort_1.model.get_inputs.return_value = [mock_input]
        
        tokenizer1 = ONNXTokenizer(tokenizer=sample_tokenizer, onnx_model=mock_ort_1)
        assert tokenizer1.fixed_sequence_length == 256
        
        # Test with session.get_inputs
        mock_ort_2 = Mock()
        mock_ort_2.path = None
        mock_ort_2.model_path = None
        del mock_ort_2.model
        mock_ort_2.session.get_inputs.return_value = [mock_input]
        
        tokenizer2 = ONNXTokenizer(tokenizer=sample_tokenizer, onnx_model=mock_ort_2)
        assert tokenizer2.fixed_sequence_length == 256
    
    def test_empty_string_handling(self, sample_tokenizer):
        """Test handling of empty strings in input."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=64
        )
        
        # Test with empty string
        result = tokenizer(["Hello", ""])
        assert result["input_ids"].shape == (2, 64)
        assert result["attention_mask"].shape == (2, 64)


@pytest.mark.unit
class TestCreateAutoShapeTokenizer:
    """Test the create_auto_shape_tokenizer function."""
    
    def test_create_auto_shape_tokenizer(self, sample_tokenizer, sample_onnx_model):
        """Test creating tokenizer with auto-detected shapes."""
        tokenizer = create_auto_shape_tokenizer(sample_tokenizer, sample_onnx_model)
        
        assert isinstance(tokenizer, ONNXTokenizer)
        assert tokenizer.fixed_batch_size == 1
        assert tokenizer.fixed_sequence_length == 128
        assert tokenizer.tokenizer == sample_tokenizer
    
    def test_create_with_ort_model(self, sample_tokenizer):
        """Test creating tokenizer with ORTModel."""
        mock_ort_model = Mock()
        mock_ort_model.path = Path("/fake/model.onnx")
        
        with patch('onnx_tokenizer.parse_onnx_input_shapes') as mock_parse:
            mock_parse.return_value = {"input_ids": [4, 512]}
            
            tokenizer = create_auto_shape_tokenizer(sample_tokenizer, mock_ort_model)
            
            assert isinstance(tokenizer, ONNXTokenizer)
            assert tokenizer.fixed_batch_size == 4
            assert tokenizer.fixed_sequence_length == 512


@pytest.mark.integration
class TestONNXTokenizerIntegration:
    """Integration tests for ONNXTokenizer with real-like scenarios."""
    
    def test_end_to_end_workflow(self, mock_onnx_model_with_file, sample_texts):
        """Test complete end-to-end workflow."""
        from transformers import AutoTokenizer
        
        # Load tokenizer (might fail in test environment, use mock if needed)
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(str(mock_onnx_model_with_file))
        except Exception:
            base_tokenizer = Mock()
            base_tokenizer.pad_token_id = 0
            base_tokenizer.vocab_size = 1000
            base_tokenizer.return_value = {
                "input_ids": torch.zeros((len(sample_texts), 128), dtype=torch.long),
                "attention_mask": torch.ones((len(sample_texts), 128), dtype=torch.long)
            }
            base_tokenizer.__call__ = lambda self, *args, **kwargs: self.return_value
        
        # Create ONNX tokenizer with auto-detection
        onnx_model_path = mock_onnx_model_with_file / "model.onnx"
        tokenizer = ONNXTokenizer(
            tokenizer=base_tokenizer,
            onnx_model=onnx_model_path
        )
        
        # Test processing various inputs
        for text in sample_texts:
            if text:  # Skip empty string for this test
                result = tokenizer(text)
                assert result["input_ids"].shape == (1, 128)
                assert result["attention_mask"].shape == (1, 128)
    
    def test_batch_processing_workflow(self, sample_tokenizer, sample_onnx_model):
        """Test batch processing workflow."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            onnx_model=sample_onnx_model
        )
        
        # Process different batch sizes
        batch_sizes = [1, 3, 5]
        for batch_size in batch_sizes:
            texts = [f"Text {i}" for i in range(batch_size)]
            result = tokenizer(texts)
            
            # Should always return fixed batch size (1) due to auto-detection
            assert result["input_ids"].shape[0] == 1  # Auto-detected from model
            assert result["input_ids"].shape[1] == 128  # Auto-detected sequence length


@pytest.mark.smoke
class TestONNXTokenizerSmoke:
    """Smoke tests for basic functionality."""
    
    def test_basic_instantiation(self, sample_tokenizer):
        """Test basic instantiation doesn't crash."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=128
        )
        assert tokenizer is not None
    
    def test_basic_tokenization(self, sample_tokenizer):
        """Test basic tokenization doesn't crash."""
        tokenizer = ONNXTokenizer(
            tokenizer=sample_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=128
        )
        
        result = tokenizer("Hello world!")
        assert result is not None
        assert "input_ids" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])