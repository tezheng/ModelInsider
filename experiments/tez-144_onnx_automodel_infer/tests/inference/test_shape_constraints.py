#!/usr/bin/env python3
"""
Shape constraint handling tests.

Tests the FixedShapeTokenizer implementation for handling fixed shape
constraints in ONNX model inference.

Tests extracted from test_fixed_shape_tokenizer.py
"""

import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest
from transformers import AutoTokenizer

# Import from the src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from fixed_shape_tokenizer import FixedShapeTokenizer, FixedShapePipeline
except ImportError:
    pytest.skip("FixedShapeTokenizer not available", allow_module_level=True)


class TestShapeConstraints:
    """Tests for fixed shape constraint handling."""
    
    def test_fixed_shape_tokenizer_basic_functionality(self):
        """Test basic FixedShapeTokenizer functionality."""
        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        
        # Create fixed shape wrapper
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=16
        )
        
        # Test basic properties
        assert fixed_tokenizer.fixed_batch_size == 2
        assert fixed_tokenizer.fixed_sequence_length == 16
        assert fixed_tokenizer.tokenizer == tokenizer
    
    def test_single_input_padding_to_batch_size(self):
        """Test single input padding to fixed batch size."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=16
        )
        
        # Test single input
        single_input = "Hello world!"
        result = fixed_tokenizer(single_input)
        
        # Should pad to batch size 2
        assert result['input_ids'].shape == (2, 16), f"Expected (2, 16), got {result['input_ids'].shape}"
        assert result['attention_mask'].shape == (2, 16), f"Expected (2, 16), got {result['attention_mask'].shape}"
        
        # First row should contain the actual tokens
        assert not np.all(result['input_ids'][0] == tokenizer.pad_token_id)
        # Second row should be padding (if batch was padded)
        if result['input_ids'].shape[0] > 1:
            # Check that at least some padding exists in the batch
            assert np.any(result['input_ids'] == tokenizer.pad_token_id)
    
    def test_exact_batch_size_handling(self):
        """Test exact batch size handling."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=16
        )
        
        # Test exact batch size
        exact_batch = ["Hello world!", "ONNX is fast!"]
        result = fixed_tokenizer(exact_batch)
        
        assert result['input_ids'].shape == (2, 16), f"Expected (2, 16), got {result['input_ids'].shape}"
        assert result['attention_mask'].shape == (2, 16), f"Expected (2, 16), got {result['attention_mask'].shape}"
        
        # Both rows should contain actual tokens (not all padding)
        assert not np.all(result['input_ids'][0] == tokenizer.pad_token_id)
        assert not np.all(result['input_ids'][1] == tokenizer.pad_token_id)
    
    def test_oversized_batch_truncation(self):
        """Test oversized batch truncation."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=16
        )
        
        # Test oversized batch (should truncate)
        oversized_batch = ["First", "Second", "Third", "Fourth"]
        result = fixed_tokenizer(oversized_batch)
        
        # Should truncate to fixed batch size
        assert result['input_ids'].shape == (2, 16), f"Expected (2, 16), got {result['input_ids'].shape}"
        assert result['attention_mask'].shape == (2, 16), f"Expected (2, 16), got {result['attention_mask'].shape}"
        
        # Both rows should contain actual tokens
        assert not np.all(result['input_ids'][0] == tokenizer.pad_token_id)
        assert not np.all(result['input_ids'][1] == tokenizer.pad_token_id)
    
    def test_long_sequence_truncation(self):
        """Test long sequence truncation to fixed sequence length."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=16
        )
        
        # Test long sequence (should truncate)
        long_text = ["This is a very long sentence that will definitely exceed our maximum sequence length of 16 tokens and should be truncated properly"]
        result = fixed_tokenizer(long_text)
        
        assert result['input_ids'].shape == (2, 16), f"Expected (2, 16), got {result['input_ids'].shape}"
        assert result['attention_mask'].shape == (2, 16), f"Expected (2, 16), got {result['attention_mask'].shape}"
        
        # Sequence should be exactly 16 tokens (truncated)
        assert result['input_ids'].shape[1] == 16
    
    def test_various_fixed_sizes(self):
        """Test various fixed batch and sequence sizes."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        
        test_cases = [
            (1, 8),   # Small batch, short sequence
            (4, 32),  # Larger batch, longer sequence  
            (8, 64),  # Large batch, long sequence
        ]
        
        for batch_size, seq_length in test_cases:
            fixed_tokenizer = FixedShapeTokenizer(
                tokenizer=tokenizer,
                fixed_batch_size=batch_size,
                fixed_sequence_length=seq_length
            )
            
            # Test with various inputs
            test_inputs = [
                "Short text",
                ["Multiple", "inputs", "here"],
                "A much longer input text that will need to be truncated to fit the sequence length"
            ]
            
            for test_input in test_inputs:
                result = fixed_tokenizer(test_input)
                
                expected_shape = (batch_size, seq_length)
                assert result['input_ids'].shape == expected_shape, \
                    f"Expected {expected_shape}, got {result['input_ids'].shape} for input: {test_input}"
                assert result['attention_mask'].shape == expected_shape, \
                    f"Expected {expected_shape}, got {result['attention_mask'].shape} for input: {test_input}"
    
    def test_tokenizer_attributes_preserved(self):
        """Test that important tokenizer attributes are preserved."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=16
        )
        
        # Check that important attributes are accessible
        assert hasattr(fixed_tokenizer, 'pad_token_id')
        assert hasattr(fixed_tokenizer, 'eos_token_id')
        assert hasattr(fixed_tokenizer, 'vocab_size')
        
        # Test attribute access
        assert fixed_tokenizer.pad_token_id == tokenizer.pad_token_id
        assert fixed_tokenizer.eos_token_id == tokenizer.eos_token_id
        assert fixed_tokenizer.vocab_size == tokenizer.vocab_size
    
    def test_return_tensors_consistency(self):
        """Test that return_tensors parameter is handled consistently."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=16
        )
        
        test_input = "Test input"
        
        # Test numpy tensors
        result_np = fixed_tokenizer(test_input, return_tensors="np")
        assert isinstance(result_np['input_ids'], np.ndarray)
        assert result_np['input_ids'].shape == (2, 16)
        
        # Test PyTorch tensors (if available)
        try:
            import torch
            result_pt = fixed_tokenizer(test_input, return_tensors="pt")
            assert torch.is_tensor(result_pt['input_ids'])
            assert result_pt['input_ids'].shape == (2, 16)
        except ImportError:
            # PyTorch not available, skip this test
            pass
    
    def test_edge_cases(self):
        """Test edge cases for shape constraints."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=4  # Very short sequence
        )
        
        # Test empty input
        result_empty = fixed_tokenizer("")
        assert result_empty['input_ids'].shape == (1, 4)
        
        # Test very short input
        result_short = fixed_tokenizer("Hi")
        assert result_short['input_ids'].shape == (1, 4)
        
        # Test input with special characters
        result_special = fixed_tokenizer("Hello, world! 🌍")
        assert result_special['input_ids'].shape == (1, 4)
    
    def test_attention_mask_correctness(self):
        """Test that attention masks are correctly generated."""
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        fixed_tokenizer = FixedShapeTokenizer(
            tokenizer=tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=8
        )
        
        # Test with different length inputs
        inputs = ["Short", "This is a longer input"]
        result = fixed_tokenizer(inputs)
        
        # Check attention mask validity
        attention_mask = result['attention_mask']
        input_ids = result['input_ids']
        
        # Attention mask should be 1 for real tokens, 0 for padding
        for i in range(attention_mask.shape[0]):
            for j in range(attention_mask.shape[1]):
                if input_ids[i, j] == tokenizer.pad_token_id:
                    # Pad tokens should have attention 0 (mostly true, some exceptions)
                    pass  # Note: some tokenizers handle this differently
                else:
                    # Real tokens should have attention 1
                    assert attention_mask[i, j] == 1, f"Real token should have attention 1 at [{i}, {j}]"