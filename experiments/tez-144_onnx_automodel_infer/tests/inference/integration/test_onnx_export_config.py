#!/usr/bin/env python3
"""
ONNX export configuration tests.

Tests the UniversalOnnxConfig implementation and ONNX export workflow.

Tests extracted from test_universal_config.py
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import pytest
import torch
import onnx
from transformers import AutoConfig, AutoModel

# Import from the src directory 
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from onnx_config import UniversalOnnxConfig


class TestOnnxExportConfig:
    """Tests for UniversalOnnxConfig implementation and ONNX export."""
    
    def test_task_detection_various_models(self):
        """Test task detection for various model architectures."""
        test_cases = [
            ("prajjwal1/bert-tiny", "feature-extraction"),
            ("distilbert-base-uncased-finetuned-sst-2-english", "text-classification"),
            ("gpt2", "text-generation"),
            ("t5-small", "text2text-generation"),
            ("facebook/bart-base", "text2text-generation"),
        ]
        
        for model_name, expected_task_family in test_cases:
            try:
                config = AutoConfig.from_pretrained(model_name)
                onnx_config = UniversalOnnxConfig(config)
                
                # Check if task family matches expectation
                success = (expected_task_family in onnx_config.task or 
                          onnx_config.task == expected_task_family)
                
                assert success, f"Task detection failed for {model_name}: expected {expected_task_family}, got {onnx_config.task}"
                
            except Exception as e:
                pytest.fail(f"Task detection failed for {model_name}: {e}")
    
    def test_bert_tiny_config_generation(self):
        """Test detailed config generation for BERT-tiny model."""
        model_name = "prajjwal1/bert-tiny"
        
        # Load config and create ONNX config
        config = AutoConfig.from_pretrained(model_name)
        onnx_config = UniversalOnnxConfig(config)
        
        # Test basic properties
        assert onnx_config.task is not None, "Task should be detected"
        assert onnx_config.task_family is not None, "Task family should be detected"
        
        # Test input/output names
        input_names = onnx_config.get_input_names()
        output_names = onnx_config.get_output_names()
        
        assert len(input_names) > 0, "Should have input names"
        assert len(output_names) > 0, "Should have output names"
        assert "input_ids" in input_names, "Should have input_ids input"
        
        # Test dynamic axes
        dynamic_axes = onnx_config.get_dynamic_axes()
        assert isinstance(dynamic_axes, dict), "Dynamic axes should be dict"
        
        # Test dummy input generation
        dummy_inputs = onnx_config.generate_dummy_inputs(
            batch_size=2,
            seq_length=64
        )
        
        assert len(dummy_inputs) > 0, "Should generate dummy inputs"
        assert "input_ids" in dummy_inputs, "Should have input_ids in dummy inputs"
        
        # Verify input shapes
        input_ids = dummy_inputs["input_ids"]
        assert input_ids.shape == (2, 64), f"Unexpected input_ids shape: {input_ids.shape}"
        assert input_ids.dtype == torch.long, f"Unexpected input_ids dtype: {input_ids.dtype}"
    
    def test_onnx_export_with_universal_config(self):
        """Test actual ONNX export using UniversalOnnxConfig."""
        model_name = "prajjwal1/bert-tiny"
        
        # Load model and config
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Create UniversalOnnxConfig
        onnx_config = UniversalOnnxConfig(config)
        
        # Generate dummy inputs
        dummy_inputs = onnx_config.generate_dummy_inputs(
            batch_size=1,
            seq_length=128
        )
        
        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                tmp.name,
                input_names=onnx_config.get_input_names(),
                output_names=onnx_config.get_output_names(),
                dynamic_axes=onnx_config.get_dynamic_axes(),
                opset_version=onnx_config.DEFAULT_ONNX_OPSET,
                do_constant_folding=True,
            )
            
            # Verify the export
            onnx_model = onnx.load(tmp.name)
            onnx.checker.check_model(onnx_model)
            
            # Check file size
            file_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
            assert file_size_mb > 0.1, f"ONNX file too small: {file_size_mb:.2f} MB"
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_config_compatibility_multiple_models(self):
        """Test UniversalOnnxConfig with multiple model types."""
        test_models = [
            "prajjwal1/bert-tiny",  # BERT encoder
            "gpt2",                 # GPT decoder  
            "t5-small",             # T5 encoder-decoder
        ]
        
        results = {}
        for model_name in test_models:
            try:
                # Load config
                config = AutoConfig.from_pretrained(model_name)
                
                # Create UniversalOnnxConfig
                onnx_config = UniversalOnnxConfig(config)
                
                # Basic validation
                assert onnx_config.task is not None
                assert onnx_config.task_family is not None
                
                # Input/output validation
                input_names = onnx_config.get_input_names()
                output_names = onnx_config.get_output_names()
                assert len(input_names) > 0
                assert len(output_names) > 0
                
                # Dynamic axes validation
                dynamic_axes = onnx_config.get_dynamic_axes()
                assert isinstance(dynamic_axes, dict)
                
                # Dummy input generation
                dummy_inputs = onnx_config.generate_dummy_inputs(
                    batch_size=2,
                    seq_length=64
                )
                assert len(dummy_inputs) > 0
                
                results[model_name] = True
                
            except Exception as e:
                results[model_name] = False
                pytest.fail(f"Config generation failed for {model_name}: {e}")
        
        # All models should pass
        assert all(results.values()), f"Some models failed: {results}"
    
    def test_dummy_input_generation_shapes(self):
        """Test dummy input generation with various shapes."""
        model_name = "prajjwal1/bert-tiny"
        config = AutoConfig.from_pretrained(model_name)
        onnx_config = UniversalOnnxConfig(config)
        
        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 16),
            (2, 32),
            (4, 64),
            (8, 128),
        ]
        
        for batch_size, seq_length in test_cases:
            dummy_inputs = onnx_config.generate_dummy_inputs(
                batch_size=batch_size,
                seq_length=seq_length
            )
            
            # Verify input_ids shape
            assert "input_ids" in dummy_inputs
            input_ids = dummy_inputs["input_ids"]
            assert input_ids.shape == (batch_size, seq_length), \
                f"Wrong shape for batch_size={batch_size}, seq_length={seq_length}: {input_ids.shape}"
            
            # Check other common inputs
            if "attention_mask" in dummy_inputs:
                attention_mask = dummy_inputs["attention_mask"]
                assert attention_mask.shape == (batch_size, seq_length)
            
            if "token_type_ids" in dummy_inputs:
                token_type_ids = dummy_inputs["token_type_ids"]
                assert token_type_ids.shape == (batch_size, seq_length)
    
    def test_dynamic_axes_configuration(self):
        """Test dynamic axes configuration for ONNX export."""
        model_name = "prajjwal1/bert-tiny"
        config = AutoConfig.from_pretrained(model_name)
        onnx_config = UniversalOnnxConfig(config)
        
        dynamic_axes = onnx_config.get_dynamic_axes()
        
        # Should be a dictionary
        assert isinstance(dynamic_axes, dict)
        
        # Should have entries for inputs
        input_names = onnx_config.get_input_names()
        for input_name in input_names:
            if input_name in dynamic_axes:
                axes = dynamic_axes[input_name]
                assert isinstance(axes, dict), f"Dynamic axes for {input_name} should be dict"
                
                # Common dynamic axes for text models
                expected_dynamic_dims = {0: "batch_size", 1: "sequence_length"}
                for dim, name in expected_dynamic_dims.items():
                    if dim in axes:
                        assert isinstance(axes[dim], str), f"Axis name should be string for {input_name}[{dim}]"
    
    def test_input_output_names_consistency(self):
        """Test that input/output names are consistent and valid."""
        model_name = "prajjwal1/bert-tiny"
        config = AutoConfig.from_pretrained(model_name)
        onnx_config = UniversalOnnxConfig(config)
        
        input_names = onnx_config.get_input_names()
        output_names = onnx_config.get_output_names()
        
        # Check input names
        assert len(input_names) > 0, "Should have at least one input"
        for name in input_names:
            assert isinstance(name, str), f"Input name should be string: {name}"
            assert len(name) > 0, f"Input name should not be empty: {name}"
        
        # Check output names  
        assert len(output_names) > 0, "Should have at least one output"
        for name in output_names:
            assert isinstance(name, str), f"Output name should be string: {name}"
            assert len(name) > 0, f"Output name should not be empty: {name}"
        
        # No duplicates
        assert len(set(input_names)) == len(input_names), "Input names should be unique"
        assert len(set(output_names)) == len(output_names), "Output names should be unique"
        
        # No overlap between inputs and outputs
        overlap = set(input_names) & set(output_names)
        assert len(overlap) == 0, f"Input and output names should not overlap: {overlap}"