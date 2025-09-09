#!/usr/bin/env python3
"""
Test cases for improved input generation system.

This test suite validates that the enhanced input generation system:
1. Follows CARDINAL RULE #1: No hardcoded logic
2. Uses universal design principles
3. Generates better input dimensions than defaults
4. Maintains backward compatibility
"""

import pytest
from modelexport.core.model_input_generator import (
    generate_dummy_inputs,
    _optimize_input_shapes_for_model,
)
# Import the hacky functions from the new module
# FIXME: These imports are from the temporary hacky module
from modelexport.core.input_shape_optimizer_hack import (
    _is_text_model_hack as _is_text_model,
    _is_vision_model_hack as _is_vision_model,
    _is_multimodal_model_hack as _is_multimodal_model,
)
from transformers import AutoConfig
from optimum.utils import DEFAULT_DUMMY_SHAPES


class TestImprovedInputGeneration:
    """Test the improved input generation system."""

    def test_cardinal_rule_no_hardcoded_models(self):
        """
        CARDINAL RULE #1: Test that no model names are hardcoded.
        
        The system should work by analyzing model config and type,
        not by matching against hardcoded model name strings.
        """
        # Test that our functions work with any model name
        # as long as the config and model_type are correct
        
        models_to_test = [
            ('prajjwal1/bert-tiny', 'bert', 'text'),
            ('microsoft/resnet-18', 'resnet', 'vision'),
            ('google/vit-base-patch16-224', 'vit', 'vision'),
            ('openai/clip-vit-base-patch32', 'clip', 'multimodal'),
        ]
        
        for model_name, expected_type, expected_domain in models_to_test:
            config = AutoConfig.from_pretrained(model_name)
            
            # Test domain classification functions
            if expected_domain == 'text':
                assert _is_text_model(expected_type, config)
                assert not _is_vision_model(expected_type, config)
                assert not _is_multimodal_model(expected_type, config)
            elif expected_domain == 'vision':
                assert not _is_text_model(expected_type, config)
                assert _is_vision_model(expected_type, config)
                assert not _is_multimodal_model(expected_type, config)
            elif expected_domain == 'multimodal':
                assert not _is_text_model(expected_type, config)
                assert not _is_vision_model(expected_type, config)
                assert _is_multimodal_model(expected_type, config)

    def test_bert_input_improvements(self):
        """Test that BERT models get better input dimensions."""
        inputs = generate_dummy_inputs('prajjwal1/bert-tiny')
        
        # Should have proper tensor names
        assert 'input_ids' in inputs
        assert 'attention_mask' in inputs
        assert 'token_type_ids' in inputs
        
        # Should have improved dimensions
        assert inputs['input_ids'].shape[0] == 1  # batch_size = 1
        assert inputs['input_ids'].shape[1] == 128  # sequence_length = 128 (better than 16)
        
        # All text inputs should have same shape
        for tensor in inputs.values():
            assert tensor.shape == (1, 128)

    def test_resnet_input_improvements(self):
        """Test that ResNet models get ImageNet standard dimensions."""
        inputs = generate_dummy_inputs('microsoft/resnet-18')
        
        # Should have proper tensor name
        assert 'pixel_values' in inputs
        
        # Should have ImageNet standard dimensions
        pixel_values = inputs['pixel_values']
        assert pixel_values.shape == (1, 3, 224, 224)  # [batch, channels, height, width]

    def test_vit_input_improvements(self):
        """Test that ViT models get correct dimensions from config."""
        inputs = generate_dummy_inputs('google/vit-base-patch16-224')
        
        # Should have proper tensor name
        assert 'pixel_values' in inputs
        
        # Should use image_size from config (224)
        pixel_values = inputs['pixel_values']
        assert pixel_values.shape == (1, 3, 224, 224)

    def test_clip_multimodal_improvements(self):
        """Test that CLIP gets proper text and vision dimensions."""
        inputs = generate_dummy_inputs('openai/clip-vit-base-patch32')
        
        # Should have both text and vision inputs
        assert 'input_ids' in inputs
        assert 'attention_mask' in inputs
        assert 'pixel_values' in inputs
        
        # Text inputs should have CLIP's standard length (77)
        assert inputs['input_ids'].shape == (1, 77)
        assert inputs['attention_mask'].shape == (1, 77)
        
        # Vision input should be standard ImageNet
        assert inputs['pixel_values'].shape == (1, 3, 224, 224)

    def test_universal_batch_size_optimization(self):
        """Test that all models get batch_size=1 for ONNX export."""
        models = [
            'prajjwal1/bert-tiny',
            'microsoft/resnet-18',
            'google/vit-base-patch16-224',
            'openai/clip-vit-base-patch32'
        ]
        
        for model_name in models:
            inputs = generate_dummy_inputs(model_name)
            
            # All tensors should have batch_size = 1
            for name, tensor in inputs.items():
                assert tensor.shape[0] == 1, f"Model {model_name}, tensor {name} has batch_size {tensor.shape[0]}, expected 1"

    def test_backward_compatibility_with_user_overrides(self):
        """Test that user-provided shape_kwargs still override our optimizations."""
        
        # Test with custom batch size
        inputs = generate_dummy_inputs(
            'prajjwal1/bert-tiny',
            batch_size=3,
            sequence_length=64
        )
        
        # User values should take precedence
        assert inputs['input_ids'].shape[0] == 3  # User override
        assert inputs['input_ids'].shape[1] == 64  # User override

    def test_shape_optimization_function_directly(self):
        """Test the _optimize_input_shapes_for_model function directly."""
        base_shapes = DEFAULT_DUMMY_SHAPES.copy()
        
        # Test BERT optimization
        config = AutoConfig.from_pretrained('prajjwal1/bert-tiny')
        optimized = _optimize_input_shapes_for_model(
            shapes=base_shapes,
            model_name_or_path='prajjwal1/bert-tiny',
            export_config=None,  # Not used in our implementation
            task='feature-extraction'
        )
        
        # Should have better defaults
        assert optimized['batch_size'] == 1  # Better than 2
        assert optimized['sequence_length'] == 128  # Better than 16

    def test_error_handling_graceful_fallback(self):
        """Test that the system gracefully falls back on errors."""
        base_shapes = DEFAULT_DUMMY_SHAPES.copy()
        
        # Test with invalid model path
        result = _optimize_input_shapes_for_model(
            shapes=base_shapes,
            model_name_or_path='nonexistent/model',
            export_config=None,
            task=None
        )
        
        # Should return original shapes on error
        assert result == base_shapes

    def test_extensibility_new_model_types(self):
        """Test that the system can handle new model types gracefully."""
        # The classification functions should handle unknown model types
        unknown_config = type('Config', (), {'model_type': 'unknown_future_model'})()
        
        # Should not crash, should return False for all domain checks
        assert not _is_text_model('unknown_future_model', unknown_config)
        assert not _is_vision_model('unknown_future_model', unknown_config)
        assert not _is_multimodal_model('unknown_future_model', unknown_config)

class TestUniversalDesignCompliance:
    """Test that the solution follows universal design principles."""

    def test_no_hardcoded_model_names_in_code(self):
        """Verify that no model names are hardcoded in the implementation."""
        from modelexport.core import model_input_generator
        from modelexport.core import input_shape_optimizer_hack
        import inspect
        
        # Get source code of our optimization functions
        # Note: Now checking the hacky module where the functions live
        source = inspect.getsource(model_input_generator._optimize_input_shapes_for_model)
        source += inspect.getsource(input_shape_optimizer_hack._apply_universal_shape_optimizations_hack)
        
        # These model names should NOT appear in the source code
        forbidden_names = [
            'prajjwal1/bert-tiny',
            'microsoft/resnet-18',
            'google/vit-base-patch16-224',
            'openai/clip-vit-base-patch32',
            'facebook/sam-vit-base',
            'bert-base-uncased',
            'resnet-50',
        ]
        
        for name in forbidden_names:
            assert name not in source, f"Hardcoded model name '{name}' found in optimization code"

    def test_config_based_decisions(self):
        """Test that decisions are based on config attributes, not model names."""
        
        # Test that image size comes from config.image_size
        config = AutoConfig.from_pretrained('google/vit-base-patch16-224')
        assert hasattr(config, 'image_size')
        assert config.image_size == 224
        
        # Our system should use this value
        base_shapes = {'height': 64, 'width': 64, 'num_channels': 3}
        optimized = _optimize_input_shapes_for_model(
            shapes=base_shapes,
            model_name_or_path='google/vit-base-patch16-224',
            export_config=None,
            task=None
        )
        
        assert optimized['height'] == 224  # From config.image_size
        assert optimized['width'] == 224   # From config.image_size

    def test_domain_classification_logic(self):
        """Test that domain classification uses model_type, not names."""
        
        test_cases = [
            ('bert', True, False, False),
            ('gpt2', True, False, False),
            ('vit', False, True, False),
            ('resnet', False, True, False), 
            ('clip', False, False, True),
            ('unknown_type', False, False, False),
        ]
        
        for model_type, is_text, is_vision, is_multimodal in test_cases:
            fake_config = type('Config', (), {})()
            
            assert _is_text_model(model_type, fake_config) == is_text
            assert _is_vision_model(model_type, fake_config) == is_vision
            assert _is_multimodal_model(model_type, fake_config) == is_multimodal