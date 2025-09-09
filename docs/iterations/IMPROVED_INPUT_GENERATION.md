# Improved Input Generation System

## Overview

This document describes the enhanced input generation system for generating optimal input tensors based on model architectures. The system intelligently determines appropriate input dimensions without hardcoding specific model names, following universal design principles.

## Problem Solved

Previously, the system used Optimum's `DEFAULT_DUMMY_SHAPES` which had suboptimal dimensions:
- **BERT models**: `[2, 16]` → should be `[1, 128]` for realistic sequence lengths
- **ResNet models**: `[2, 3, 64, 64]` → should be `[1, 3, 224, 224]` for ImageNet standard
- **ViT models**: Already good at `[2, 3, 224, 224]` but batch size should be 1
- **CLIP models**: Needed proper text length (77) and vision dimensions

## Solution Architecture

### Universal Design Principles

1. **Architecture-Agnostic**: Uses model config attributes (`image_size`, `max_position_embeddings`) instead of hardcoded model names
2. **Domain-Based Logic**: Classifies models as text, vision, or multimodal based on `model_type`
3. **Config-Driven**: Leverages HuggingFace model configurations for optimal dimensions
4. **Extensible**: Easily supports new model types without code changes
5. **Backward Compatible**: User overrides still work via `shape_kwargs`

### Implementation

The solution adds intelligent shape optimization between Optimum's default shapes and user overrides:

```python
# Original flow:
shapes = DEFAULT_DUMMY_SHAPES.copy()
shapes.update(shape_kwargs)  # User overrides

# Enhanced flow:  
shapes = DEFAULT_DUMMY_SHAPES.copy()
shapes = _optimize_input_shapes_for_model(...)  # Intelligent optimization
shapes.update(shape_kwargs)  # User overrides (still take precedence)
```

### Key Functions

1. **`_optimize_input_shapes_for_model()`**: Main optimization function
2. **`_apply_universal_shape_optimizations()`**: Domain-specific logic
3. **`_is_text_model()`, `_is_vision_model()`, `_is_multimodal_model()`**: Domain classification
4. **`_optimize_text_model_shapes()`**: Text model optimizations
5. **`_optimize_vision_model_shapes()`**: Vision model optimizations  
6. **`_optimize_multimodal_model_shapes()`**: Multimodal model optimizations

## Results

### Before vs After Comparison

| Model Type | Before | After | Improvement |
|------------|--------|-------|-------------|
| BERT | `input_ids=[2, 16]` | `input_ids=[1, 128]` | Realistic sequence length |
| ResNet | `pixel_values=[2, 3, 64, 64]` | `pixel_values=[1, 3, 224, 224]` | ImageNet standard |
| ViT | `pixel_values=[2, 3, 224, 224]` | `pixel_values=[1, 3, 224, 224]` | Better batch size |
| CLIP | N/A | `input_ids=[1, 77], pixel_values=[1, 3, 224, 224]` | Proper multimodal dimensions |

### Key Improvements

- **✅ Universal batch size**: All models use `batch_size=1` (better for ONNX export)
- **✅ Config-driven dimensions**: Uses `config.image_size`, `config.max_position_embeddings`
- **✅ Domain-specific defaults**: ImageNet 224x224, BERT 128 sequence length, CLIP 77 text length
- **✅ No hardcoded model names**: Follows CARDINAL RULE #1
- **✅ Extensible design**: New model types supported automatically

## Testing

Comprehensive test suite validates:

1. **CARDINAL RULE compliance**: No hardcoded model names
2. **Universal design**: Config-based decisions, domain classification
3. **Functional correctness**: Proper dimensions for each model type
4. **Backward compatibility**: User overrides still work
5. **Error handling**: Graceful fallback on failures
6. **Extensibility**: New model types handled gracefully

## Usage

The improvements are automatic and transparent:

```python
# Automatic optimization based on model config
inputs = generate_dummy_inputs('prajjwal1/bert-tiny')
# Returns: input_ids=[1, 128], attention_mask=[1, 128], token_type_ids=[1, 128]

inputs = generate_dummy_inputs('microsoft/resnet-18') 
# Returns: pixel_values=[1, 3, 224, 224]

# User overrides still work
inputs = generate_dummy_inputs('prajjwal1/bert-tiny', batch_size=4, sequence_length=256)
# Returns: input_ids=[4, 256], attention_mask=[4, 256], token_type_ids=[4, 256]
```

## Files Modified

1. **`modelexport/core/model_input_generator.py`**: Main implementation
2. **`tests/test_improved_input_generation.py`**: Comprehensive test suite

## Compliance

- **✅ CARDINAL RULE #1**: No hardcoded logic - uses universal `nn.Module` hierarchy and config attributes
- **✅ CARDINAL RULE #2**: All testing via pytest - comprehensive test suite added
- **✅ Universal Design**: Works with ANY model architecture automatically
- **✅ Maintainable**: Clean, documented, extensible code
- **✅ Backward Compatible**: Existing functionality preserved

## Future Extensions

The system can be easily extended for new model architectures by:

1. Adding new model types to domain classification functions
2. Adding config attribute mappings for new dimension types
3. Adding domain-specific optimization logic
4. All without hardcoding model names or breaking existing functionality

This design ensures the system will work with future HuggingFace models automatically while providing optimal input dimensions for better ONNX export quality.