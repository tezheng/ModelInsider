# ONNX Auto Processor Testing Guide

**Status**: Production Testing Framework  
**Project**: ModelExport - Inference Module  

## Overview

This document provides comprehensive testing strategies for the ONNXAutoProcessor system, focusing on validation of auto-detection, processor wrapping, and performance optimization capabilities.

## Quick Test Reference

| Test Type | Purpose | Duration | When to Run | Pass Criteria |
|-----------|---------|----------|-------------|---------------|
| **🚬 Smoke Tests** | Basic functionality check | 5 min | Every commit | 100% pass |
| **✅ Sanity Tests** | Core feature validation | 15 min | Before PR merge | 100% pass |
| **🔧 Unit Tests** | Component isolation | 30 min | During development | 95% pass |
| **🔄 Integration Tests** | End-to-end workflows | 45 min | Before release | 98% pass |
| **⚡ Performance Tests** | Speed/memory benchmarks | 20 min | Weekly/Release | Meet targets |

## Test Execution Commands

```bash
# Quick validation (5 min)
pytest -m smoke -v

# Core features (20 min total)
pytest -m "smoke or sanity" -v

# Full development suite
pytest -m "not slow" -v

# Integration tests
pytest -m integration -v

# Performance benchmarks
pytest -m performance -v

# CI/CD pipeline
pytest -m smoke --exitfirst        # Fail fast on smoke tests
pytest -m sanity --exitfirst       # Then sanity tests
pytest -m "not slow"               # Comprehensive but fast
```

## Pytest Configuration

```toml
# In pyproject.toml
[tool.pytest.ini_options]
markers = [
    "smoke: Core functionality tests that must always pass",
    "sanity: Sanity check tests for basic functionality", 
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (may require external services)",
    "performance: Speed and memory benchmark tests",
    "slow: Tests that take > 1 second to run",
    "multimodal: Tests specifically for multimodal models",
    "requires_gpu: Tests requiring GPU/CUDA",
    "requires_models: Tests that need pre-downloaded models"
]
```

## Test Implementation Strategy

### Comprehensive Test Class Structure

```python
import pytest
from modelexport.inference import ONNXAutoProcessor

class TestONNXAutoProcessor:
    """Complete test suite for ONNXAutoProcessor with pytest markers."""
    
    # === SMOKE TESTS (5 min) ===
    
    @pytest.mark.smoke
    def test_import_onnx_auto_processor(self):
        """Verify ONNXAutoProcessor can be imported."""
        assert ONNXAutoProcessor is not None
        
    @pytest.mark.smoke
    def test_create_processor_from_bert_onnx(self):
        """Test basic BERT processor creation."""
        processor = ONNXAutoProcessor.from_model("test_models/bert_tiny.onnx")
        assert processor is not None
        assert hasattr(processor, '_onnx_processor')
        
    @pytest.mark.smoke
    def test_process_simple_text_input(self):
        """Test processing simple text input."""
        processor = ONNXAutoProcessor.from_model("test_models/bert_tiny.onnx")
        output = processor("Hello world")
        assert 'input_ids' in output
        
    @pytest.mark.smoke
    @pytest.mark.multimodal
    def test_multimodal_detection_clip(self):
        """Test CLIP multimodal detection."""
        processor = ONNXAutoProcessor.from_model("test_models/clip_tiny.onnx")
        assert processor.onnx_config.get('is_multimodal') == True
```

### Sanity Tests (15 minutes)

```python
    # === SANITY TESTS ===
    
    @pytest.mark.sanity
    def test_all_five_processor_types(self):
        """Test all 5 processor types can be created."""
        test_models = {
            "text": "bert_tiny.onnx",
            "image": "vit_tiny.onnx", 
            "audio": "wav2vec2_tiny.onnx",
            "video": "videomae_tiny.onnx",
            "multimodal": "clip_tiny.onnx"
        }
        
        for modality, model_path in test_models.items():
            processor = ONNXAutoProcessor.from_model(f"test_models/{model_path}")
            assert processor is not None
            print(f"✓ {modality} processor created successfully")
    
    @pytest.mark.sanity
    def test_processor_wrapping_correct(self):
        """Test processors wrapped with correct ONNX wrapper."""
        from transformers import AutoTokenizer
        
        base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        processor = ONNXAutoProcessor.from_model(
            "test_models/bert.onnx",
            base_processor=base_tokenizer
        )
        assert hasattr(processor._onnx_processor, 'batch_size')
        assert hasattr(processor._onnx_processor, 'sequence_length')
    
    @pytest.mark.sanity
    def test_fixed_shape_enforcement(self):
        """Test fixed shapes are enforced correctly."""
        processor = ONNXAutoProcessor.from_model("test_models/bert.onnx")
        
        # Process varying length inputs
        output1 = processor("Hi")
        output2 = processor("This is a much longer sentence for padding test")
        
        # Both should have same shape
        assert output1['input_ids'].shape == output2['input_ids'].shape
```

### Unit Tests (30 minutes)

```python
    # === UNIT TESTS ===
    
    @pytest.mark.unit
    def test_extract_onnx_info_text_modality(self):
        """Test _extract_onnx_info for text inputs."""
        mock_model = create_mock_onnx_model(
            inputs=[("input_ids", [1, 128], "int64"),
                   ("attention_mask", [1, 128], "int64")]
        )
        info = ONNXAutoProcessor._extract_onnx_info(mock_model)
        
        assert info['modalities']['text']['batch_size'] == 1
        assert info['modalities']['text']['sequence_length'] == 128
        assert len(info['modalities']['text']['tensors']) == 2
    
    @pytest.mark.unit
    @pytest.mark.multimodal
    def test_extract_onnx_info_multimodal(self):
        """Test _extract_onnx_info for CLIP-like model."""
        mock_model = create_mock_onnx_model(
            inputs=[("input_ids", [1, 77], "int64"),
                   ("pixel_values", [1, 3, 224, 224], "float32")]
        )
        info = ONNXAutoProcessor._extract_onnx_info(mock_model)
        
        assert info['is_multimodal'] == True
        assert 'text' in info['modalities']
        assert 'image' in info['modalities']
    
    @pytest.mark.unit
    def test_detect_modality_by_name(self):
        """Test modality detection by tensor name."""
        test_cases = [
            ("input_ids", "text"),
            ("pixel_values", "image"),
            ("input_values", "audio"),
            ("video_frames", "video"),
            ("unknown_tensor", "unknown")
        ]
        
        for tensor_name, expected_modality in test_cases:
            detected = detect_modality_from_name(tensor_name)
            assert detected == expected_modality
    
    @pytest.mark.unit
    def test_detect_modality_by_shape(self):
        """Test modality detection by tensor shape."""
        test_cases = [
            ([1, 128], "text"),           # 2D likely text
            ([1, 3, 224, 224], "image"),  # 4D NCHW
            ([1, 16000], "audio"),        # 2D waveform
            ([1, 3, 16, 224, 224], "video"), # 5D NCTHW
        ]
        
        for shape, expected_modality in test_cases:
            detected = detect_modality_from_shape(shape)
            assert detected == expected_modality
```

## Test Categories Detail

### 1. Modality Detection Tests

```python
class TestModalityDetection:
    """Test processor type detection across all modalities."""
    
    @pytest.mark.unit
    def test_text_processor_detection(self):
        """Test BERT-like text model detection."""
        # Input: BERT with input_ids, attention_mask, token_type_ids
        # Expected: text modality, batch_size=1, sequence_length=128
        
    @pytest.mark.unit 
    def test_image_processor_detection(self):
        """Test ViT-like image model detection."""
        # Input: ViT with pixel_values [1, 3, 224, 224]
        # Expected: image modality, height=224, width=224, channels=3
        
    @pytest.mark.unit
    def test_audio_processor_detection(self):
        """Test Wav2Vec2-like audio model detection."""
        # Input: Audio with input_values [1, 16000]
        # Expected: audio modality, sequence_length=16000
        
    @pytest.mark.unit
    def test_video_processor_detection(self):
        """Test VideoMAE-like video model detection."""
        # Input: Video with frames [1, 3, 16, 224, 224]
        # Expected: video modality, num_frames=16
        
    @pytest.mark.multimodal
    def test_multimodal_detection(self):
        """Test CLIP-like multimodal detection."""
        # Input: Text + image tensors
        # Expected: is_multimodal=True, both modalities detected
```

### 2. Configuration Loading Tests

```python
class TestConfigurationLoading:
    """Test configuration hierarchy and loading."""
    
    def test_metadata_loading_priority(self):
        """Test metadata loading hierarchy."""
        # Priority: ONNX metadata > JSON file > auto-detection
        
    def test_hf_config_preservation(self):
        """Test HuggingFace config preservation."""
        # Processor configs always from original HF model
        
    def test_shape_extraction_from_onnx(self):
        """Test shape extraction from ONNX model."""
        # Always extract shapes from actual ONNX graph
```

### 3. Error Handling Tests

```python
class TestErrorHandling:
    """Test error scenarios and recovery."""
    
    def test_missing_processor_configs(self):
        """Test missing HF configuration files."""
        # Scenarios: No tokenizer_config.json, no preprocessor_config.json
        
    def test_corrupted_onnx_model(self):
        """Test corrupted ONNX file handling."""
        # Invalid format, missing graph info, corrupted tensors
        
    def test_unsupported_model_types(self):
        """Test unknown model type handling."""
        # Custom models with no recognizable patterns
        
    def test_fallback_mechanisms(self):
        """Test fallback strategies."""
        # Auto-detection when metadata missing
```

### 4. Integration Tests

```python
class TestEndToEnd:
    """Test complete processor creation workflows."""
    
    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.parametrize("model_name", [
        "bert-base-uncased",
        "openai/clip-vit-base-patch32", 
        "facebook/wav2vec2-base",
        "MCG-NJU/videomae-base"
    ])
    def test_popular_models(self, model_name):
        """Test with popular HuggingFace models."""
        # Export to ONNX → Create processor → Validate → Run inference
        
    @pytest.mark.integration
    def test_pipeline_integration(self):
        """Test enhanced pipeline integration."""
        # Create processor → Use with pipeline → Validate outputs
        
    @pytest.mark.integration
    @pytest.mark.multimodal
    def test_multimodal_pipeline_integration(self):
        """Test multimodal pipeline workflows."""
        # CLIP-like models with text and image inputs
```

### 5. Performance Tests

```python
class TestPerformance:
    """Test performance characteristics and benchmarks."""
    
    @pytest.mark.performance
    def test_processor_creation_speed(self):
        """Benchmark processor instantiation time."""
        # Target: < 100ms for standard models
        
    @pytest.mark.performance
    def test_preprocessing_speedup(self):
        """Benchmark preprocessing performance."""
        # Compare: HuggingFace dynamic vs ONNX fixed shapes
        # Target: 40x+ speedup for batch processing
        
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory consumption patterns."""
        # No memory leaks, reasonable footprint
        
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_model_handling(self):
        """Test with large models (>1GB)."""
        # Efficient metadata extraction for large models
```

## Test Data Requirements

### Required Test Models

```python
TEST_MODELS = {
    "text": {
        "bert": "bert-base-uncased",
        "gpt2": "gpt2",
        "t5": "t5-small"
    },
    "vision": {
        "vit": "google/vit-base-patch16-224",
        "resnet": "microsoft/resnet-50",
        "detr": "facebook/detr-resnet-50"
    },
    "audio": {
        "wav2vec2": "facebook/wav2vec2-base", 
        "whisper": "openai/whisper-base"
    },
    "video": {
        "videomae": "MCG-NJU/videomae-base",
        "timesformer": "facebook/timesformer-base"
    },
    "multimodal": {
        "clip": "openai/clip-vit-base-patch32",
        "layoutlm": "microsoft/layoutlm-base-uncased"
    }
}
```

### Test Fixtures

```python
@pytest.fixture
def bert_onnx_model():
    """Fixture providing BERT ONNX model."""
    return create_test_onnx_model("bert")

@pytest.fixture
def clip_onnx_model():
    """Fixture providing CLIP ONNX model."""
    return create_test_onnx_model("clip")

@pytest.fixture
def mock_processors():
    """Fixture providing mock processors for testing."""
    return {
        "tokenizer": create_mock_tokenizer(),
        "image_processor": create_mock_image_processor(),
        "feature_extractor": create_mock_feature_extractor()
    }
```

## Compatibility Testing

### Version Matrix Testing

```python
@pytest.mark.parametrize("transformers_version", [
    "4.30.0", "4.35.0", "4.40.0", "4.45.0"
])
def test_transformers_versions(self, transformers_version):
    """Test compatibility across transformers versions."""
    
@pytest.mark.parametrize("onnx_version", [
    "1.14.0", "1.15.0", "1.16.0"
])
def test_onnx_versions(self, onnx_version):
    """Test compatibility across ONNX versions."""
```

### Export Tool Compatibility

```python
def test_optimum_exported_models(self):
    """Test models exported with Optimum."""
    
def test_torch_onnx_exported_models(self):
    """Test models exported with torch.onnx."""
    
def test_modelexport_htp_models(self):
    """Test models exported with ModelExport HTP."""
```

## Performance Baselines

### Speed Requirements
- **Processor Creation**: < 100ms for standard models
- **Text Processing**: 40x+ speedup vs HuggingFace dynamic
- **Image Processing**: 25x+ speedup vs standard preprocessing
- **Audio Processing**: 30x+ speedup vs feature extraction

### Memory Requirements
- **No Memory Leaks**: Stable memory usage across iterations
- **Reasonable Footprint**: < 2x base model memory usage
- **Efficient Caching**: Smart metadata caching strategies

## Coverage Requirements

- **Line Coverage**: Minimum 90%
- **Branch Coverage**: Minimum 85%
- **Critical Path Coverage**: 100% for error handling
- **Modality Coverage**: All 5 processor types tested
- **Integration Coverage**: End-to-end workflows verified

## CI/CD Integration

```yaml
# .github/workflows/test_inference.yml
name: Inference Module Tests

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
        
    steps:
      - name: Run smoke tests
        run: pytest -m smoke --exitfirst
        
      - name: Run sanity tests  
        run: pytest -m sanity --exitfirst
        
      - name: Run comprehensive tests
        run: pytest -m "not slow" --cov=modelexport.inference
        
      - name: Run performance tests
        run: pytest -m performance
```

## Test Maintenance

### Regular Maintenance Schedule
- **Weekly**: Run full test suite with latest models
- **Per PR**: Run smoke + sanity tests for quick feedback
- **Monthly**: Update test models to latest HuggingFace versions
- **Quarterly**: Review test coverage and update requirements

### Test Data Management
- **Model Caching**: Cache downloaded models for faster testing
- **Version Pinning**: Pin model versions for reproducible tests
- **Size Optimization**: Use tiny/small models for unit tests
- **Storage Management**: Clean up large test artifacts regularly

This comprehensive testing strategy ensures the ONNXAutoProcessor system is robust, performant, and compatible across the entire HuggingFace ecosystem while maintaining the promised 40x+ performance improvements.