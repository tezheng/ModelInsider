# ONNXAutoProcessor Test Design Document

## Table of Contents

- [Quick Test Type Reference](#quick-test-type-reference)
  - [Test Execution with Pytest Markers](#test-execution-with-pytest-markers)
  - [Pytest Configuration](#pytest-configuration-pytestini)
- [Test Strategy Overview](#test-strategy-overview)
  - [Testing Philosophy](#testing-philosophy)
  - [Test Categories](#test-categories)
- [Test Classification](#test-classification)
  - [🚬 Smoke Tests (Quick Validation - 5 minutes)](#-smoke-tests-quick-validation---5-minutes)
  - [✅ Sanity Tests (Core Features - 15 minutes)](#-sanity-tests-core-features---15-minutes)
  - [🔧 Unit Tests (Component Testing - 30 minutes)](#-unit-tests-component-testing---30-minutes)
  - [🔄 Integration Tests (End-to-End - 45 minutes)](#-integration-tests-end-to-end---45-minutes)
  - [⚡ Performance Tests (Benchmarks - 20 minutes)](#-performance-tests-benchmarks---20-minutes)
- [Detailed Test Specifications](#detailed-test-specifications)
  - [1. Metadata Extraction Tests](#1-metadata-extraction-tests)
  - [2. Processor Detection Tests](#2-processor-detection-tests)
  - [3. Processor Creation Tests](#3-processor-creation-tests)
  - [4. Error Handling Tests](#4-error-handling-tests)
  - [5. Multimodal Tests](#5-multimodal-tests)
  - [6. Performance Benchmarks](#6-performance-benchmarks)
  - [7. Backward Compatibility Tests](#7-backward-compatibility-tests)
  - [8. Edge Case Tests](#8-edge-case-tests)
- [Test Data Requirements](#test-data-requirements)
  - [Mock ONNX Models](#mock-onnx-models)
  - [Sample Inputs](#sample-inputs)
  - [Expected Outputs](#expected-outputs)
- [Test File Organization](#test-file-organization)
- [Test Utilities](#test-utilities)
- [Performance Baseline Requirements](#performance-baseline-requirements)
- [CI/CD Integration](#cicd-integration)
- [Test Coverage Requirements](#test-coverage-requirements)
- [Security Testing Requirements](#security-testing-requirements)
- [Test Documentation](#test-documentation)

## Quick Test Type Reference

| Test Type | Purpose | Time | When to Run | Pass Criteria |
|-----------|---------|------|-------------|---------------|
| **🚬 Smoke Tests** | Basic functionality check | 5 min | Every commit/deploy | 100% pass |
| **✅ Sanity Tests** | Core feature validation | 15 min | Before PR merge | 100% pass |
| **🔧 Unit Tests** | Component isolation testing | 30 min | During development | 95% pass |
| **🔄 Integration Tests** | End-to-end workflows | 45 min | Before release | 98% pass |
| **⚡ Performance Tests** | Speed/memory benchmarks | 20 min | Weekly/Release | Meet targets |

### Test Execution with Pytest Markers

```bash
# Quick validation (5 min)
pytest -m smoke -v

# Core features (20 min total)
pytest -m "smoke or sanity" -v

# Full unit tests (30 min)
pytest -m unit -v

# Integration tests only (45 min)
pytest -m integration -v

# Performance benchmarks (20 min)
pytest -m performance -v

# Everything except slow tests
pytest -m "not slow" -v

# CI/CD pipeline
pytest -m smoke --exitfirst        # Fail fast on smoke tests
pytest -m sanity --exitfirst        # Then sanity
pytest -m "not slow"                # Then comprehensive

# Custom combinations
pytest -m "smoke or (sanity and not slow)" -v
pytest -m "unit and not performance" -v
```

### Pytest Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
# Test markers are defined in the main pyproject.toml
markers = [
    "smoke: Core functionality tests that must always pass",
    "sanity: Sanity check tests for basic functionality",
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (may require external services)",
    "performance: Speed and memory benchmark tests (20 min)",
    "slow: Tests that take > 1 second to run",
    "multimodal: Tests specifically for multimodal models",
    "requires_gpu: Tests requiring GPU/CUDA",
    "requires_models: Tests that need pre-downloaded models"
]
```

**Note**: Configuration is centralized in the root `pyproject.toml` file following modern Python best practices.

## Test Strategy Overview

### Testing Philosophy

- **Comprehensive Coverage**: Test all 5 processor types and multimodal combinations
- **Real-World Scenarios**: Use actual model patterns from HuggingFace
- **Edge Case Focus**: Test boundary conditions and error scenarios
- **Performance Validation**: Ensure 40x+ speedup is maintained
- **Backward Compatibility**: Verify existing code continues to work

### Test Categories

1. **Smoke Tests**: Basic functionality verification (5 min)
2. **Sanity Tests**: Core feature validation (15 min)
3. **Unit Tests**: Individual component testing (30 min)
4. **Integration Tests**: End-to-end processor creation (45 min)
5. **Performance Tests**: Speed and memory benchmarks (20 min)
6. **Compatibility Tests**: HuggingFace version compatibility (60 min)
7. **Security Tests**: Input validation and safety (15 min)

## Test Classification

### 🚬 Smoke Tests (Quick Validation - 5 minutes)

**Purpose**: Verify basic functionality works after deployment/changes. First line of defense.

```python
import pytest

class TestONNXAutoProcessor:
    """All tests for ONNXAutoProcessor in a single file with markers."""
    
    @pytest.mark.smoke
    def test_import_onnx_auto_processor(self):
        """Test that ONNXAutoProcessor can be imported."""
        from onnx_auto_processor import ONNXAutoProcessor
        assert ONNXAutoProcessor is not None
        # Time: <1 second
    
    @pytest.mark.smoke
    def test_create_processor_from_bert_onnx(self):
        """Test basic BERT processor creation."""
        processor = ONNXAutoProcessor.from_model("test_models/bert_tiny.onnx")
        assert processor is not None
        assert hasattr(processor, '_onnx_processor')
        # Time: <5 seconds
    
    @pytest.mark.smoke
    def test_process_simple_text_input(self):
        """Test processing a simple text input."""
        processor = ONNXAutoProcessor.from_model("test_models/bert_tiny.onnx")
        output = processor("Hello world")
        assert 'input_ids' in output
        # Time: <3 seconds
    
    @pytest.mark.smoke
    def test_metadata_extraction_works(self):
        """Test that metadata can be extracted from ONNX."""
        import onnx
        model = onnx.load("test_models/bert_tiny.onnx")
        info = ONNXAutoProcessor._extract_onnx_info(model)
        assert 'modalities' in info
        # Time: <2 seconds
    
    @pytest.mark.smoke
    @pytest.mark.multimodal
    def test_multimodal_detection_clip(self):
        """Test that CLIP is detected as multimodal."""
        processor = ONNXAutoProcessor.from_model("test_models/clip_tiny.onnx")
        assert processor.onnx_config.get('is_multimodal') == True
        # Time: <5 seconds
```

**Total Smoke Test Time**: ~5 minutes
**Pass Criteria**: 100% pass rate required

### ✅ Sanity Tests (Core Features - 15 minutes)

**Purpose**: Validate core functionality works correctly. More thorough than smoke tests.

```python
    # === SANITY TESTS ===
    
    @pytest.mark.sanity
    def test_all_five_processor_types(self):
        """Test that all 5 processor types can be created."""
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
        # Time: <30 seconds
    
    @pytest.mark.sanity
    def test_processor_wrapping_correct(self):
        """Test that processors are wrapped with correct ONNX wrapper."""
        from transformers import AutoTokenizer
        
        # Text processor should be wrapped with ONNXTokenizer
        base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        processor = ONNXAutoProcessor.from_model(
            "test_models/bert.onnx",
            base_processor=base_tokenizer
        )
        assert hasattr(processor._onnx_processor, 'batch_size')
        assert hasattr(processor._onnx_processor, 'sequence_length')
        # Time: <20 seconds
    
    @pytest.mark.sanity
    def test_fixed_shape_enforcement(self):
        """Test that fixed shapes are enforced."""
        processor = ONNXAutoProcessor.from_model("test_models/bert.onnx")
        
        # Process varying length inputs
        output1 = processor("Hi")
        output2 = processor("This is a much longer sentence that should be padded")
        
        # Both should have same shape
        assert output1['input_ids'].shape == output2['input_ids'].shape
        # Time: <10 seconds
    
    @pytest.mark.sanity
    def test_metadata_priority_order(self):
        """Test metadata loading priority: ONNX > JSON > auto-detect."""
        # Create model with metadata
        model_with_metadata = create_test_model_with_metadata()
        processor = ONNXAutoProcessor.from_model(model_with_metadata)
        
        # Verify metadata was used (not auto-detected)
        assert processor.onnx_config['source'] == 'onnx_metadata'
        # Time: <15 seconds
    
    @pytest.mark.sanity
    def test_backward_compatibility(self):
        """Test that existing code patterns still work."""
        # Old style - explicit processor
        from onnx_tokenizer import ONNXTokenizer
        old_processor = ONNXTokenizer(...)
        
        # New style - auto processor
        new_processor = ONNXAutoProcessor.from_model("test_models/bert.onnx")
        
        # Both should work with pipeline
        assert callable(old_processor)
        assert callable(new_processor)
        # Time: <10 seconds
```

**Total Sanity Test Time**: ~15 minutes
**Pass Criteria**: 100% pass rate required

### 🔧 Unit Tests (Component Testing - 30 minutes)

**Purpose**: Test individual methods and components in isolation.

```python
    # === UNIT TESTS ===
    
    @pytest.mark.unit
    def test_extract_onnx_info_text_modality(self):
        """Unit test: _extract_onnx_info for text inputs."""
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
        """Unit test: _extract_onnx_info for CLIP-like model."""
        mock_model = create_mock_onnx_model(
            inputs=[("input_ids", [1, 77], "int64"),
                   ("pixel_values", [1, 3, 224, 224], "float32")]
        )
        info = ONNXAutoProcessor._extract_onnx_info(mock_model)
        
        assert info['is_multimodal'] == True
        assert 'text' in info['modalities']
        assert 'image' in info['modalities']
    
    # === Modality Detection Unit Tests ===
    
    @pytest.mark.unit
    def test_detect_modality_by_name(self):
        """Unit test: Modality detection by tensor name."""
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
        """Unit test: Modality detection by tensor shape."""
        test_cases = [
            ([1, 128], "text"),           # 2D likely text
            ([1, 3, 224, 224], "image"),  # 4D NCHW
            ([1, 16000], "audio"),         # 2D waveform
            ([1, 3, 16, 224, 224], "video"), # 5D NCTHW
        ]
        
        for shape, expected_modality in test_cases:
            detected = detect_modality_from_shape(shape)
            assert detected == expected_modality
    
    # === Configuration Building Unit Tests ===
    
    @pytest.mark.unit
    def test_build_text_config(self):
        """Unit test: Build configuration for text modality."""
        modality_info = {
            "batch_size": 1,
            "sequence_length": 128,
            "tensors": [{"name": "input_ids", "shape": [1, 128]}]
        }
        config = build_processor_config("text", modality_info)
        
        assert config['processor_type'] == 'tokenizer'
        assert config['batch_size'] == 1
        assert config['sequence_length'] == 128
    
    # === Wrapper Creation Unit Tests ===
    
    @pytest.mark.unit
    def test_create_tokenizer_wrapper(self):
        """Unit test: Create ONNXTokenizer wrapper."""
        from transformers import AutoTokenizer
        
        base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config = {"batch_size": 1, "sequence_length": 128}
        
        wrapper = create_onnx_wrapper(base_tokenizer, config)
        assert wrapper is not None
        assert hasattr(wrapper, 'process_with_fixed_shapes')
    
    # === Error Handling Unit Tests ===
    
    @pytest.mark.unit
    def test_handle_missing_metadata(self):
        """Unit test: Handle missing metadata gracefully."""
        mock_model = create_mock_onnx_model(inputs=[])
        info = ONNXAutoProcessor._extract_onnx_info(mock_model)
        
        assert 'modalities' in info
        assert info['input_count'] == 0
    
    @pytest.mark.unit
    def test_validate_shapes(self):
        """Unit test: Shape validation logic."""
        valid_shapes = [[1, 128], [1, 3, 224, 224]]
        invalid_shapes = [[], [-1, -1], None]
        
        for shape in valid_shapes:
            assert validate_shape(shape) == True
        
        for shape in invalid_shapes:
            assert validate_shape(shape) == False
```

**Total Unit Test Time**: ~30 minutes
**Pass Criteria**: 95% pass rate minimum

## Detailed Test Cases

### 1. Core Functionality Tests

#### 1.1 Single Modality Processor Detection

```python
class TestSingleModalityDetection:
    """Test detection and configuration for single modality models."""
    
    def test_text_processor_detection(self):
        """Test BERT-like text model detection."""
        # Test Case: BERT model with input_ids, attention_mask, token_type_ids
        # Expected: Correctly identifies as text modality
        # Validates: batch_size=1, sequence_length=128
        
    def test_image_processor_detection(self):
        """Test ViT-like image model detection."""
        # Test Case: ViT model with pixel_values [1, 3, 224, 224]
        # Expected: Correctly identifies as image modality
        # Validates: batch_size=1, height=224, width=224, channels=3
        
    def test_audio_processor_detection(self):
        """Test Wav2Vec2-like audio model detection."""
        # Test Case: Audio model with input_values [1, 16000]
        # Expected: Correctly identifies as audio modality
        # Validates: batch_size=1, sequence_length=16000
        
    def test_video_processor_detection(self):
        """Test VideoMAE-like video model detection."""
        # Test Case: Video model with frames [1, 3, 16, 224, 224]
        # Expected: Correctly identifies as video modality
        # Validates: batch_size=1, num_frames=16, height=224, width=224
```

#### 1.2 Multimodal Processor Detection

```python
class TestMultimodalDetection:
    """Test detection for multimodal models."""
    
    def test_clip_text_image_detection(self):
        """Test CLIP model with text and image inputs."""
        # Inputs: input_ids[1,77], attention_mask[1,77], pixel_values[1,3,224,224]
        # Expected: is_multimodal=True, modalities=['text', 'image']
        # Validates: Separate metadata for each modality
        
    def test_whisper_audio_text_detection(self):
        """Test Whisper model with audio and text outputs."""
        # Inputs: input_features[1,80,3000], decoder_input_ids[1,448]
        # Expected: is_multimodal=True, modalities=['audio', 'text']
        
    def test_layoutlm_text_image_bbox_detection(self):
        """Test LayoutLM with text, image, and bbox inputs."""
        # Complex multimodal with layout information
        # Expected: Correct handling of 3+ modality types
```

### 2. Metadata Extraction Tests

#### 2.1 ONNX Metadata Parsing

```python
class TestMetadataExtraction:
    """Test metadata extraction from ONNX models."""
    
    def test_metadata_from_model_properties(self):
        """Test reading metadata from ONNX model.metadata_props."""
        # Test Case: Model with processor.* metadata keys
        # Validates: Correct parsing of all metadata fields
        
    def test_metadata_from_companion_json(self):
        """Test loading from companion JSON file."""
        # Test Case: model.onnx + model_metadata.json
        # Validates: JSON loading and merging with ONNX info
        
    def test_metadata_priority_order(self):
        """Test metadata loading hierarchy."""
        # Test Case: Conflicting metadata in multiple sources
        # Expected: ONNX metadata > JSON file > auto-detection
```

#### 2.2 Shape Inference Tests

```python
class TestShapeInference:
    """Test shape extraction and inference."""
    
    def test_dynamic_dimension_handling(self):
        """Test handling of dynamic dimensions (-1)."""
        # Test Case: Model with batch_size=-1
        # Expected: Graceful handling, warning messages
        
    def test_shape_validation(self):
        """Test shape consistency validation."""
        # Test Case: Mismatched shapes across tensors
        # Expected: Error with clear message
        
    def test_unusual_shape_patterns(self):
        """Test non-standard shape configurations."""
        # Test Cases: 
        # - 6D tensors
        # - Single dimension tensors
        # - Empty shapes
```

### 3. Error Handling Tests

#### 3.1 Missing Information Handling

```python
class TestErrorHandling:
    """Test error scenarios and recovery."""
    
    def test_missing_processor_configs(self):
        """Test when HF processor configs are missing."""
        # Scenarios:
        # - No tokenizer_config.json
        # - No preprocessor_config.json
        # - Missing model directory
        
    def test_corrupted_onnx_model(self):
        """Test handling of corrupted ONNX files."""
        # Test Cases:
        # - Invalid ONNX format
        # - Missing graph information
        # - Corrupted tensor definitions
        
    def test_unsupported_model_types(self):
        """Test handling of unknown model types."""
        # Test Case: Custom model with no recognizable patterns
        # Expected: Clear error message with fallback options
```

#### 3.2 Fallback Mechanisms

```python
class TestFallbackMechanisms:
    """Test fallback strategies."""
    
    def test_auto_detection_fallback(self):
        """Test fallback when metadata is missing."""
        # Test Case: No metadata, rely on tensor name patterns
        # Expected: Best-effort detection with warnings
        
    def test_default_configuration_fallback(self):
        """Test using default configs when specific ones missing."""
        # Expected: Reasonable defaults for common cases
```

### 4. Integration Tests

#### 4.1 End-to-End Processor Creation

```python
class TestEndToEnd:
    """Test complete processor creation workflow."""
    
    @pytest.mark.parametrize("model_name", [
        "bert-base-uncased",
        "openai/clip-vit-base-patch32", 
        "facebook/wav2vec2-base",
        "MCG-NJU/videomae-base",
        "whisper-base"
    ])
    def test_popular_models(self, model_name):
        """Test with popular HF models."""
        # Export model to ONNX
        # Create processor with ONNXAutoProcessor
        # Validate processor configuration
        # Run inference test
        
    def test_pipeline_integration(self):
        """Test integration with enhanced pipeline."""
        # Create processor
        # Use with pipeline
        # Validate outputs
```

#### 4.2 Processor Wrapping Tests

```python
class TestProcessorWrapping:
    """Test wrapping of base processors."""
    
    def test_tokenizer_wrapping(self):
        """Test ONNXTokenizer wrapper creation."""
        # Input: HF tokenizer
        # Expected: Wrapped with fixed shapes
        
    def test_image_processor_wrapping(self):
        """Test ONNXImageProcessor wrapper creation."""
        # Input: HF image processor
        # Expected: Wrapped with fixed dimensions
        
    def test_multimodal_processor_wrapping(self):
        """Test ONNXProcessor wrapper for multimodal."""
        # Input: HF ProcessorMixin subclass
        # Expected: Correctly wrapped sub-processors
```

### 5. Performance Tests

#### 5.1 Speed Benchmarks

```python
class TestPerformance:
    """Test performance characteristics."""
    
    def test_processor_creation_speed(self):
        """Benchmark processor instantiation time."""
        # Measure: Time to create processor from ONNX
        # Target: < 100ms for standard models
        
    def test_preprocessing_speed(self):
        """Benchmark preprocessing performance."""
        # Compare: HF dynamic vs ONNX fixed shapes
        # Target: 40x+ speedup for batch processing
        
    def test_memory_usage(self):
        """Test memory consumption."""
        # Measure: Memory usage during processing
        # Target: No memory leaks, reasonable footprint
```

#### 5.2 Scalability Tests

```python
class TestScalability:
    """Test scalability characteristics."""
    
    def test_large_model_handling(self):
        """Test with large models (>1GB)."""
        # Test Case: Large language models
        # Validates: Efficient metadata extraction
        
    def test_many_inputs_handling(self):
        """Test models with many input tensors."""
        # Test Case: Models with 10+ input tensors
        # Validates: Correct modality grouping
```

### 6. Compatibility Tests

#### 6.1 Version Compatibility

```python
class TestCompatibility:
    """Test compatibility across versions."""
    
    @pytest.mark.parametrize("transformers_version", [
        "4.30.0", "4.35.0", "4.40.0", "4.45.0"
    ])
    def test_transformers_versions(self, transformers_version):
        """Test with different transformers versions."""
        # Validates: Works across version range
        
    @pytest.mark.parametrize("onnx_version", [
        "1.14.0", "1.15.0", "1.16.0"
    ])
    def test_onnx_versions(self, onnx_version):
        """Test with different ONNX versions."""
        # Validates: ONNX format compatibility
```

#### 6.2 Export Tool Compatibility

```python
class TestExportCompatibility:
    """Test models from different export tools."""
    
    def test_optimum_exported_models(self):
        """Test models exported with Optimum."""
        # Validates: Metadata format compatibility
        
    def test_torch_onnx_exported_models(self):
        """Test models exported with torch.onnx."""
        # Validates: Different naming conventions
        
    def test_custom_exported_models(self):
        """Test models with custom export scripts."""
        # Validates: Robustness to variations
```

### 7. Edge Cases and Boundary Tests

#### 7.1 Unusual Model Configurations

```python
class TestEdgeCases:
    """Test edge cases and unusual configurations."""
    
    def test_single_input_models(self):
        """Test models with only one input tensor."""
        
    def test_many_output_models(self):
        """Test models with many output tensors."""
        
    def test_custom_tensor_names(self):
        """Test models with non-standard tensor names."""
        # Test Case: Tensors named "x", "y", "z"
        # Expected: Falls back to shape-based detection
        
    def test_mixed_precision_models(self):
        """Test models with mixed dtypes."""
        # Test Case: float16 and int32 tensors
        # Expected: Correct dtype preservation
```

#### 7.2 Boundary Value Tests

```python
class TestBoundaryValues:
    """Test boundary conditions."""
    
    def test_minimum_shape_sizes(self):
        """Test with minimum valid shapes."""
        # Test Case: batch_size=1, sequence_length=1
        
    def test_maximum_shape_sizes(self):
        """Test with very large shapes."""
        # Test Case: sequence_length=4096
        
    def test_zero_dimension_handling(self):
        """Test handling of 0-dim tensors."""
        # Test Case: Scalar outputs
```

### 8. Security and Validation Tests

#### 8.1 Input Validation

```python
class TestSecurity:
    """Test security and input validation."""
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        # Test Case: "../../../etc/passwd" style paths
        # Expected: Proper sanitization
        
    def test_malformed_input_handling(self):
        """Test handling of malicious inputs."""
        # Test Case: Extremely long tensor names
        # Expected: Input validation and limits
        
    def test_resource_limits(self):
        """Test resource consumption limits."""
        # Test Case: Huge ONNX files
        # Expected: Graceful handling with limits
```

## Test Data Requirements

### Required Test Models

1. **Text Models**
   - BERT (bert-base-uncased)
   - GPT-2 (gpt2)
   - T5 (t5-small)

2. **Vision Models**
   - ViT (google/vit-base-patch16-224)
   - ResNet (microsoft/resnet-50)
   - DETR (facebook/detr-resnet-50)

3. **Audio Models**
   - Wav2Vec2 (facebook/wav2vec2-base)
   - Whisper (openai/whisper-base)

4. **Video Models**
   - VideoMAE (MCG-NJU/videomae-base)
   - TimeSformer (facebook/timesformer-base)

5. **Multimodal Models**
   - CLIP (openai/clip-vit-base-patch32)
   - LayoutLM (microsoft/layoutlm-base-uncased)
   - ALIGN (kakaobrain/align-base)

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
def corrupted_onnx_model():
    """Fixture providing corrupted ONNX for error testing."""
    return create_corrupted_onnx()
```

## Test Execution Strategy

### Continuous Integration

```yaml
# .github/workflows/test_onnx_processor.yml
name: ONNXAutoProcessor Tests

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
        transformers-version: [4.35, 4.40, latest]
    
    steps:
      - name: Run unit tests
        run: pytest tests/unit/test_onnx_auto_processor.py
      
      - name: Run integration tests
        run: pytest tests/integration/test_onnx_processor_e2e.py
      
      - name: Run performance benchmarks
        run: pytest tests/performance/test_processor_benchmarks.py
```

### Test Coverage Requirements

- **Line Coverage**: Minimum 90%
- **Branch Coverage**: Minimum 85%
- **Critical Path Coverage**: 100% for error handling
- **Multimodal Coverage**: All 5 processor types tested

## Mock and Stub Strategy

```python
class MockONNXModel:
    """Mock ONNX model for testing."""
    
    def __init__(self, modality="text", num_inputs=1):
        self.graph = self._create_mock_graph(modality, num_inputs)
        self.metadata_props = []

class StubAutoProcessor:
    """Stub HF AutoProcessor for testing."""
    
    @classmethod
    def from_pretrained(cls, model_name):
        return create_stub_processor(model_name)
```

## Success Criteria

1. **All tests pass** on supported Python versions
2. **Performance targets met** (40x+ speedup maintained)
3. **No regressions** in existing functionality
4. **Edge cases handled** gracefully
5. **Security vulnerabilities** addressed
6. **Documentation coverage** for all test scenarios

## Complete Test File Example

```python
# tests/test_onnx_auto_processor.py

import pytest
import onnx
from pathlib import Path
from onnx_auto_processor import ONNXAutoProcessor

class TestONNXAutoProcessor:
    """Comprehensive test suite with pytest markers."""
    
    # === SMOKE TESTS (5 min) ===
    
    @pytest.mark.smoke
    def test_import(self):
        """Basic import test."""
        assert ONNXAutoProcessor is not None
    
    @pytest.mark.smoke
    def test_basic_creation(self):
        """Create processor from ONNX."""
        processor = ONNXAutoProcessor.from_model("bert.onnx")
        assert processor is not None
    
    # === SANITY TESTS (15 min) ===
    
    @pytest.mark.sanity
    def test_all_processor_types(self):
        """Test all 5 processor types."""
        # Test implementation
    
    @pytest.mark.sanity
    @pytest.mark.multimodal
    def test_multimodal_support(self):
        """Test multimodal processor creation."""
        # Test implementation
    
    # === UNIT TESTS (30 min) ===
    
    @pytest.mark.unit
    def test_metadata_extraction(self):
        """Test _extract_onnx_info method."""
        # Test implementation
    
    @pytest.mark.unit
    def test_modality_detection(self):
        """Test modality detection logic."""
        # Test implementation
    
    # === INTEGRATION TESTS (45 min) ===
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.requires_models
    def test_e2e_bert(self):
        """End-to-end test with BERT."""
        # Test implementation
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.multimodal
    @pytest.mark.requires_models
    def test_e2e_clip(self):
        """End-to-end test with CLIP."""
        # Test implementation
    
    # === PERFORMANCE TESTS (20 min) ===
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_processing_speed(self):
        """Benchmark processing speed."""
        # Test implementation
    
    @pytest.mark.performance
    @pytest.mark.requires_gpu
    def test_gpu_acceleration(self):
        """Test GPU acceleration if available."""
        # Test implementation
```

## Running Tests by Category

```bash
# Development workflow
pytest -m "smoke" -v                    # Quick check (5 min)
pytest -m "not slow" -v                  # Fast tests only
pytest -m "unit and not performance" -v  # Unit tests without benchmarks

# CI/CD workflow
pytest -m smoke --exitfirst              # Fail fast
pytest -m "smoke or sanity" --maxfail=1  # Core features
pytest -m "not (slow or requires_gpu)"   # CI-friendly tests

# Comprehensive testing
pytest --cov=onnx_auto_processor -v      # With coverage
pytest --benchmark-only -v               # Performance tests only
pytest -m multimodal -v                  # Multimodal tests only
```

## Test Maintenance

- **Weekly**: Run full test suite (`pytest -v`)
- **Per PR**: Run relevant test subset (`pytest -m "not slow"`)
- **Monthly**: Update test models to latest versions
- **Quarterly**: Review and update test coverage