# Test Extraction Plan: Experimental Tests Review

## Executive Summary

After thorough review of all experimental test files, we discovered **critical test coverage** that is NOT present in the main test suite. These tests validate real-world production workflows that the mock-based main suite cannot cover.

## Critical Finding

**The main test suite (`tests/test_onnx_auto_processor.py`) uses mocks and doesn't test:**
- Actual ONNX export workflows
- Real shape constraint handling
- Optimum ORTModel integration
- End-to-end production deployment
- Pipeline compatibility

## Test Files to Extract (Priority Order)

### 🔴 CRITICAL - Must Extract

#### 1. `final_validation_test.py`
- **Coverage**: Complete end-to-end workflow validation
- **Unique Value**: Tests entire production cycle from export to deployment
- **Extract to**: `tests/inference/integration/test_end_to_end_validation.py`
- **Key Tests**:
  - Export with HTP strategy
  - Config file generation
  - Optimum compatibility validation
  - Performance benchmarking

#### 2. `test_clean_onnx_optimum.py`
- **Coverage**: Clean ONNX export → config addition → Optimum validation
- **Unique Value**: Tests production deployment pattern
- **Extract to**: `tests/inference/integration/test_clean_export_workflow.py`
- **Key Tests**:
  - Clean ONNX export (without metadata)
  - Config file addition post-export
  - Optimum ORTModel loading
  - Inference validation

### 🟠 HIGH Priority - Strong Value

#### 3. `test_universal_config.py`
- **Coverage**: UniversalOnnxConfig implementation
- **Unique Value**: Tests actual ONNX export configuration
- **Extract to**: `tests/inference/integration/test_onnx_export_config.py`
- **Key Tests**:
  - Task detection from model config
  - ONNX export workflow
  - Config generation for various models

#### 4. `test_existing_onnx.py`
- **Coverage**: Adding configs to existing ONNX models
- **Unique Value**: Retrofitting deployment pattern
- **Extract to**: `tests/inference/integration/test_config_addition.py`
- **Key Tests**:
  - Config file generation for existing models
  - Compatibility validation
  - Deployment readiness checks

#### 5. `test_fixed_shape_tokenizer.py`
- **Coverage**: FixedShapeTokenizer implementation
- **Unique Value**: Real shape constraint handling
- **Extract to**: `tests/inference/test_shape_constraints.py`
- **Key Tests**:
  - Shape enforcement in tokenization
  - Padding/truncation validation
  - Performance optimization tests

#### 6. `test_enhanced_pipeline.py`
- **Coverage**: Enhanced pipeline with data_processor
- **Unique Value**: Tests improved API design
- **Extract to**: `tests/inference/test_enhanced_pipeline_api.py`
- **Key Tests**:
  - Universal data_processor parameter
  - Pipeline compatibility
  - Performance comparisons

#### 7. `test_auto_shape_detection.py`
- **Coverage**: Automatic ONNX shape detection
- **Unique Value**: Intelligent shape inference
- **Extract to**: `tests/inference/test_shape_auto_detection.py`
- **Key Tests**:
  - ONNX model introspection
  - Shape inference logic
  - Fallback mechanisms

### 🟡 MEDIUM Priority - Useful Coverage

#### 8. `test_ort_pipeline.py`
- **Coverage**: ORTModel with transformers pipeline
- **Unique Value**: Pipeline integration patterns
- **Extract to**: `tests/inference/integration/test_ort_pipeline_integration.py`

#### 9. `test_bert_model_ort.py`
- **Coverage**: BERT-specific ORTModel compatibility
- **Unique Value**: Model-specific edge cases
- **Extract to**: `tests/inference/test_model_specific_compatibility.py`

#### 10. `test_pipeline_with_fixed_tokenizer.py`
- **Coverage**: Standard pipeline with fixed shapes
- **Unique Value**: Real-world usage patterns
- **Extract to**: `tests/inference/test_pipeline_fixed_shapes.py`

#### 11. `test_export_simple.py`
- **Coverage**: Basic export functionality
- **Unique Value**: Simple validation scenarios
- **Extract to**: `tests/inference/test_basic_export.py`

### ⚪ LOW Priority - Can Delete

#### 12. `test_processor_as_tokenizer.py`
- **Coverage**: Parameter routing
- **Decision**: Merge into pipeline tests

#### 13. `test_simple_auto_tokenizer.py`
- **Coverage**: Basic usage demonstration
- **Decision**: DELETE - covered by other tests

## Extraction Strategy

### Phase 1: Critical Tests (Week 1, Day 1-2)
1. Create `tests/inference/integration/` directory
2. Extract end-to-end validation tests
3. Convert to pytest format
4. Ensure all fixtures are available

### Phase 2: High Priority Tests (Week 1, Day 3-4)
1. Extract shape handling tests
2. Extract pipeline enhancement tests
3. Create shared fixtures for ONNX models

### Phase 3: Medium Priority Tests (Week 1, Day 5)
1. Extract model-specific tests
2. Consolidate overlapping test cases
3. Create comprehensive test suite

## Test Organization Structure

```bash
tests/
└── inference/
    ├── unit/                          # Fast, isolated tests
    │   ├── test_shape_constraints.py
    │   ├── test_shape_auto_detection.py
    │   └── test_enhanced_pipeline_api.py
    │
    ├── integration/                    # Real model tests
    │   ├── test_end_to_end_validation.py    # CRITICAL
    │   ├── test_clean_export_workflow.py    # CRITICAL
    │   ├── test_onnx_export_config.py       # HIGH
    │   ├── test_config_addition.py          # HIGH
    │   ├── test_ort_pipeline_integration.py # MEDIUM
    │   └── test_model_compatibility.py      # MEDIUM
    │
    └── fixtures/
        ├── models/                     # Test ONNX models
        └── configs/                    # Test configurations
```

## Key Test Patterns to Preserve

### 1. Real ONNX Export Tests
```python
def test_export_with_htp_strategy():
    """Test actual ONNX export with HTP strategy"""
    # This validates the core export functionality
    
def test_export_without_metadata():
    """Test clean ONNX export for production"""
    # Critical for production deployment
```

### 2. Config Generation Tests
```python
def test_config_generation_for_existing_onnx():
    """Test adding configs to pre-exported models"""
    # Important for retrofit scenarios
```

### 3. Shape Handling Tests
```python
def test_fixed_shape_enforcement():
    """Test real shape constraints, not mocks"""
    # Critical for performance optimization
```

### 4. Pipeline Integration Tests
```python
def test_ort_model_with_pipeline():
    """Test real ORTModel in transformers pipeline"""
    # Validates production usage
```

## Coverage Comparison

| Test Category | Main Suite | Experimental Tests | Action Required |
|---------------|------------|-------------------|-----------------|
| ONNX Export | ❌ None | ✅ Complete | **EXTRACT** |
| Shape Handling | 🟡 Mocked | ✅ Real | **EXTRACT** |
| Optimum Integration | ❌ None | ✅ Complete | **EXTRACT** |
| Pipeline Usage | 🟡 Basic | ✅ Enhanced | **EXTRACT** |
| End-to-End | ❌ None | ✅ Complete | **EXTRACT** |
| Config Generation | ❌ None | ✅ Complete | **EXTRACT** |

## Summary

**CRITICAL DECISION**: Do NOT delete these experimental tests without extracting their unique coverage. They contain production-critical validation that the main test suite lacks.

**Recommended Actions**:
1. **IMMEDIATE**: Extract critical end-to-end tests
2. **HIGH PRIORITY**: Extract shape handling and config generation tests
3. **MEDIUM PRIORITY**: Extract pipeline integration tests
4. **LOW PRIORITY**: Delete only truly redundant tests (2 files)

**Files to Extract**: 11 out of 13 experimental test files
**Files to Delete**: Only 2 truly redundant files
**New Test Coverage**: ~500+ lines of critical production validation