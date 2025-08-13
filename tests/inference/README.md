# ONNXAutoProcessor Test Suite

Comprehensive test suite for the ONNX Auto Processor system with 100+ tests covering all functionality.

## Quick Start

```bash
# Quick validation (5 min)
pytest -m smoke -v

# Core functionality (15 min) 
pytest -m sanity -v

# Full test suite
pytest -v

# Use the test runner
python run_tests.py --progressive
```

## Test Categories

| Category | Purpose | Time | Command |
|----------|---------|------|---------|
| 🚬 **Smoke** | Basic functionality | 5 min | `pytest -m smoke -v` |
| ✅ **Sanity** | Core features | 15 min | `pytest -m sanity -v` |
| 🔧 **Unit** | Component isolation | 30 min | `pytest -m unit -v` |
| 🔄 **Integration** | End-to-end workflows | 45 min | `pytest -m integration -v` |
| ⚡ **Performance** | Speed/memory benchmarks | 20 min | `pytest -m performance -v` |

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                    # Pytest configuration
├── run_tests.py                   # Test runner with different modes
├── test_utils.py                  # Mock models and utilities
├── test_onnx_auto_processor.py    # Main comprehensive test suite
└── README.md                      # This file
```

## Key Features

### Mock Model Generation
- **Text Models**: BERT-like with configurable shapes
- **Image Models**: ViT-like with NCHW format
- **Audio Models**: Wav2Vec2/Whisper-like with waveform/spectrogram inputs
- **Video Models**: VideoMAE-like with NCTHW format
- **Multimodal Models**: CLIP-like with text + image inputs

### Test Coverage
- ✅ All 5 processor types (text, image, audio, video, multimodal)
- ✅ Metadata extraction from ONNX models
- ✅ Fixed-shape enforcement and validation
- ✅ Error handling and edge cases
- ✅ Performance benchmarks and memory usage
- ✅ Concurrent processing safety
- ✅ Integration with HuggingFace processors

### Test Data
- **Fixtures**: Pre-configured ONNX models and mock processors
- **Performance Data**: Realistic test datasets for benchmarking
- **Edge Cases**: Boundary conditions and error scenarios

## Running Tests

### Using pytest directly

```bash
# All tests
pytest -v

# Specific categories
pytest -m "smoke or sanity" -v
pytest -m "unit and not slow" -v
pytest -m multimodal -v

# With coverage (if pytest-cov installed)
pytest --cov=src --cov-report=html -v

# Fast tests only (excludes slow ones)
pytest -m "not slow" -v

# CI-friendly (excludes GPU/model requirements)
pytest -m "not (requires_gpu or requires_models)" -v
```

### Using the test runner

```bash
# Progressive validation (recommended)
python run_tests.py --progressive

# Individual categories
python run_tests.py --smoke
python run_tests.py --sanity
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --performance

# Development workflow
python run_tests.py --fast

# CI workflow
python run_tests.py --ci
```

## Test Development

### Adding New Tests

1. **Choose Category**: Determine appropriate test markers
2. **Use Fixtures**: Leverage existing fixtures for models and data
3. **Follow Patterns**: Follow existing test naming and structure
4. **Add Markers**: Use `@pytest.mark.{category}` decorators

Example:
```python
@pytest.mark.unit
def test_new_functionality(self, bert_onnx_model, mock_tokenizer):
    """Test new functionality - unit test."""
    # Test implementation
    pass

@pytest.mark.integration
@pytest.mark.multimodal
def test_multimodal_workflow(self, clip_onnx_model):
    """Test multimodal workflow - integration test."""
    # Test implementation
    pass
```

### Creating Test Data

```python
# Use existing utilities
from test_utils import create_text_onnx_model, create_mock_base_processor

# Create custom models
model = (MockONNXModel("custom_test")
         .add_input("custom_input", [1, 64], TensorType.FLOAT32)
         .add_output("custom_output", [1, 10], TensorType.FLOAT32)
         .build())

# Create mock processors
mock_processor = create_mock_base_processor("tokenizer")
```

## Performance Benchmarks

Performance tests validate:

- **Processor Creation**: <100ms for standard models
- **Text Processing**: <10ms per text sample
- **Image Processing**: <50ms per image
- **Memory Usage**: No memory leaks over 1000+ iterations
- **Concurrent Safety**: Thread-safe processing

## Test Configuration

Key settings in `pytest.ini`:
- **Markers**: All test categories defined
- **Timeout**: 300 seconds per test
- **Output**: Verbose with short tracebacks
- **Warnings**: Filtered for clean output

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src directory is in Python path
   export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
   ```

2. **Missing Dependencies**
   ```bash
   # Install test dependencies
   pip install pytest numpy onnx transformers
   pip install pytest-cov pytest-timeout  # Optional
   ```

3. **Slow Tests**
   ```bash
   # Skip slow tests during development
   pytest -m "not slow" -v
   ```

4. **GPU Tests Failing**
   ```bash
   # Skip GPU tests if no GPU available
   pytest -m "not requires_gpu" -v
   ```

### Debug Mode

```bash
# Run with maximum verbosity and no capture
pytest -vvv -s --tb=long test_onnx_auto_processor.py::TestClass::test_method

# Run single test with debugging
pytest --pdb -s test_onnx_auto_processor.py::test_specific_function
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
- name: Run Tests
  run: |
    # Quick validation
    python tests/run_tests.py --smoke
    
    # Core functionality  
    python tests/run_tests.py --sanity
    
    # Fast comprehensive tests
    python tests/run_tests.py --fast
```

## Test Metrics

Target metrics:
- **Line Coverage**: >90%
- **Branch Coverage**: >85%
- **Test Count**: 100+ tests
- **Test Speed**: <2 minutes for fast tests
- **Success Rate**: 100% for smoke/sanity tests

## Contributing

When adding new functionality:

1. Add corresponding tests in appropriate categories
2. Update test utilities if needed
3. Ensure all test categories still pass
4. Update documentation if test interface changes

The test suite is designed to grow with the codebase while maintaining fast feedback loops for development.