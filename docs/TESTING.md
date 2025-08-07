# Testing Documentation

This document provides comprehensive information about the testing infrastructure for the modelexport project.

## Overview

The modelexport project has a comprehensive test suite designed to ensure reliability, security, and universal compatibility across different model architectures and use cases.

## Test Structure

### Test Categories

Our test suite is organized into several categories, each addressing specific aspects of the system:

#### 1. Core Functionality Tests (`tests/test_core_functionality.py`)
- **Purpose**: Test fundamental components like hierarchy builders, node taggers, and core algorithms
- **Coverage**: ONNX node tagging, tracing hierarchy builder, parameter mapping, tag propagation
- **Key Features**:
  - CARDINAL RULES validation (MUST-001, MUST-002, MUST-003)
  - Universal architecture support testing
  - Core algorithm correctness validation

#### 2. Export Strategies Tests (`tests/unit/test_strategies/htp/test_htp_hierarchy_exporter.py`)
- **Purpose**: Test HTP (Hierarchical Trace-and-Project) export strategy
- **Coverage**: HTP strategy functionality, unified export pipeline, reporting modes
- **Key Features**:
  - HTP strategy validation
  - Hierarchy preservation testing
  - Reporting and metadata generation
  - Universal design compliance

#### 3. CLI Interface Tests (`tests/test_cli_interface.py`)
- **Purpose**: Test command-line interface functionality
- **Coverage**: Export, analyze, validate, compare commands
- **Key Features**:
  - End-to-end CLI workflows
  - Command validation and error handling
  - Output format testing

#### 4. Integration Workflow Tests (`tests/test_comprehensive_integration.py`)
- **Purpose**: Test complete workflows across different scenarios
- **Coverage**: Full export pipelines, configuration workflows, quality assurance
- **Key Features**:
  - Complete export workflows
  - Configuration file handling
  - Multi-step validation processes
  - CLI integration testing

#### 5. Comprehensive End-to-End Tests (`tests/test_comprehensive_e2e.py`)
- **Purpose**: Test against 8 carefully selected models across domains and architectures
- **Coverage**: Model diversity, architecture-specific handling, scalability
- **Key Features**:
  - **8 models** across 3 domains (language, vision, multimodal)
  - **8 different architectures** (BERT, LLaMA, Qwen, ResNet, ViT, SAM, YOLO, CLIP)
  - **5 size categories** (tiny, small, medium, base, large)
  - Architecture-specific validation (universal design compliance)
  - Performance scaling analysis

#### 6. Security Tests (`tests/test_security.py`) 🆕
- **Purpose**: Test security aspects and vulnerability protection
- **Coverage**: Path validation, input sanitization, file system security, resource protection
- **Key Features**:
  - Path traversal attack prevention
  - Symlink attack protection
  - Malicious input handling
  - Resource exhaustion protection
  - File permission validation

#### 7. Resource Management Tests (`tests/test_resource_management.py`) 🆕
- **Purpose**: Test memory management, timeouts, and resource cleanup
- **Coverage**: Memory leaks, pressure testing, timeout handling, concurrent operations
- **Key Features**:
  - Memory leak detection
  - Memory pressure handling
  - GPU memory management
  - Timeout limits validation
  - Concurrent export testing
  - Resource cleanup verification

#### 8. Error Recovery Tests (`tests/test_error_recovery.py`) 🆕
- **Purpose**: Test error handling and system resilience
- **Coverage**: Corruption handling, network failures, state recovery, graceful degradation
- **Key Features**:
  - Corrupted file handling
  - Network failure scenarios
  - Interrupted export recovery
  - State consistency validation
  - Graceful degradation testing

## Test Execution

### Running Tests

#### Basic Test Execution
```bash
# Run all tests
uv run pytest tests/

# Run specific test category
uv run pytest tests/test_core_functionality.py
uv run pytest tests/test_comprehensive_e2e.py

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=modelexport --cov-report=html
```

#### Test Markers

Following pytest best practices (2025), our tests use a comprehensive marker strategy to categorize and filter tests efficiently:

**Test Categories:**
```bash
# Core functionality validation
uv run pytest -m "smoke"                    # Critical tests that must always pass
uv run pytest -m "unit"                     # Fast, isolated unit tests
uv run pytest -m "integration"              # Tests with external dependencies
uv run pytest -m "e2e"                      # End-to-end workflow tests

# Run regression suite (comprehensive validation)
uv run pytest -m "regression"
```

**Feature-Specific Markers:**
```bash
# Version management tests
uv run pytest -m "version"

# CLI functionality tests
uv run pytest -m "cli"

# GraphML format and validation tests
uv run pytest -m "graphml"

# HTP strategy-specific tests
uv run pytest -m "htp"

# Security and resource management
uv run pytest -m "security"
uv run pytest -m "resource"
uv run pytest -m "error_recovery"
```

**Test Attributes:**
```bash
# Performance-based filtering
uv run pytest -m "not slow"                 # Skip slow tests (>1 second)
uv run pytest -m "slow"                     # Run only slow tests

# Skip flaky or network-dependent tests
uv run pytest -m "not flaky"
uv run pytest -m "not requires_network"

# Environment-specific tests
uv run pytest -m "requires_transformers"    # Tests needing transformers package
uv run pytest -m "requires_gpu"             # GPU/CUDA-dependent tests
```

**Advanced Filtering (Combined Markers):**
```bash
# Fast smoke tests for development
uv run pytest -m "smoke and not slow"

# Integration tests without network dependencies
uv run pytest -m "integration and not requires_network"

# All version tests except slow ones
uv run pytest -m "version and not slow"

# CLI tests that require transformers
uv run pytest -m "cli and requires_transformers"
```

#### Parallel Test Execution
```bash
# Run tests in parallel (requires pytest-xdist)
uv run pytest tests/ -n auto

# Run specific number of workers
uv run pytest tests/ -n 4
```

### Test Configuration

#### pyproject.toml Configuration (Best Practice 2025)

Following modern Python standards, all pytest configuration is centralized in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = [
    "-v",
    "--tb=short", 
    "--strict-markers",
    "--disable-warnings",
    "--color=yes",
    "--durations=10"
]
minversion = "6.0"

markers = [
    # Test Categories
    "smoke: Core functionality tests that must always pass",
    "unit: Unit tests (fast, isolated)", 
    "integration: Integration tests (may require external services)",
    "e2e: End-to-end tests (slower, full workflows)",
    
    # Feature-Specific Markers
    "version: Tests related to version management and detection",
    "cli: Command-line interface tests",
    "graphml: GraphML format and validation tests",
    "htp: HTP strategy-specific tests",
    "security: Security and vulnerability tests",
    
    # Test Attributes
    "slow: Tests that take > 1 second to run",
    "requires_transformers: Tests that require transformers library"
]

filterwarnings = [
    "ignore::UserWarning",
    "ignore::DeprecationWarning"
]
```

**Key Benefits of pyproject.toml approach:**
- ✅ Single source of truth for all project configuration
- ✅ Modern Python standard (PEP 518)
- ✅ Better IDE and tool integration
- ✅ Centralized configuration management

## Test Data and Fixtures

### Model Fixtures
The test suite includes fixtures for various model types:

- **Language Models**: BERT variants, RoBERTa, DistilBERT, ALBERT
- **Vision Models**: ResNet, ViT, EfficientNet, ConvNeXt
- **Custom Models**: Synthetic models for specific test scenarios
- **Edge Case Models**: Models with unusual architectures or properties

### Test Data Generation
```bash
# Generate test data for specific models
uv run python tests/data/generate_test_data.py --model prajjwal1/bert-tiny

# Generate baseline data
uv run python tests/data/generate_test_data.py --model prajjwal1/bert-tiny --output-dir temp/baseline/
```

## CARDINAL RULES Testing

All tests must validate compliance with CARDINAL RULES:

### MUST-001: No Hardcoded Logic
- Tests verify universal design across all model architectures
- No architecture-specific assumptions in test logic
- Universal approaches validated across model diversity

### MUST-002: Torch.nn Filtering
- Tests validate proper filtering of torch.nn modules
- Whitelist-based approach verification
- Universal module handling validation

### MUST-003: Universal Design
- Tests ensure architecture-agnostic functionality
- Cross-architecture consistency validation
- Universal compatibility verification

### MUST-004: Mandatory Test Verification
- All feature implementations must be verified with `uv run pytest tests/`
- All test case revisions must be confirmed with `uv run pytest tests/`
- No code changes considered complete without pytest verification

## Performance Benchmarks

### Test Performance Standards
- **Core functionality tests**: < 60 seconds total
- **Export strategy tests**: < 120 seconds total
- **CLI interface tests**: < 90 seconds total
- **Individual model export**: < 30 seconds for small models
- **Memory usage**: < 500MB increase per export
- **Coverage requirement**: 100% for all successful exports

### Performance Monitoring
Tests include performance monitoring to detect regressions:

```python
# Example performance assertion
assert export_time < 30.0, f"Export took {export_time:.2f}s, should be <30s"
assert memory_increase < 100 * 1024 * 1024, f"Memory increase {memory_increase}MB too high"
```

## Security Testing

### Security Test Categories
1. **Path Validation**: Directory traversal, symlink attacks
2. **Input Sanitization**: Malicious model names, input specs
3. **File System Security**: Permissions, access controls
4. **Resource Protection**: DoS prevention, memory limits

### Security Test Execution
```bash
# Run all security tests
uv run pytest tests/test_security.py -v

# Run specific security test categories
uv run pytest tests/test_security.py::TestPathValidation -v
uv run pytest tests/test_security.py::TestInputSanitization -v
```

## Resource Management Testing

### Resource Test Categories
1. **Memory Management**: Leak detection, pressure testing
2. **Timeout Handling**: Operation limits, hang detection
3. **Resource Cleanup**: Temp files, GPU memory, processes
4. **Concurrent Operations**: Thread safety, parallel exports

### Resource Test Execution
```bash
# Run all resource management tests
uv run pytest tests/test_resource_management.py -v

# Run memory-specific tests
uv run pytest tests/test_resource_management.py::TestMemoryManagement -v

# Run concurrency tests
uv run pytest tests/test_resource_management.py::TestConcurrentOperations -v
```

## Error Recovery Testing

### Error Recovery Test Categories
1. **Corruption Handling**: Corrupted ONNX files, metadata
2. **Network Failures**: Download failures, timeouts
3. **State Recovery**: Interrupted exports, inconsistent states
4. **Graceful Degradation**: Limited functionality operation

### Error Recovery Test Execution
```bash
# Run all error recovery tests
uv run pytest tests/test_error_recovery.py -v

# Run corruption handling tests
uv run pytest tests/test_error_recovery.py::TestCorruptionHandling -v

# Run network failure tests
uv run pytest tests/test_error_recovery.py::TestNetworkFailures -v
```

## Comprehensive End-to-End Testing

### Model Coverage
Our comprehensive E2E test suite validates against **8 carefully selected models** representing the most important architectural paradigms:

#### **Current Model Matrix (Updated 2024)**

| **Model** | **HuggingFace ID** | **Architecture** | **Domain** | **Size** | **Notes** |
|-----------|-------------------|------------------|------------|----------|-----------|
| **BERT** | `prajjwal1/bert-tiny` | BERT | Language | Tiny | Fast baseline for testing |
| **LLaMA** | `meta-llama/Llama-3.2-1B` | LLaMA 3.2 | Language | Small (1B) | Latest LLM architecture |
| **Qwen** | `Qwen/Qwen1.5-0.5B` | Qwen | Language | Small (0.5B) | Chinese LLM with different tokenization |
| **ResNet** | `microsoft/resnet-18` | ResNet | Vision | Small | Classic CNN architecture |
| **ViT** | `google/vit-base-patch16-224` | ViT | Vision | Base | Vision Transformer |
| **SAM** | `facebook/sam-vit-base` | SAM | Vision | Base | Segment Anything Model |
| **YOLO** | `Ultralytics/YOLO11` | YOLOv11 | Vision | Small | Latest object detection (2024) |
| **CLIP** | `openai/clip-vit-base-patch32` | CLIP | Multimodal | Base | Vision-language model |

#### **Coverage Summary**
- **8 models** across **3 domains**: language, vision, multimodal  
- **8 different architectures**: BERT, LLaMA, Qwen, ResNet, ViT, SAM, YOLO, CLIP
- **5 size categories**: tiny (33M), small (9-11M), medium (0.5B), base (86-151M), large (375M)
- **Latest versions**: LLaMA 3.2 (2024), YOLOv11 (Sept 2024)

#### **Architecture Diversity**
- **Language Models**: Traditional (BERT) + Modern LLMs (LLaMA 3.2, Qwen)
- **Vision Models**: CNN (ResNet) + Transformers (ViT) + Specialized (SAM, YOLO)  
- **Multimodal**: Vision-language understanding (CLIP)
- **Universal Design**: All models tested with same universal export strategy

### E2E Test Execution
```bash
# Run all comprehensive E2E tests
uv run pytest tests/test_comprehensive_e2e.py -v

# Run specific model tests
uv run pytest "tests/test_comprehensive_e2e.py::TestComprehensiveEndToEnd::test_popular_model_export[bert_tiny]" -v
uv run pytest "tests/test_comprehensive_e2e.py::TestComprehensiveEndToEnd::test_popular_model_export[resnet18]" -v

# Run architecture-specific tests
uv run pytest tests/test_comprehensive_e2e.py::TestModelArchitectureSpecific -v

# Run performance scaling tests
uv run pytest tests/test_comprehensive_e2e.py::TestScalabilityAndPerformance -v

# Validate model diversity coverage
uv run pytest "tests/test_comprehensive_e2e.py::TestComprehensiveEndToEnd::test_model_diversity_coverage" -v
```

## Continuous Integration

### CI/CD Pipeline
The test suite is designed for CI/CD environments:

```yaml
# Example GitHub Actions workflow
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          uv venv
          uv pip install -e .
          
      - name: Run MUST tests
        run: uv run pytest tests/ -m "must" --tb=short
        
      - name: Run core functionality tests
        run: uv run pytest tests/test_core_functionality.py -v
        
      - name: Run security tests
        run: uv run pytest tests/test_security.py -v
        
      - name: Run comprehensive E2E tests (sample)
        run: |
          uv run pytest "tests/test_comprehensive_e2e.py::TestComprehensiveEndToEnd::test_popular_model_export[bert_tiny]" -v
          uv run pytest "tests/test_comprehensive_e2e.py::TestComprehensiveEndToEnd::test_popular_model_export[resnet18]" -v
```

### Test Environment Requirements
- **Python**: 3.12+
- **Memory**: 4GB+ recommended for comprehensive tests
- **Disk**: 2GB+ for model downloads and temp files
- **Network**: Required for model downloads (some tests can run offline)
- **GPU**: Optional, but enables additional CUDA-specific tests

## Test Maintenance

### Adding New Tests
When adding new functionality, ensure:

1. **Test Coverage**: New code should have corresponding tests
2. **CARDINAL RULES**: All tests must validate CARDINAL RULES compliance
3. **Documentation**: Test purpose and expected behavior documented
4. **Performance**: Performance benchmarks included for critical paths
5. **Security**: Security implications considered and tested

### Test Review Checklist
- [ ] CARDINAL RULES compliance validated
- [ ] Universal design principles followed
- [ ] Security considerations addressed
- [ ] Performance benchmarks included
- [ ] Error handling tested
- [ ] Documentation updated
- [ ] CI/CD integration verified

## Troubleshooting

### Common Test Issues

#### Model Download Failures
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Run tests with offline models only
uv run pytest tests/ -k "not download"
```

#### Memory Issues
```bash
# Run tests with limited parallelism
uv run pytest tests/ -n 1

# Skip memory-intensive tests
uv run pytest tests/ -m "not slow"
```

#### Permission Issues
```bash
# Fix temp directory permissions
sudo chmod 755 /tmp

# Run tests with user temp directory
TMPDIR=~/tmp uv run pytest tests/
```

### Debug Mode
```bash
# Run tests with maximum verbosity
uv run pytest tests/ -vvv --tb=long --showlocals

# Run specific test with debugging
uv run pytest tests/test_core_functionality.py::TestTracingHierarchyBuilder::test_universal_hierarchy_extraction -vvv --pdb
```

## Test Results Interpretation

### Success Criteria
- **Coverage**: 100% tag coverage for all model exports
- **Empty Tags**: 0 empty tags (CARDINAL RULE compliance)
- **Performance**: Within specified time and memory limits
- **Security**: No vulnerabilities detected
- **Resource Cleanup**: Proper resource cleanup verified

### Failure Analysis
When tests fail:

1. **Check CARDINAL RULES**: Ensure no hardcoded logic violations
2. **Verify Universal Design**: Confirm architecture-agnostic approach
3. **Review Error Messages**: Analyze specific failure reasons
4. **Check Resource Usage**: Monitor memory and file handle usage
5. **Validate Test Environment**: Ensure proper setup and dependencies

## Future Test Enhancements

### Planned Additions
- **Fuzzing Tests**: Automated testing with random inputs
- **Property-Based Testing**: Hypothesis-driven test generation
- **Load Testing**: High-volume concurrent operation testing
- **Regression Testing**: Automated detection of performance regressions
- **Model Compatibility Matrix**: Systematic testing across model versions

### Enhancement Priorities
1. **Automated Model Discovery**: Dynamic test generation for new models
2. **Performance Regression Detection**: Automated benchmarking
3. **Security Scanning Integration**: Automated vulnerability scanning
4. **Test Environment Isolation**: Container-based test execution
5. **Real-world Scenario Testing**: Production-like test scenarios

---

*This documentation is updated regularly to reflect the current state of the test suite. For the most current information, refer to the test code and commit history.*