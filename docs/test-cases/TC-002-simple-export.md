# Test Case: Simple Export

## Type
**Smoke Test** 🚭

## Purpose
Verify that the HierarchyExporter can perform a basic export operation with a simple PyTorch model without crashing.

## Test Data (Fixtures)
- Simple PyTorch model (Linear layers)
- Basic tensor input
- Temporary output directory

## Test Command
```bash
uv run python -m pytest tests/test_param_mapping.py::TestParameterMapping::test_parameter_mapping_simple_model -v
```

## Expected Behavior
- Export completes without exceptions
- ONNX file is created
- Sidecar JSON file is generated
- Model has at least one tagged operation

## Failure Modes
- **ONNX Export Error**: PyTorch → ONNX conversion issues
- **File Creation Error**: Permission or path issues
- **Tagging Error**: Hook registration or execution tracing failures

## Dependencies
- torch
- onnx
- Simple model class definition
- Temporary file system access

## Notes
- Uses SimpleModel with Embedding + Linear layers
- Tests universal tagging (not model-specific)
- Should complete in <10 seconds