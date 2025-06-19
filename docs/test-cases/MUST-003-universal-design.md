# Test Case: Universal Design Validation

## Type
**MUST Test** ⚠️ **CRITICAL - CORE DESIGN PRINCIPLE**

## Purpose
Verify that the exporter truly works with ANY PyTorch model without modification, demonstrating universal design principles. This validates that our approach is architecture-agnostic.

## Test Data (Fixtures)
- Multiple diverse model architectures:
  - BERT (transformer)
  - SimpleModel (basic nn.Module)
  - Custom model with unique structure
  - Vision model (if available)
- Various input types and shapes

## Test Command
```bash
# Test multiple architectures
uv run python -m pytest tests/test_param_mapping.py::TestParameterMapping::test_parameter_mapping_simple_model -v
uv run python -m pytest tests/test_param_mapping.py::TestParameterMapping::test_parameter_mapping_transformers_model -v

# Custom model test
uv run python -c "
import torch.nn as nn
import torch
from modelexport import HierarchyExporter

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unique_module = nn.Sequential(
            nn.Conv1d(10, 20, 3),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )
    def forward(self, x):
        return self.unique_module(x)

model = CustomModel()
exporter = HierarchyExporter()
inputs = torch.randn(1, 10, 50)
exporter.export(model, inputs, 'custom.onnx')
print('Universal design validated!')
"
```

## Expected Behavior
- Same exporter code works across all model types
- No model-specific modifications needed
- Universal module detection logic applies consistently
- Hierarchical paths generated appropriately for each architecture
- No hardcoded assumptions about model structure

## Failure Modes
- **Architecture-Specific Code**: Exporter fails on unexpected model types
- **Hardcoded Assumptions**: Logic assumes specific module names or structures
- **Limited Module Support**: Fails with certain nn.Module subclasses
- **Input Format Issues**: Cannot handle diverse input types

## Dependencies
- Multiple model architectures
- torch.nn module classes
- Various input tensor shapes and types
- No model-specific libraries required for core functionality

## Notes
- **🚨 MUST TEST**: This validates the fundamental promise of universal design - must work with EVERY code change
- **🏗️ CORE PRINCIPLE**: Architecture-agnostic design is the foundation of the entire system
- Should work with:
  - ✅ Transformer models (BERT, GPT, T5, etc.)
  - ✅ Vision models (ResNet, VGG, etc.)
  - ✅ Custom nn.Module subclasses
  - ✅ Sequential models
  - ✅ Complex nested architectures
- Uses only fundamental PyTorch concepts:
  - `nn.Module` hierarchy
  - `named_modules()` iteration
  - Forward hooks
  - Module class inspection

## Test Matrix

| Model Type | Input Type | Expected Result |
|------------|------------|-----------------|
| BERT | Dict (BatchEncoding) | ✅ Hierarchical tags |
| SimpleModel | Tensor | ✅ Universal tags |
| Custom Sequential | Tensor | ✅ Nested hierarchy |
| Vision CNN | Image tensor | ✅ Conv hierarchy |

## Validation Checklist
- [ ] Same HierarchyExporter instance works across model types
- [ ] No model-specific imports in core logic
- [ ] Module detection logic applies universally
- [ ] Hook registration works with any nn.Module
- [ ] Path building adapts to any hierarchy structure
- [ ] No failures on unexpected model architectures
- [ ] Tag format consistent across all model types