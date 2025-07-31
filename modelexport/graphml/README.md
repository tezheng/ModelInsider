# GraphML Module

**Status**: ✅ **IMPLEMENTED** - TEZ-127 Implementation Complete  
**Format**: GraphML v1.1 with bidirectional ONNX conversion  
**Testing**: 96/96 tests passing  

## Overview

The GraphML module provides universal ONNX to GraphML conversion with complete bidirectional support. It implements GraphML v1.1 specification with universal structural validation - **no hardcoded logic** for any model architectures.

## Key Features

- ✅ **Universal Model Support**: Works with any ONNX model architecture  
- ✅ **Bidirectional Conversion**: GraphML ↔ ONNX with round-trip validation  
- ✅ **Parameter Storage**: Sidecar, embedded, and reference strategies  
- ✅ **Hierarchical Structure**: Preserves PyTorch module hierarchy with HTP metadata  
- ✅ **Comprehensive Testing**: 96/96 tests with structural validation

## Core Components (Bidirectional)

### Primary API
- **`EnhancedGraphMLConverter`** - ONNX → GraphML conversion with parameter support
- **`GraphMLToONNXConverter`** - GraphML → ONNX reconstruction
- **`RoundTripValidator`** - Validates bidirectional conversion integrity

### Supporting Components
- **`parameter_manager.py`** - Handles parameter storage strategies (sidecar/embedded/reference)
- **`metadata_reader.py`** - Reads HTP metadata for hierarchy information
- **`onnx_parser.py`** - Parses ONNX model structure
- **`graphml_writer.py`** - Low-level GraphML XML generation
- **`utils.py`** - Common data structures and utilities

## Legacy Components

These components are maintained for backward compatibility but are not part of the core bidirectional conversion:
- **`converter.py`** - Basic one-way ONNX → GraphML converter
- **`hierarchical_converter.py`** - Hierarchical GraphML converter (extends basic converter)

## Usage

### Bidirectional Conversion
```python
from modelexport.graphml import EnhancedGraphMLConverter, GraphMLToONNXConverter

# ONNX → GraphML
converter = EnhancedGraphMLConverter(
    htp_metadata_path="model_metadata.json",
    parameter_strategy="sidecar"
)
result = converter.convert("model.onnx", output_base="model")

# GraphML → ONNX
reconstructor = GraphMLToONNXConverter()
reconstructor.convert("model.graphml", "model_reconstructed.onnx")
```

### Round-Trip Validation
```python
from modelexport.graphml import RoundTripValidator

validator = RoundTripValidator()
result = validator.validate("original.onnx", "reconstructed.onnx")
print(f"Validation passed: {result.is_valid}")
```

## Universal Design Principles

### No Hardcoded Logic
**CRITICAL**: This module contains **zero hardcoded model architectures, node names, or operation types**.

```python
# ❌ WRONG - Hardcoded approach
if op_type in ["Add", "MatMul", "Gather"]:  # Violates MUST RULE #1

# ✅ CORRECT - Universal approach  
has_nested_graph = node.find('.//graph') is not None
if not has_nested_graph:  # Universal ONNX operation detection
```

### Architecture Agnostic
- Works with **any** ONNX model (BERT, ResNet, GPT, custom models)
- Uses PyTorch's universal `nn.Module` hierarchy structure
- Leverages ONNX's standard protocol buffer format
- No model-specific code paths or assumptions

## GraphML v1.1 Format

### Key Features
- **Complete ONNX Attributes**: All node attributes preserved as JSON
- **Tensor Information**: Type and shape data on edges  
- **Parameter Storage**: Flexible strategies (sidecar/embedded/reference)
- **Model Metadata**: Opset imports, producer info, graph specifications
- **Round-trip Ready**: Full ONNX reconstruction capability

### Example Structure
See comprehensive format examples and specifications in [`docs/specs/graphml-format-specification.md`](../../../docs/specs/graphml-format-specification.md#example-enhanced-node).

## Testing Coverage

- **Structural Tests**: 8 comprehensive validation tests
- **Round-trip Tests**: GraphML → ONNX → validation
- **Edge Cases**: Input/Output nodes, parameter reconstruction  
- **CLI Integration**: End-to-end workflow testing
- **Quality Gates**: 100% test pass rate maintained

## Documentation References

- **Format Specification**: `docs/specs/graphml-format-specification.md`
- **System Design**: `docs/design/bidirectional_onnx_graphml_conversion.md`
- **Technical Workflow**: `docs/design/onnx_to_graphml_workflow.md`
- **Implementation Summary**: `docs/design/TEZ-127-implementation-summary.md`