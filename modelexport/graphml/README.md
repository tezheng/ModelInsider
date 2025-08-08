# GraphML Bidirectional Conversion Module

## Overview

This module provides complete bidirectional conversion between ONNX and GraphML formats, enabling:
- ONNX → GraphML conversion with full parameter preservation
- GraphML → ONNX reconstruction with validation
- Round-trip validation to ensure conversion integrity

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

## Architecture

The bidirectional conversion system uses GraphML v1.1 format specification with:
- Complete ONNX node attributes preservation
- Parameter storage management (sidecar files)
- Hierarchical structure preservation from HTP metadata
- Full model reconstruction capability