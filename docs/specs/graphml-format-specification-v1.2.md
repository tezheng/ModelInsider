# GraphML Format Specification v1.2 (Authoritative)

**Status**: ✅ **AUTHORITATIVE SPECIFICATION**  
**Format Version**: 1.2  
**Consolidation Date**: 2025-08-04  
**Linear Task**: TEZ-134  
**Supersedes**: v1.1, v1.1.1 design notes, v2.0 proposals

## Important Notice

This is the **ONLY** authoritative GraphML specification for the modelexport project. All other specifications have been archived. This consolidates the implemented v1.1 features with clarifications and fixes key numbering conflicts.

## Overview

GraphML v1.2 defines the format for ONNX model export with complete bidirectional conversion support. The format enables visualization in tools like yEd, Gephi, and Cytoscape while preserving all information necessary for ONNX model reconstruction with 85%+ accuracy.

## Key Features

1. **Universal Compatibility** - Works with any ONNX model architecture
2. **Bidirectional Conversion** - Full ONNX ↔ GraphML round-trip capability  
3. **Hierarchical Visualization** - Preserves PyTorch module structure
4. **Parameter Storage** - Flexible strategies for weight management
5. **Tool Compatibility** - Works with major graph visualization tools

## Document Structure

### 1. Root Structure

```xml
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" 
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns 
                             http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <!-- Key definitions -->
  <key id="..." for="..." attr.name="..." attr.type="..."/>
  
  <!-- Main graph -->
  <graph id="G" edgedefault="directed">
    <!-- Graph metadata -->
    <data key="...">...</data>
    
    <!-- Nodes and edges -->
    <node id="...">...</node>
    <edge source="..." target="...">...</edge>
  </graph>
</graphml>
```

## Key Definitions

### Graph Attributes (Compound Nodes)

| Key ID | Attribute Name | Type | Description | Required |
|--------|----------------|------|-------------|----------|
| d0 | class_name | string | PyTorch module class name | MUST |
| d1 | module_type | string | "pytorch" or "huggingface" | MUST |
| d2 | execution_order | int | Module execution order | MUST |
| d3 | traced_tag | string | Full module path | MUST |

### Node Attributes

| Key ID | Attribute Name | Type | Description | Required |
|--------|----------------|------|-------------|----------|
| n0 | op_type | string | ONNX operator type or module type | MUST |
| n1 | hierarchy_tag | string | Module hierarchy path from HTP | MUST |
| n2 | onnx_attributes | string | JSON string of ONNX operator attributes | MUST |
| n3 | name | string | Original ONNX node name | MUST |
| n4 | input_names | string | JSON array of input tensor names | MUST |
| n5 | output_names | string | JSON array of output tensor names | MUST |
| n6 | domain | string | ONNX operator domain | OPTIONAL |

### Edge Attributes

| Key ID | Attribute Name | Type | Description | Required |
|--------|----------------|------|-------------|----------|
| e0 | tensor_name | string | Name of tensor flowing through edge | MUST |
| t0 | tensor_type | string | ONNX tensor data type | MUST |
| t1 | tensor_shape | string | JSON array of tensor dimensions | OPTIONAL |
| t2 | tensor_data_ref | string | Reference to tensor data | OPTIONAL |

### Model Metadata

| Key ID | Attribute Name | Type | Description | Required |
|--------|----------------|------|-------------|----------|
| m0 | source_onnx_text | string | Original ONNX filename | OPTIONAL |
| m1 | source_htp | string | HTP metadata filename | OPTIONAL |
| m2 | format_version | string | GraphML format version ("1.2") | MUST |
| m3 | export_timestamp | string | ISO 8601 timestamp | MUST |
| m4 | opset_imports | string | JSON array of ONNX opset imports | MUST |
| m5 | producer_name | string | Producer name (e.g., "pytorch") | MUST |
| m6 | producer_version | string | Producer version | MUST |
| m7 | model_version | string | Model version number | OPTIONAL |
| m8 | doc_string | string | Model documentation | OPTIONAL |

### Graph I/O Specifications

| Key ID | Attribute Name | Type | Description | Required |
|--------|----------------|------|-------------|----------|
| g0 | graph_inputs | string | JSON array of graph input specs | MUST |
| g1 | graph_outputs | string | JSON array of graph output specs | MUST |
| g2 | value_info | string | JSON array of intermediate tensors | OPTIONAL |
| g3 | initializers_ref | string | Reference to parameter data | CONDITIONAL |

### Parameter Storage

| Key ID | Attribute Name | Type | Description | Required |
|--------|----------------|------|-------------|----------|
| p0 | parameter_strategy | string | "sidecar", "embedded", or "reference" | MUST |
| p1 | parameter_file | string | Path to parameter file | CONDITIONAL |
| p2 | parameter_checksum | string | SHA256 checksum of parameters | CONDITIONAL |

## Node Types

### 1. Operation Nodes
Regular ONNX operations (Conv, Add, MatMul, etc.)

```xml
<node id="/embeddings/word_embeddings/Gather">
  <data key="n0">Gather</data>
  <data key="n1">/BertModel/BertEmbeddings/Embedding</data>
  <data key="n2">{"axis": 0}</data>
  <data key="n3">/embeddings/word_embeddings/Gather</data>
  <data key="n4">["word_embeddings.weight", "input_ids"]</data>
  <data key="n5">["embeddings_output"]</data>
  <data key="n6"></data>
</node>
```

### 2. Input/Output Nodes
Model inputs and outputs with special naming convention:

```xml
<!-- Input node -->
<node id="input_input_ids">
  <data key="n0">Input</data>
  <data key="n1"></data>
  <data key="n2">{}</data>
  <data key="n3">input_ids</data>
  <data key="n4">[]</data>
  <data key="n5">["input_ids"]</data>
  <!-- input_ids: [2, 16] -->
</node>

<!-- Output node -->
<node id="output_logits">
  <data key="n0">Output</data>
  <data key="n1"></data>
  <data key="n2">{}</data>
  <data key="n3">logits</data>
  <data key="n4">["logits"]</data>
  <data key="n5">[]</data>
  <!-- logits: [2, 16, 30522] -->
</node>
```

### 3. Compound Nodes (Hierarchical Mode)
Represent PyTorch module hierarchy:

```xml
<node id="embeddings">
  <data key="n0">BertEmbeddings</data>
  <data key="n1">/BertModel/BertEmbeddings</data>
  <data key="n2">{"module_type": "huggingface", "execution_order": 1}</data>
  <data key="n3">embeddings</data>
  <graph id="embeddings::" edgedefault="directed">
    <data key="d0">BertEmbeddings</data>
    <data key="d1">huggingface</data>
    <data key="d2">1</data>
    <data key="d3">/BertModel/BertEmbeddings</data>
    <!-- Nested nodes -->
  </graph>
</node>
```

## Edge Specification

```xml
<edge source="/embeddings/Add" target="/embeddings/LayerNorm/ReduceMean">
  <data key="e0">embeddings_output</data>
  <data key="t0">float32</data>
  <data key="t1">[2, 16, 128]</data>
</edge>
```

## Parameter Storage Strategies

### Strategy 1: Sidecar (Default, Recommended)
```xml
<graph id="BertModel">
  <data key="p0">sidecar</data>
  <data key="p1">model.onnxdata</data>
  <data key="p2">sha256:a1b2c3d4...</data>
  <data key="g3">model.onnxdata</data>
</graph>
```

### Strategy 2: Embedded (Small Models Only)
```xml
<graph id="BertModel">
  <data key="p0">embedded</data>
  <data key="g3">{"weight": {"dims": [128, 256], "data_b64": "..."}}</data>
</graph>
```

### Strategy 3: Reference (Original ONNX)
```xml
<graph id="BertModel">
  <data key="p0">reference</data>
  <data key="p1">original_model.onnx</data>
  <data key="p2">sha256:e5f6g7h8...</data>
</graph>
```

## Export Modes

### 1. Flat Export (Default without HTP)
- All nodes at root level
- No compound nodes
- Simpler structure for basic visualization

### 2. Hierarchical Export (With HTP Metadata)
- Compound nodes represent module hierarchy
- Nested graph structure
- Requires HTP metadata JSON file

## Bidirectional Conversion

### ONNX → GraphML Process
1. Parse ONNX model structure
2. Extract all node attributes and metadata
3. Apply HTP hierarchy tags if available
4. Store parameters using selected strategy
5. Generate GraphML with complete information

### GraphML → ONNX Process
1. Parse GraphML structure
2. Filter out compound nodes (module containers)
3. Reconstruct ONNX nodes from attributes
4. Load parameters from storage
5. Rebuild ONNX graph with validation

### Round-Trip Accuracy
- **Target**: 85%+ size preservation
- **Node Count**: May differ due to compound node filtering
- **Parameters**: 100% preservation with sidecar strategy
- **Attributes**: Complete preservation in n2 field

## Custom Attributes Philosophy

### Core Principle: Separation of Concerns

1. **ONNX Attributes** (n2): Original operator attributes preserved exactly
2. **GraphML Metadata**: Additional visualization/analysis data kept separate
3. **Filtering Rule**: Only ONNX-native attributes go back to ONNX

### Example
```xml
<!-- n2 contains ONLY ONNX operator attributes -->
<data key="n2">{"kernel_shape": [3, 3], "strides": [1, 1]}</data>

<!-- Module metadata stored separately -->
<data key="n1">/Model/Layer1/Conv</data>  <!-- Hierarchy tag -->
```

## Tool Compatibility

### Supported Tools
- **yEd Graph Editor**: Full support including compound nodes
- **NetworkX (Python)**: Basic graph structure preserved
- **Gephi**: Node/edge attributes visible
- **Cytoscape**: Import with some limitations

### Compatibility Notes
- Some tools may not support nested graphs
- Large models may require memory adjustments
- Compound nodes may not be collapsible in all tools

## Validation Framework

### 1. Schema Validation
```python
def validate_graphml_schema(graphml_file):
    """Validate against GraphML XSD schema."""
    # Check XML well-formedness
    # Verify namespace declarations
    # Validate against XSD
```

### 2. Content Validation
```python
def validate_content(graphml_file):
    """Validate required attributes and structure."""
    # Check all MUST keys present
    # Verify node/edge connectivity
    # Validate JSON fields parseable
```

### 3. Round-Trip Validation
```python
def validate_round_trip(original_onnx, graphml_file):
    """Validate bidirectional conversion."""
    # Convert GraphML back to ONNX
    # Compare node counts (allowing for filtering)
    # Verify parameter integrity
    # Check size preservation ≥85%
```

## CLI Commands

### Export ONNX to GraphML
```bash
# Flat export (no hierarchy)
uv run modelexport export MODEL_NAME output.graphml

# Hierarchical export with HTP
uv run modelexport export MODEL_NAME output.graphml --with-graphml

# With custom parameter strategy
uv run modelexport export MODEL_NAME output.graphml --param-strategy embedded
```

### Convert GraphML to ONNX
```bash
# Basic conversion
uv run modelexport graphml-to-onnx input.graphml output.onnx

# With validation
uv run modelexport graphml-to-onnx input.graphml output.onnx --validate
```

## Performance Characteristics

| Model Size | Export Time | File Size | Round-Trip Time |
|------------|-------------|-----------|-----------------|
| Small (<100 nodes) | <1s | ~100KB | <2s |
| Medium (100-1K nodes) | <3s | ~1MB | <5s |
| Large (1K-10K nodes) | <30s | ~10MB | <60s |

## Migration from Previous Versions

### From v1.1
- Format largely compatible
- Key IDs reorganized for clarity
- Update format_version to "1.2"

### From v1.1.1 Design Notes
- Not an implemented version
- Concepts already included in v1.2

### From v2.0 Proposals
- Never implemented
- Breaking changes not adopted
- Consider for future major version

## Known Limitations

1. **Scale**: Performance degrades beyond 10K nodes (see TEZ-133)
2. **Round-Trip**: 85% accuracy acceptable for visualization use cases
3. **Module Filtering**: Compound nodes removed during ONNX reconstruction
4. **Tool Support**: Not all GraphML features supported by all tools

## Version History

- **1.0** (2024-01-28): Initial specification
- **1.1** (2025-07-29): Added bidirectional conversion
- **1.2** (2025-08-04): Consolidated specification, fixed conflicts

## References

- Implementation: `/modelexport/graphml/`
- Tests: `/tests/graphml/`
- Archived Specs: `/docs/archive/`
- Linear Task: TEZ-134

---

**This is the authoritative specification. All implementation and tests must conform to this document.**