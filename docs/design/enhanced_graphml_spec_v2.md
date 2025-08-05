# Enhanced GraphML Specification v2.0 for Bidirectional ONNX Conversion

> ⚠️ **DEPRECATED**: This document has been superseded by the authoritative [GraphML Format Specification v1.2](/docs/specs/graphml-format-specification-v1.2.md).
> 
> **Status**: ARCHIVED - Future proposal, never implemented
> **Archived**: 2025-08-04
> **Reason**: Consolidated into v1.2 specification (TEZ-134)
> 
> This document contains proposals for a v2.0 redesign that were evaluated but not adopted. The v1.2 specification incorporates the useful ideas while maintaining backward compatibility.

**Original Status**: REQUIRED REDESIGN  
**Original Reason**: Current spec insufficient for ONNX reconstruction  
**Original Linear Task**: [TEZ-124](https://linear.app/tezheng/issue/TEZ-124)

## Critical Analysis Summary

**Current GraphML**: ❌ Cannot reconstruct ONNX
- Missing: Node attributes, parameter data, tensor types/shapes, model metadata
- File size: ~100KB (structure only)
- ONNX size: 16.76MB (mostly parameters)

## Enhanced Specification

### 1. **Model Metadata Keys**
```xml
<!-- Required for ONNX ModelProto -->
<key id="m4" for="graph" attr.name="opset_imports" attr.type="string" />
<key id="m5" for="graph" attr.name="producer_name" attr.type="string" />
<key id="m6" for="graph" attr.name="producer_version" attr.type="string" />
<key id="m7" for="graph" attr.name="model_version" attr.type="string" />
<key id="m8" for="graph" attr.name="doc_string" attr.type="string" />
```

### 2. **Enhanced Node Keys**
```xml
<!-- Current keys (keep) -->
<key id="n0" for="node" attr.name="op_type" attr.type="string" />
<key id="n1" for="node" attr.name="hierarchy_tag" attr.type="string" />
<key id="n3" for="node" attr.name="name" attr.type="string" />

<!-- REDESIGNED: n2 now contains actual ONNX attributes -->
<key id="n2" for="node" attr.name="onnx_attributes" attr.type="string" />

<!-- NEW: Essential ONNX node information -->
<key id="n4" for="node" attr.name="input_names" attr.type="string" />
<key id="n5" for="node" attr.name="output_names" attr.type="string" />
<key id="n6" for="node" attr.name="domain" attr.type="string" />
```

### 3. **Tensor Information Keys**
```xml
<!-- NEW: Tensor type and shape information -->
<key id="t0" for="edge" attr.name="tensor_type" attr.type="string" />
<key id="t1" for="edge" attr.name="tensor_shape" attr.type="string" />
<key id="t2" for="edge" attr.name="tensor_data_ref" attr.type="string" />
```

### 4. **Graph Structure Keys**
```xml
<!-- NEW: ONNX graph specifications -->
<key id="g0" for="graph" attr.name="graph_inputs" attr.type="string" />
<key id="g1" for="graph" attr.name="graph_outputs" attr.type="string" />
<key id="g2" for="graph" attr.name="value_info" attr.type="string" />
<key id="g3" for="graph" attr.name="initializers_ref" attr.type="string" />
```

### 5. **Parameter Storage Keys**
```xml
<!-- NEW: Parameter management -->
<key id="p0" for="graph" attr.name="parameter_strategy" attr.type="string" />
<key id="p1" for="graph" attr.name="parameter_file" attr.type="string" />
<key id="p2" for="graph" attr.name="parameter_checksum" attr.type="string" />
```

## Example Enhanced Node
```xml
<node id="/embeddings/word_embeddings/Gather">
  <data key="n0">Gather</data>
  <data key="n1">/BertModel/BertEmbeddings</data>
  <data key="n2">{"axis": 0, "indices_dtype": "int64"}</data>
  <data key="n3">/embeddings/word_embeddings/Gather</data>
  <data key="n4">["word_embeddings.weight", "input_ids"]</data>
  <data key="n5">["embeddings_output"]</data>
  <data key="n6">""</data>
</node>
```

## Example Enhanced Edge
```xml
<edge source="input_input_ids" target="/embeddings/word_embeddings/Gather">
  <data key="e0">input_ids</data>
  <data key="t0">int64</data>
  <data key="t1">[2, 16]</data>
  <data key="t2">null</data>
</edge>
```

## Parameter Storage Strategies

### Strategy 1: Embedded (Small Models)
```xml
<graph id="BertModel">
  <data key="p0">embedded</data>
  <data key="p1">null</data>
  <data key="g3">{"word_embeddings.weight": "base64_data_here"}</data>
</graph>
```

### Strategy 2: Sidecar File (Recommended)
```xml
<graph id="BertModel">
  <data key="p0">sidecar</data>
  <data key="p1">bert-correct.onnxdata</data>
  <data key="p2">sha256_checksum</data>
</graph>
```

### Strategy 3: Reference Original
```xml
<graph id="BertModel">
  <data key="p0">reference</data>
  <data key="p1">bert-correct.onnx</data>
  <data key="p2">sha256_checksum</data>
</graph>
```

## Implementation Impact

### File Size Implications
- **Current**: ~100KB GraphML only
- **Strategy 1**: ~17MB (embedded parameters)
- **Strategy 2**: ~100KB GraphML + ~16MB .onnxdata
- **Strategy 3**: ~100KB GraphML + reference to 16MB .onnx

### Conversion Performance
- **Forward**: ONNX → Enhanced GraphML (~2x current time)
- **Reverse**: Enhanced GraphML → ONNX (~3x forward time)
- **Round-trip**: ~6x original export time

### Compatibility
- **Breaking change**: Requires v2.0 format version
- **Migration**: Convert v1.0 GraphML to v2.0 (with warnings about missing data)
- **Tools**: Update all GraphML readers/writers

## Implementation Phases

### Phase 1: Schema Enhancement
1. Define new key specifications
2. Update GraphML writer to capture ONNX attributes
3. Implement parameter extraction and storage
4. Add model metadata capture

### Phase 2: Reader Implementation  
1. Parse enhanced GraphML format
2. Reconstruct ONNX graph topology
3. Restore node attributes and types
4. Load and integrate parameters

### Phase 3: Validation
1. Round-trip testing framework
2. Numerical accuracy validation
3. Performance benchmarking
4. Error handling for incomplete data

## Migration Strategy

### v1.0 → v2.0 Migration
```python
def migrate_graphml_v1_to_v2(v1_path: str, onnx_path: str) -> str:
    """
    Migrate v1.0 GraphML to v2.0 by extracting missing data from original ONNX.
    
    Args:
        v1_path: Path to v1.0 GraphML file
        onnx_path: Path to original ONNX file for data extraction
        
    Returns:
        Path to migrated v2.0 GraphML file
    """
    # 1. Parse v1.0 GraphML structure
    # 2. Load original ONNX for missing data
    # 3. Extract node attributes, types, parameters
    # 4. Generate v2.0 GraphML with complete information
    pass
```

## Validation Requirements

### Functional Equivalence
- Same computational behavior (within floating-point precision)
- Identical output shapes and types
- Numerical difference < 1e-6 for same inputs

### Structural Preservation
- All nodes, edges, and hierarchy preserved
- ONNX metadata accurately restored
- Parameter integrity maintained

### Performance Targets  
- Conversion time < 5x original export
- Memory usage < 2x model size during conversion
- File size reasonable for storage/transmission

## Risk Mitigation

### Data Loss Prevention
- Comprehensive validation during GraphML generation
- Checksum verification for parameters
- Graceful degradation for incomplete v1.0 files

### Performance Optimization
- Lazy parameter loading
- Streaming for large models
- Compression for parameter storage

### Backward Compatibility
- Support both v1.0 and v2.0 formats
- Clear migration path
- Deprecation warnings for v1.0

---

**Conclusion**: Current GraphML specification is fundamentally insufficient for bidirectional conversion. Enhanced v2.0 specification required with proper ONNX attribute storage, tensor type information, and parameter management system.