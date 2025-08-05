# GraphML Specification Comparison Matrix

**Date**: 2025-08-04  
**Task**: TEZ-134 - Consolidate GraphML Format Specifications  
**Status**: IN PROGRESS

## Executive Summary

Three competing GraphML specifications exist in the codebase, creating significant implementation confusion. This matrix analyzes differences, conflicts, and contradictions to determine the authoritative version.

## Specification Versions Overview

| Version | File Location | Status | Last Updated | Key Focus |
|---------|--------------|---------|--------------|-----------|
| **v1.1** | `/docs/specs/graphml-format-specification.md` | ✅ IMPLEMENTED | 2025-07-31 | Complete bidirectional ONNX ↔ GraphML |
| **v1.1.1** | `/docs/design/graphml_v1.1_format_specification.md` | 📝 DESIGN DOC | Unknown | Custom attributes design |
| **v2.0** | `/docs/design/enhanced_graphml_spec_v2.md` | ❌ REQUIRED REDESIGN | Unknown | Enhanced for full ONNX reconstruction |

## Key Definition Comparison

### Graph Attributes (Compound Nodes)

| Key ID | Attribute | v1.1 | v1.1.1 | v2.0 | Conflict? |
|--------|-----------|------|--------|------|-----------|
| d0 | class_name | ✅ MUST | ✅ Implied | Not specified | ❌ v2.0 missing |
| d1 | module_type | ✅ MUST | ✅ Implied | Not specified | ❌ v2.0 missing |
| d2 | execution_order | ✅ MUST | ✅ Implied | Not specified | ❌ v2.0 missing |
| d3 | traced_tag | ✅ MUST | ✅ Implied | Not specified | ❌ v2.0 missing |

### Node Attributes

| Key ID | Attribute | v1.1 | v1.1.1 | v2.0 | Conflict? |
|--------|-----------|------|--------|------|-----------|
| n0 | op_type | ✅ MUST | ✅ | ✅ | ✅ Consistent |
| n1 | hierarchy_tag | ✅ MUST | ✅ | ✅ | ✅ Consistent |
| n2 | node_attributes/onnx_attributes | ✅ JSON (module info) | ❓ Not clear | ✅ JSON (ONNX attrs) | 🔴 **MAJOR CONFLICT** |
| n3 | name | ✅ MUST | ✅ | ✅ | ✅ Consistent |
| n4 | input_names | ✅ MUST (v1.1 extended) | ❌ | ✅ NEW | ⚠️ v1.1.1 missing |
| n5 | output_names | ✅ MUST (v1.1 extended) | ❌ | ✅ NEW | ⚠️ v1.1.1 missing |
| n6 | domain | ✅ OPTIONAL | ❌ | ✅ NEW | ⚠️ v1.1.1 missing |

### Edge Attributes

| Key ID | Attribute | v1.1 | v1.1.1 | v2.0 | Conflict? |
|--------|-----------|------|--------|------|-----------|
| e0 | tensor_name | ✅ MUST | ✅ Implied | ✅ | ✅ Consistent |
| t0 | tensor_type | ✅ MUST (extended) | ❌ | ✅ NEW | ⚠️ v1.1.1 missing |
| t1 | tensor_shape | ✅ OPTIONAL | ❌ | ✅ NEW | ⚠️ v1.1.1 missing |
| t2 | tensor_data_ref | ❌ | ❌ | ✅ NEW | 🟡 v2.0 only |

### Model Metadata

| Key ID | Attribute | v1.1 | v1.1.1 | v2.0 | Conflict? |
|--------|-----------|------|--------|------|-----------|
| m0 | source_onnx_text | ✅ OPTIONAL | ✅ | Not specified | ❌ v2.0 missing |
| m1 | source_htp | ✅ OPTIONAL | ✅ | Not specified | ❌ v2.0 missing |
| m2 | format_version | ✅ MUST ("1.1") | ✅ ("1.1") | ❌ Different ("2.0") | 🔴 **VERSION CONFLICT** |
| m3 | export_timestamp | ✅ MUST | ✅ | Not specified | ❌ v2.0 missing |
| m4 | opset_imports | ✅ MUST (extended) | ❌ | ✅ NEW | ⚠️ v1.1.1 missing |
| m5 | producer_name/graph_inputs | ✅ graph_inputs | ❌ | ✅ producer_name | 🔴 **PURPOSE CONFLICT** |
| m6 | producer_version/graph_outputs | ✅ graph_outputs | ❌ | ✅ producer_version | 🔴 **PURPOSE CONFLICT** |
| m7 | model_version/value_info | ✅ value_info | ❌ | ✅ model_version | 🔴 **PURPOSE CONFLICT** |
| m8 | doc_string/initializers_ref | ✅ initializers_ref | ❌ | ✅ doc_string | 🔴 **PURPOSE CONFLICT** |

### Parameter Storage

| Key ID | Attribute | v1.1 | v1.1.1 | v2.0 | Conflict? |
|--------|-----------|------|--------|------|-----------|
| p0 | parameter_strategy | ✅ MUST | ✅ Implied | ✅ NEW | ✅ Consistent concept |
| p1 | parameter_file | ✅ CONDITIONAL | ✅ Implied | ✅ NEW | ✅ Consistent |
| p2 | parameter_checksum | ✅ CONDITIONAL | ❌ | ✅ NEW | ⚠️ v1.1.1 missing |

## Critical Conflicts Analysis

### 🔴 CONFLICT 1: Format Version
- **v1.1 spec**: Claims version "1.1" with status "IMPLEMENTED"
- **v1.1.1 design**: Claims version "1.1" but is just a design doc
- **v2.0 spec**: Claims version "2.0" with status "REQUIRED REDESIGN"
- **Impact**: Developers don't know which version is actually implemented

### 🔴 CONFLICT 2: n2 Attribute Purpose
- **v1.1 spec**: `node_attributes` contains `{"module_type": "...", "execution_order": ...}`
- **v2.0 spec**: `onnx_attributes` contains `{"axis": 0, "indices_dtype": "int64"}`
- **Impact**: Same key used for completely different data!

### 🔴 CONFLICT 3: Metadata Keys m5-m8
- **v1.1 spec**: Uses m5-m8 for graph I/O specifications
- **v2.0 spec**: Uses m5-m8 for ONNX model metadata
- **Impact**: Complete incompatibility between formats

### 🔴 CONFLICT 4: Implementation Status
- **v1.1 spec**: Claims "✅ IMPLEMENTED - TEZ-127 Complete"
- **v1.1.1 design**: Just a design document, not implemented
- **v2.0 spec**: Says "REQUIRED REDESIGN" implying v1.1 insufficient
- **Impact**: Implementation doesn't match any single specification

## Feature Comparison

| Feature | v1.1 | v1.1.1 | v2.0 | Notes |
|---------|------|--------|------|-------|
| **Bidirectional Conversion** | ✅ Full support | ⚠️ Design only | ✅ Enhanced | v1.1 claims implemented |
| **Parameter Storage** | ✅ 3 strategies | ✅ Sidecar | ✅ 3 strategies | Consistent |
| **Custom Attributes** | ✅ Complete | ✅ Focus area | ✅ Enhanced | v1.1.1 is subset |
| **Round-trip Validation** | ✅ 85%+ accuracy | ❌ | ✅ Target 100% | v1.1 has lower target |
| **HTP Integration** | ✅ Full | ✅ Implied | ⚠️ Not mentioned | v2.0 missing HTP |
| **Tool Compatibility** | ✅ yEd, Gephi, etc | ⚠️ Not specified | ⚠️ Breaking change | v2.0 breaks tools |
| **Test Coverage** | ✅ 96/96 tests | ❌ | ❌ | Only v1.1 tested |

## Implementation Reality Check

Looking at the actual implementation files:

### `/modelexport/graphml/onnx_to_graphml_converter.py`
```python
# Actual implementation shows:
- Format version: "1.1" (matches v1.1 spec)
- Uses keys: n0-n6, e0, t0-t1, m0-m8, p0-p2, g0-g3
- Parameter strategies: sidecar, embedded, reference
- Has bidirectional conversion support
```

### `/modelexport/graphml/graphml_to_onnx_converter.py`
```python
# Round-trip implementation shows:
- Reads format version "1.1"
- Expects keys matching v1.1 extended specification
- Successfully performs round-trip conversion
```

## Findings

### 1. Implementation Matches v1.1 Extended
The actual code implementation most closely matches the **v1.1 specification** from `/docs/specs/graphml-format-specification.md`, specifically the extended section (lines 657-850) that describes the enhanced v1.1 features.

### 2. v1.1.1 is a Subset Design Doc
The v1.1.1 document appears to be a design document focusing on custom attributes, which is already incorporated into the v1.1 spec. It's not a separate version but rather design notes.

### 3. v2.0 is a Proposed Redesign
The v2.0 specification is marked "REQUIRED REDESIGN" but has never been implemented. It proposes breaking changes that conflict with the current implementation.

### 4. Key Numbering Conflicts
The specs use the same key IDs (m5-m8) for completely different purposes, creating dangerous ambiguity.

## Recommendation

### Authoritative Version: v1.1 (Extended)

**Rationale**:
1. **It's actually implemented** - Code matches this spec
2. **It's tested** - 96/96 tests pass against this spec
3. **It works** - Round-trip conversion functional at 85%+ accuracy
4. **It's complete** - Includes all features from v1.1.1 design notes

### Action Plan

#### Immediate Actions (This Week)
1. **Consolidate to Single v1.2 Specification**:
   - Base on current v1.1 implementation
   - Incorporate useful clarifications from v1.1.1
   - Fix key numbering conflicts
   - Add missing documentation

2. **Archive Conflicting Specs**:
   - Move v1.1.1 to `/docs/archive/design_notes/`
   - Move v2.0 to `/docs/archive/future_proposals/`
   - Add deprecation notices to both

3. **Create Migration Guide**:
   - Document differences between specs
   - Explain why v1.2 is authoritative
   - Provide upgrade path

#### Specification v1.2 Structure
```
/docs/specs/graphml-format-specification-v1.2.md  (NEW - Authoritative)
/docs/archive/
  ├── design_notes/
  │   └── graphml_v1.1_custom_attributes.md (renamed from v1.1.1)
  └── future_proposals/
      └── graphml_v2.0_proposal.md (renamed from enhanced_graphml_spec_v2.md)
```

## Conclusion

The three competing specifications create dangerous confusion. The v1.1 extended specification (lines 657-850 of `/docs/specs/graphml-format-specification.md`) is the de facto standard as it matches the implementation. We should consolidate to a single v1.2 specification that clarifies all ambiguities and archives the competing versions.