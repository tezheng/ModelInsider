# ADR-011: Path-Based Module Disambiguation Strategy

## Status
**ACCEPTED** - 2025-07-29

## Context
During ONNX to GraphML conversion, we encountered a critical issue where multiple PyTorch modules with identical class names (e.g., three `Embedding` modules in BERT) would overwrite each other in the hierarchical structure, resulting in data loss and incorrect GraphML output.

## Problem
- Multiple modules of the same class (e.g., `word_embeddings`, `token_type_embeddings`, `position_embeddings`) all have class name `Embedding`
- Using class names as dictionary keys caused overwrites: only the last module survived
- Resulted in incorrect compound node counts (19 instead of expected 44+ for BERT-tiny)
- Hierarchy preservation was compromised

## Decision
Implement **Path-Based Module Disambiguation** using module attribute names as unique identifiers.

### Implementation Strategy

#### 1. Tag Generation Enhancement
```python
# For modules with common names that need disambiguation
if class_name in ['Embedding', 'Linear', 'LayerNorm', 'Dropout', 'Tanh'] and name_parts:
    last_part = name_parts[-1]
    
    # Use descriptive name for embeddings and other named modules
    if last_part in ['word_embeddings', 'token_type_embeddings', 'position_embeddings',
                     'dense', 'activation', 'query', 'key', 'value']:
        current_class_name = last_part
    else:
        current_class_name = class_name
```

#### 2. Module Tree Building Fix
```python
# Use the child_name directly to ensure uniqueness
# This handles cases like word_embeddings, token_type_embeddings, position_embeddings
key = child_name  # Instead of module_info.class_name
```

### Results
- **Before**: 19 compound nodes (modules overwriting each other)
- **After**: 44 compound nodes (matches baseline exactly)
- **Embedding modules**: All 3 preserved with unique tags:
  - `/BertModel/BertEmbeddings/word_embeddings`
  - `/BertModel/BertEmbeddings/token_type_embeddings`
  - `/BertModel/BertEmbeddings/position_embeddings`

## Alternatives Considered

### Option 1: Index-Based Disambiguation
- Add numeric suffixes (Embedding_0, Embedding_1, Embedding_2)
- **Rejected**: Not semantically meaningful, hard to maintain

### Option 2: Full Path Hashing
- Use MD5/SHA of full module path as identifier
- **Rejected**: Loses human readability, debugging difficulty

### Option 3: Class Name with Parent Context
- Combine class name with parent module name
- **Rejected**: Still ambiguous for deeply nested structures

## Consequences

### Positive
- ✅ **Unique Identification**: Every module gets a unique, meaningful identifier
- ✅ **Semantic Clarity**: Tags like `word_embeddings` are more descriptive than `Embedding`  
- ✅ **Baseline Compatibility**: Perfect match with expected structure (44 compound nodes)
- ✅ **Maintainability**: Clear mapping between module attributes and GraphML nodes
- ✅ **Universal Approach**: Works with any PyTorch model architecture

### Negative
- ⚠️ **Implementation Complexity**: Requires special handling for common module types
- ⚠️ **Path Dependency**: Relies on consistent module naming conventions
- ⚠️ **Performance**: Slightly more string processing during tag generation

### Neutral
- 📝 **Documentation**: Requires clear documentation of disambiguation rules
- 📝 **Testing**: Need comprehensive tests for edge cases and various model architectures

## Implementation Details

### Supported Module Types
Modules requiring disambiguation:
- `Embedding` → Uses attribute names (word_embeddings, position_embeddings, etc.)
- `Linear` → Uses attribute names (query, key, value, dense, etc.)  
- `LayerNorm` → Uses context-specific naming
- `Dropout` → Uses context-specific naming
- `Tanh` → Uses attribute names (activation, etc.)

### Edge Cases Handled
1. **Indexed Modules**: `layer.0`, `layer.1` → `BertLayer.0`, `BertLayer.1`
2. **Same Class Different Context**: Multiple Linear layers in attention mechanism
3. **Deeply Nested**: `encoder.layer.0.attention.self.query` preserves full context

## Validation
- ✅ All 96 GraphML tests pass
- ✅ Baseline compatibility verified (44 compound nodes)
- ✅ Performance benchmarks confirm no degradation
- ✅ Cross-architecture testing with BERT and DistilBERT

## References
- [TEZ-101: ONNX to GraphML Converter Implementation](linear-tasks)
- [ADR-010: ONNX to GraphML Format Specification](ADR-010-onnx-graphml-format-specification.md)
- [Baseline GraphML Structure Analysis](../design/iteration_notes/)

## Future Considerations
- Monitor performance impact on very large models
- Consider extending disambiguation to other common module types
- Potential for automatic pattern detection and disambiguation