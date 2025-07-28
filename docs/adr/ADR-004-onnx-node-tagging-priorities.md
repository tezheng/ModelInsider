# ADR-004: ONNX Node Tagging Priority System

| Status | Date | Decision Maker(s) | Consulted | Informed |
|--------|------|-------------------|-----------|----------|
| Accepted | 2025-07-08 | Development Team | Architecture Team | Project Team |

## Context and Problem Statement

The ModelExport framework requires a systematic approach to tag ONNX nodes with their corresponding PyTorch module hierarchy. This is critical for:

1. **Hierarchy Preservation**: Maintaining the original model structure in exported ONNX
2. **Subgraph Filtering**: Enabling extraction of semantic portions by hierarchy tags
3. **100% Coverage**: Ensuring no nodes are left untagged (empty tags break graph filtering)
4. **Universal Design**: Working across all model architectures without hardcoded logic

**Current Challenge**: ONNX nodes have varying levels of scope information, from fully qualified scope paths to no scope at all. We need a robust system to handle all cases while guaranteeing no empty tags.

## Decision Drivers

- **Zero Empty Tags**: CARDINAL RULE - no untagged nodes allowed
- **Universal Applicability**: Must work with any PyTorch model architecture
- **No Hardcoded Logic**: Dynamic root extraction, no model-specific patterns
- **Semantic Accuracy**: Tags should reflect actual module relationships where possible
- **Performance**: Efficient tagging without expensive graph analysis
- **Fallback Guarantee**: Always provide a valid tag even in edge cases

## Considered Options

1. **Single-Strategy Approach**: Use only direct scope mapping
2. **Binary Fallback**: Direct mapping with root fallback only
3. **Multi-Priority System**: Progressive fallback through multiple strategies
4. **ML-Based Inference**: Use pattern recognition for tag assignment

## Decision Outcome

**Chosen option**: Multi-Priority System with 4 progressive priorities

### Rationale

The 4-priority system provides the optimal balance of accuracy, coverage, and performance:
- **Priority 1-2**: Handle well-structured nodes with clear scope information
- **Priority 3**: Optional enhancement for specialized operations
- **Priority 4**: Absolute guarantee against empty tags

## Implementation Details

### Priority 1: Direct Scope Match
```python
def _direct_scope_match(self, node: onnx.NodeProto) -> Optional[str]:
    """
    Extract scope from node name and map directly to hierarchy.
    
    Examples:
    - Node: "/encoder/layer.0/attention/self/MatMul"
    - Scope: "encoder.layer.0.attention.self"  
    - Tag: "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"
    """
    scope = self._extract_scope_from_node(node)
    return self.scope_to_tag.get(scope)
```

**Success Rate**: ~40-60% of nodes
**Use Case**: Well-structured nodes with full scope paths

### Priority 2: Parent Scope Match
```python
def _find_parent_scope_tag(self, scope: str) -> Optional[str]:
    """
    Progressive parent scope matching for hierarchical fallback.
    
    Examples:
    - Try: "encoder.layer.0.attention.self"
    - Try: "encoder.layer.0.attention"  
    - Try: "encoder.layer.0"
    - Try: "encoder"
    """
    while scope:
        if scope in self.scope_to_tag:
            return self.scope_to_tag[scope]
        scope = '.'.join(scope.split('.')[:-1])
    return None
```

**Success Rate**: ~20-30% of nodes
**Use Case**: Nodes with partial scope information

### Priority 3: Operation-Based Fallback (Optional)
```python
def _operation_based_fallback(self, node: onnx.NodeProto) -> Optional[str]:
    """
    Context-aware operation type matching for specialized cases.
    
    Examples:
    - MatMul in attention context
    - Add in residual connections
    - LayerNorm in normalization layers
    """
    if not self.enable_operation_fallback:
        return None
    
    # Implementation depends on operation type and context
    return self._analyze_operation_context(node)
```

**Success Rate**: ~5-10% of nodes
**Use Case**: Specialized operations with predictable patterns
**Default**: Disabled (enable via `enable_operation_fallback=True`)

### Priority 4: Root Fallback (Never Empty)
```python
def _root_fallback(self) -> str:
    """
    Guaranteed fallback to dynamically extracted model root.
    
    Examples:
    - BertModel → "/BertModel"
    - ResNet → "/ResNet"  
    - GPT2LMHeadModel → "/GPT2LMHeadModel"
    """
    return self.model_root_tag
```

**Success Rate**: 100% of remaining nodes
**Use Case**: Absolute guarantee against empty tags
**Implementation**: Dynamic root extraction from hierarchy data

## Consequences

### Positive Consequences
- ✅ **100% ONNX Node Coverage**: Every node gets a valid tag
- ✅ **Zero Empty Tags**: Eliminates graph filtering failures
- ✅ **Universal Design**: Works with any model architecture
- ✅ **No Hardcoded Logic**: Dynamic root extraction
- ✅ **Semantic Accuracy**: Progressive degradation maintains meaning
- ✅ **Performance**: Efficient O(1) lookups with fallback chains

### Negative Consequences
- ⚠️ **Tag Dilution**: Some nodes may get overly generic tags (root fallback)
- ⚠️ **Implementation Complexity**: Multiple code paths to maintain
- ⚠️ **Optional Complexity**: Priority 3 adds optional complexity

### Neutral Consequences
- **Configurable Fallback**: Priority 3 can be enabled/disabled as needed
- **Progressive Degradation**: Maintains best possible semantic accuracy

## Implementation Notes

### Core Algorithm
```python
def tag_single_node(self, node: onnx.NodeProto) -> str:
    """Tag a single ONNX node using 4-priority system."""
    
    # Priority 1: Direct scope match
    if tag := self._direct_scope_match(node):
        return tag
    
    # Priority 2: Parent scope match  
    if tag := self._find_parent_scope_tag(self._extract_scope_from_node(node)):
        return tag
        
    # Priority 3: Operation-based fallback (optional)
    if self.enable_operation_fallback:
        if tag := self._operation_based_fallback(node):
            return tag
    
    # Priority 4: Root fallback (never empty)
    return self._root_fallback()
```

### Scope Extraction
```python
def _extract_scope_from_node(self, node: onnx.NodeProto) -> str:
    """
    Extract scope from ONNX node name.
    
    Examples:
    - "/encoder/layer.0/attention/self/MatMul" → "encoder.layer.0.attention.self"
    - "/embeddings/Add" → "embeddings"
    - "/Constant_123" → "__root__"
    """
    node_name = node.name or ""
    if not node_name or not node_name.startswith('/'):
        return "__root__"
    
    parts = node_name.strip('/').split('/')
    if len(parts) <= 1:
        return "__root__"
    
    # Remove operation name, keep scope path
    scope_parts = parts[:-1]
    return '.'.join(scope_parts) if scope_parts else "__root__"
```

### Root Extraction
```python
def _extract_model_root_tag(self) -> str:
    """Dynamically extract model root from hierarchy data."""
    for module_path, module_info in self.hierarchy_data.items():
        if not module_path:  # Root module has empty path
            class_name = module_info.get('class_name', 'Model')
            return f"/{class_name}"
    
    # Fallback if no root found
    return "/Model"
```

## Validation/Confirmation

### Success Metrics
- **100% Node Coverage**: All ONNX nodes receive valid tags
- **Zero Empty Tags**: No nodes with empty or null tags
- **Tag Distribution**: Reasonable distribution across priority levels
- **Performance**: <5% overhead on export time
- **Universal Compatibility**: Works with BERT, ResNet, GPT, custom models

### Test Cases
1. **BERT-tiny**: 254 nodes → 100% coverage, 0 empty tags
2. **ResNet-50**: CNN architecture with different scope patterns
3. **GPT-2**: Transformer with attention-heavy operations
4. **Custom Models**: Edge cases and unusual architectures
5. **Scope Variations**: Models with different naming conventions

### Monitoring
```python
def get_tagging_statistics(self, onnx_model: onnx.ModelProto) -> Dict[str, int]:
    """Generate statistics for tagging performance analysis."""
    return {
        'total_nodes': len(onnx_model.graph.node),
        'direct_matches': self._count_direct_matches(),
        'parent_matches': self._count_parent_matches(),
        'operation_matches': self._count_operation_matches(),
        'root_fallbacks': self._count_root_fallbacks(),
        'empty_tags': 0  # Always 0 by design
    }
```

## Detailed Analysis of Options

### Option 1: Single-Strategy Approach
- **Description**: Use only direct scope mapping
- **Pros**: Simple, fast, clear semantics
- **Cons**: Low coverage (~40-60%), many empty tags
- **Technical Impact**: Fails CARDINAL RULE of no empty tags

### Option 2: Binary Fallback
- **Description**: Direct mapping with root fallback only
- **Pros**: Simple, guaranteed coverage
- **Cons**: Loses semantic information, too many root tags
- **Technical Impact**: Poor tag distribution, limited usefulness

### Option 3: Multi-Priority System ✅
- **Description**: Progressive fallback through 4 priorities
- **Pros**: Optimal accuracy/coverage balance, configurable
- **Cons**: More complex implementation
- **Technical Impact**: Best semantic preservation with guaranteed coverage

### Option 4: ML-Based Inference
- **Description**: Pattern recognition for tag assignment
- **Pros**: Potentially high accuracy
- **Cons**: Complex, training overhead, unpredictable
- **Technical Impact**: Violates universal design principles

## Related Decisions

- ADR-001: Record Architecture Decisions
- ADR-002: Auxiliary Operations Tagging Strategy
- ADR-003: Standard vs Function Export

## More Information

- [ONNX Node Structure Documentation](https://onnx.ai/onnx/intro/python.html)
- [PyTorch Module Hierarchy](https://pytorch.org/docs/stable/nn.html)
- Implementation: `modelexport/core/onnx_node_tagger.py`
- Examples: `examples/demo_onnx_node_tagging.py`

---
*Last updated: 2025-07-08*
*Next review: 2025-10-08*