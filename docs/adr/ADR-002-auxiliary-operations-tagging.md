# ADR-002: Universal Auxiliary Operations Tagging Strategy

| Status | Date | Decision Maker(s) | Consulted | Informed |
|--------|------|-------------------|-----------|----------|
| Proposed | 2025-06-26 | Development Team | Stakeholders | Project Team |

## Context and Problem Statement

The ModelExport framework faces a critical issue with auxiliary operations in ONNX graphs that don't directly correspond to PyTorch modules. These operations (Shape, Constant, Cast, Reshape, etc.) are currently left with empty tag lists, causing two major problems:

1. **Graph Filtering Malformation**: When filtering ONNX graphs by hierarchy tags, nodes with empty tags are excluded, creating malformed graphs that cannot execute
2. **Functionality Regression**: Previous implementation provided 100% operation coverage, but recent changes (built-in tracking) lost this capability

**Current State:**
- 31 out of 254 operations (12.2%) have empty tags in BERT-tiny export
- Operations like `/Shape`, `/Constant`, `/Unsqueeze`, `/Cast` are essential for graph execution
- Subgraph filtering by hierarchy tags would break computation graphs

**Core Challenge:** Design a universal solution for tagging auxiliary operations that:
- Ensures 100% operation coverage (no empty tags)
- Maintains graph execution validity during filtering
- Provides meaningful hierarchy context for semantic analysis
- Works universally across all model architectures

## Research Summary

### Existing Infrastructure Analysis

The codebase contains sophisticated infrastructure for handling auxiliary operations:

#### **ONNX Operation Categorization** (`onnx_categorization.py`)
- **Shape/Metadata Operations**: `Shape`, `Size`, `Range` (80% criticality threshold)
- **Parameter Constants**: `Constant`, `ConstantOfShape` (70% threshold)  
- **Type Conversion**: `Cast`, `CastLike` (60% threshold)
- **Tensor Manipulation**: `Reshape`, `Transpose`, `Concat` (85% threshold, critical=False)

#### **Operation Configuration** (`operation_config.py`)
```python
# Auxiliary operations identified by empty patch_targets
'slice': {
    'patch_targets': [],  # No patchable PyTorch function
    'onnx_types': ['Slice'],
    'priority': 3
},
'size': {
    'patch_targets': [],
    'onnx_types': ['Shape'], 
    'priority': 3
}
```

#### **Current Detection Patterns**
- **Empty patch targets**: Operations with no corresponding PyTorch functions
- **Low criticality**: Operations below 90% threshold are often auxiliary
- **Priority levels**: Priority 3-5 operations are typically auxiliary

### Problem Analysis

**Auxiliary Operation Categories Identified:**

1. **Infrastructure Operations**
   - `Shape`, `Size`, `ConstantOfShape`, `NonZero`
   - Extract tensor metadata, don't belong to specific modules
   - **Issue**: Essential for dynamic shape handling

2. **Export Constants**
   - `Constant` nodes generated during ONNX export
   - Provide parameters and values to other operations
   - **Issue**: Required dependencies for computation

3. **Type Conversions**
   - `Cast`, `CastLike` operations
   - ONNX-specific data type handling
   - **Issue**: Critical for tensor compatibility

4. **Tensor Manipulations**
   - `Reshape`, `Transpose`, `Unsqueeze`, `Squeeze`, `Concat`
   - May or may not belong to specific modules
   - **Issue**: Ambiguous ownership, context-dependent

5. **Input Preprocessing**
   - `Slice`, `Gather`, `Where`, `Equal` operations
   - Handle input tensor preparation
   - **Issue**: Bridge between model inputs and first module

## Decision Drivers

### **Primary Requirements:**
1. **Graph Execution Validity**: Filtered graphs must remain executable
2. **100% Operation Coverage**: No operations with empty tags
3. **Semantic Meaningfulness**: Tags should provide useful hierarchy context
4. **Universal Applicability**: Works across all model architectures
5. **Filtering Safety**: Multiple filtering strategies for different use cases

### **Use Cases (from stakeholder feedback):**
1. **Subgraph Filtering**: Extract semantic portions of ONNX models by hierarchy tags for analysis
2. **Structure Analysis**: Understand semantic structure of ONNX models through hierarchy relationships
3. **Targeted Optimization**: Evaluate/optimize specific subgraphs while preserving auxiliary dependencies
4. **Model Manipulation**: Refine/manipulate only portions of ONNX models with proper auxiliary operation handling

### **Stakeholder Requirements:**
- **Tag Format**: Must follow existing `/BertModel/xxx/xxx` hierarchy format (no auxiliary-specific prefixes)
- **Filtering Safety**: Filtered subgraphs must remain executable (auxiliary operations preserved)
- **Semantic Meaning**: Tags should reflect actual model structure relationships
- **Universal Compatibility**: No conflicts with existing non-auxiliary operation tagging

### **Design Constraints:**
- Must integrate with existing categorization infrastructure
- Follow current tagging mechanism format (`/BertModel/xxx/xxx`)
- No specific performance requirements
- Maintain backward compatibility with existing non-auxiliary tagging

## Considered Options

### Option 1: Context Inheritance Strategy
**Approach**: Auxiliary operations inherit tags from connected primary operations

**Pros:**
- Maintains semantic relationships
- Follows data flow dependencies
- Natural hierarchy assignment

**Cons:**
- Complex dependency analysis required
- Ambiguous cases with multiple connections
- May assign auxiliary operations incorrectly

### Option 2: Auxiliary Category Assignment
**Approach**: Create dedicated auxiliary hierarchy paths (e.g., `/auxiliary/shape`, `/auxiliary/constants`)

**Pros:**
- Clear separation of auxiliary vs. primary operations
- Systematic categorization
- Easy to filter out auxiliaries when needed

**Cons:**
- Creates artificial hierarchy not reflecting model structure
- May not provide meaningful semantic context
- Filtering could still break graphs if auxiliaries excluded
- **REJECTED**: Stakeholder feedback requires following existing `/BertModel/xxx/xxx` format

### Option 3: Nearest Module Assignment
**Approach**: Assign auxiliary operations to nearest primary module in graph

**Pros:**
- Simple implementation
- Guarantees assignment
- Maintains module-based hierarchy

**Cons:**
- Arbitrary assignments may mislead semantic analysis
- Doesn't reflect true operational relationships
- Complex "nearest" calculation in large graphs

### Option 4: Hybrid Multi-Strategy Approach
**Approach**: Combine multiple strategies with fallback hierarchy

**Pros:**
- Handles different auxiliary operation types appropriately
- Robust fallback for edge cases
- Leverages existing infrastructure

**Cons:**
- More complex implementation
- Multiple code paths to maintain
- Potential inconsistencies between strategies

## Decision

**Selected Option: Hybrid Multi-Strategy Approach** with the following framework:

### **Hierarchical Auxiliary Tagging Framework**

#### **1. Auxiliary Operation Classification System**
```python
AUXILIARY_OPERATION_TYPES = {
    "infrastructure": {
        "operations": ["Shape", "Size", "ConstantOfShape", "NonZero"],
        "strategy": "context_inheritance_with_fallback"
    },
    "constants": {
        "operations": ["Constant"],
        "strategy": "context_inheritance"  
    },
    "type_conversion": {
        "operations": ["Cast", "CastLike"],
        "strategy": "context_inheritance"
    },
    "tensor_manipulation": {
        "operations": ["Reshape", "Transpose", "Unsqueeze", "Squeeze", "Concat"],
        "strategy": "context_analysis"
    },
    "preprocessing": {
        "operations": ["Slice", "Gather", "Where", "Equal"],
        "strategy": "input_pattern_matching"
    }
}
```

#### **2. Multi-Level Tagging Strategy**
```python
def tag_auxiliary_operation(node, graph_context):
    """
    Tag auxiliary operations using priority-ordered strategies:
    
    1. Context inheritance (data flow analysis)
    2. Input/output pattern matching  
    3. Nearest module assignment
    4. Default module fallback
    """
    
    # Strategy 1: Context Inheritance
    if inherited_tag := analyze_data_flow_context(node, graph_context):
        return inherited_tag
    
    # Strategy 2: Input Pattern Matching
    if input_tag := analyze_input_patterns(node):
        return input_tag
    
    # Strategy 3: Nearest Module Assignment
    if nearest_tag := find_nearest_module_tag(node, graph_context):
        return nearest_tag
    
    # Strategy 4: Default Fallback
    return get_default_module_tag(graph_context)
```

#### **3. Context Analysis Framework**
- **Data Flow Analysis**: Analyze producers and consumers to inherit context
- **Input Pattern Recognition**: Special handling for input preprocessing operations
- **Dependency Chain**: Follow tensor dependencies to determine ownership
- **Fallback Assignment**: Default to embedding or first available module tag

#### **4. Filtering-Safe Design**
```python
class GraphFilteringStrategy:
    def filter_by_hierarchy(self, graph, target_hierarchies, 
                           auxiliary_strategy="include_context"):
        """
        Options:
        - "include_all": Include all auxiliary operations (safest)
        - "include_context": Include auxiliaries supporting selected modules  
        - "exclude_safe": Advanced mode with dependency analysis
        """
```

### **Tag Format Consistency**
- **Follow existing mechanism**: Use `/BertModel/xxx/xxx` format (per stakeholder requirement)
- **No auxiliary-specific prefixes**: Auxiliary operations inherit regular hierarchy tags
- **Semantic context preserved**: Tags reflect actual model structure relationships through inheritance
- **Example**: `/Shape` operation supporting `/BertModel/BertEmbeddings` gets tag `/BertModel/BertEmbeddings`

### **Implementation Integration**
- **Enhance existing coverage**: Extend `_ensure_complete_coverage()` method in HTP strategy
- **Leverage existing infrastructure**: Use categorization and operation config systems
- **Zero conflicts**: Auxiliary operations currently have empty tags, so we're filling gaps not overriding
- **Backward compatibility**: No changes to existing non-auxiliary tagging logic (223 successful operations unchanged)

## Consequences

### **Positive Consequences:**
- ✅ **100% Operation Coverage**: No more empty tag lists
- ✅ **Graph Execution Validity**: Filtered graphs remain executable
- ✅ **Semantic Analysis**: Meaningful hierarchy context for all operations
- ✅ **Universal Applicability**: Works across all model architectures
- ✅ **Robust Filtering**: Multiple filtering strategies for different use cases

### **Negative Consequences:**
- ⚠️ **Increased Complexity**: Multiple tagging strategies to maintain
- ⚠️ **Performance Overhead**: Additional graph analysis required
- ⚠️ **Ambiguous Cases**: Some auxiliary operations may have unclear ownership
- ⚠️ **Tag Inheritance**: Auxiliary operations inherit tags that don't directly correspond to their function

### **Risks and Mitigations:**
- **Risk**: Complex dependency analysis may introduce bugs
  - **Mitigation**: Comprehensive testing with fallback strategies
- **Risk**: Performance impact on large models  
  - **Mitigation**: Optimize graph analysis algorithms, consider caching
- **Risk**: Incorrect tag assignments misleading semantic analysis
  - **Mitigation**: Clear documentation, auxiliary operation metadata tracking

## Implementation Plan

### **Phase 1: Core Framework** (Week 1)
1. Implement auxiliary operation classification system
2. Create multi-strategy tagging framework
3. Integrate with existing `_ensure_complete_coverage()` method

### **Phase 2: Context Analysis** (Week 2)  
1. Implement data flow analysis for context inheritance
2. Add input pattern recognition for preprocessing operations
3. Create nearest module assignment algorithm

### **Phase 3: Filtering Integration** (Week 3)
1. Develop filtering-safe graph extraction
2. Implement multiple auxiliary handling strategies
3. Add comprehensive validation and testing

### **Phase 4: Validation & Optimization** (Week 4)
1. Test across multiple model architectures
2. Performance optimization and profiling
3. Documentation and examples

## Validation Criteria

### **Success Metrics:**
- ✅ **Zero Empty Tags**: No operations with empty tag lists across test models
- ✅ **Graph Validity**: All filtered subgraphs remain executable
- ✅ **Coverage Consistency**: 100% operation coverage maintained across strategies
- ✅ **Performance Acceptable**: <10% overhead on export time
- ✅ **Universal Compatibility**: Works with BERT, ResNet, GPT, custom models

### **Test Cases:**
1. **BERT-tiny**: Verify all 31 previously empty operations get meaningful tags
2. **ResNet-50**: Test with CNN-based auxiliary operations
3. **Custom Models**: Validate universal applicability
4. **Subgraph Filtering**: Test multiple filtering scenarios for graph validity
5. **Performance Benchmarks**: Measure overhead across model sizes

---

**Next Steps:**
1. Stakeholder review and approval of design approach
2. Implementation of Phase 1 core framework
3. Integration testing with existing HTP strategy
4. Documentation of auxiliary operation tagging behavior

**Dependencies:**
- Existing ONNX categorization system
- Current operation configuration infrastructure  
- HTP strategy implementation
- Graph analysis utilities

## Related Decisions

- ADR-001: Record Architecture Decisions
- ADR-003: Graph Filtering Strategies (future)
- ADR-004: Performance Optimization for Large Models (future)

## More Information

- [ONNX Graph Structure Documentation](https://onnx.ai/onnx/intro/python.html)
- [PyTorch ONNX Export Documentation](https://pytorch.org/docs/stable/onnx.html)
- Issue: Auxiliary Operations Regression in HTP Strategy

---
*Last updated: 2025-06-26*
*Next review: 2025-07-26*