# ADR-008: ONNX to GraphML Package Selection

| Status | Date | Decision Maker(s) | Consulted | Informed |
|--------|------|-------------------|-----------|----------|
| Accepted | 2025-07-28 | Development Team | Research Team | Project Team |

## Context and Problem Statement

We need to select a package/approach for converting ONNX models to GraphML format for hierarchical model visualization. GraphML is chosen as the canonical format for visualizing neural network architectures with proper hierarchy preservation. The solution must handle models with 10,000+ nodes efficiently while preserving the hierarchical structure captured by our HTP (Hierarchical Trace-and-Project) strategy.

## Decision Drivers

- **Hierarchy Support**: Must preserve and represent hierarchical/compound graph structures
- **Performance**: Handle large models (10K+ nodes) efficiently
- **Integration**: Clean integration with existing ModelExport infrastructure
- **Maintainability**: Well-documented, stable API
- **Flexibility**: Ability to customize output format and attributes
- **License**: Compatible with project requirements

## Considered Options

### 1. NetworkX with GraphML Writer
- **Description**: Use NetworkX's built-in GraphML read/write capabilities
- **Pros**: 
  - Mature, well-tested library
  - Native GraphML support
  - Excellent documentation
  - Large community
  - Supports compound graphs via nested subgraphs
- **Cons**: 
  - Heavyweight dependency for simple conversion
  - May require additional processing for ONNX specifics
- **Technical Impact**: Medium complexity, good performance

### 2. Custom XML.etree Implementation
- **Description**: Direct GraphML generation using Python's built-in XML library
- **Pros**: 
  - No external dependencies
  - Full control over output format
  - Lightweight and fast
  - Already prototyped in experiments
- **Cons**: 
  - More code to maintain
  - Need to ensure GraphML spec compliance
- **Technical Impact**: Low complexity, excellent performance

### 3. ONNX GraphSurgeon + NetworkX
- **Description**: NVIDIA's ONNX manipulation tool combined with NetworkX
- **Pros**: 
  - Specialized for ONNX graphs
  - Good ONNX node/edge extraction
  - Can simplify graph before export
- **Cons**: 
  - Additional dependency
  - May not preserve all metadata
  - Primarily for graph modification, not export
- **Technical Impact**: Higher complexity, moderate performance

### 4. pygraphviz with GraphML Export
- **Description**: Python interface to Graphviz with export capabilities
- **Pros**: 
  - Powerful graph layout algorithms
  - Multiple export formats
- **Cons**: 
  - Requires Graphviz system installation
  - Complex setup
  - GraphML is not primary format
- **Technical Impact**: High complexity, setup challenges

### 5. onnx.helper.printable_graph() + Custom Parser
- **Description**: Use ONNX's built-in text representation as intermediate format
- **Pros**: 
  - Native ONNX support
  - No external dependencies for ONNX part
- **Cons**: 
  - Requires custom parser for text format
  - Two-step conversion process
  - May lose information in text representation
- **Technical Impact**: Medium complexity, potential data loss

## Decision Outcome

**Chosen option**: **Option 2: Custom XML.etree Implementation**

### Rationale

1. **No External Dependencies**: Uses only Python standard library, reducing complexity and deployment issues
2. **Full Control**: Complete control over GraphML output format and hierarchical structure representation
3. **Performance**: Direct XML generation is fast and memory-efficient
4. **Proven Approach**: Already successfully prototyped in `json_hierarchy_to_graphml.py`
5. **Maintainability**: Simple, readable code that's easy to debug and extend
6. **Integration**: Clean integration with existing HTP metadata format

### Consequences

**Positive:**
- Zero additional dependencies
- Fast and efficient conversion
- Complete control over output format
- Easy to customize for specific visualization tool requirements
- Can directly map HTP hierarchy to GraphML compound nodes

**Negative:**
- Need to maintain GraphML spec compliance ourselves
- No built-in graph algorithms (not needed for conversion)
- Must implement attribute handling and edge creation

**Neutral:**
- Requires understanding of GraphML specification
- Need to write comprehensive tests for spec compliance

## Implementation Notes

```python
# Core implementation structure
class ONNXToGraphMLConverter:
    def __init__(self, exclude_initializers=True):
        self.exclude_initializers = exclude_initializers
        
    def convert(self, onnx_model, htp_metadata=None):
        """Convert ONNX to GraphML with optional hierarchy."""
        graphml_root = self._create_graphml_root()
        self._define_attribute_keys(graphml_root)
        
        if htp_metadata:
            root_graph = self._build_hierarchical_graph(onnx_model, htp_metadata)
        else:
            root_graph = self._build_flat_graph(onnx_model)
            
        graphml_root.append(root_graph)
        return ET.tostring(graphml_root, encoding='unicode')
```

### GraphML Structure for Hierarchical Graphs
```xml
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="op_type" attr.type="string"/>
  <key id="d1" for="node" attr.name="hierarchy_tag" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="n0">
      <graph id="g0"> <!-- Compound node -->
        <node id="n1">...</node>
      </graph>
    </node>
  </graph>
</graphml>
```

## Validation/Confirmation

- Export bert-tiny model and validate with yEd Graph Editor
- Verify compound node structure preserves HTP hierarchy
- Performance test with models containing 10K+ nodes
- Validate GraphML output against official schema

## Related Decisions

- ADR-004: ONNX Node Tagging Priorities (defines metadata to preserve)
- TEZ-102: Integrate HTP metadata with GraphML (implementation details)

## More Information

- GraphML Specification: http://graphml.graphdrawing.org/
- Existing prototype: `/mnt/d/BYOM/modelexport/experiments/model_architecture_visualization/json_hierarchy_to_graphml.py`
- Visualization tool research: `graphml_viewers_research.md`

---
*Last updated: 2025-07-28*
*Next review: 2025-08-28*