# ADR-009: GraphML Converter Architecture

| Status | Date | Decision Maker(s) | Consulted | Informed |
|--------|------|-------------------|-----------|----------|
| Proposed | 2025-07-28 | Development Team | Architecture Team | Project Team |

## Context and Problem Statement

We need to design a clean, maintainable architecture for converting ONNX models to GraphML format. The architecture must support both flat ONNX graph export and hierarchical export with HTP metadata integration, while maintaining separation of concerns and following SOLID principles.

## Decision Drivers

- **Separation of Concerns**: Clean separation between ONNX parsing, GraphML generation, and hierarchy handling
- **Extensibility**: Easy to add new features without modifying existing code
- **Testability**: Components should be independently testable
- **Performance**: Efficient handling of large models (10K+ nodes)
- **No Coupling**: GraphML module must not depend on strategy implementations

## Architecture Design

### Module Structure

```
modelexport/graphml/
├── __init__.py           # Module exports
├── converter.py          # Base converter interface and implementation
├── onnx_parser.py        # ONNX graph parsing logic
├── graphml_writer.py     # GraphML XML generation
├── hierarchical_converter.py  # HTP metadata integration
├── metadata_reader.py    # HTP metadata file reader
├── styling.py            # Visual styling configuration
└── utils.py              # Helper functions and constants
```

### Core Components

#### 1. Base Converter (`converter.py`)
```python
class ONNXToGraphMLConverter:
    """Base converter for ONNX to GraphML transformation."""
    
    def __init__(self, exclude_initializers=True, exclude_attributes=None):
        self.parser = ONNXGraphParser()
        self.writer = GraphMLWriter()
        
    def convert(self, onnx_model_path: str) -> str:
        """Convert ONNX model to GraphML string."""
        
    def save(self, onnx_model_path: str, output_path: str) -> None:
        """Convert and save to file."""
```

#### 2. ONNX Parser (`onnx_parser.py`)
```python
class ONNXGraphParser:
    """Extract graph structure from ONNX models."""
    
    def parse(self, onnx_model) -> GraphData:
        """Parse ONNX model into internal graph representation."""
        
    def extract_nodes(self, graph) -> List[NodeData]:
        """Extract computational nodes (exclude initializers)."""
        
    def extract_edges(self, graph) -> List[EdgeData]:
        """Extract tensor connections between nodes."""
```

#### 3. GraphML Writer (`graphml_writer.py`)
```python
class GraphMLWriter:
    """Generate GraphML XML from graph data."""
    
    def write(self, graph_data: GraphData) -> ET.Element:
        """Convert graph data to GraphML XML structure."""
        
    def define_attributes(self) -> List[AttributeDef]:
        """Define GraphML attribute keys."""
        
    def create_node_element(self, node: NodeData) -> ET.Element:
        """Create GraphML node element."""
```

#### 4. Hierarchical Converter (`hierarchical_converter.py`)
```python
class HierarchicalGraphMLConverter(ONNXToGraphMLConverter):
    """Extended converter with HTP hierarchy support."""
    
    def __init__(self, htp_metadata_path: str, **kwargs):
        super().__init__(**kwargs)
        self.metadata_reader = MetadataReader(htp_metadata_path)
        
    def convert(self, onnx_model_path: str) -> str:
        """Convert with hierarchy preservation."""
        
    def build_compound_nodes(self) -> Dict[str, CompoundNode]:
        """Create nested graph structure from hierarchy."""
```

### Data Flow

```
ONNX Model → ONNXGraphParser → GraphData → GraphMLWriter → GraphML
                                    ↑
                                    |
                            HTP Metadata Reader
                                    |
                            (for hierarchical mode)
```

### Design Patterns

1. **Strategy Pattern**: Different converters for flat vs hierarchical export
2. **Builder Pattern**: GraphML construction with incremental element addition
3. **Adapter Pattern**: ONNX model to internal graph representation
4. **Template Method**: Base converter with extension points

## Decision Outcome

**Chosen approach**: Modular architecture with clear separation of concerns

### Implementation Strategy

1. **Phase 1**: Core components (parser, writer, base converter)
2. **Phase 2**: Hierarchical extension with metadata reader
3. **Phase 3**: Styling and visual enhancements
4. **Phase 4**: CLI integration

### Consequences

**Positive:**
- Clean, testable components
- Easy to extend with new features
- No coupling to existing strategies
- Reusable components

**Negative:**
- More files to maintain
- Slightly more complex than monolithic approach

**Neutral:**
- Requires careful interface design
- Need comprehensive test coverage

## Validation/Confirmation

- Unit tests for each component
- Integration tests with real ONNX models
- Performance benchmarks with large models
- GraphML schema validation

## Related Decisions

- ADR-008: ONNX to GraphML Package Selection
- ADR-003: Standard vs Function Export (no coupling requirement)

---
*Last updated: 2025-07-28*
*Next review: 2025-08-28*