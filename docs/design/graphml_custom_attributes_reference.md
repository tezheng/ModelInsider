# GraphML Custom Attributes Quick Reference

## Overview

This document provides implementation-specific details for the modelexport project's use of GraphML v1.1 custom attributes. This is NOT part of the generic GraphML v1.1 specification but rather how this specific project implements it.

## Custom Attribute Keys

### Node Attributes (`n` prefix)
| Key | Attribute Name | Type | Description | Example |
|-----|---------------|------|-------------|---------|
| n0 | op_type | string | ONNX operator type | "Conv", "Add", "MatMul" |
| n1 | hierarchy_tag | string | Full module path | "/BertModel/Encoder/Layer.0/Attention" |
| n2 | node_attributes | string | JSON-encoded ONNX node attributes | '{"kernel_shape": [3, 3]}' |
| n3 | name | string | Original ONNX node name | "Conv_123" |

### Graph Attributes (`g` prefix)
| Key | Attribute Name | Type | Description | Example |
|-----|---------------|------|-------------|---------|
| g0 | class_name | string | Model class name | "BertModel" |
| g1 | module_type | string | Framework type | "huggingface" |
| g2 | execution_order | string | Execution order | "0" |
| g3 | traced_tag | string | Traced tag path | "/BertModel" |
| g4 | graph_inputs | string | JSON-encoded inputs | '[{"name": "input", "shape": [1, 512]}]' |
| g5 | graph_outputs | string | JSON-encoded outputs | '[{"name": "output", "shape": [1, 768]}]' |

### Edge Attributes (`e` prefix)
| Key | Attribute Name | Type | Description | Example |
|-----|---------------|------|-------------|---------|
| e0 | tensor_name | string | Tensor name | "hidden_states" |

### Metadata Attributes (`m` prefix)
| Key | Attribute Name | Type | Description | Example |
|-----|---------------|------|-------------|---------|
| m0 | source_onnx | string | Source ONNX path | "model.onnx" |
| m1 | source_htp | string | HTP metadata path | "metadata.json" |
| m2 | format_version | string | GraphML format version | "1.1" |
| m3 | export_timestamp | string | ISO timestamp | "2025-01-31T10:30:00" |

## Filtering Rules

### Core Principle: "What comes from ONNX goes back to ONNX intact"

### Attributes Excluded from ONNX (GraphML-only metadata)
These are attributes ADDED by the GraphML converter, NOT from the original ONNX:
```python
GRAPHML_METADATA_ATTRS = {
    "hierarchy_tag",      # Added for visualization
    "module_type",        # Added from HTP metadata
    "execution_order",    # Added from tracing
    "scope",             # Added from module analysis
    "traced_tag",        # Added from HTP metadata
    "class_name",        # Added from model structure
}
```

### Attributes Preserved (ALL original ONNX attributes)
- ALL attributes from the original ONNX model are preserved
- This includes standard and non-standard attributes
- Custom domain attributes are preserved
- Vendor-specific attributes are preserved

### Storage Strategy
1. Original ONNX attributes → stored in `node_attributes` JSON
2. GraphML metadata → stored as separate node data elements
3. During conversion back: extract from `node_attributes`, ignore metadata

## Usage Example

```python
# In ONNXToGraphMLConverter
def _add_node_to_graph(self, node, parent_graph):
    node_elem = ET.SubElement(parent_graph, "node", {"id": node.id})
    
    # Standard ONNX attribute
    self._add_data(node_elem, GC.NODE_OP_TYPE, node.op_type)
    
    # Custom attributes (preserved in GraphML, filtered from ONNX)
    self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, node.hierarchy_tag)
    self._add_data(node_elem, GC.NODE_MODULE_TYPE, node.module_type)
    
# In GraphMLToONNXConverter  
def _should_include_in_onnx(self, op_type, attr_name):
    # Filter out custom attributes
    if attr_name in GRAPHML_ONLY_ATTRS:
        return False
    # Filter out attributes with GraphML key prefixes
    if attr_name.startswith(('n', 'g', 'e', 'm', 'p', 't')):
        return False
    return True
```

## Testing Custom Attributes

Test files verify:
1. Custom attributes are preserved in GraphML export
2. Custom attributes are filtered during GraphML→ONNX conversion
3. Valid ONNX attributes pass through round-trip conversion
4. Attribute filtering logic works correctly

See: `tests/graphml/test_custom_attributes.py`