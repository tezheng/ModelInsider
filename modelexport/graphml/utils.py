"""
Utilities and Data Structures for GraphML Conversion

This module contains shared data structures, constants, and utility functions
used across the GraphML conversion pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class GraphMLConstants:
    """GraphML namespace and attribute constants (v1.3)."""
    GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
    GRAPHML_NS_MAP: ClassVar[dict[str, str]] = {"": GRAPHML_NS}
    
    # Graph attributes (for compound nodes)
    GRAPH_CLASS_NAME = "d0"
    GRAPH_MODULE_TYPE = "d1"
    GRAPH_EXECUTION_ORDER = "d2"
    GRAPH_TRACED_TAG = "d3"
    
    # Node attributes
    NODE_OP_TYPE = "n0"
    NODE_HIERARCHY_TAG = "n1"
    NODE_ATTRIBUTES_JSON = "n2"
    NODE_NAME = "n3"
    NODE_INPUT_NAMES = "n4"  # v1.3
    NODE_OUTPUT_NAMES = "n5"  # v1.3
    NODE_DOMAIN = "n6"  # v1.3
    
    # Edge attributes (v1.3 naming)
    EDGE_TENSOR_NAME = "e0"
    EDGE_TENSOR_TYPE = "e1"  # v1.3 (was t0)
    EDGE_TENSOR_SHAPE = "e2"  # v1.3 (was t1)
    EDGE_TENSOR_DATA_REF = "e3"  # v1.3 (was t2)
    
    # Metadata keys (v1.3 naming - resolves conflicts)
    META_SOURCE_ONNX = "meta0"  # v1.3 (was m0)
    META_SOURCE_HTP = "meta1"  # v1.3 (was m1)
    META_FORMAT_VERSION = "meta2"  # v1.3 (was m2)
    META_TIMESTAMP = "meta3"  # v1.3 (was m3)
    META_OPSET_IMPORTS = "meta4"  # v1.3
    META_PRODUCER_NAME = "meta5"  # v1.3 (was m5 - conflict resolved)
    META_PRODUCER_VERSION = "meta6"  # v1.3 (was m6 - conflict resolved)
    META_MODEL_VERSION = "meta7"  # v1.3 (was m7 - conflict resolved)
    META_DOC_STRING = "meta8"  # v1.3 (was m8 - conflict resolved)
    
    # Parameter keys (v1.3 naming)
    PARAM_STRATEGY = "param0"  # v1.3 (was p0)
    PARAM_FILE = "param1"  # v1.3 (was p1)
    PARAM_CHECKSUM = "param2"  # v1.3 (was p2)
    
    # Graph I/O metadata keys (v1.3 naming)
    GRAPH_INPUTS = "io0"  # v1.3 (was g0)
    GRAPH_OUTPUTS = "io1"  # v1.3 (was g1)
    GRAPH_VALUE_INFO = "io2"  # v1.3 (was g2)
    GRAPH_INITIALIZERS_REF = "io3"  # v1.3 (was g3)


class NodeType(Enum):
    """Types of nodes in the graph."""
    OPERATION = "operation"
    INPUT = "input"
    OUTPUT = "output"
    COMPOUND = "compound"


@dataclass
class NodeData:
    """Data structure for a graph node."""
    id: str
    name: str
    op_type: str
    node_type: NodeType = NodeType.OPERATION
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    hierarchy_tag: str | None = None
    module_type: str | None = None
    execution_order: int | None = None


@dataclass
class EdgeData:
    """Data structure for a graph edge."""
    source_id: str
    target_id: str
    tensor_name: str
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None


@dataclass
class CompoundNode:
    """Data structure for a hierarchical compound node."""
    id: str
    name: str
    module_path: str
    class_name: str
    children: list[str] = field(default_factory=list)
    parent: str | None = None


@dataclass
class GraphData:
    """Complete graph data structure."""
    nodes: list[NodeData] = field(default_factory=list)
    edges: list[EdgeData] = field(default_factory=list)
    inputs: list[NodeData] = field(default_factory=list)
    outputs: list[NodeData] = field(default_factory=list)
    compounds: dict[str, CompoundNode] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def sanitize_node_id(name: str) -> str:
    """
    Sanitize a node name to be a valid GraphML ID.
    
    Args:
        name: Original node name
        
    Returns:
        Sanitized ID safe for GraphML
    """
    # Replace invalid characters with underscores
    sanitized = name.replace("/", "_").replace(".", "_").replace(":", "_")
    sanitized = sanitized.replace(" ", "_").replace("-", "_")
    
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    
    return sanitized or "node"


def get_tensor_dtype_name(dtype_int: int) -> str:
    """
    Convert ONNX data type integer to human-readable name.
    
    Args:
        dtype_int: ONNX data type constant
        
    Returns:
        Human-readable type name
    """
    dtype_map = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        8: "string",
        9: "bool",
        10: "float16",
        11: "float64",
        12: "uint32",
        13: "uint64",
    }
    return dtype_map.get(dtype_int, f"unknown_{dtype_int}")


def format_tensor_shape(shape: list[Any]) -> str:
    """
    Format tensor shape for display.
    
    Args:
        shape: List of dimension values
        
    Returns:
        Formatted shape string
    """
    if not shape:
        return "scalar"
    
    formatted_dims = []
    for dim in shape:
        if isinstance(dim, int):
            formatted_dims.append(str(dim))
        elif hasattr(dim, 'dim_value'):
            formatted_dims.append(str(dim.dim_value) if dim.dim_value else "?")
        elif hasattr(dim, 'dim_param'):
            formatted_dims.append(dim.dim_param)
        else:
            formatted_dims.append("?")
    
    return f"[{', '.join(formatted_dims)}]"