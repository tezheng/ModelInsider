"""
Utilities and Data Structures for GraphML Conversion

This module contains shared data structures, constants, and utility functions
used across the GraphML conversion pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class GraphMLConstants:
    """GraphML namespace and attribute constants."""
    GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
    GRAPHML_NS_MAP = {"": GRAPHML_NS}
    
    # Standard attribute keys
    NODE_OP_TYPE = "n0"
    NODE_HIERARCHY_TAG = "n1"
    NODE_ATTRIBUTES_JSON = "n2"
    NODE_NAME = "n3"
    NODE_INPUT_NAMES = "n4"
    NODE_OUTPUT_NAMES = "n5"
    NODE_DOMAIN = "n6"
    NODE_MODULE_TYPE = "n7"  # Not used in v1.1, kept for compatibility
    NODE_EXECUTION_ORDER = "n8"  # Not used in v1.1, kept for compatibility
    EDGE_TENSOR_NAME = "e0"
    EDGE_TENSOR_SHAPE = "t1"
    EDGE_TENSOR_DTYPE = "t0"
    
    # Graph attribute keys
    GRAPH_CLASS_NAME = "g0"
    GRAPH_MODULE_TYPE = "g1"
    GRAPH_EXECUTION_ORDER = "g2"
    GRAPH_TRACED_TAG = "g3"
    GRAPH_INPUTS = "g4"
    GRAPH_OUTPUTS = "g5"
    GRAPH_VALUE_INFO = "g2"  # Reusing g2 for value_info metadata (execution_order is for compound nodes)
    GRAPH_INITIALIZERS_REF = "g3"  # Reusing g3 for initializers reference (traced_tag is for compound nodes)
    
    # Metadata keys
    META_SOURCE_FILE = "m0"
    META_HTP_FILE = "m1"
    META_FORMAT_VERSION = "m2"
    META_TIMESTAMP = "m3"
    META_OPSET_IMPORTS = "m4"
    META_PRODUCER_NAME = "m5"
    META_PRODUCER_VERSION = "m6"
    META_MODEL_VERSION = "m7"
    META_DOC_STRING = "m8"
    
    # Parameter keys
    PARAM_STRATEGY = "p0"
    PARAM_FILE = "p1"
    PARAM_CHECKSUM = "p2"
    PARAM_COUNT = "p3"


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
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    hierarchy_tag: Optional[str] = None
    module_type: Optional[str] = None
    execution_order: Optional[int] = None
    domain: Optional[str] = None


@dataclass
class EdgeData:
    """Data structure for a graph edge."""
    source_id: str
    target_id: str
    tensor_name: str
    tensor_shape: Optional[List[int]] = None
    tensor_dtype: Optional[str] = None


@dataclass
class CompoundNode:
    """Data structure for a hierarchical compound node."""
    id: str
    name: str
    module_path: str
    class_name: str
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None


@dataclass
class GraphData:
    """Complete graph data structure."""
    nodes: List[NodeData] = field(default_factory=list)
    edges: List[EdgeData] = field(default_factory=list)
    inputs: List[NodeData] = field(default_factory=list)
    outputs: List[NodeData] = field(default_factory=list)
    compounds: Dict[str, CompoundNode] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


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


def format_tensor_shape(shape: List[Any]) -> str:
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