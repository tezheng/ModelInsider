"""
GraphML Constants and Configuration

This module defines all constants used across the GraphML conversion system.
Centralizes version numbers, key mappings, and other shared configuration.
Eliminates magic numbers and provides a single source of truth.

Linear Tasks: 
- TEZ-135 (GraphML v1.3: Schema-Driven Specification)
- TEZ-133 (Code Quality Improvements)
"""

from dataclasses import dataclass
from ..version import GRAPHML_VERSION

# GraphML Format Version
GRAPHML_FORMAT_VERSION = GRAPHML_VERSION
GRAPHML_FORMAT_DESCRIPTION = "Schema-Driven Specification"

# GraphML v1.3 Key Definitions
GRAPHML_V13_KEYS = {
    # Graph keys (compound nodes)
    "d0": {"for": "graph", "attr.name": "class_name", "attr.type": "string"},
    "d1": {"for": "graph", "attr.name": "module_type", "attr.type": "string"},
    "d2": {"for": "graph", "attr.name": "execution_order", "attr.type": "int"},
    "d3": {"for": "graph", "attr.name": "traced_tag", "attr.type": "string"},
    
    # Node keys
    "n0": {"for": "node", "attr.name": "op_type", "attr.type": "string"},
    "n1": {"for": "node", "attr.name": "hierarchy_tag", "attr.type": "string"},
    "n2": {"for": "node", "attr.name": "onnx_attributes", "attr.type": "string"},
    "n3": {"for": "node", "attr.name": "name", "attr.type": "string"},
    "n4": {"for": "node", "attr.name": "input_names", "attr.type": "string"},
    "n5": {"for": "node", "attr.name": "output_names", "attr.type": "string"},
    "n6": {"for": "node", "attr.name": "domain", "attr.type": "string"},
    
    # Edge keys (v1.3 naming)
    "e0": {"for": "edge", "attr.name": "tensor_name", "attr.type": "string"},
    "e1": {"for": "edge", "attr.name": "tensor_type", "attr.type": "string"},
    "e2": {"for": "edge", "attr.name": "tensor_shape", "attr.type": "string"},
    "e3": {"for": "edge", "attr.name": "tensor_data_ref", "attr.type": "string"},
    
    # Metadata keys (v1.3 naming)
    "meta0": {"for": "graph", "attr.name": "source_onnx_file", "attr.type": "string"},
    "meta1": {"for": "graph", "attr.name": "source_htp_file", "attr.type": "string"},
    "meta2": {"for": "graph", "attr.name": "format_version", "attr.type": "string"},
    "meta3": {"for": "graph", "attr.name": "export_timestamp", "attr.type": "string"},
    "meta4": {"for": "graph", "attr.name": "opset_imports", "attr.type": "string"},
    "meta5": {"for": "graph", "attr.name": "producer_name", "attr.type": "string"},
    "meta6": {"for": "graph", "attr.name": "producer_version", "attr.type": "string"},
    "meta7": {"for": "graph", "attr.name": "model_version", "attr.type": "string"},
    "meta8": {"for": "graph", "attr.name": "doc_string", "attr.type": "string"},
    
    # Parameter keys (v1.3 naming)
    "param0": {"for": "graph", "attr.name": "parameter_strategy", "attr.type": "string"},
    "param1": {"for": "graph", "attr.name": "parameter_file", "attr.type": "string"},
    "param2": {"for": "graph", "attr.name": "parameter_checksum", "attr.type": "string"},
    
    # I/O keys (v1.3 naming)
    "io0": {"for": "graph", "attr.name": "graph_inputs", "attr.type": "string"},
    "io1": {"for": "graph", "attr.name": "graph_outputs", "attr.type": "string"},
    "io2": {"for": "graph", "attr.name": "value_info", "attr.type": "string"},
    "io3": {"for": "graph", "attr.name": "initializers_ref", "attr.type": "string"},
}

# Total number of required keys in v1.3
GRAPHML_V13_KEY_COUNT = len(GRAPHML_V13_KEYS)

# Key mappings for easier lookup (key_id -> attr_name)
GRAPHML_V13_KEY_MAPPINGS = {
    key_id: info["attr.name"] 
    for key_id, info in GRAPHML_V13_KEYS.items()
}

# Reverse mappings (attr_name -> key_id)
GRAPHML_V13_ATTR_TO_KEY = {
    info["attr.name"]: key_id 
    for key_id, info in GRAPHML_V13_KEYS.items()
}

# JSON field keys (require JSON parsing)
GRAPHML_JSON_KEYS = {
    "n2",  # onnx_attributes
    "n4",  # input_names
    "n5",  # output_names
    "meta4",  # opset_imports
    "io0",  # graph_inputs
    "io1",  # graph_outputs
    "io2",  # value_info
    "io3",  # initializers_ref
    "e2",  # tensor_shape
}

# Required metadata keys
GRAPHML_REQUIRED_METADATA = {
    "meta2",  # format_version
    "meta3",  # export_timestamp
    "param0",  # parameter_strategy
}

# Custom attributes that should not be included in ONNX
GRAPHML_CUSTOM_ATTRIBUTES = {
    "hierarchy_tag",
    "module_type",
    "execution_order",
    "scope",
    "traced_tag",
    "class_name",
}


@dataclass(frozen=True)
class GraphMLConstants:
    """Immutable GraphML constants to eliminate magic numbers."""
    
    # Test data generation
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_SEQUENCE_LENGTH: int = 10
    DEFAULT_INT_RANGE: int = 1000
    
    # Tolerance values (with documentation)
    SIZE_TOLERANCE_STRICT: float = 0.01  # 1% - for critical validations
    SIZE_TOLERANCE_NORMAL: float = 0.05  # 5% - for standard checks
    
    # Memory constants
    BYTES_PER_KB: int = 1024
    BYTES_PER_MB: int = 1024 * 1024
    BYTES_PER_GB: int = 1024 * 1024 * 1024
    
    # Report formatting
    REPORT_WIDTH: int = 80
    SECTION_WIDTH: int = 40
    REPORT_SEPARATOR: str = "="
    SECTION_SEPARATOR: str = "-"
    
    # Graph depth limits
    MAX_GRAPH_DEPTH: int = 100  # Hard limit to prevent stack overflow
    WARN_GRAPH_DEPTH: int = 50   # Warning threshold for deep hierarchies
    
    # Performance thresholds
    MAX_NODES_INTERACTIVE: int = 10_000    # Real-time processing limit
    MAX_NODES_BATCH: int = 100_000         # Batch processing limit
    MAX_NODES_STREAMING: int = 1_000_000   # Requires streaming architecture
    
    # Operation timeouts (seconds)
    CONVERSION_TIMEOUT: int = 300  # 5 minutes for large models
    VALIDATION_TIMEOUT: int = 60   # 1 minute for validation
    
    # Buffer sizes for streaming operations
    STREAMING_BUFFER_SIZE: int = 8192  # 8KB chunks
    MAX_XML_ELEMENT_SIZE: int = 100 * 1024 * 1024  # 100MB limit
    
    # Validation thresholds
    MIN_NODE_COUNT: int = 1
    MAX_ATTRIBUTE_LENGTH: int = 65536  # 64KB for attribute values
    MAX_HIERARCHY_TAG_LENGTH: int = 1024  # Reasonable path length
    
    # Default values
    DEFAULT_OPSET_VERSION: int = 17
    DEFAULT_IR_VERSION: int = 8
    
    # Parameter strategies
    PARAM_STRATEGY_SIDECAR: str = "sidecar"
    PARAM_STRATEGY_EMBEDDED: str = "embedded"
    PARAM_STRATEGY_REFERENCE: str = "reference"
    
    # File extensions
    GRAPHML_EXTENSION: str = ".graphml"
    ONNX_EXTENSION: str = ".onnx"
    ONNXDATA_EXTENSION: str = ".onnxdata"
    
    # XML namespaces
    GRAPHML_NAMESPACE: str = "http://graphml.graphdrawing.org/xmlns"
    XSI_NAMESPACE: str = "http://www.w3.org/2001/XMLSchema-instance"
    SCHEMA_LOCATION: str = "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"


# Global instance for easy access
GRAPHML_CONST = GraphMLConstants()


# Parameter strategy validation
VALID_PARAMETER_STRATEGIES = {
    GRAPHML_CONST.PARAM_STRATEGY_SIDECAR,
    GRAPHML_CONST.PARAM_STRATEGY_EMBEDDED,
    GRAPHML_CONST.PARAM_STRATEGY_REFERENCE,
}


# Legacy key mappings for backward compatibility detection
LEGACY_KEY_MAPPINGS = {
    # v1.1 to v1.3 mappings
    'm5': 'meta5',
    'm6': 'meta6', 
    'm7': 'meta7',
    'm8': 'meta8',
    'p0': 'param0',
    'p1': 'param1',
    'p2': 'param2',
    'g0': 'io0',
    'g1': 'io1',
    'g2': 'io2',
    'g3': 'io3',
    't0': 'e1',
    't1': 'e2',
    't2': 'e3',
}