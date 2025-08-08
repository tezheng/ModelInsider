"""
GraphML Bidirectional Conversion Module

This module provides bidirectional conversion between ONNX and GraphML formats
with full parameter support and round-trip validation.

Main components:
- ONNXToGraphMLConverter: Unified ONNX → GraphML conversion (flat or hierarchical)
- GraphMLToONNXConverter: GraphML → ONNX reconstruction  
- RoundTripValidator: Validates bidirectional conversion integrity
"""

# GraphML format version (defined before imports to avoid circular dependencies)
__version__ = "1.3.0"  # GraphML format/schema version
__spec_version__ = ".".join(__version__.split(".")[:2])  # "1.3"

# Primary exports
from .graphml_to_onnx_converter import GraphMLToONNXConverter
from .onnx_to_graphml_converter import ONNXToGraphMLConverter
from .round_trip_validator import RoundTripValidator

__all__ = [
    "GraphMLToONNXConverter",
    "ONNXToGraphMLConverter",
    "RoundTripValidator",
    "__version__",
    "__spec_version__",
]