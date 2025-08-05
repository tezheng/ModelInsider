"""
GraphML Bidirectional Conversion Module

This module provides bidirectional conversion between ONNX and GraphML formats
with full parameter support and round-trip validation.

Main components:
- ONNXToGraphMLConverter: Unified ONNX → GraphML conversion (flat or hierarchical)
- GraphMLToONNXConverter: GraphML → ONNX reconstruction  
- RoundTripValidator: Validates bidirectional conversion integrity
"""

# Primary exports
from .graphml_to_onnx_converter import GraphMLToONNXConverter
from .onnx_to_graphml_converter import ONNXToGraphMLConverter
from .round_trip_validator import RoundTripValidator

__all__ = [
    "GraphMLToONNXConverter",
    "ONNXToGraphMLConverter",
    "RoundTripValidator",
]

__version__ = "0.1.0"