"""
GraphML Export Module for ModelExport

This module provides functionality to export ONNX models to GraphML format
with optional hierarchical structure preservation from HTP metadata.

Main components:
- ONNXToGraphMLConverter: Base converter for ONNX to GraphML
- HierarchicalGraphMLConverter: Extended converter with HTP integration
- MetadataReader: HTP metadata file reader
- GraphMLWriter: Low-level GraphML XML generation
"""

from .converter import ONNXToGraphMLConverter
from .hierarchical_converter import EnhancedHierarchicalConverter, HierarchicalGraphMLConverter
from .metadata_reader import MetadataReader
from .utils import CompoundNode, EdgeData, GraphData, NodeData, NodeType

__all__ = [
    "CompoundNode",
    "EdgeData",
    "GraphData",
    "HierarchicalGraphMLConverter",
    "EnhancedHierarchicalConverter",
    "MetadataReader",
    "NodeData",
    "NodeType",
    "ONNXToGraphMLConverter",
]

__version__ = "0.1.0"