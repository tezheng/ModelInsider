"""
GraphML Export Module for ModelExport

This module provides functionality to export ONNX models to GraphML format
with optional hierarchical structure preservation from HTP metadata.

Main components:
- ONNXToGraphMLConverter: Base converter for ONNX to GraphML
- HierarchicalGraphMLConverter: Extended converter with HTP integration
- GraphMLWriter: Low-level GraphML XML generation
"""

from .converter import ONNXToGraphMLConverter

__all__ = [
    "ONNXToGraphMLConverter",
]

# HierarchicalGraphMLConverter will be added in Phase 4

__version__ = "0.1.0"