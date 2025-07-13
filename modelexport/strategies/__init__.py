"""
Modelexport Strategies Package

This package contains the HTP (Hierarchical Trace-and-Project) strategy 
for hierarchy-preserving ONNX export, optimized for complex models including
HuggingFace transformers.
"""

from .htp import HTPHierarchyExporter

__all__ = [
    "HTPHierarchyExporter",
]