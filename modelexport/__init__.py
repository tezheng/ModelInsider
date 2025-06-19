"""
Universal Hierarchy-Preserving ONNX Export for PyTorch Models

A universal exporter that works with ANY PyTorch model by leveraging 
the inherent nn.Module hierarchy structure.

Key Features:
- Universal: Works with any PyTorch model (no hardcoded architectures)
- Usage-based tagging: Operations tagged only when actually used
- Recursive propagation: Captures full dependency chains
- Stack-based context: Preserves module execution hierarchy
"""

from .hierarchy_exporter import HierarchyExporter
from . import tag_utils

__version__ = "0.1.0"
__all__ = ["HierarchyExporter", "tag_utils"]