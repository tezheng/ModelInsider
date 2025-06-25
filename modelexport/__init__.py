"""
Universal Hierarchy-Preserving ONNX Export for PyTorch Models

A universal exporter with multiple strategies for different model types and use cases.

Strategies:
- FX: Symbolic tracing for pure PyTorch models (95%+ coverage, fast)
- HTP: Execution tracing for complex models with control flow (HuggingFace compatible)
- Usage-based: Legacy strategy for backward compatibility

Key Features:
- Universal: Works with any PyTorch model (no hardcoded architectures)
- Strategy-based: Optimized approach for different model types
- Hierarchy preservation: Complete module structure maintained in ONNX
- Production ready: Comprehensive testing and validation
"""

from .strategies.fx import FXHierarchyExporter
from .strategies.htp import HTPHierarchyExporter
from .strategies.usage_based import UsageBasedExporter
from .core import tag_utils
from .core.base import BaseHierarchyExporter

# Backward compatibility
HierarchyExporter = HTPHierarchyExporter

__version__ = "0.1.0"
__all__ = [
    "FXHierarchyExporter",
    "HTPHierarchyExporter", 
    "UsageBasedExporter",
    "BaseHierarchyExporter",
    "HierarchyExporter",  # Backward compatibility
    "tag_utils"
]