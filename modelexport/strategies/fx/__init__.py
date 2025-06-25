"""
FX Graph-based Strategy

This strategy uses PyTorch's FX (functional transform) framework for symbolic tracing
and hierarchy preservation. Optimized for pure PyTorch models without control flow.

Key Features:
- Symbolic graph representation
- 95%+ coverage on supported models
- Fast export times
- Universal design (no hardcoded logic)

Limitations:
- Cannot handle dynamic control flow
- HuggingFace models often fail due to input validation
"""

from .fx_hierarchy_exporter import FXHierarchyExporter

__all__ = ["FXHierarchyExporter"]