"""
HTP (Hierarchical Trace-and-Project) Strategy with IO/ABC Architecture

This strategy uses execution tracing with PyTorch hooks to capture module context
during forward pass, then projects this onto ONNX operations.

Key Features:
- Works with complex models and control flow
- Built-in module tracking for better accuracy
- Conservative tag propagation
- Optimized for HuggingFace transformers
- New IO/ABC-based monitoring architecture

Variations:
- Standard HTP: Hook-based execution tracing
- Built-in HTP: Uses PyTorch's internal module tracking
"""

from .base_writer import ExportStep
from .export_monitor import HTPExportMonitor
from .htp_exporter import HTPExporter, export_with_htp, export_with_htp_reporting

__all__ = [
    "ExportStep",
    "HTPExportMonitor",
    "HTPExporter",
    "export_with_htp",
    "export_with_htp_reporting",
]