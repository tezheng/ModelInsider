"""
ModelExport - Universal Hierarchy-Preserving ONNX Export

A production-ready framework for exporting PyTorch models to ONNX with preserved
module hierarchy information. Features intelligent strategy selection and 
comprehensive optimizations.

Quick Start:
    >>> import modelexport
    >>> report = modelexport.export_model(
    ...     model,
    ...     torch.randn(1, 3, 224, 224),
    ...     "model.onnx"
    ... )
    >>> print(f"Exported using {report['summary']['final_strategy']} strategy")

Strategies:
- Enhanced Semantic: HuggingFace-level semantic mapping, 97% coverage (recommended)
- Usage-Based: Fastest strategy (2.5s), good for production  
- HTP: Comprehensive tracing for complex models (4-6s)
- FX: Limited compatibility, not recommended for HuggingFace models
"""

# Main export interface
from .unified_export import export_model, UnifiedExporter

# Core components for advanced usage
from .core.strategy_selector import ExportStrategy, StrategySelector, select_best_strategy
from .core.unified_optimizer import UnifiedOptimizer, create_optimized_exporter

# Individual strategies for direct access
from .strategies.htp import HTPHierarchyExporter

# Utilities
from .core.base import BaseHierarchyExporter, should_tag_module, build_hierarchy_path
from .core.onnx_utils import ONNXUtils
from .core import tag_utils

# Backward compatibility
HierarchyExporter = HTPHierarchyExporter

__version__ = "0.1.0"
__all__ = [
    # Main interface (recommended)
    "export_model",
    "UnifiedExporter",
    
    # Strategy selection
    "ExportStrategy", 
    "StrategySelector",
    "select_best_strategy",
    
    # Optimization framework
    "UnifiedOptimizer",
    "create_optimized_exporter",
    
    # Individual strategies
    "HTPHierarchyExporter",
    
    # Utilities
    "BaseHierarchyExporter",
    "should_tag_module",
    "build_hierarchy_path",
    "ONNXUtils",
    "tag_utils",
    
    # Backward compatibility
    "HierarchyExporter",
]