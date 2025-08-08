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
from .core import tag_utils

# Utilities
from .core.base import BaseHierarchyExporter, should_include_in_hierarchy
from .core.onnx_utils import ONNXUtils

# Core components for advanced usage
from .core.strategy_selector import (
    ExportStrategy,
    StrategySelector,
    select_best_strategy,
)
from .core.unified_optimizer import UnifiedOptimizer, create_optimized_exporter

# Individual strategies for direct access
from .strategies.htp import HTPExporter
from .unified_export import UnifiedExporter, export_model

# Backward compatibility
HierarchyExporter = HTPExporter

# Package version management with hybrid approach
import os
from pathlib import Path


def _get_version() -> str:
    """
    Get package version using hybrid approach.
    
    Priority:
    1. Try importlib.metadata (installed package)
    2. Try pyproject.toml (development/editable install)  
    3. Fallback to "unknown"
    
    Returns:
        str: Package version string
    """
    # Check for verbose mode
    verbose = os.environ.get("MODELEXPORT_VERBOSE", "").lower() in ("1", "true", "yes")
    
    # Try importlib.metadata first (works for installed packages)
    try:
        from importlib.metadata import version, PackageNotFoundError
        pkg_version = version("modelexport")
        if verbose:
            print(f"[modelexport] Version {pkg_version} from package metadata")
        return pkg_version
    except PackageNotFoundError:
        if verbose:
            print("[modelexport] Package not installed, checking development mode")
    except ImportError:
        if verbose:
            print("[modelexport] importlib.metadata not available")
    
    # Development mode: read from pyproject.toml
    try:
        import tomllib  # Python 3.11+
        root = Path(__file__).resolve().parent.parent
        pyproject = root / "pyproject.toml"
        
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                version_str = data["project"]["version"]
                dev_version = f"{version_str}.dev0"
                if verbose:
                    print(f"[modelexport] Development version {dev_version} from pyproject.toml")
                return dev_version
    except Exception as e:
        if verbose:
            print(f"[modelexport] Could not read pyproject.toml: {e}")
    
    # Final fallback
    if verbose:
        print("[modelexport] Using fallback version 'unknown'")
    return "unknown"


__version__ = _get_version()
__all__ = [
    "BaseHierarchyExporter",
    "create_optimized_exporter",
    "ExportStrategy",
    "export_model",
    "HierarchyExporter",
    "HTPExporter",
    "ONNXUtils",
    "select_best_strategy",
    "should_include_in_hierarchy",
    "StrategySelector",
    "tag_utils",
    "UnifiedExporter",
    "UnifiedOptimizer",
]