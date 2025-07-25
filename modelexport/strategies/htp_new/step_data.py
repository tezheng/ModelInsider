"""
Structured step data classes for HTP Export Monitor.

This module defines typed dataclasses for each export step,
replacing the loose dict[str, dict[str, Any]] structure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def _timestamp_field():
    """Create a timestamp field that captures time at instance creation."""
    return field(default_factory=lambda: time.time())


@dataclass
class ModuleInfo:
    """Information about a module in the hierarchy."""
    
    class_name: str
    traced_tag: str
    execution_order: int | None = None
    source: str | None = None


@dataclass
class TensorInfo:
    """Information about a tensor input."""
    
    shape: list[int]
    dtype: str


@dataclass
class ModelPrepData:
    """Data for model preparation step."""
    
    model_class: str
    total_modules: int
    total_parameters: int
    timestamp: float = _timestamp_field()


@dataclass
class InputGenData:
    """Data for input generation step."""
    
    method: str
    model_type: str | None = None
    task: str | None = None
    inputs: dict[str, TensorInfo] = field(default_factory=dict)
    timestamp: float = _timestamp_field()


@dataclass
class HierarchyData:
    """Data for hierarchy building step."""
    
    hierarchy: dict[str, ModuleInfo]
    execution_steps: int
    module_list: list[tuple[str, str]] = field(default_factory=list)
    timestamp: float = _timestamp_field()


@dataclass
class ONNXExportData:
    """Data for ONNX export step."""
    
    opset_version: int = 17
    do_constant_folding: bool = True
    verbose: bool = False
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] | None = None
    onnx_size_mb: float = 0.0
    timestamp: float = _timestamp_field()


@dataclass
class NodeTaggingData:
    """Data for node tagging step."""
    
    total_nodes: int
    tagged_nodes: dict[str, str]
    tagging_stats: dict[str, int]
    coverage: float
    op_counts: dict[str, int] = field(default_factory=dict)
    timestamp: float = _timestamp_field()


@dataclass
class TagInjectionData:
    """Data for tag injection step."""
    
    tags_injected: bool
    tags_stripped: bool = False
    timestamp: float = _timestamp_field()