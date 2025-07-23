"""
Structured step data classes for HTP Export Monitor.

This module defines typed dataclasses for each export step,
replacing the loose dict[str, dict[str, Any]] structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModuleInfo:
    """Information about a module in the hierarchy."""
    
    class_name: str
    traced_tag: str
    execution_order: int | None = None
    source: str | None = None  # "DIRECT", "FORWARD_HOOK", etc.


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


@dataclass
class InputGenData:
    """Data for input generation step."""
    
    method: str  # "provided" or "auto_generated"
    model_type: str | None = None
    task: str | None = None
    inputs: dict[str, TensorInfo] = field(default_factory=dict)
    # inputs format: {"input_ids": TensorInfo(shape=[1, 128], dtype="int64")}


@dataclass
class HierarchyData:
    """Data for hierarchy building step."""
    
    hierarchy: dict[str, ModuleInfo]  # Module path -> ModuleInfo
    execution_steps: int
    module_list: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ONNXExportData:
    """Data for ONNX export step."""
    
    opset_version: int = 17
    do_constant_folding: bool = True
    verbose: bool = False
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] | None = None
    onnx_size_mb: float = 0.0


@dataclass
class NodeTaggingData:
    """Data for node tagging step."""
    
    total_nodes: int
    tagged_nodes: dict[str, str]  # node_name -> hierarchy_tag
    tagging_stats: dict[str, int]  # statistics
    coverage: float
    op_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class TagInjectionData:
    """Data for tag injection step."""
    
    tags_injected: bool
    tags_stripped: bool = False  # For --clean-onnx mode