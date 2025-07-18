"""
Pydantic models for HTP metadata generation.

This module provides Pydantic models as an alternative to dataclasses,
offering better validation and JSON schema generation capabilities.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExportContextModel(BaseModel):
    """Export session context information."""
    model_config = ConfigDict(populate_by_name=True)
    
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        description="Export timestamp in ISO format"
    )
    strategy: str = Field(default="htp", description="Export strategy name")
    version: str = Field(default="1.0", description="Metadata version")
    exporter: str = Field(default="HTPExporter", description="Exporter class name")
    embed_hierarchy_attributes: bool = Field(
        default=True,
        description="Whether hierarchy tags are embedded in ONNX"
    )


class ModelInfoModel(BaseModel):
    """Model information."""
    model_config = ConfigDict(populate_by_name=True)
    
    name_or_path: str = Field(description="Model name or path")
    class_name: str = Field(alias="class", description="Model class name")
    framework: str = Field(default="transformers", description="ML framework")
    total_modules: int = Field(default=0, description="Total number of modules")
    total_parameters: int = Field(default=0, description="Total parameter count")


class InputTensorInfo(BaseModel):
    """Input tensor information."""
    shape: list[int] = Field(description="Tensor shape")
    dtype: str = Field(description="Tensor data type")


class TracingInfoModel(BaseModel):
    """Tracing execution information."""
    model_config = ConfigDict(populate_by_name=True)
    
    builder: str = Field(
        default="TracingHierarchyBuilder",
        description="Hierarchy builder class"
    )
    modules_traced: int = Field(default=0, description="Number of modules traced")
    execution_steps: int = Field(default=0, description="Execution step count")
    model_type: str | None = Field(default=None, description="Model type (e.g., bert)")
    task: str | None = Field(default=None, description="Task type (e.g., feature-extraction)")
    inputs: dict[str, InputTensorInfo] | None = Field(
        default=None,
        description="Input tensor specifications"
    )
    outputs: list[str] | None = Field(
        default=None,
        description="Output tensor names"
    )


class TaggingCoverageModel(BaseModel):
    """Tagging coverage statistics."""
    total_onnx_nodes: int = Field(default=0, description="Total ONNX nodes")
    tagged_nodes: int = Field(default=0, description="Number of tagged nodes")
    coverage_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Coverage percentage"
    )
    empty_tags: int = Field(default=0, description="Number of empty tags")


class TaggingInfoModel(BaseModel):
    """Node tagging information."""
    tagged_nodes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of node names to hierarchy tags"
    )
    statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="Tagging statistics"
    )
    coverage: TaggingCoverageModel = Field(
        default_factory=TaggingCoverageModel,
        description="Coverage information"
    )


class OnnxModelOutputModel(BaseModel):
    """ONNX model output information."""
    path: str = Field(description="ONNX file name")
    size_mb: float = Field(default=0.0, ge=0.0, description="File size in MB")
    opset_version: int = Field(default=17, ge=1, description="ONNX opset version")
    output_names: list[str] | None = Field(
        default=None,
        description="Model output tensor names"
    )


class OutputFilesModel(BaseModel):
    """Output file information."""
    onnx_model: OnnxModelOutputModel = Field(description="ONNX model info")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Metadata file info"
    )


class ExportStepModel(BaseModel):
    """Export process step information."""
    status: str = Field(default="pending", description="Step status")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Step details"
    )


class QualityGuaranteesModel(BaseModel):
    """Export quality guarantees."""
    no_hardcoded_logic: bool = Field(
        default=True,
        description="No model-specific hardcoded logic"
    )
    universal_module_tracking: str = Field(
        default="TracingHierarchyBuilder",
        description="Module tracking method"
    )
    empty_tags_guarantee: int = Field(
        default=0,
        description="Number of empty tags (should be 0)"
    )
    coverage_guarantee: str = Field(
        default="0.0%",
        description="Coverage percentage guarantee"
    )
    optimum_compatible: bool = Field(
        default=True,
        description="Optimum library compatibility"
    )


class ExportReportModel(BaseModel):
    """Export process report."""
    export_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Export duration in seconds"
    )
    steps: dict[str, Any] = Field(
        default_factory=dict,
        description="Export process steps"
    )
    quality_guarantees: QualityGuaranteesModel = Field(
        default_factory=QualityGuaranteesModel,
        description="Quality guarantees"
    )


class StatisticsModel(BaseModel):
    """Export statistics summary."""
    export_time: float = Field(default=0.0, ge=0.0, description="Export time")
    hierarchy_modules: int = Field(default=0, ge=0, description="Hierarchy module count")
    onnx_nodes: int = Field(default=0, ge=0, description="ONNX node count")
    tagged_nodes: int = Field(default=0, ge=0, description="Tagged node count")
    empty_tags: int = Field(default=0, ge=0, description="Empty tag count")
    coverage_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Coverage percentage"
    )
    module_types: list[str] = Field(
        default_factory=list,
        description="Unique module types"
    )


class HTPMetadataModel(BaseModel):
    """Complete HTP metadata model."""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "title": "HTP Export Metadata",
            "description": "Metadata for Hierarchy-preserving Tags Protocol ONNX export"
        }
    )
    
    export_context: ExportContextModel = Field(description="Export context")
    model: ModelInfoModel = Field(description="Model information")
    tracing: TracingInfoModel = Field(description="Tracing information")
    modules: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Module hierarchy data"
    )
    tagging: TaggingInfoModel = Field(description="Tagging information")
    outputs: OutputFilesModel = Field(description="Output files")
    report: ExportReportModel = Field(description="Export report")
    statistics: StatisticsModel = Field(description="Statistics summary")
    
    def model_dump_json(self, **kwargs) -> str:
        """Override to ensure proper JSON formatting."""
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)