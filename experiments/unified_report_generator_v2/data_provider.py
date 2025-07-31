"""
Data provider implementation for testing.
"""

from dataclasses import dataclass, field
from typing import Any

from interfaces import IDataProvider, StatisticsInfo, StepInfo


@dataclass
class ExportSessionData:
    """Container for all export session data."""
    # Model info
    model_name_or_path: str = ""
    model_class: str = ""
    total_modules: int = 0
    total_parameters: int = 0
    output_path: str = ""
    
    # Export config
    export_time: float = 0.0
    embed_hierarchy_attributes: bool = True
    opset_version: int = 17
    timestamp: str = ""
    
    # Steps
    steps: dict[str, StepInfo] = field(default_factory=dict)
    
    # Data collections
    hierarchy_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    tagged_nodes: dict[str, str] = field(default_factory=dict)
    
    # Statistics
    tagging_statistics: dict[str, int] = field(default_factory=dict)
    
    # File info
    onnx_path: str = ""
    onnx_size_mb: float = 0.0
    metadata_path: str = ""
    report_path: str = ""
    
    # Additional
    input_names: list = field(default_factory=list)
    output_names: list = field(default_factory=list)


class SessionDataProvider(IDataProvider):
    """Data provider backed by ExportSessionData."""
    
    def __init__(self, session_data: ExportSessionData):
        self.session = session_data
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "name_or_path": self.session.model_name_or_path,
            "class_name": self.session.model_class,
            "total_modules": self.session.total_modules,
            "total_parameters": self.session.total_parameters,
            "output_path": self.session.output_path,
            "eval_mode": True  # Assume always true for exports
        }
    
    def get_step_info(self, step_name: str) -> StepInfo | None:
        """Get information about a specific step."""
        return self.session.steps.get(step_name)
    
    def get_hierarchy_data(self) -> dict[str, dict[str, Any]]:
        """Get module hierarchy data."""
        return self.session.hierarchy_data
    
    def get_tagged_nodes(self) -> dict[str, str]:
        """Get ONNX node to tag mappings."""
        return self.session.tagged_nodes
    
    def get_statistics(self) -> StatisticsInfo:
        """Get tagging statistics."""
        stats = self.session.tagging_statistics
        total = stats.get("total_nodes", 0)
        tagged = stats.get("tagged_nodes", total)
        
        return StatisticsInfo(
            total_onnx_nodes=total,
            tagged_nodes=tagged,
            coverage_percentage=(tagged / total * 100) if total > 0 else 0.0,
            direct_matches=stats.get("direct_matches", 0),
            parent_matches=stats.get("parent_matches", 0),
            operation_matches=stats.get("operation_matches", 0),
            root_fallbacks=stats.get("root_fallbacks", 0),
            empty_tags=stats.get("empty_tags", 0)
        )
    
    def get_export_config(self) -> dict[str, Any]:
        """Get export configuration."""
        return {
            "export_time": self.session.export_time,
            "embed_hierarchy_attributes": self.session.embed_hierarchy_attributes,
            "opset_version": self.session.opset_version,
            "timestamp": self.session.timestamp
        }
    
    def get_file_info(self) -> dict[str, Any]:
        """Get output file information."""
        return {
            "onnx_path": self.session.onnx_path,
            "onnx_size_mb": self.session.onnx_size_mb,
            "metadata_path": self.session.metadata_path,
            "report_path": self.session.report_path,
            "ONNX model": self.session.onnx_path,
            "Metadata": self.session.metadata_path,
            "Report": self.session.report_path
        }