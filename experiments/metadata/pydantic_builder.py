"""
Pydantic-based metadata builder for HTP Exporter.

This module provides a cleaner approach using Pydantic models
with full JSON schema support and validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metadata_models import (
    ExportContextModel,
    ExportReportModel,
    HTPMetadataModel,
    ModelInfoModel,
    OnnxModelOutputModel,
    OutputFilesModel,
    QualityGuaranteesModel,
    StatisticsModel,
    TaggingCoverageModel,
    TaggingInfoModel,
    TracingInfoModel,
)


class HTPMetadataBuilderPydantic:
    """
    Pydantic-based builder for HTP metadata with schema support.
    
    This builder uses Pydantic models which provide:
    - Automatic validation
    - JSON schema generation
    - Type safety
    - Serialization/deserialization
    """
    
    def __init__(self):
        """Initialize builder with empty models."""
        self._data = {}
    
    @classmethod
    def from_exporter_state(
        cls,
        export_report: dict[str, Any],
        export_stats: dict[str, Any],
        hierarchy_data: dict[str, dict[str, Any]],
        tagged_nodes: dict[str, str],
        tagging_stats: dict[str, Any],
        hierarchy_builder: Any,
        output_path: str,
        metadata_path: str,
        embed_hierarchy_attributes: bool,
        strategy: str = "htp",
    ) -> HTPMetadataModel:
        """
        Create metadata directly from exporter state.
        
        This is the cleanest approach - one method that builds
        the entire metadata model with validation.
        """
        # Extract data
        model_info = export_report["model_info"]
        input_gen_details = export_report["export_report"]["input_generation"]["details"]
        hierarchy_details = export_report["export_report"]["hierarchy_building"]["details"]
        onnx_details = export_report["export_report"]["onnx_export"]["details"]
        
        # Get output names if available
        outputs = hierarchy_builder.get_outputs() if hierarchy_builder else None
        output_names = None
        if outputs:
            # Import here to avoid circular dependency
            from ...core.onnx_utils import infer_output_names
            output_names = infer_output_names(outputs)
        
        # Get module types
        module_types = list(
            {
                info.get("class_name", "")
                for info in hierarchy_data.values()
                if info.get("class_name")
            }
        )
        
        # Build the complete model with validation
        return HTPMetadataModel(
            export_context=ExportContextModel(
                strategy=strategy,
                embed_hierarchy_attributes=embed_hierarchy_attributes
            ),
            model=ModelInfoModel(
                name_or_path=model_info.get("model_name_or_path", "unknown"),
                class_name=model_info.get("model_class", "unknown"),
                framework=model_info.get("framework", "transformers"),
                total_modules=model_info.get("total_modules", 0),
                total_parameters=model_info.get("total_parameters", 0),
            ),
            tracing=TracingInfoModel(
                modules_traced=len(hierarchy_data),
                execution_steps=hierarchy_details.get("execution_steps", 0),
                model_type=input_gen_details.get("model_type"),
                task=input_gen_details.get("detected_task"),
                inputs=input_gen_details.get("inputs"),
                outputs=output_names,
            ),
            modules=hierarchy_data,
            tagging=TaggingInfoModel(
                tagged_nodes=tagged_nodes,
                statistics=tagging_stats or {},
                coverage=TaggingCoverageModel(
                    total_onnx_nodes=export_stats.get("onnx_nodes", 0),
                    tagged_nodes=export_stats.get("tagged_nodes", 0),
                    coverage_percentage=export_stats.get("coverage_percentage", 0.0),
                    empty_tags=export_stats.get("empty_tags", 0),
                ),
            ),
            outputs=OutputFilesModel(
                onnx_model=OnnxModelOutputModel(
                    path=Path(output_path).name,
                    size_mb=onnx_details.get("file_size_mb", 0),
                    opset_version=onnx_details["export_config"].get("opset_version", 17),
                    output_names=output_names,
                ),
                metadata={"path": Path(metadata_path).name},
            ),
            report=ExportReportModel(
                export_time_seconds=round(export_stats.get("export_time", 0), 2),
                steps={
                    "model_preparation": export_report["export_report"]["model_preparation"],
                    "input_generation": {
                        "status": export_report["export_report"]["input_generation"]["status"],
                        "method": input_gen_details.get("method", "unknown"),
                    },
                    "hierarchy_building": export_report["export_report"]["hierarchy_building"],
                    "onnx_export": {
                        "status": export_report["export_report"]["onnx_export"]["status"],
                        "export_config": onnx_details["export_config"],
                    },
                    "node_tagging": {
                        "status": export_report["export_report"]["node_tagging"]["status"],
                        "top_hierarchies": export_report["export_report"]["node_tagging"]["details"].get("top_hierarchies", []),
                    },
                    "tag_injection": export_report["export_report"]["tag_injection"],
                },
                quality_guarantees=QualityGuaranteesModel(
                    empty_tags_guarantee=export_stats.get("empty_tags", 0),
                    coverage_guarantee=f"{export_stats.get('coverage_percentage', 0):.1f}%",
                ),
            ),
            statistics=StatisticsModel(
                export_time=export_stats.get("export_time", 0),
                hierarchy_modules=export_stats.get("hierarchy_modules", 0),
                onnx_nodes=export_stats.get("onnx_nodes", 0),
                tagged_nodes=export_stats.get("tagged_nodes", 0),
                empty_tags=export_stats.get("empty_tags", 0),
                coverage_percentage=export_stats.get("coverage_percentage", 0.0),
                module_types=module_types,
            ),
        )
    
    @staticmethod
    def save_schema(output_path: str = "htp_metadata_schema.json") -> None:
        """
        Save the JSON schema for HTP metadata.
        
        This schema can be used for:
        - Validation in other tools
        - Documentation
        - Code generation
        - API contracts
        """
        schema = HTPMetadataModel.model_json_schema()
        
        # Add metadata about the schema
        schema["$id"] = "https://github.com/user/modelexport/schemas/htp-metadata-v1.0.json"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["title"] = "HTP Export Metadata Schema"
        schema["description"] = "Schema for Hierarchy-preserving Tags Protocol (HTP) ONNX export metadata"
        
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)
    
    @staticmethod
    def validate_metadata(metadata: dict[str, Any] | str | Path) -> HTPMetadataModel:
        """
        Validate metadata against the schema.
        
        Args:
            metadata: Dictionary, JSON string, or path to JSON file
            
        Returns:
            Validated HTPMetadataModel
            
        Raises:
            ValidationError: If metadata doesn't match schema
        """
        if isinstance(metadata, str | Path):
            # Load from file
            with open(metadata) as f:
                data = json.load(f)
        else:
            data = metadata
        
        return HTPMetadataModel.model_validate(data)