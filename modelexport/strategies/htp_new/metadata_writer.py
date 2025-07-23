"""
Metadata writer for HTP export monitoring.

This module provides JSON metadata writing using the HTPMetadataBuilder,
following the new structure specified in REPORT_IMPROVEMENTS.md.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from .base_writer import ExportData, ExportStep, StepAwareWriter, step
from .metadata_builder import HTPMetadataBuilder
from .step_data import ModuleInfo


class MetadataWriter(StepAwareWriter):
    """JSON metadata writer using HTPMetadataBuilder."""
    
    def __init__(self, output_path: str):
        """
        Initialize metadata writer.
        
        Args:
            output_path: Base output path for the ONNX model
        """
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.metadata_path = f"{self.output_path}_htp_metadata.json"
        self.builder = HTPMetadataBuilder()
        
        # Store data for final building
        self._model_info_set = False
        self._export_time = 0.0
        self._steps_data = {}
    
    def _get_step_timestamp(self) -> str:
        """Generate a unique timestamp for the current step with millisecond precision."""
        dt = datetime.datetime.now(datetime.UTC)
        # Format with milliseconds (3 digits) instead of microseconds (6 digits)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    
    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default handler - record step completion."""
        self._steps_data[export_step.value] = {
            "completed": True,
            "timestamp": self._get_step_timestamp(),
        }
        return 0
    
    @step(ExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: ExportStep, data: ExportData) -> int:
        """Record model information."""
        if not data.model_prep:
            return 0
        
        self.builder.with_model_info(
            name_or_path=data.model_name,
            class_name=data.model_prep.model_class,
            total_modules=data.model_prep.total_modules,
            total_parameters=data.model_prep.total_parameters,
        )
        self._model_info_set = True
        
        self.builder.with_export_context(
            embed_hierarchy_attributes=data.embed_hierarchy,
            export_time_seconds=data.export_time,
        )
        
        # Record step completion
        self._steps_data["model_preparation"] = {
            "completed": True,
            "timestamp": self._get_step_timestamp(),
            "model_class": data.model_prep.model_class,
            "total_modules": data.model_prep.total_modules,
            "total_parameters": data.model_prep.total_parameters,
        }
        
        return 1
    
    @step(ExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: ExportStep, data: ExportData) -> int:
        """Record input generation details."""
        if not data.input_gen:
            return 0
        
        # Convert TensorInfo to dict format for builder
        inputs_dict = {}
        for name, tensor_info in data.input_gen.inputs.items():
            inputs_dict[name] = {
                "shape": tensor_info.shape,
                "dtype": tensor_info.dtype,
            }
        
        # This will be part of tracing info
        self._steps_data["input_generation"] = {
            "timestamp": self._get_step_timestamp(),
            "method": data.input_gen.method,
            "model_type": data.input_gen.model_type,
            "task": data.input_gen.task,
            "inputs": inputs_dict,
        }
        
        return 1
    
    @step(ExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
        """Record hierarchy data."""
        if not data.hierarchy:
            return 0
        
        # Build hierarchical module structure
        modules_dict = self._build_hierarchical_modules(data.hierarchy.hierarchy)
        
        self.builder.with_modules(modules_dict)
        
        # Get input info from previous step data
        input_data = self._steps_data.get("input_generation", {})
        
        self.builder.with_tracing_info(
            modules_traced=len(data.hierarchy.hierarchy),
            execution_steps=data.hierarchy.execution_steps,
            model_type=input_data.get("model_type"),
            task=input_data.get("task"),
            inputs=input_data.get("inputs"),
        )
        
        # Record step completion
        self._steps_data["hierarchy_building"] = {
            "completed": True,
            "timestamp": self._get_step_timestamp(),
            "modules_traced": len(data.hierarchy.hierarchy),
            "execution_steps": data.hierarchy.execution_steps,
        }
        
        return 1
    
    @step(ExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: ExportStep, data: ExportData) -> int:
        """Record ONNX export details."""
        if not data.onnx_export:
            return 0
        
        # Store ONNX details for final output
        self._steps_data["onnx_export"] = {
            "timestamp": self._get_step_timestamp(),
            "opset_version": data.onnx_export.opset_version,
            "do_constant_folding": data.onnx_export.do_constant_folding,
            "onnx_size_mb": data.onnx_export.onnx_size_mb,
            "output_names": data.onnx_export.output_names,
        }
        
        return 1
    
    @step(ExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: ExportStep, data: ExportData) -> int:
        """Record node tagging results."""
        if not data.node_tagging:
            return 0
        
        # Use the builder method for tagging info
        self.builder.with_tagging_info(
            tagged_nodes=data.node_tagging.tagged_nodes,
            statistics=data.node_tagging.tagging_stats,
            total_onnx_nodes=data.node_tagging.total_nodes,
            tagged_nodes_count=len(data.node_tagging.tagged_nodes),
            coverage_percentage=data.node_tagging.coverage,
            empty_tags=data.node_tagging.tagging_stats.get("empty_tags", 0),
        )
        
        # Record step completion with enhanced structure per spec
        self._steps_data["node_tagging"] = {
            "completed": True,
            "timestamp": self._get_step_timestamp(),
            "total_nodes": data.node_tagging.total_nodes,
            "tagged_nodes_count": len(data.node_tagging.tagged_nodes),
            "coverage_percentage": data.node_tagging.coverage,
            "statistics": {
                "root_nodes": data.node_tagging.tagging_stats.get("root_nodes", 0),
                "scoped_nodes": data.node_tagging.tagging_stats.get("scoped_nodes", 0),
                "unique_scopes": data.node_tagging.tagging_stats.get("unique_scopes", 0),
                "direct_matches": data.node_tagging.tagging_stats.get("direct_matches", 0),
                "parent_matches": data.node_tagging.tagging_stats.get("parent_matches", 0),
                "operation_matches": data.node_tagging.tagging_stats.get("operation_matches", 0),
                "root_fallbacks": data.node_tagging.tagging_stats.get("root_fallbacks", 0),
            },
            "coverage": {
                "percentage": data.node_tagging.coverage,
                "total_onnx_nodes": data.node_tagging.total_nodes,
                "tagged_nodes": len(data.node_tagging.tagged_nodes),
            }
        }
        
        return 1
    
    @step(ExportStep.TAG_INJECTION)
    def write_tag_injection(self, export_step: ExportStep, data: ExportData) -> int:
        """Record tag injection status."""
        if not data.tag_injection:
            return 0
        
        self._steps_data["tag_injection"] = {
            "timestamp": self._get_step_timestamp(),
            "tags_injected": data.tag_injection.tags_injected,
            "tags_stripped": data.tag_injection.tags_stripped,
        }
        
        # This is the final step, so capture export time
        self._export_time = data.export_time
        
        return 1
    
    def flush(self) -> None:
        """Build and write metadata to file."""
        # Get ONNX info from steps data
        onnx_data = self._steps_data.get("onnx_export", {})
        
        # Determine report path if it exists
        report_path = f"{self.output_path}_htp_export_report.md"
        report_exists = Path(report_path).exists()
        
        # Set output files info
        self.builder.with_output_files(
            onnx_path=f"{self.output_path}",
            onnx_size_mb=onnx_data.get("onnx_size_mb", 0.0),
            metadata_path=self.metadata_path,
            opset_version=onnx_data.get("opset_version", 17),
            output_names=onnx_data.get("output_names"),
            report_path=report_path if report_exists else None,
        )
        
        # Set export report
        empty_tags = 0
        coverage = 0.0
        if hasattr(self.builder, "_tagging_info") and self.builder._tagging_info:
            empty_tags = self.builder._tagging_info.coverage.empty_tags
            coverage = self.builder._tagging_info.coverage.coverage_percentage
        
        self.builder.with_export_report(
            export_time_seconds=self._export_time,
            steps=self._steps_data,
            empty_tags_guarantee=empty_tags,
            coverage_percentage=coverage,
        )
        
        # Set statistics
        traced_modules = self._count_modules(self.builder._modules) if self.builder._modules else 0
        # Get total modules from model info
        total_modules = 0
        if hasattr(self.builder, "_model_info") and self.builder._model_info:
            total_modules = self.builder._model_info.total_modules
        
        onnx_nodes = 0
        tagged_nodes = 0
        module_types = []
        
        if hasattr(self.builder, "_tagging_info") and self.builder._tagging_info:
            onnx_nodes = self.builder._tagging_info.coverage.total_onnx_nodes
            tagged_nodes = self.builder._tagging_info.coverage.tagged_nodes
        
        if self.builder._modules:
            # Extract module types from hierarchical structure
            module_types = self._extract_module_types(self.builder._modules)
        
        self.builder.with_statistics(
            export_time=self._export_time,
            hierarchy_modules=total_modules,
            traced_modules=traced_modules,
            onnx_nodes=onnx_nodes,
            tagged_nodes=tagged_nodes,
            empty_tags=empty_tags,
            coverage_percentage=coverage,
            module_types=sorted(module_types),
        )
        
        # Build the metadata
        try:
            metadata = self.builder.build()
            
            # Validate metadata before writing
            from .validation_utils import validate_before_write
            validate_before_write(metadata)
            
            # Ensure output directory exists
            Path(self.metadata_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except ValueError as e:
            # If builder validation fails, write minimal metadata
            minimal_metadata = {
                "export_context": {
                    "timestamp": self._steps_data.get("model_preparation", {}).get("timestamp", ""),
                    "strategy": "htp",
                    "version": "1.0",
                },
                "error": str(e),
            }
            # Ensure output directory exists
            Path(self.metadata_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w') as f:
                json.dump(minimal_metadata, f, indent=2)
    
    def _build_hierarchical_modules(self, flat_hierarchy: dict[str, ModuleInfo]) -> dict[str, Any]:
        """
        Build hierarchical module structure from flat hierarchy.
        
        Args:
            flat_hierarchy: Flat dict of module path -> ModuleInfo
            
        Returns:
            Hierarchical dict structure with children
        """
        if not flat_hierarchy:
            return {}
        
        # Find root module (empty path)
        root_info = flat_hierarchy.get("")
        if not root_info:
            # No root, shouldn't happen but handle gracefully
            return {}
        
        # Build root structure
        root = {
            "class_name": root_info.class_name,
            "traced_tag": root_info.traced_tag,
            "scope": "",
        }
        if root_info.execution_order is not None:
            root["execution_order"] = root_info.execution_order
        if root_info.source:
            root["source"] = root_info.source
        
        # Build children recursively
        children = self._build_children_for_parent("", flat_hierarchy)
        if children:
            root["children"] = children
        
        return root
    
    def _build_children_for_parent(self, parent_path: str, flat_hierarchy: dict[str, ModuleInfo]) -> dict[str, Any] | None:
        """
        Build children dict for a parent module.
        
        Args:
            parent_path: Parent module path (e.g., "", "encoder", "encoder.layer.0")
            flat_hierarchy: Flat dict of all modules
            
        Returns:
            Dict of children or None if no children
        """
        children = {}
        
        # Find all direct children of this parent
        for path, module_info in flat_hierarchy.items():
            if path == parent_path:
                continue  # Skip self
            
            # Check if this is a direct child
            if parent_path == "":
                # For root, direct children have no dots
                if "." not in path:
                    child_name = path
                else:
                    continue
            else:
                # For non-root, check if path starts with parent and has exactly one more segment
                if not path.startswith(parent_path + "."):
                    continue
                
                # Get the path after the parent
                remainder = path[len(parent_path) + 1:]
                
                # Check if this is a direct child by counting the depth
                # For indexed modules like "layer.0", we need to check the full indexed name
                # Split remainder into segments considering indexed modules
                segments = []
                current_segment = ""
                for part in remainder.split("."):
                    if current_segment and part.isdigit():
                        # This is an index, append to previous segment
                        current_segment = f"{current_segment}.{part}"
                    else:
                        # Start new segment
                        if current_segment:
                            segments.append(current_segment)
                        current_segment = part
                if current_segment:
                    segments.append(current_segment)
                
                # Direct child has exactly one segment
                if len(segments) != 1:
                    continue
                    
                child_name = segments[0]
            
            # Determine the key to use
            # For indexed modules like layer.0, layer.1, use class_name.index
            if "." in child_name and child_name.split(".")[-1].isdigit():
                # This is an indexed module like layer.0
                index = child_name.split(".")[-1]
                key = f"{module_info.class_name}.{index}"
            else:
                # Regular module, use class name
                key = module_info.class_name
            
            # Build child structure
            child = {
                "class_name": module_info.class_name,
                "traced_tag": module_info.traced_tag,
                "scope": path,  # Full path from root
            }
            if module_info.execution_order is not None:
                child["execution_order"] = module_info.execution_order
            if module_info.source:
                child["source"] = module_info.source
            
            # Recursively build children for this child
            grandchildren = self._build_children_for_parent(path, flat_hierarchy)
            if grandchildren:
                child["children"] = grandchildren
            
            children[key] = child
        
        return children if children else None
    
    def _count_modules(self, module: dict[str, Any]) -> int:
        """Count total modules in hierarchical structure."""
        if not module:
            return 0
        
        count = 1  # Count current module
        
        # Count children recursively
        if module.get('children'):
            for child in module['children'].values():
                count += self._count_modules(child)
        
        return count
    
    def _extract_module_types(self, module: dict[str, Any]) -> list[str]:
        """Extract unique module types from hierarchical structure."""
        types = set()
        
        if module and 'class_name' in module:
            types.add(module['class_name'])
        
        # Extract from children recursively
        if module.get('children'):
            for child in module['children'].values():
                types.update(self._extract_module_types(child))
        
        return sorted(types)