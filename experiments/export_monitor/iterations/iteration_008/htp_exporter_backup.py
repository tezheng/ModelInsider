"""
HTP (Hierarchy-preserving Tags Protocol) Exporter.

This exporter preserves the hierarchical structure of HuggingFace models
when converting to ONNX format by tracing module execution and tagging
ONNX nodes with their source module information.

Key Features:
- Direct module context capture during execution
- Precise hierarchy tag generation
- Comprehensive metadata export
- Optional detailed reporting
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

import onnx
import torch
import torch.nn as nn
from rich.console import Console

from ...core.onnx_node_tagger import create_node_tagger_from_hierarchy
from ...core.onnx_utils import infer_output_names
from ...core.tracing_hierarchy_builder import TracingHierarchyBuilder
from .export_monitor import HTPExportMonitor, HTPExportStep
from .metadata_builder import HTPMetadataBuilder

logger = logging.getLogger(__name__)


class HTPConfig:
    """Configuration constants for HTP Exporter."""

    # Strategy and file naming
    STRATEGY_NAME = "htp"
    ONNX_EXTENSION = ".onnx"
    REPORT_SUFFIX = "_htp_export_report.txt"
    METADATA_SUFFIX = "_htp_metadata.json"

    # Console formatting
    CONSOLE_WIDTH = 80

    # Export defaults
    DEFAULT_TASK = "feature-extraction"

    # Default ONNX export configuration
    DEFAULT_EXPORT_CONFIG: ClassVar[dict[str, Any]] = {
        "opset_version": 17,
        "do_constant_folding": True,
        "verbose": False,  # ONNX internal verbose
    }

    # Default export statistics structure
    DEFAULT_EXPORT_STATS: ClassVar[dict[str, Any]] = {
        "export_time": 0.0,
        "hierarchy_modules": 0,
        "onnx_nodes": 0,
        "tagged_nodes": 0,
        "empty_tags": sys.maxsize,  # CARDINAL RULE: Must be 0, default to max int to catch violations
        "coverage_percentage": 0.0,  # Will be calculated based on actual tagging
        "strategy": STRATEGY_NAME,
    }


class HTPExporter:
    """
    HTP Exporter with proper verbose console output.

    This implementation properly separates:
    - verbose: Controls console output (8-step format)
    - enable_reporting: Controls report file generation
    """

    def __init__(
        self,
        verbose: bool = False,
        enable_reporting: bool = False,
        embed_hierarchy_attributes: bool = True,
        include_torch_nn_children: bool = False,
    ):
        """
        Initialize HTP exporter.

        Args:
            verbose: Enable verbose console output (8-step format)
            enable_reporting: Enable report file generation
            embed_hierarchy_attributes: Whether to embed hierarchy_tag attributes in ONNX
                                       (disabled by --clean-onnx or --no-hierarchy-attrs)
            include_torch_nn_children: Include torch.nn children of HF modules in hierarchy
                                      for proper operation attribution (e.g., ResNet)
        """
        self.verbose = verbose
        self.enable_reporting = enable_reporting
        self.embed_hierarchy_attributes = embed_hierarchy_attributes
        self.include_torch_nn_children = include_torch_nn_children
        self.strategy = HTPConfig.STRATEGY_NAME

        # Core components
        self._hierarchy_builder = None
        self._node_tagger = None
        self._hierarchy_data = {}
        self._tagged_nodes = {}
        self._tagging_stats = {}

        # Example inputs
        self.example_inputs = None

        # Export statistics
        self._export_stats = HTPConfig.DEFAULT_EXPORT_STATS.copy()

        # Export monitor will be initialized in export()
        self._monitor = None
        
        # Rich console for tree rendering
        self.console = Console(width=HTPConfig.CONSOLE_WIDTH)

        # Configure logging based on verbose mode
        if verbose:
            # Suppress INFO messages when verbose console output is enabled
            logging.getLogger().setLevel(logging.WARNING)
        else:
            # Suppress INFO messages when not in verbose mode too
            logging.getLogger().setLevel(logging.WARNING)


    def export(
        self,
        model: nn.Module | None = None,
        output_path: str = "",
        model_name_or_path: str | None = None,
        input_specs: dict[str, dict[str, Any]] | None = None,
        export_config: dict[str, Any] | None = None,
        enable_operation_fallback: bool = False,
        metadata_filename: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Export model to ONNX with hierarchy-preserving tags."""
        start_time = time.time()
        
        # Initialize export monitor
        self._monitor = HTPExportMonitor(
            output_path=output_path,
            verbose=self.verbose,
            enable_report=self.enable_reporting,
            console=self.console,
            embed_hierarchy=self.embed_hierarchy_attributes
        )

        # Use monitor as context manager
        with self._monitor as monitor:
            # Auto-load model if needed
            if model is None:
                if model_name_or_path is None:
                    raise ValueError(
                        "Either 'model' or 'model_name_or_path' must be provided."
                    )

                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_name_or_path)

            # Step 1: Model Preparation
            model.eval()
            
            # Update monitor with model info
            monitor.update(
                HTPExportStep.MODEL_PREP,
                model_name=model_name_or_path or "unknown",
                model_class=type(model).__name__,
                total_modules=len(list(model.modules())),
                total_parameters=sum(p.numel() for p in model.parameters())
            )

            # Step 2: Input Generation
            self._create_example_inputs(model_name_or_path, input_specs)
            
            # Prepare input generation data
            input_gen_data = {
                "method": "provided" if input_specs else "auto_generated",
                "inputs": {}
            }
            
            # Try to add model type and task info
            if model_name_or_path and not input_specs:
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_name_or_path)
                    input_gen_data["model_type"] = config.model_type
                    
                    # Try to detect task
                    task = None
                    try:
                        from optimum.exporters import TasksManager
                        supported_tasks = TasksManager.get_supported_tasks_for_model_type(
                            config.model_type, exporter="onnx", library_name="transformers"
                        )
                        if supported_tasks:
                            task = next(iter(supported_tasks.keys()))
                        else:
                            task = HTPConfig.DEFAULT_TASK
                    except Exception:
                        task = HTPConfig.DEFAULT_TASK
                    
                    input_gen_data["task"] = task
                except Exception:
                    # If we can't get model info, just continue
                    pass
            
            # Add input shapes and dtypes
            if self.example_inputs:
                for name, tensor in self.example_inputs.items():
                    input_gen_data["inputs"][name] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype)
                    }
            
            # Update monitor
            monitor.update(HTPExportStep.INPUT_GEN, **input_gen_data)

            # Step 3: Hierarchy Building
            self._trace_model_hierarchy(model)
            
            # Update monitor with hierarchy data
            execution_steps = self._hierarchy_builder.get_execution_summary().get("execution_steps", 0) if self._hierarchy_builder else 0
            monitor.update(
                HTPExportStep.HIERARCHY,
                hierarchy=self._hierarchy_data,
                execution_steps=execution_steps
            )

            # Step 4: ONNX Export
            export_kwargs = {
                **HTPConfig.DEFAULT_EXPORT_CONFIG,
                **(export_config or {}),
                **kwargs,
            }

            self._convert_model_to_onnx(model, output_path, export_kwargs)
            
            # Update monitor with ONNX export info
            onnx_size_mb = round(Path(output_path).stat().st_size / (1024 * 1024), 2) if Path(output_path).exists() else 0
            monitor.update(
                HTPExportStep.ONNX_EXPORT,
                opset_version=export_kwargs.get("opset_version", 17),
                do_constant_folding=export_kwargs.get("do_constant_folding", True),
                onnx_size_mb=onnx_size_mb
            )

            # Step 5: Node Tagger Creation
            onnx_model = onnx.load(output_path)

            self._initialize_node_tagger(enable_operation_fallback)

            # Update monitor
            monitor.update(
                HTPExportStep.TAGGER_CREATION,
                tagger_type="HierarchyNodeTagger",
                enable_operation_fallback=enable_operation_fallback
            )

            # Step 6: Node Tagging
            self._apply_hierarchy_tags(onnx_model)

            # Update monitor with tagging results
            monitor.update(
                HTPExportStep.NODE_TAGGING,
                total_nodes=len(onnx_model.graph.node),
                tagged_nodes=self._tagged_nodes,
                tagging_stats=self._tagging_stats
            )

            # Step 7: Tag Injection
            self._embed_tags_in_onnx(output_path, onnx_model)
            
            # Update monitor
            monitor.update(HTPExportStep.TAG_INJECTION)

            # Calculate final statistics before metadata generation
            self._export_stats["export_time"] = time.time() - start_time
            self._export_stats["hierarchy_modules"] = len(self._hierarchy_data)
            self._export_stats["onnx_nodes"] = len(onnx_model.graph.node)
            self._export_stats["tagged_nodes"] = len(self._tagged_nodes)
            
            # Calculate empty tags (should be 0 with our implementation)
            empty_tag_count = sum(1 for tag in self._tagged_nodes.values() if not tag or not tag.strip())
            self._export_stats["empty_tags"] = empty_tag_count
            
            # Calculate coverage percentage
            total_nodes = len(onnx_model.graph.node)
            tagged_nodes = len(self._tagged_nodes)
            coverage = (tagged_nodes / total_nodes * 100.0) if total_nodes > 0 else 0.0
            self._export_stats["coverage_percentage"] = coverage
            
            # Step 8: Metadata Generation
            metadata_path = self._generate_metadata_file(output_path, metadata_filename)
            
            # Update monitor
            monitor.update(
                HTPExportStep.METADATA_GEN,
                metadata_path=metadata_path
            )
            
            # Store output names if available
            outputs = self._hierarchy_builder.get_outputs() if self._hierarchy_builder else None
            output_names = infer_output_names(outputs) if outputs else []
            monitor.data.output_names = output_names or []

        # The monitor's context manager will handle finalization
        return self._export_stats.copy()














    # Internal implementation methods
    def _create_example_inputs(
        self, model_name_or_path: str, input_specs: dict | None
    ) -> None:
        """Generate inputs internally."""
        from ...core.model_input_generator import generate_dummy_inputs

        # The model_input_generator already logs these details internally
        # We should NOT duplicate or hardcode them here

        self.example_inputs = generate_dummy_inputs(
            model_name_or_path=model_name_or_path,
            input_specs=input_specs,
            exporter="onnx",
        )

    def _trace_model_hierarchy(self, model: nn.Module) -> None:
        """Build hierarchy internally."""
        # Determine if we need torch.nn exceptions for this model
        exceptions = None
        if self.include_torch_nn_children:
            # Common torch.nn modules that might be children of HF modules
            exceptions = [
                "Conv1d",
                "Conv2d",
                "Conv3d",
                "BatchNorm1d",
                "BatchNorm2d",
                "BatchNorm3d",
                "Linear",
                "Embedding",
                "ReLU",
                "GELU",
                "Tanh",
                "Sigmoid",
                "Dropout",
                "LayerNorm",
                "MaxPool1d",
                "MaxPool2d",
                "MaxPool3d",
                "AvgPool1d",
                "AvgPool2d",
                "AvgPool3d",
            ]

        self._hierarchy_builder = TracingHierarchyBuilder(exceptions=exceptions)

        # Convert inputs
        if isinstance(self.example_inputs, dict):
            input_args = self.example_inputs
        else:
            input_args = (self.example_inputs,)

        self._hierarchy_builder.trace_model_execution(model, input_args)

        summary = self._hierarchy_builder.get_execution_summary()
        self._hierarchy_data = summary["module_hierarchy"]
        self._export_stats["hierarchy_modules"] = len(self._hierarchy_data)

    def _convert_model_to_onnx(
        self, model: nn.Module, output_path: str, export_kwargs: dict
    ) -> None:
        """Export to ONNX internally."""
        import warnings

        # Filter out non-ONNX export keys
        filtered_kwargs = {
            k: v
            for k, v in export_kwargs.items()
            if k
            not in {
                "input_specs",
                "export_params",
                "training",
                "input_generation_kwargs",
            }
        }

        # Handle input/output names for Optimum compatibility
        if "input_names" not in filtered_kwargs and isinstance(
            self.example_inputs, dict
        ):
            filtered_kwargs["input_names"] = list(self.example_inputs.keys())

        # Universal output naming: Only try to infer names if not already provided
        # This approach is model-agnostic and doesn't hardcode any specific names
        if "output_names" not in filtered_kwargs:
            # Get outputs from the tracing hierarchy builder to avoid duplicate execution
            outputs = self._hierarchy_builder.get_outputs() if self._hierarchy_builder else None
            
            # Use the universal utility function to infer output names
            output_names = infer_output_names(outputs)
            if output_names:
                filtered_kwargs["output_names"] = output_names

        # Convert inputs to tuple for ONNX export if needed
        if isinstance(self.example_inputs, dict):
            example_inputs_tuple = tuple(self.example_inputs.values())
        else:
            example_inputs_tuple = self.example_inputs

        # Suppress TracerWarnings during export
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(
                model, example_inputs_tuple, output_path, **filtered_kwargs
            )

    def _initialize_node_tagger(self, enable_operation_fallback: bool) -> None:
        """Create node tagger internally."""
        self._node_tagger = create_node_tagger_from_hierarchy(
            self._hierarchy_data, enable_operation_fallback=enable_operation_fallback
        )

    def _apply_hierarchy_tags(self, onnx_model: onnx.ModelProto) -> None:
        """Tag nodes internally."""
        self._tagged_nodes = self._node_tagger.tag_all_nodes(onnx_model)

        # Get statistics
        stats = self._node_tagger.get_tagging_statistics(onnx_model)
        self._tagging_stats = stats

        # Update export stats
        self._export_stats["onnx_nodes"] = len(onnx_model.graph.node)
        self._export_stats["tagged_nodes"] = len(self._tagged_nodes)
        
        # Calculate empty tags (should be 0 with our implementation)
        empty_tag_count = sum(1 for tag in self._tagged_nodes.values() if not tag or not tag.strip())
        self._export_stats["empty_tags"] = empty_tag_count
        
        # Calculate coverage percentage
        total_nodes = len(onnx_model.graph.node)
        tagged_nodes = len(self._tagged_nodes)
        coverage = (tagged_nodes / total_nodes * 100.0) if total_nodes > 0 else 0.0
        self._export_stats["coverage_percentage"] = coverage

    def _embed_tags_in_onnx(
        self, output_path: str, onnx_model: onnx.ModelProto
    ) -> None:
        """Inject tags internally."""
        if self.embed_hierarchy_attributes:
            # Add hierarchy tags as node attributes
            for node in onnx_model.graph.node:
                node_name = node.name or f"{node.op_type}_{id(node)}"
                if node_name in self._tagged_nodes:
                    tag = self._tagged_nodes[node_name]
                    metadata_attr = onnx.helper.make_attribute("hierarchy_tag", tag)
                    node.attribute.append(metadata_attr)
            # Save model
            onnx.save(onnx_model, output_path)

    def _generate_metadata_file(
        self, output_path: str, metadata_filename: str | None
    ) -> str:
        """Create metadata using the clean builder pattern."""
        if metadata_filename:
            metadata_path = metadata_filename
        else:
            metadata_path = str(output_path).replace(
                HTPConfig.ONNX_EXTENSION, HTPConfig.METADATA_SUFFIX
            )

        # Extract data from monitor
        monitor_data = self._monitor.data
        
        # Get output names if available
        outputs = self._hierarchy_builder.get_outputs() if self._hierarchy_builder else None
        output_names = infer_output_names(outputs) if outputs else None
        
        # Get module types
        module_types = list(
            {
                info.get("class_name", "")
                for info in self._hierarchy_data.values()
                if info.get("class_name")
            }
        )
        
        # Get input generation details from steps
        input_gen_step = monitor_data.steps.get("input_generation", {})
        onnx_export_step = monitor_data.steps.get("onnx_export", {})
        
        # Build metadata - TODO: Switch to Pydantic when added as dependency
        # from .pydantic_builder import HTPMetadataBuilderPydantic
        # metadata_model = HTPMetadataBuilderPydantic.from_exporter_state(
        #     export_report=self._export_report,
        #     export_stats=self._export_stats,
        #     hierarchy_data=self._hierarchy_data,
        #     tagged_nodes=self._tagged_nodes,
        #     tagging_stats=self._tagging_stats,
        #     hierarchy_builder=self._hierarchy_builder,
        #     output_path=output_path,
        #     metadata_path=metadata_path,
        #     embed_hierarchy_attributes=self.embed_hierarchy_attributes,
        #     strategy=self.strategy,
        # )
        # metadata = metadata_model.model_dump(exclude_none=True)
        
        # For now, use dataclass builder until Pydantic is added
        builder = HTPMetadataBuilder()
        
        metadata = (
            builder
            .with_export_context(
                strategy=self.strategy,
                embed_hierarchy_attributes=self.embed_hierarchy_attributes
            )
            .with_model_info(
                name_or_path=monitor_data.model_name,
                class_name=monitor_data.model_class,
                total_modules=monitor_data.total_modules,
                total_parameters=monitor_data.total_parameters,
                framework="transformers"
            )
            .with_tracing_info(
                modules_traced=len(self._hierarchy_data),
                execution_steps=monitor_data.execution_steps,
                model_type=input_gen_step.get("model_type"),
                task=input_gen_step.get("task"),
                inputs=input_gen_step.get("inputs"),
                outputs=output_names
            )
            .with_modules(self._hierarchy_data)
            .with_tagging_info(
                tagged_nodes=self._tagged_nodes,
                statistics=self._tagging_stats if hasattr(self, "_tagging_stats") else {},
                total_onnx_nodes=self._export_stats.get("onnx_nodes", 0),
                tagged_nodes_count=self._export_stats.get("tagged_nodes", 0),
                coverage_percentage=self._export_stats.get("coverage_percentage", 0.0),
                empty_tags=self._export_stats.get("empty_tags", 0)
            )
            .with_output_files(
                onnx_path=output_path,
                onnx_size_mb=monitor_data.onnx_size_mb,
                metadata_path=metadata_path,
                opset_version=onnx_export_step.get("opset_version", 17),
                output_names=output_names
            )
            .with_export_report(
                export_time_seconds=round(self._export_stats.get("export_time", 0), 2),
                steps={
                    "model_preparation": {
                        "status": "completed",
                        "details": {
                            "model_class": monitor_data.model_class,
                            "module_count": monitor_data.total_modules,
                            "parameter_count": monitor_data.total_parameters,
                            "export_target": output_path,
                            "model_mode": "eval"
                        }
                    },
                    "input_generation": {
                        "status": "completed",
                        "method": input_gen_step.get("method", "unknown"),
                    },
                    "hierarchy_building": {
                        "status": "completed",
                        "details": {
                            "builder": "TracingHierarchyBuilder",
                            "modules_traced": len(self._hierarchy_data),
                            "execution_steps": monitor_data.execution_steps
                        }
                    },
                    "onnx_export": {
                        "status": "completed",
                        "export_config": {
                            "opset_version": onnx_export_step.get("opset_version", 17),
                            "do_constant_folding": onnx_export_step.get("do_constant_folding", True)
                        }
                    },
                    "node_tagging": {
                        "status": "completed",
                        "top_hierarchies": []  # TODO: Calculate from tagged_nodes
                    },
                    "tag_injection": {
                        "status": "completed",
                        "details": {
                            "hierarchy_attributes_embedded": self.embed_hierarchy_attributes,
                            "injection_method": "onnx_node_attributes" if self.embed_hierarchy_attributes else "none",
                            "nodes_with_tags": len(self._tagged_nodes) if self.embed_hierarchy_attributes else 0
                        }
                    }
                },
                empty_tags_guarantee=self._export_stats.get("empty_tags", 0),
                coverage_percentage=self._export_stats.get("coverage_percentage", 0.0)
            )
            .with_statistics(
                export_time=self._export_stats.get("export_time", 0),
                hierarchy_modules=self._export_stats.get("hierarchy_modules", 0),
                onnx_nodes=self._export_stats.get("onnx_nodes", 0),
                tagged_nodes=self._export_stats.get("tagged_nodes", 0),
                empty_tags=self._export_stats.get("empty_tags", 0),
                coverage_percentage=self._export_stats.get("coverage_percentage", 0.0),
                module_types=module_types
            )
            .build()
        )

        # Restructure metadata to improve organization
        # Move tagged_nodes to root as "nodes"
        if "tagging" in metadata and "tagged_nodes" in metadata["tagging"]:
            metadata["nodes"] = metadata["tagging"]["tagged_nodes"]
            del metadata["tagging"]["tagged_nodes"]
        
        # Move statistics and coverage to report/node_tagging
        if "tagging" in metadata:
            if "report" not in metadata:
                metadata["report"] = {}
            
            # Ensure node_tagging section exists in report
            if "node_tagging" not in metadata["report"]:
                metadata["report"]["node_tagging"] = {}
            
            # Move statistics and coverage
            if "statistics" in metadata["tagging"]:
                metadata["report"]["node_tagging"]["statistics"] = metadata["tagging"]["statistics"]
            if "coverage" in metadata["tagging"]:
                metadata["report"]["node_tagging"]["coverage"] = metadata["tagging"]["coverage"]
            
            # Remove the now-empty tagging section if it only had these fields
            remaining_fields = [k for k in metadata["tagging"] if k not in ["tagged_nodes", "statistics", "coverage"]]
            if not remaining_fields:
                del metadata["tagging"]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata_path


# Export functions for backward compatibility
def export_with_htp(
    model: nn.Module,
    output_path: str = "",
    model_name_or_path: str | None = None,
    input_specs: dict[str, dict[str, Any]] | None = None,
    verbose: bool = False,
    embed_hierarchy_attributes: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Export with HTP strategy."""
    exporter = HTPExporter(
        verbose=verbose, embed_hierarchy_attributes=embed_hierarchy_attributes
    )
    return exporter.export(
        model=model,
        output_path=output_path,
        model_name_or_path=model_name_or_path,
        input_specs=input_specs,
        **kwargs,
    )


def export_with_htp_reporting(
    model: nn.Module,
    output_path: str = "",
    model_name_or_path: str | None = None,
    input_specs: dict[str, dict[str, Any]] | None = None,
    verbose: bool = False,
    embed_hierarchy_attributes: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Export with HTP strategy and reporting."""
    exporter = HTPExporter(
        verbose=verbose,
        enable_reporting=True,
        embed_hierarchy_attributes=embed_hierarchy_attributes,
    )
    return exporter.export(
        model=model,
        output_path=output_path,
        model_name_or_path=model_name_or_path,
        input_specs=input_specs,
        **kwargs,
    )
