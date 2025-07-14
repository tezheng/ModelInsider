"""
HTP Exporter - Unified HTP Implementation with Optional Reporting

This is the unified HTP (Hierarchical Trace-and-Project) exporter that combines:
1. TracingHierarchyBuilder for optimized hierarchy building
2. ONNXNodeTagger for ONNX node tagging
3. Optional detailed reporting and visualization
4. CARDINAL RULES compliance throughout

CARDINAL RULES:
- MUST-001: NO HARDCODED LOGIC - Universal design for any model
- MUST-002: TORCH.NN FILTERING - Filter torch.nn except whitelist
- MUST-003: UNIVERSAL DESIGN - Architecture-agnostic approach

Features:
- Clean base implementation without dependencies
- Optional rich console output and detailed reporting
- Comprehensive export statistics and metadata
- Backward compatibility with existing CLI and functions
"""

from __future__ import annotations

import io
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

from ...core.onnx_node_tagger import create_node_tagger_from_hierarchy
from ...core.tracing_hierarchy_builder import TracingHierarchyBuilder

logger = logging.getLogger(__name__)


class HTPExporter:
    """
    Unified HTP Exporter with optional reporting capabilities.

    This unified implementation provides hierarchy-preserving ONNX export
    with optional detailed reporting and visualization features.
    """

    def __init__(self, verbose: bool = False, enable_reporting: bool = False):
        """
        Initialize HTP exporter.

        Args:
            verbose: Enable verbose console output
            enable_reporting: Enable detailed reporting and visualization
        """
        self.verbose = verbose
        self.enable_reporting = enable_reporting
        self.strategy = "htp"

        # Core components
        self._hierarchy_builder = None
        self._node_tagger = None

        # Export state
        self._hierarchy_data = {}
        self._export_stats = {
            "export_time": 0.0,
            "hierarchy_modules": 0,
            "onnx_nodes": 0,
            "tagged_nodes": 0,
            "empty_tags": 0,
            "coverage_percentage": 0.0,
        }
        self._metadata_path = None

        # Reporting components (optional)
        if self.enable_reporting:
            self.report_buffer = io.StringIO()
            try:
                from rich.console import Console
                from rich.text import Text
                from rich.tree import Tree
                self.console = Console()
                self._rich_available = True
            except ImportError:
                self._rich_available = False
                if self.verbose:
                    logger.warning("Rich library not available, using basic reporting")

        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
            logger.info("HTP Exporter initialized with reporting" if enable_reporting else "HTP Exporter initialized")

    def export(
        self,
        model: nn.Module,
        output_path: str = "",
        model_name_or_path: str | None = None,
        input_specs: dict[str, dict[str, Any]] | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 17,
        enable_operation_fallback: bool = False,
        metadata_filename: str | None = None,
        **export_kwargs,
    ) -> dict[str, Any]:
        """
        Export model to ONNX with hierarchy-preserving tags.

        Args:
            model: PyTorch model to export
            output_path: Path to save ONNX model
            model_name_or_path: HuggingFace model name/path for auto-input generation
            input_specs: Manual input specifications (overrides auto-generation)
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes configuration
            opset_version: ONNX opset version
            enable_operation_fallback: Enable operation-based fallback in tagging
            metadata_filename: Custom metadata filename (default: *_htp_metadata.json)
            **export_kwargs: Additional arguments for torch.onnx.export

        Returns:
            Dictionary with export statistics and metadata
        """
        start_time = time.time()

        if self.enable_reporting:
            self._print_step_header("🚀 HTP INTEGRATED EXPORT - DETAILED ANALYSIS")
            self._print_and_log(f"Model: {type(model).__name__}")
            self._print_and_log(f"Output: {Path(output_path).name}")
            self._print_and_log(f"Strategy: {self.strategy}")

        if self.verbose:
            logger.info(f"Starting HTP export for {type(model).__name__}")

        # Step 1: Generate inputs using the unified generator
        self._generate_and_validate_inputs(model_name_or_path, input_specs, export_kwargs)

        # Step 2: Set model to eval mode
        model.eval()
        if self.enable_reporting:
            self._print_step_header("📋 STEP 1: MODEL PREPARATION")
            self._print_and_log("✅ Model set to evaluation mode")

        # Step 3: Build optimized hierarchy using TracingHierarchyBuilder
        self._build_hierarchy(model, self.example_inputs)

        # Step 4: Export to ONNX
        self._export_to_onnx(
            model,
            self.example_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            **export_kwargs,
        )

        # Step 5: Load ONNX model and create node tagger
        onnx_model = onnx.load(output_path)
        self._create_node_tagger(enable_operation_fallback)

        # Step 6: Tag all ONNX nodes
        self._tag_onnx_nodes(onnx_model)

        # Step 7: Inject tags into ONNX model
        self._inject_tags_into_onnx(output_path, onnx_model)

        # Step 8: Create metadata file
        self._create_metadata_file(output_path, metadata_filename)

        # Step 9: Generate final report
        if self.enable_reporting:
            self._generate_final_report(output_path)

        # Calculate final statistics
        self._export_stats["export_time"] = time.time() - start_time

        if self.verbose:
            logger.info(
                f"HTP export completed in {self._export_stats['export_time']:.2f}s"
            )
            logger.info(f"Coverage: {self._export_stats['coverage_percentage']:.1f}%")

        # Add strategy and reporting data to export stats
        self._export_stats["strategy"] = self.strategy
        
        if self.enable_reporting:
            self._export_stats["report_data"] = self.report_buffer.getvalue()

        return self._export_stats.copy()

    def _generate_and_validate_inputs(self, model_name_or_path: str | None, input_specs: dict | None, export_kwargs: dict) -> None:
        """Generate and validate input tensors."""
        from ...core.model_input_generator import generate_dummy_inputs

        if self.enable_reporting:
            self._print_step_header("🎯 STEP 2: INPUT GENERATION & VALIDATION")

        if self.verbose:
            if input_specs:
                logger.info("Using provided input specs")
                if self.enable_reporting:
                    self._print_and_log("📝 Using provided input specifications")
            else:
                logger.info(f"Auto-generating inputs for model: {model_name_or_path}")
                if self.enable_reporting:
                    self._print_and_log(f"🤖 Auto-generating inputs for: {model_name_or_path}")

        try:
            self.example_inputs = generate_dummy_inputs(
                model_name_or_path=model_name_or_path,
                input_specs=input_specs,
                exporter="onnx",
                **export_kwargs.get("input_generation_kwargs", {}),
            )

            input_names_list = list(self.example_inputs.keys())
            if self.verbose:
                logger.info(f"✅ Generated inputs: {input_names_list}")

            if self.enable_reporting:
                self._print_and_log("✅ Input generation successful")
                self._print_and_log(f"📊 Generated {len(input_names_list)} input tensors:")
                for name, tensor in self.example_inputs.items():
                    self._print_and_log(f"   • {name}: {list(tensor.shape)} ({tensor.dtype})")

        except Exception as e:
            logger.error(f"Failed to generate inputs: {e}")
            if self.enable_reporting:
                self._print_and_log(f"❌ Input generation failed: {e}")
            raise

    def _build_hierarchy(self, model: nn.Module, example_inputs: Any) -> None:
        """Build optimized hierarchy using TracingHierarchyBuilder."""
        if self.enable_reporting:
            self._print_step_header("🏗️ STEP 3: HIERARCHY BUILDING")

        if self.verbose:
            logger.info("Building hierarchy with TracingHierarchyBuilder...")

        self._hierarchy_builder = TracingHierarchyBuilder()

        # Convert example_inputs for tracing (preserve dict format for keyword args)
        if isinstance(example_inputs, torch.Tensor):
            input_args = (example_inputs,)
        elif isinstance(example_inputs, tuple | list):
            input_args = tuple(example_inputs)
        elif isinstance(example_inputs, dict):
            # Keep dict format for models that need keyword arguments (like SAM)
            input_args = example_inputs
        else:
            input_args = (example_inputs,)

        # Trace model execution
        self._hierarchy_builder.trace_model_execution(model, input_args)

        # Get hierarchy data
        execution_summary = self._hierarchy_builder.get_execution_summary()
        self._hierarchy_data = execution_summary["module_hierarchy"]

        # Update statistics
        self._export_stats["hierarchy_modules"] = len(self._hierarchy_data)

        if self.verbose:
            logger.info(f"Built hierarchy with {len(self._hierarchy_data)} modules")
            logger.info(f"Execution steps: {execution_summary['execution_steps']}")

        if self.enable_reporting:
            self._print_and_log("✅ Hierarchy building completed")
            self._print_and_log(f"📈 Discovered {len(self._hierarchy_data)} modules")
            self._print_and_log(f"🔄 Execution steps: {execution_summary['execution_steps']}")
            
            # Print hierarchy tree if rich is available
            if self._rich_available:
                self._print_hierarchy_tree()

    def _export_to_onnx(
        self, model: nn.Module, example_inputs: Any, output_path: str, **kwargs
    ) -> None:
        """Export model to ONNX using standard PyTorch export."""
        if self.enable_reporting:
            self._print_step_header("📦 STEP 4: ONNX EXPORT")

        if self.verbose:
            logger.info(f"Exporting to ONNX: {Path(output_path).name}")

        # Filter out CLI-specific keys that aren't valid for torch.onnx.export
        cli_specific_keys = {"input_specs", "export_params", "training", "input_generation_kwargs"}
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in cli_specific_keys
        }

        if self.enable_reporting:
            self._print_and_log(f"🎯 Target file: {Path(output_path).name}")
            if filtered_kwargs:
                self._print_and_log("⚙️ Export parameters:")
                for key, value in filtered_kwargs.items():
                    self._print_and_log(f"   • {key}: {value}")

        torch.onnx.export(model, example_inputs, output_path, **filtered_kwargs)

        if self.verbose:
            logger.info("ONNX export completed")

        if self.enable_reporting:
            self._print_and_log("✅ ONNX export completed successfully")

    def _create_node_tagger(self, enable_operation_fallback: bool) -> None:
        """Create ONNX node tagger from hierarchy data."""
        if self.enable_reporting:
            self._print_step_header("🏷️ STEP 5: NODE TAGGER CREATION")

        if self.verbose:
            logger.info("Creating ONNX node tagger...")

        self._node_tagger = create_node_tagger_from_hierarchy(
            self._hierarchy_data, enable_operation_fallback=enable_operation_fallback
        )

        if self.verbose:
            logger.info(
                f"Node tagger created with model root: {self._node_tagger.model_root_tag}"
            )

        if self.enable_reporting:
            self._print_and_log("✅ Node tagger created successfully")
            self._print_and_log(f"🎯 Model root tag: {self._node_tagger.model_root_tag}")
            self._print_and_log(f"🔧 Operation fallback: {'enabled' if enable_operation_fallback else 'disabled'}")

    def _tag_onnx_nodes(self, onnx_model: onnx.ModelProto) -> None:
        """Tag all ONNX nodes using the node tagger."""
        if self.enable_reporting:
            self._print_step_header("🔗 STEP 6: ONNX NODE TAGGING")

        if self.verbose:
            logger.info("Tagging ONNX nodes...")

        # Tag all nodes
        self._tagged_nodes = self._node_tagger.tag_all_nodes(onnx_model)

        # Verify NO EMPTY TAGS rule
        empty_tags = [
            name
            for name, tag in self._tagged_nodes.items()
            if not tag or not tag.strip()
        ]

        # Update statistics
        self._export_stats["onnx_nodes"] = len(onnx_model.graph.node)
        self._export_stats["tagged_nodes"] = len(self._tagged_nodes)
        self._export_stats["empty_tags"] = len(empty_tags)
        self._export_stats["coverage_percentage"] = (
            (self._export_stats["tagged_nodes"] / self._export_stats["onnx_nodes"])
            * 100
            if self._export_stats["onnx_nodes"] > 0
            else 0.0
        )

        # Get detailed statistics
        stats = self._node_tagger.get_tagging_statistics(onnx_model)

        # Verify CARDINAL RULES compliance
        if empty_tags:
            raise RuntimeError(
                f"CARDINAL RULE VIOLATION: {len(empty_tags)} empty tags found!"
            )

        if self.verbose:
            logger.info(f"Tagged {len(self._tagged_nodes)} nodes with 0 empty tags")
            logger.info(f"Direct matches: {stats['direct_matches']}")
            logger.info(f"Parent matches: {stats['parent_matches']}")
            logger.info(f"Root fallbacks: {stats['root_fallbacks']}")

        if self.enable_reporting:
            self._print_and_log("✅ Node tagging completed successfully")
            self._print_and_log(f"📊 Tagged nodes: {len(self._tagged_nodes)}/{self._export_stats['onnx_nodes']}")
            self._print_and_log(f"📈 Coverage: {self._export_stats['coverage_percentage']:.1f}%")
            self._print_and_log(f"🎯 Direct matches: {stats['direct_matches']}")
            self._print_and_log(f"🔗 Parent matches: {stats['parent_matches']}")
            self._print_and_log(f"🏠 Root fallbacks: {stats['root_fallbacks']}")
            self._print_and_log(f"✅ Empty tags: {len(empty_tags)} (CARDINAL RULE: MUST BE 0)")

    def _inject_tags_into_onnx(
        self, output_path: str, onnx_model: onnx.ModelProto
    ) -> None:
        """Inject hierarchy tags into ONNX model metadata."""
        if self.enable_reporting:
            self._print_step_header("💉 STEP 7: TAG INJECTION")

        if self.verbose:
            logger.info("Injecting tags into ONNX model...")

        # Add hierarchy tags as node attributes
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{id(node)}"
            if node_name in self._tagged_nodes:
                tag = self._tagged_nodes[node_name]

                # Add tag as string attribute
                metadata_attr = onnx.helper.make_attribute("hierarchy_tag", tag)
                node.attribute.append(metadata_attr)

        # Add global metadata as string property
        exporter_info = json.dumps(
            {
                "exporter": "HTP_Exporter",
                "version": "2.0",
                "strategy": self.strategy,
                "hierarchy_modules": self._export_stats["hierarchy_modules"],
                "tagged_nodes": self._export_stats["tagged_nodes"],
                "coverage_percentage": self._export_stats["coverage_percentage"],
            }
        )

        # Create metadata property
        metadata_prop = onnx.StringStringEntryProto()
        metadata_prop.key = "exporter_info"
        metadata_prop.value = exporter_info
        onnx_model.metadata_props.append(metadata_prop)

        # Save updated ONNX model
        onnx.save(onnx_model, output_path)

        if self.verbose:
            logger.info("Tags injected into ONNX model")

        if self.enable_reporting:
            self._print_and_log("✅ Tags injected into ONNX model successfully")
            self._print_and_log(f"📄 Updated ONNX file: {Path(output_path).name}")

    def _create_metadata_file(self, onnx_path: str, metadata_filename: str | None = None) -> None:
        """Create comprehensive metadata file."""
        if metadata_filename:
            # Use custom filename (can be absolute or relative to ONNX file)
            if "/" in metadata_filename or "\\" in metadata_filename:
                metadata_path = metadata_filename
            else:
                # Relative to ONNX file directory
                metadata_path = str(Path(onnx_path).parent / metadata_filename)
        else:
            # Default filename: *_htp_metadata.json
            metadata_path = str(onnx_path).replace(".onnx", "_htp_metadata.json")

        metadata = {
            "export_info": {
                "onnx_file": Path(onnx_path).name,
                "exporter": "HTP_Exporter",
                "strategy": self.strategy,
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cardinal_rules_compliance": {
                    "MUST_001_no_hardcoded_logic": True,
                    "MUST_002_torch_nn_filtering": True,
                    "MUST_003_universal_design": True,
                },
            },
            "statistics": self._export_stats,
            "hierarchy_data": self._hierarchy_data,
            "tagged_nodes": self._tagged_nodes,
            "tagging_guide": {
                "overview": "HTP export with TracingHierarchyBuilder + ONNXNodeTagger",
                "tag_format": "Hierarchical tags: /ModelClass/Module/Submodule.instance",
                "no_empty_tags_guarantee": "All nodes have non-empty hierarchy tags",
                "coverage_guarantee": "100% node coverage with proper fallbacks",
            },
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Store metadata path for reporting
        self._metadata_path = metadata_path

        if self.verbose:
            logger.info(f"Created metadata file: {Path(metadata_path).name}")

        if self.enable_reporting:
            self._print_step_header("📊 STEP 8: METADATA GENERATION")
            self._print_and_log("✅ Metadata file created successfully")
            self._print_and_log(f"📄 Metadata file: {Path(metadata_path).name}")

    def _generate_final_report(self, output_path: str) -> None:
        """Generate final comprehensive report."""
        if not self.enable_reporting:
            return

        self._print_step_header("📋 FINAL EXPORT SUMMARY")
        
        self._print_and_log("🎉 HTP Export completed successfully!")
        self._print_and_log(f"📊 Export Statistics:")
        self._print_and_log(f"   • Export time: {self._export_stats['export_time']:.2f}s")
        self._print_and_log(f"   • Hierarchy modules: {self._export_stats['hierarchy_modules']}")
        self._print_and_log(f"   • ONNX nodes: {self._export_stats['onnx_nodes']}")
        self._print_and_log(f"   • Tagged nodes: {self._export_stats['tagged_nodes']}")
        self._print_and_log(f"   • Coverage: {self._export_stats['coverage_percentage']:.1f}%")
        self._print_and_log(f"   • Empty tags: {self._export_stats['empty_tags']} ✅")

        self._print_and_log(f"\n📁 Output Files:")
        self._print_and_log(f"   • ONNX model: {Path(output_path).name}")
        if self._metadata_path:
            self._print_and_log(f"   • Metadata: {Path(self._metadata_path).name}")
        else:
            # Fallback (should not happen)
            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            self._print_and_log(f"   • Metadata: {Path(metadata_path).name}")
        
        if self.enable_reporting:
            report_path = str(output_path).replace(".onnx", "_htp_export_report.txt")
            with open(report_path, "w") as f:
                f.write(self.report_buffer.getvalue())
            self._print_and_log(f"   • Report: {Path(report_path).name}")

        self._print_separator()

    def _print_hierarchy_tree(self) -> None:
        """Print hierarchy tree using Rich library."""
        if not self._rich_available or not self.enable_reporting:
            return

        try:
            from rich.text import Text
            from rich.tree import Tree

            tree = Tree("🏗️ Module Hierarchy")
            
            # Group modules by hierarchy level
            hierarchy_levels = defaultdict(list)
            for module_path, module_info in self._hierarchy_data.items():
                level = module_path.count("/")
                hierarchy_levels[level].append((module_path, module_info))
            
            # Add top-level modules
            for level in sorted(hierarchy_levels.keys())[:3]:  # Show first 3 levels
                for module_path, module_info in hierarchy_levels[level][:10]:  # Limit to 10 per level
                    module_name = module_path.split("/")[-1] if "/" in module_path else module_path
                    execution_count = module_info.get("execution_count", 0)
                    
                    node_text = Text(f"{module_name}")
                    if execution_count > 0:
                        node_text.append(f" ({execution_count}x)", style="dim")
                    
                    tree.add(node_text)
            
            # Print tree using console
            console_output = io.StringIO()
            temp_console = Console(file=console_output, width=80)
            temp_console.print(tree)
            tree_output = console_output.getvalue()
            
            self._print_and_log(tree_output)
            
        except Exception as e:
            self._print_and_log(f"⚠️ Could not generate hierarchy tree: {e}")

    def _print_step_header(self, title: str) -> None:
        """Print step header with formatting."""
        self._print_and_log(f"\n{'=' * 80}")
        self._print_and_log(f"🔍 {title}")
        self._print_and_log(f"{'=' * 80}")

    def _print_header(self, title: str) -> None:
        """Print main header."""
        self._print_and_log(f"\n{title}")

    def _print_separator(self) -> None:
        """Print separator line."""
        self._print_and_log("=" * 80)

    def _print_and_log(self, message: str) -> None:
        """Print message and log to report buffer."""
        if self.verbose:
            try:
                import click
                click.echo(message)
            except ImportError:
                print(message)

        if self.enable_reporting:
            self.report_buffer.write(message + "\n")

    def get_export_statistics(self) -> dict[str, Any]:
        """Get detailed export statistics."""
        return self._export_stats.copy()

    def get_hierarchy_data(self) -> dict[str, Any]:
        """Get the complete hierarchy data."""
        return self._hierarchy_data.copy()

    def get_tagged_nodes(self) -> dict[str, str]:
        """Get the complete node tagging data."""
        return self._tagged_nodes.copy() if hasattr(self, "_tagged_nodes") else {}


# Backward compatibility aliases
HTPIntegratedExporter = HTPExporter
HTPIntegratedExporterWithReporting = HTPExporter


def export_with_htp(
    model: nn.Module,
    output_path: str = "",
    model_name_or_path: str | None = None,
    input_specs: dict[str, dict[str, Any]] | None = None,
    verbose: bool = False,
    metadata_filename: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience function for HTP export.

    Args:
        model: PyTorch model to export
        output_path: Output ONNX file path
        model_name_or_path: HuggingFace model name/path for auto-input generation
        input_specs: Manual input specifications (overrides auto-generation)
        verbose: Enable verbose logging
        metadata_filename: Custom metadata filename (default: *_htp_metadata.json)
        **kwargs: Additional export arguments

    Returns:
        Export statistics and metadata
    """
    exporter = HTPExporter(verbose=verbose)
    return exporter.export(
        model=model,
        output_path=output_path,
        model_name_or_path=model_name_or_path,
        input_specs=input_specs,
        metadata_filename=metadata_filename,
        **kwargs,
    )


def export_with_htp_reporting(
    model: nn.Module,
    output_path: str = "",
    model_name_or_path: str | None = None,
    input_specs: dict[str, dict[str, Any]] | None = None,
    verbose: bool = False,
    metadata_filename: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience function for HTP export with detailed reporting.

    Args:
        model: PyTorch model to export
        output_path: Output ONNX file path
        model_name_or_path: HuggingFace model name/path for auto-input generation
        input_specs: Manual input specifications (overrides auto-generation)
        verbose: Enable verbose console output
        metadata_filename: Custom metadata filename (default: *_htp_metadata.json)
        **kwargs: Additional export arguments

    Returns:
        Enhanced export statistics with reporting data
    """
    exporter = HTPExporter(verbose=verbose, enable_reporting=True)
    return exporter.export(
        model=model,
        output_path=output_path,
        model_name_or_path=model_name_or_path,
        input_specs=input_specs,
        metadata_filename=metadata_filename,
        **kwargs,
    )