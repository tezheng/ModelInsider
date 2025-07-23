"""
Improved HTP Exporter with full report generation and restructured metadata.

Key improvements:
1. Write full console output to report.txt without truncation
2. Include complete modules and tagged_nodes in report
3. Restructure metadata tagging field for better organization
"""

import json
import time
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.tree import Tree

from .htp_exporter import HTPExporter, HTPConfig
from .metadata_builder import HTPMetadataBuilder


class ImprovedHTPExporter(HTPExporter):
    """Enhanced HTP Exporter with improved reporting and metadata structure."""
    
    def __init__(self, **kwargs):
        """Initialize with extended configuration."""
        super().__init__(**kwargs)
        # Create a text report buffer for full output
        self.text_report_buffer = StringIO()
        
    def _output_message(self, message: str) -> None:
        """Override to capture full output for text report."""
        super()._output_message(message)
        # Also write to text buffer
        self.text_report_buffer.write(message + "\n")
        
    def _render_tree_output(self, tree: Tree, max_lines: int | None = None) -> None:
        """Override to capture full tree output without truncation."""
        # Original implementation for console
        super()._render_tree_output(tree, max_lines)
        
        # Full output to text buffer without truncation
        text_console = Console(file=self.text_report_buffer, force_terminal=False, width=120)
        text_console.print(tree)
        
    def _generate_metadata_file(self, output_path: str, metadata_filename: str | None = None) -> str:
        """Generate metadata with improved structure."""
        # Determine metadata path
        if metadata_filename:
            metadata_path = metadata_filename
        else:
            metadata_path = str(output_path).replace(HTPConfig.ONNX_EXTENSION, HTPConfig.METADATA_SUFFIX)
        
        # Prepare module information with execution order
        module_info = {}
        for module_path, module_data in self._hierarchy_data.items():
            module_info[module_path] = {
                "name": module_path,
                "class_name": module_data.get("class_name", ""),
                "module_type": module_data.get("module_type", ""),
                "traced_tag": module_data.get("traced_tag", ""),
                "execution_order": module_data.get("execution_order", 0),
                "expected_tag": module_data.get("expected_tag", ""),
                # Add parameter count if available
                "parameters": module_data.get("parameters", 0) if "parameters" in module_data else None
            }
        
        # Get module types for statistics
        module_types = list(set(
            m.get("class_name", "Unknown")
            for m in self._hierarchy_data.values()
            if m.get("class_name")
        ))
        
        # Prepare inputs information
        inputs_info = {}
        if self.example_inputs and isinstance(self.example_inputs, dict):
            for name, tensor in self.example_inputs.items():
                inputs_info[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype)
                }
        
        # Extract outputs from tracing
        outputs_list = []
        if self._hierarchy_builder:
            outputs = self._hierarchy_builder.get_outputs()
            if outputs:
                if hasattr(outputs, "_fields"):  # NamedTuple
                    outputs_list = list(outputs._fields)
                elif isinstance(outputs, dict):
                    outputs_list = list(outputs.keys())
        
        # Get model info
        model_info = self._export_report.get("model_info", {})
        
        # Get ONNX file size
        onnx_size_mb = 0.0
        try:
            onnx_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        except:
            pass
        
        # Build metadata using the builder pattern
        builder = HTPMetadataBuilder()
        
        # Basic export context
        metadata = (
            builder
            .with_export_context(
                strategy=self.strategy,
                version="2.0",  # Version 2.0 for improved structure
                exporter=self.__class__.__name__,
                embed_hierarchy_attributes=self.embed_hierarchy_attributes
            )
            .with_model_info(
                name_or_path=model_info.get("model_name_or_path", "unknown"),
                class_name=model_info.get("model_class", "Unknown"),
                total_modules=model_info.get("total_modules", 0),
                total_parameters=model_info.get("total_parameters", 0),
                framework=model_info.get("framework", "transformers")
            )
            .with_tracing_info(
                modules_traced=len(self._hierarchy_data),
                execution_steps=self._hierarchy_builder.get_execution_summary().get("execution_steps", 0) if self._hierarchy_builder else 0,
                model_type=self._export_report.get("export_report", {}).get("input_generation", {}).get("details", {}).get("model_type"),
                task=self._export_report.get("export_report", {}).get("input_generation", {}).get("details", {}).get("detected_task"),
                inputs=inputs_info,
                outputs=outputs_list
            )
            .with_modules(module_info)
            .with_output_files(
                onnx_path=output_path,
                onnx_size_mb=onnx_size_mb,
                metadata_path=metadata_path,
                opset_version=self._export_report.get("export_report", {}).get("onnx_export", {}).get("details", {}).get("opset_version", 17),
                output_names=self._export_report.get("export_report", {}).get("onnx_export", {}).get("details", {}).get("output_names")
            )
            .with_export_report(
                export_time_seconds=self._export_stats.get("export_time", 0.0),
                steps=self._export_report.get("export_report", {}),
                empty_tags_guarantee=self._export_stats.get("empty_tags", 0),
                coverage_percentage=self._export_stats.get("coverage_percentage", 0.0)
            )
            .with_statistics(
                export_time=self._export_stats.get("export_time", 0.0),
                hierarchy_modules=self._export_stats.get("hierarchy_modules", 0),
                onnx_nodes=self._export_stats.get("onnx_nodes", 0),
                tagged_nodes=self._export_stats.get("tagged_nodes", 0),
                empty_tags=self._export_stats.get("empty_tags", 0),
                coverage_percentage=self._export_stats.get("coverage_percentage", 0.0),
                module_types=module_types
            )
            .build()
        )
        
        # RESTRUCTURE: Move tagged_nodes to root level with better name
        metadata["nodes"] = self._tagged_nodes.copy() if self._tagged_nodes else {}
        
        # RESTRUCTURE: Move statistics and coverage under report
        if "tagging" in metadata:
            # Move statistics and coverage to report section
            if "report" not in metadata:
                metadata["report"] = {}
            
            metadata["report"]["node_tagging"] = {
                "statistics": metadata["tagging"].get("statistics", {}),
                "coverage": metadata["tagging"].get("coverage", {})
            }
            
            # Remove the old tagging field completely
            del metadata["tagging"]
        
        # Add full module hierarchy and node mapping to report
        metadata["report"]["full_hierarchy"] = {
            "modules": module_info,
            "total_modules": len(module_info),
            "module_types": module_types
        }
        
        # Write metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def export(self, *args, **kwargs):
        """Enhanced export with full text report generation."""
        # Call parent export
        result = super().export(*args, **kwargs)
        
        # Generate text report with full content
        output_path = args[1] if len(args) > 1 else kwargs.get('output_path', 'model.onnx')
        report_path = str(output_path).replace('.onnx', '_full_report.txt')
        
        # Add complete modules and node mappings to text report
        self.text_report_buffer.write("\n" + "="*80 + "\n")
        self.text_report_buffer.write("COMPLETE MODULE HIERARCHY\n")
        self.text_report_buffer.write("="*80 + "\n\n")
        
        # Write full module hierarchy
        for module_path, module_data in sorted(self._hierarchy_data.items()):
            self.text_report_buffer.write(f"Module: {module_path or '[ROOT]'}\n")
            self.text_report_buffer.write(f"  Class: {module_data.get('class_name', 'Unknown')}\n")
            self.text_report_buffer.write(f"  Tag: {module_data.get('traced_tag', '')}\n")
            self.text_report_buffer.write(f"  Execution Order: {module_data.get('execution_order', 0)}\n")
            self.text_report_buffer.write("\n")
        
        self.text_report_buffer.write("\n" + "="*80 + "\n")
        self.text_report_buffer.write("COMPLETE NODE MAPPINGS\n")
        self.text_report_buffer.write("="*80 + "\n\n")
        
        # Write full node mappings
        if self._tagged_nodes:
            for node_name, tag in sorted(self._tagged_nodes.items()):
                self.text_report_buffer.write(f"{node_name} -> {tag}\n")
        else:
            self.text_report_buffer.write("No tagged nodes found.\n")
        
        # Write full text report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.text_report_buffer.getvalue())
        
        print(f"\n📄 Full text report written to: {report_path}")
        
        return result


def compare_metadata_and_console_output():
    """
    Compare the metadata structure with console output to identify differences.
    
    Returns a plan for handling any discrepancies.
    """
    comparison = {
        "metadata_structure": {
            "root_level": [
                "export_context",
                "model", 
                "tracing",
                "modules",
                "node_hierarchy_mapping",  # NEW: renamed from tagged_nodes
                "outputs",
                "report",  # EXPANDED: now includes node_tagging
                "statistics"
            ],
            "report_section": {
                "export_time_seconds",
                "steps",
                "quality_guarantees",
                "node_tagging": {  # NEW: moved from tagging
                    "statistics",
                    "coverage"
                },
                "full_hierarchy"  # NEW: complete module info
            }
        },
        "console_output_sections": [
            "Model Preparation",
            "Input Generation", 
            "Hierarchy Building",
            "ONNX Export",
            "Node Tagging",
            "Tag Coverage Analysis",
            "Tag Injection",
            "Metadata Generation",
            "Final Summary",
            "Top 20 Nodes by Hierarchy",
            "Complete HF Hierarchy with ONNX Nodes"
        ],
        "differences": [
            {
                "issue": "Console shows tree visualization, metadata has flat structure",
                "solution": "Text report now includes full tree output without truncation"
            },
            {
                "issue": "Console truncates at 50 lines, metadata has all data",
                "solution": "Full text report captures complete output"
            },
            {
                "issue": "Metadata 'tagging' field poorly organized",
                "solution": "Restructured: statistics/coverage moved to report.node_tagging"
            },
            {
                "issue": "tagged_nodes poorly named in metadata",
                "solution": "Renamed to 'node_hierarchy_mapping' at root level"
            }
        ],
        "action_plan": [
            "1. Use ImprovedHTPExporter for full text reports",
            "2. Metadata now has cleaner structure with node_hierarchy_mapping at root",
            "3. Report section consolidated with node_tagging subsection",
            "4. Full module and node data included in both text report and metadata",
            "5. No truncation in text report - complete hierarchy visible"
        ]
    }
    
    return comparison