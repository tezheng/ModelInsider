"""
HTP Integrated Exporter with Detailed Reporting

This extends the HTP integrated exporter to include comprehensive debugging
and reporting capabilities modeled after the HTP debugger (steps 1-6).

Features:
- Detailed console output when --verbose is specified
- Comprehensive report file generated alongside metadata
- Beautiful tree visualization with rich library
- Step-by-step analysis of the export process
- All original functionality preserved
"""

from __future__ import annotations

import torch
import torch.nn as nn
import onnx
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from collections import defaultdict, Counter
from rich.tree import Tree
from rich.console import Console
from rich.text import Text
import io
import contextlib

from .htp_integrated_exporter import HTPIntegratedExporter
from ...core.tracing_hierarchy_builder import TracingHierarchyBuilder
from ...core.onnx_node_tagger import create_node_tagger_from_hierarchy

logger = logging.getLogger(__name__)


class HTPIntegratedExporterWithReporting(HTPIntegratedExporter):
    """
    Enhanced HTP Integrated Exporter with comprehensive reporting.
    
    Provides all the functionality of HTPIntegratedExporter plus:
    - Detailed step-by-step reporting
    - Beautiful tree visualization
    - Comprehensive analysis output
    - Export report file generation
    """
    
    def __init__(self, verbose: bool = False, enable_reporting: bool = True):
        """
        Initialize enhanced HTP integrated exporter.
        
        Args:
            verbose: Enable verbose console output
            enable_reporting: Enable detailed reporting (always generates report file)
        """
        super().__init__(verbose=verbose)
        self.enable_reporting = enable_reporting
        self.console = Console()
        self.report_buffer = io.StringIO()
        self._report_data = {}
        
        # Track additional statistics for reporting
        self._detailed_stats = {
            "execution_summary": {},
            "tagging_statistics": {},
            "scope_analysis": {},
            "tag_distribution": {},
            "hierarchy_tree": ""
        }
    
    def export(
        self,
        model: nn.Module,
        example_inputs: Union[torch.Tensor, Tuple, Dict],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 17,
        enable_operation_fallback: bool = False,
        **export_kwargs
    ) -> Dict[str, Any]:
        """
        Export model with detailed reporting.
        
        Args:
            Same as parent class
            
        Returns:
            Enhanced export statistics with reporting data
        """
        # Initialize reporting
        if self.enable_reporting:
            self._initialize_reporting()
        
        # Call parent export method with reporting enhancements
        result = super().export(
            model=model,
            example_inputs=example_inputs,
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            enable_operation_fallback=enable_operation_fallback,
            **export_kwargs
        )
        
        # Generate final report
        if self.enable_reporting:
            self._generate_export_report(output_path)
        
        # Add reporting data to result
        result["reporting_data"] = self._detailed_stats
        
        return result
    
    def _initialize_reporting(self) -> None:
        """Initialize reporting system."""
        self._print_header("🚀 HTP Integrated Export with Detailed Reporting")
        self._print_separator()
    
    def _build_hierarchy(self, model: nn.Module, example_inputs: Any) -> None:
        """Enhanced hierarchy building with detailed reporting."""
        if self.enable_reporting:
            self._print_section_header("Step 1: Building Module Hierarchy")
        
        # Call parent method
        super()._build_hierarchy(model, example_inputs)
        
        # Enhanced reporting
        if self.enable_reporting:
            execution_summary = self._hierarchy_builder.get_execution_summary()
            self._detailed_stats["execution_summary"] = execution_summary
            
            self._report_hierarchy_building(execution_summary)
            self._print_hierarchy_tree(self._hierarchy_data)
    
    def _export_to_onnx(self, model: nn.Module, example_inputs: Any, output_path: str, **kwargs) -> None:
        """Enhanced ONNX export with reporting."""
        if self.enable_reporting:
            self._print_section_header("Step 2: ONNX Export")
        
        # Call parent method
        super()._export_to_onnx(model, example_inputs, output_path, **kwargs)
        
        # Enhanced reporting
        if self.enable_reporting:
            onnx_model = onnx.load(output_path)
            self._report_onnx_export(onnx_model, output_path)
    
    def _create_node_tagger(self, enable_operation_fallback: bool) -> None:
        """Enhanced node tagger creation with reporting."""
        if self.enable_reporting:
            self._print_section_header("Step 3: ONNX Node Tagging Analysis")
        
        # Call parent method
        super()._create_node_tagger(enable_operation_fallback)
        
        # Enhanced reporting
        if self.enable_reporting:
            self._report_node_tagger_creation()
    
    def _tag_onnx_nodes(self, onnx_model: onnx.ModelProto) -> None:
        """Enhanced ONNX node tagging with reporting."""
        # Call parent method
        super()._tag_onnx_nodes(onnx_model)
        
        # Enhanced reporting
        if self.enable_reporting:
            self._report_node_tagging(onnx_model)
            self._analyze_scope_buckets(onnx_model)
            self._analyze_tag_distribution()
            self._print_complete_hierarchy_with_nodes(onnx_model)
    
    def _report_hierarchy_building(self, execution_summary: Dict[str, Any]) -> None:
        """Report hierarchy building results."""
        hierarchy_count = len(self._hierarchy_data)
        total_modules = execution_summary.get('total_modules', 0)
        execution_steps = execution_summary.get('execution_steps', 0)
        
        message = f"✅ Traced {hierarchy_count} modules"
        self._print_and_log(message)
        
        if execution_steps > 0:
            self._print_and_log(f"   Execution steps: {execution_steps}")
        
        if total_modules > 0:
            self._print_and_log(f"   Total modules in model: {total_modules}")
            optimization_ratio = (hierarchy_count / total_modules) * 100
            self._print_and_log(f"   Optimization ratio: {hierarchy_count}/{total_modules} ({optimization_ratio:.1f}%)")
    
    def _print_hierarchy_tree(self, hierarchy_data: Dict[str, Dict]) -> None:
        """Print HuggingFace hierarchy with beautiful tree rendering."""
        if not hierarchy_data:
            return
            
        self._print_and_log("\n📊 HuggingFace Class Hierarchy:")
        self._print_and_log("-" * 60)
        
        # Sort by execution order
        sorted_modules = sorted(
            hierarchy_data.items(),
            key=lambda x: x[1].get('execution_order', 999)
        )
        
        # Find root module
        root_class = "Model"
        for module_path, module_info in sorted_modules:
            if not module_path:  # Root module
                root_class = module_info.get('class_name', 'Model')
                break
        
        if self.verbose:
            # Create rich tree for console output
            tree = Tree(
                Text(root_class, style="bold bright_cyan"),
                guide_style="bright_white"
            )
            
            # Build tree structure
            node_map = {}
            for module_path, module_info in sorted_modules:
                if not module_path:  # Skip root
                    continue
                    
                class_name = module_info.get('class_name', 'Unknown')
                
                # Create styled text for this node
                if module_path.count('.') == 0:
                    node_text = Text()
                    node_text.append(class_name, style="bold bright_green")
                    node_text.append(f": {module_path}", style="bright_cyan")
                else:
                    node_text = Text()
                    node_text.append(class_name, style="bright_yellow")
                    node_text.append(f": {module_path}", style="bright_white")
                
                # Find parent node
                path_parts = module_path.split('.')
                if len(path_parts) == 1:
                    parent_node = tree
                else:
                    parent_path = '.'.join(path_parts[:-1])
                    parent_node = node_map.get(parent_path, tree)
                
                # Add this node to its parent
                current_node = parent_node.add(node_text)
                node_map[module_path] = current_node
            
            # Render tree to console
            self.console.print(tree)
            
            # Capture tree for report
            with io.StringIO() as tree_buffer:
                tree_console = Console(file=tree_buffer, width=120)
                tree_console.print(tree)
                self._detailed_stats["hierarchy_tree"] = tree_buffer.getvalue()
        else:
            # Simple text output for report
            tree_lines = []
            tree_lines.append(f"{root_class}")
            
            for module_path, module_info in sorted_modules:
                if not module_path:
                    continue
                
                class_name = module_info.get('class_name', 'Unknown')
                depth = module_path.count('.')
                indent = "  " * depth
                tree_lines.append(f"{indent}└─ {class_name}: {module_path}")
            
            tree_text = "\n".join(tree_lines)
            self._detailed_stats["hierarchy_tree"] = tree_text
            
            # Print summary
            self._print_and_log(f"   Root: {root_class}")
            self._print_and_log(f"   Modules: {len(sorted_modules) - 1}")
    
    def _report_onnx_export(self, onnx_model: onnx.ModelProto, output_path: str) -> None:
        """Report ONNX export results."""
        node_count = len(onnx_model.graph.node)
        self._print_and_log(f"📦 ONNX model exported: {Path(output_path).name}")
        self._print_and_log(f"✅ ONNX model contains {node_count} nodes")
    
    def _report_node_tagger_creation(self) -> None:
        """Report node tagger creation."""
        self._print_and_log("📋 Node Tagger Configuration:")
        self._print_and_log(f"   Model root: {self._node_tagger.model_root_tag}")
        self._print_and_log(f"   Scope mappings: {len(self._node_tagger.scope_to_tag)}")
        
        fallback_status = "Enabled" if self._node_tagger.enable_operation_fallback else "Disabled"
        self._print_and_log(f"   Operation fallback: {fallback_status}")
    
    def _report_node_tagging(self, onnx_model: onnx.ModelProto) -> None:
        """Report node tagging results."""
        self._print_and_log(f"\n✅ Tagged {len(self._tagged_nodes)} nodes")
        
        # Get detailed tagging statistics
        stats = self._node_tagger.get_tagging_statistics(onnx_model)
        self._detailed_stats["tagging_statistics"] = stats
        
        # Calculate accuracy breakdown
        total = stats['total_nodes']
        accuracy = {
            'direct_match_rate': (stats['direct_matches'] / total * 100) if total > 0 else 0,
            'parent_match_rate': (stats['parent_matches'] / total * 100) if total > 0 else 0,
            'root_fallback_rate': (stats['root_fallbacks'] / total * 100) if total > 0 else 0,
            'operation_match_rate': (stats['operation_matches'] / total * 100) if total > 0 else 0,
        }
        
        self._print_and_log("\n📊 Tagging Statistics:")
        self._print_and_log(f"   Direct matches: {stats['direct_matches']} ({accuracy['direct_match_rate']:.1f}%)")
        self._print_and_log(f"   Parent matches: {stats['parent_matches']} ({accuracy['parent_match_rate']:.1f}%)")
        self._print_and_log(f"   Root fallbacks: {stats['root_fallbacks']} ({accuracy['root_fallback_rate']:.1f}%)")
        
        if self._node_tagger.enable_operation_fallback:
            self._print_and_log(f"   Operation matches: {stats['operation_matches']} ({accuracy['operation_match_rate']:.1f}%)")
    
    def _analyze_scope_buckets(self, onnx_model: onnx.ModelProto) -> None:
        """Analyze scope bucketization."""
        self._print_section_header("Step 4: Scope Bucketization Analysis")
        
        scope_buckets = self._node_tagger.bucketize_nodes_by_scope(onnx_model)
        
        # Convert to simple format for analysis
        simple_buckets = {}
        for scope_name, nodes in scope_buckets.items():
            simple_buckets[scope_name] = [node.name for node in nodes]
        
        self._detailed_stats["scope_analysis"] = simple_buckets
        
        self._print_and_log(f"📂 Found {len(scope_buckets)} unique scopes:")
        
        # Sort by number of nodes
        sorted_scopes = sorted(simple_buckets.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Show top 10 scopes
        for i, (scope_name, nodes) in enumerate(sorted_scopes[:10]):
            self._print_and_log(f"   {i+1}. {scope_name}: {len(nodes)} nodes")
            if len(nodes) <= 3:
                for node in nodes:
                    self._print_and_log(f"      └─ {node}")
            else:
                self._print_and_log(f"      └─ {nodes[0]}")
                self._print_and_log(f"      └─ ...")
                self._print_and_log(f"      └─ {nodes[-1]}")
        
        if len(sorted_scopes) > 10:
            self._print_and_log(f"   ... and {len(sorted_scopes) - 10} more scopes")
    
    def _analyze_tag_distribution(self) -> None:
        """Analyze tag distribution."""
        self._print_section_header("Step 5: Tag Distribution Analysis")
        
        tag_distribution = Counter(self._tagged_nodes.values())
        self._detailed_stats["tag_distribution"] = dict(tag_distribution.most_common())
        
        self._print_and_log(f"🏷️ Unique tags: {len(tag_distribution)}")
        self._print_and_log("\nTop 10 most common tags:")
        
        for i, (tag, count) in enumerate(tag_distribution.most_common(10)):
            percentage = (count / len(self._tagged_nodes)) * 100
            self._print_and_log(f"   {i+1}. {tag}: {count} nodes ({percentage:.1f}%)")
    
    def _print_complete_hierarchy_with_nodes(self, onnx_model: onnx.ModelProto) -> None:
        """Print complete hierarchy with ONNX nodes."""
        self._print_section_header("Step 6: Complete HF Hierarchy with ONNX Nodes")
        
        if not self.verbose:
            self._print_and_log("🌳 Complete hierarchy tree (see detailed version in report file)")
            # Still generate the tree for the report, but don't show on console
            self._generate_complete_hierarchy_tree(onnx_model, show_console=False)
            return
        
        # Generate and show tree for both console and report
        self._generate_complete_hierarchy_tree(onnx_model, show_console=True)
    
    def _generate_complete_hierarchy_tree(self, onnx_model: onnx.ModelProto, show_console: bool = True) -> None:
        """Generate complete hierarchy tree with ONNX nodes."""
        
        # Group nodes by their tags
        nodes_by_tag = defaultdict(list)
        for node_name, tag in self._tagged_nodes.items():
            nodes_by_tag[tag].append(node_name)
        
        # Create mapping for additional node info
        node_info_map = {}
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{id(node)}"
            node_info_map[node_name] = {
                'op_type': node.op_type,
                'scope': self._node_tagger._extract_scope_from_node(node),
            }
        
        # Find root class
        root_class = "Model"
        for module_path, module_info in self._hierarchy_data.items():
            if not module_path:
                root_class = module_info.get('class_name', 'Model')
                break
        
        # Create rich tree
        tree = Tree(
            Text(f"{root_class} ({len(self._tagged_nodes)} ONNX nodes)", style="bold bright_cyan"),
            guide_style="bright_white"
        )
        
        # Build tree with ONNX nodes
        node_map = {}
        sorted_hierarchy = sorted(
            self._hierarchy_data.items(),
            key=lambda x: x[1].get('execution_order', 999)
        )
        
        for module_path, module_info in sorted_hierarchy:
            if not module_path:
                continue
                
            class_name = module_info.get('class_name', 'Unknown')
            module_tag = module_info.get('traced_tag', '')
            
            # Count ONNX nodes for this module
            module_nodes = nodes_by_tag.get(module_tag, [])
            node_count = len(module_nodes)
            
            # Create styled text
            module_text = Text()
            if module_path.count('.') == 0:
                module_text.append(class_name, style="bold bright_green")
                module_text.append(f": {module_path}", style="bright_cyan")
            else:
                module_text.append(class_name, style="bright_yellow")
                module_text.append(f": {module_path}", style="bright_white")
            
            module_text.append(f" ({node_count} nodes)", style="dim bright_white")
            
            # Find parent node
            path_parts = module_path.split('.')
            if len(path_parts) == 1:
                parent_node = tree
            else:
                parent_path = '.'.join(path_parts[:-1])
                parent_node = node_map.get(parent_path, tree)
            
            # Add module to tree
            current_node = parent_node.add(module_text)
            node_map[module_path] = current_node
            
            # Add ONNX operations as children (limit to avoid clutter)
            if module_nodes:
                ops_by_type = defaultdict(list)
                for node_name in module_nodes:
                    if node_name in node_info_map:
                        op_type = node_info_map[node_name]['op_type']
                        ops_by_type[op_type].append(node_name)
                
                for op_type, op_nodes in sorted(ops_by_type.items()):
                    if len(op_nodes) == 1:
                        op_text = Text()
                        op_text.append(f"{op_type}", style="bright_magenta")
                        op_text.append(f": {op_nodes[0]}", style="dim bright_cyan")
                        current_node.add(op_text)
                    else:
                        group_text = Text()
                        group_text.append(f"{op_type}", style="bright_magenta")
                        group_text.append(f" ({len(op_nodes)} ops)", style="dim bright_white")
                        current_node.add(group_text)
        
        # Always capture tree for report
        with io.StringIO() as tree_buffer:
            tree_console = Console(file=tree_buffer, width=120, legacy_windows=False)
            tree_console.print(tree)
            tree_text = tree_buffer.getvalue()
        
        # Store the complete hierarchy tree in detailed stats
        self._detailed_stats["complete_hierarchy_tree"] = tree_text
        
        # Render tree to console if requested
        if show_console:
            self.console.print(tree)
            # Also capture the tree text in the report buffer for console output section
            self._print_and_log("(Tree visualization shown above)")
        else:
            # For quiet mode, add simplified text version to report buffer
            self._print_and_log("\n📊 Complete HF Hierarchy with ONNX Nodes:")
            self._print_and_log("-" * 60)
            self._add_simplified_tree_to_report()
    
    def _add_simplified_tree_to_report(self) -> None:
        """Add a simplified text version of the hierarchy tree to the report."""
        # Find root class
        root_class = "Model"
        for module_path, module_info in self._hierarchy_data.items():
            if not module_path:
                root_class = module_info.get('class_name', 'Model')
                break
        
        # Group nodes by their tags
        nodes_by_tag = defaultdict(list)
        for node_name, tag in self._tagged_nodes.items():
            nodes_by_tag[tag].append(node_name)
        
        # Create simplified text tree
        tree_lines = []
        tree_lines.append(f"{root_class} ({len(self._tagged_nodes)} ONNX nodes)")
        
        sorted_hierarchy = sorted(
            self._hierarchy_data.items(),
            key=lambda x: x[1].get('execution_order', 999)
        )
        
        for module_path, module_info in sorted_hierarchy:
            if not module_path:
                continue
                
            class_name = module_info.get('class_name', 'Unknown')
            module_tag = module_info.get('traced_tag', '')
            module_nodes = nodes_by_tag.get(module_tag, [])
            node_count = len(module_nodes)
            
            depth = module_path.count('.')
            indent = "  " * depth
            tree_lines.append(f"{indent}└─ {class_name}: {module_path} ({node_count} nodes)")
            
            # Add top operation types for this module
            if module_nodes:
                ops_by_type = defaultdict(int)
                for node_name in module_nodes:
                    # Extract operation type from node name
                    op_type = node_name.split('/')[-1].split('_')[0]
                    ops_by_type[op_type] += 1
                
                # Show top 3 operation types
                top_ops = sorted(ops_by_type.items(), key=lambda x: x[1], reverse=True)[:3]
                for op_type, count in top_ops:
                    tree_lines.append(f"{indent}    • {op_type}: {count} ops")
        
        # Add to report buffer
        for line in tree_lines:
            self._print_and_log(line)
    
    def _generate_export_report(self, output_path: str) -> None:
        """Generate comprehensive export report."""
        report_path = str(output_path).replace(".onnx", "_htp_export_report.txt")
        
        # Prepare report content
        report_content = []
        report_content.append("HTP Integrated Export Report")
        report_content.append("=" * 60)
        report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"Model: {Path(output_path).stem}")
        report_content.append(f"Strategy: {self.strategy}")
        report_content.append("")
        
        # Add export statistics
        report_content.append("Export Statistics:")
        report_content.append("-" * 30)
        for key, value in self._export_stats.items():
            if isinstance(value, float):
                report_content.append(f"{key}: {value:.2f}")
            else:
                report_content.append(f"{key}: {value}")
        report_content.append("")
        
        # Add detailed analysis
        if self._detailed_stats.get("execution_summary"):
            report_content.append("Execution Summary:")
            report_content.append("-" * 30)
            for key, value in self._detailed_stats["execution_summary"].items():
                report_content.append(f"{key}: {value}")
            report_content.append("")
        
        if self._detailed_stats.get("tagging_statistics"):
            report_content.append("Tagging Statistics:")
            report_content.append("-" * 30)
            for key, value in self._detailed_stats["tagging_statistics"].items():
                report_content.append(f"{key}: {value}")
            report_content.append("")
        
        # Add hierarchy tree
        if self._detailed_stats.get("hierarchy_tree"):
            report_content.append("Module Hierarchy Tree:")
            report_content.append("-" * 30)
            report_content.append(self._detailed_stats["hierarchy_tree"])
            report_content.append("")
        
        # Add scope analysis (top 20)
        if self._detailed_stats.get("scope_analysis"):
            report_content.append("Top 20 Scopes by Node Count:")
            report_content.append("-" * 30)
            sorted_scopes = sorted(
                self._detailed_stats["scope_analysis"].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            for i, (scope_name, nodes) in enumerate(sorted_scopes[:20]):
                report_content.append(f"{i+1:2d}. {scope_name}: {len(nodes)} nodes")
            report_content.append("")
        
        # Add tag distribution (top 20)
        if self._detailed_stats.get("tag_distribution"):
            report_content.append("Top 20 Tags by Node Count:")
            report_content.append("-" * 30)
            total_nodes = sum(self._detailed_stats["tag_distribution"].values())
            for i, (tag, count) in enumerate(list(self._detailed_stats["tag_distribution"].items())[:20]):
                percentage = (count / total_nodes) * 100
                report_content.append(f"{i+1:2d}. {tag}: {count} nodes ({percentage:.1f}%)")
            report_content.append("")
        
        # Add complete hierarchy tree (Step 6)
        if self._detailed_stats.get("complete_hierarchy_tree"):
            report_content.append("Complete HF Hierarchy with ONNX Nodes:")
            report_content.append("-" * 30)
            report_content.append(self._detailed_stats["complete_hierarchy_tree"])
            report_content.append("")
        
        # Add buffer content (console output)
        if self.report_buffer.getvalue():
            report_content.append("Console Output:")
            report_content.append("-" * 30)
            report_content.append(self.report_buffer.getvalue())
        
        # Write report file
        with open(report_path, 'w') as f:
            f.write("\n".join(report_content))
        
        if self.verbose:
            self._print_and_log(f"\n📄 Export report saved: {Path(report_path).name}")
    
    def _print_section_header(self, title: str) -> None:
        """Print a formatted section header."""
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


def export_with_htp_integrated_reporting(
    model: nn.Module,
    example_inputs: Any,
    output_path: str,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for HTP integrated export with detailed reporting.
    
    Args:
        model: PyTorch model to export
        example_inputs: Example inputs for tracing
        output_path: Output ONNX file path
        verbose: Enable verbose console output
        **kwargs: Additional export arguments
        
    Returns:
        Enhanced export statistics with reporting data
    """
    exporter = HTPIntegratedExporterWithReporting(verbose=verbose, enable_reporting=True)
    return exporter.export(model, example_inputs, output_path, **kwargs)