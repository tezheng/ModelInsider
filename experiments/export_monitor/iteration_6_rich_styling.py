#!/usr/bin/env python3
"""
Iteration 6: Add rich text styling support to match baseline console output.
Replace print() with rich console for colored output.
"""

import io
import json
import sys
from pathlib import Path


def create_rich_export_monitor():
    """Create export monitor with rich text styling."""
    
    code = '''"""
HTP Export Monitor - Iteration 6 with Rich text styling.
Uses rich library for colored console output matching baseline.
"""

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from rich.console import Console
from rich.text import Text


# Configuration - All magic numbers/strings in one place
class Config:
    """All configuration values centralized."""
    # Separators
    MAJOR_SEP = "=" * 80
    MINOR_SEP = "-" * 60
    SHORT_SEP = "-" * 30
    
    # Display limits
    MODULE_TREE_MAX = 100
    NODE_TREE_MAX = 30
    TOP_NODES_COUNT = 20
    
    # File suffixes
    METADATA_SUFFIX = "_htp_metadata.json"
    REPORT_SUFFIX = "_full_report.txt"
    
    # Steps
    TOTAL_STEPS = 8
    
    # Messages - exactly as in baseline
    MESSAGES = {
        "model_loaded": "✅ Model loaded: {model_class} ({modules} modules, {params:.1f}M parameters)",
        "export_target": "🎯 Export target: {path}",
        "strategy": "⚙️ Strategy: HTP (Hierarchy-Preserving)",
        "hierarchy_enabled": "✅ Hierarchy attributes will be embedded in ONNX",
        "eval_mode": "✅ Model set to evaluation mode",
        "auto_inputs": "🤖 Auto-generating inputs for: {model}",
        "export_config": "✅ Created onnx export config for {model_type} with task {task}",
        "generated_tensors": "🔧 Generated {count} input tensors:",
        "hierarchy_complete": "✅ Hierarchy building completed with TracingHierarchyBuilder",
        "traced_modules": "📈 Traced {count} modules",
        "execution_steps": "🔄 Execution steps: {count}",
        "target_file": "🎯 Target file: {path}",
        "export_complete": "✅ ONNX export completed successfully",
        "tagger_created": "✅ Node tagger created successfully",
        "model_root_tag": "🏷️ Model root tag: /{class_name}",
        "operation_fallback": "🔧 Operation fallback: {status}",
        "tagging_complete": "✅ Node tagging completed successfully",
        "coverage": "📈 Coverage: {percent:.1f}%",
        "tagged_nodes": "📊 Tagged nodes: {tagged}/{total}",
        "empty_tags_ok": "✅ Empty tags: {count}",
        "empty_tags_error": "❌ Empty tags: {count}",
        "tag_injection_enabled": "🏷️ Hierarchy tag attributes: enabled",
        "tag_injection_complete": "✅ Tags injected into ONNX model successfully",
        "updated_file": "📄 Updated ONNX file: {path}",
        "metadata_created": "✅ Metadata file created successfully",
        "metadata_file": "📄 Metadata file: {path}",
        "export_success": "🎉 HTP Export completed successfully in {time:.2f}s!",
        "export_stats": "📊 Export Statistics:",
        "output_files": "📁 Output Files:",
    }


@dataclass
class ExportState:
    """Simple state container for export data."""
    # Model info
    model_name: str = ""
    model_class: str = ""
    total_modules: int = 0
    total_parameters: int = 0
    
    # Paths
    output_path: str = ""
    metadata_path: str = ""
    report_path: str = ""
    
    # Settings
    embed_hierarchy_attributes: bool = True
    
    # Data
    hierarchy: dict = None
    tagged_nodes: dict = None
    tagging_stats: dict = None
    
    # Computed
    start_time: float = 0.0
    export_time: float = 0.0
    total_nodes: int = 0
    execution_steps: int = 0
    onnx_size_mb: float = 0.0
    
    # Input/output
    input_names: list = None
    output_names: list = None
    
    # Step data
    input_generation: dict = None
    onnx_export: dict = None
    tagger_creation: dict = None
    
    def __post_init__(self):
        self.start_time = time.time()
        self.hierarchy = self.hierarchy or {}
        self.tagged_nodes = self.tagged_nodes or {}
        self.tagging_stats = self.tagging_stats or {}
        self.input_names = self.input_names or []
        self.output_names = self.output_names or []
    
    @property
    def coverage(self) -> float:
        """Coverage percentage."""
        if self.total_nodes == 0:
            return 0.0
        return len(self.tagged_nodes) / self.total_nodes * 100


class HTPExportMonitor:
    """Export monitor with Rich text styling for colored console output."""
    
    def __init__(self, output_path: str, model_name: str = "", verbose: bool = True, enable_report: bool = True):
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.verbose = verbose
        self.enable_report = enable_report
        
        # State
        self.state = ExportState(
            model_name=model_name,
            output_path=str(output_path)
        )
        
        # Setup paths
        base_name = self.output_path.stem
        output_dir = self.output_path.parent
        
        self.state.metadata_path = str(output_dir / f"{base_name}{Config.METADATA_SUFFIX}")
        if enable_report:
            self.state.report_path = str(output_dir / f"{base_name}{Config.REPORT_SUFFIX}")
        
        # Console and buffers
        self.console = Console(force_terminal=True, width=80)
        self.console_buffer = io.StringIO()
        self.report_buffer = io.StringIO() if enable_report else None
        
        # Custom string buffer to capture rich output
        self.string_buffer = io.StringIO()
        self.capture_console = Console(file=self.string_buffer, force_terminal=True, width=80)
        
        # Add header to report
        if self.report_buffer:
            self._write_report_header()
    
    def _print(self, msg: str = "", style: str = None, highlight: bool = False) -> None:
        """Print to console and buffer with optional styling."""
        if self.verbose:
            if style:
                self.console.print(msg, style=style, highlight=highlight)
            else:
                self.console.print(msg, highlight=highlight)
        
        # Also capture to buffer
        if style:
            self.capture_console.print(msg, style=style, highlight=highlight)
        else:
            self.capture_console.print(msg, highlight=highlight)
        
        # Plain text for report
        self.console_buffer.write(msg + "\\n")
    
    def _header(self, text: str) -> None:
        """Print section header with styling."""
        self._print()
        self._print(Config.MAJOR_SEP, style="bright_blue")
        self._print(text, style="bold bright_white")
        self._print(Config.MAJOR_SEP, style="bright_blue")
    
    def _write_report_header(self) -> None:
        """Write report file header."""
        self.report_buffer.write(f"\\n{Config.MAJOR_SEP}\\n")
        self.report_buffer.write("COMPLETE MODULE HIERARCHY\\n")
        self.report_buffer.write(f"{Config.MAJOR_SEP}\\n")
    
    # Step 1: Model Preparation
    def model_preparation(self, model_class: str, total_modules: int, total_parameters: int,
                         embed_hierarchy_attributes: bool = True) -> None:
        """Step 1: Model preparation."""
        self.state.model_class = model_class
        self.state.total_modules = total_modules
        self.state.total_parameters = total_parameters
        self.state.embed_hierarchy_attributes = embed_hierarchy_attributes
        
        self._header("📋 STEP 1/8: MODEL PREPARATION")
        
        self._print(Config.MESSAGES["model_loaded"].format(
            model_class=model_class,
            modules=total_modules,
            params=total_parameters/1e6
        ), style="green")
        self._print(Config.MESSAGES["export_target"].format(path=self.state.output_path), style="cyan")
        self._print(Config.MESSAGES["strategy"], style="yellow")
        
        if embed_hierarchy_attributes:
            self._print(Config.MESSAGES["hierarchy_enabled"], style="green")
        
        self._print(Config.MESSAGES["eval_mode"], style="green")
        
        # Report
        if self.report_buffer:
            self.report_buffer.write(f"\\nModel Name: {self.model_name}\\n")
            self.report_buffer.write(f"Model Class: {model_class}\\n")
            self.report_buffer.write(f"Total Modules: {total_modules}\\n")
            self.report_buffer.write(f"Total Parameters: {total_parameters:,}\\n")
    
    # Step 2: Input Generation
    def input_generation(self, model_type: str, task: str, inputs: dict) -> None:
        """Step 2: Input generation."""
        self.state.input_generation = {
            "model_type": model_type,
            "task": task,
            "inputs": inputs
        }
        self.state.input_names = list(inputs.keys())
        
        self._header("🔧 STEP 2/8: INPUT GENERATION & VALIDATION")
        
        self._print(Config.MESSAGES["auto_inputs"].format(model=self.model_name), style="blue")
        self._print(f"   • Model type: {model_type}")
        self._print(f"   • Auto-detected task: {task}")
        self._print(Config.MESSAGES["export_config"].format(model_type=model_type, task=task), style="green")
        self._print(Config.MESSAGES["generated_tensors"].format(count=len(inputs)), style="yellow")
        
        for name, spec in inputs.items():
            self._print(f"   • {name}: {spec['shape']} ({spec['dtype']})", style="dim")
        
        # Report
        if self.report_buffer:
            self.report_buffer.write("\\nINPUT GENERATION\\n")
            self.report_buffer.write("-" * 40 + "\\n")
            self.report_buffer.write(f"Model Type: {model_type}\\n")
            self.report_buffer.write(f"Task: {task}\\n")
            self.report_buffer.write("Method: auto_generated\\n")
            self.report_buffer.write("\\nGenerated Inputs:\\n")
            for name, spec in inputs.items():
                self.report_buffer.write(f"  {name}: shape={spec['shape']}, dtype={spec['dtype']}\\n")
    
    # Step 3: Hierarchy Building
    def hierarchy_building(self, hierarchy: dict, execution_steps: int) -> None:
        """Step 3: Hierarchy building."""
        self.state.hierarchy = hierarchy
        self.state.execution_steps = execution_steps
        
        self._header("🏗️ STEP 3/8: HIERARCHY BUILDING")
        
        self._print(Config.MESSAGES["hierarchy_complete"], style="green")
        self._print(Config.MESSAGES["traced_modules"].format(count=len(hierarchy)), style="cyan")
        self._print(Config.MESSAGES["execution_steps"].format(count=execution_steps), style="yellow")
        
        # Print hierarchy tree
        self._print_hierarchy_tree(hierarchy)
        
        # Report - write full hierarchy
        if self.report_buffer:
            sorted_modules = sorted(hierarchy.items(), key=lambda x: x[1].get("execution_order", 0))
            for path, info in sorted_modules:
                display_path = path if path else "[ROOT]"
                self.report_buffer.write(f"\\nModule: {display_path}\\n")
                self.report_buffer.write(f"  Class: {info.get('class_name', 'Unknown')}\\n")
                self.report_buffer.write(f"  Tag: {info.get('traced_tag', 'N/A')}\\n")
    
    # ... [Rest of the methods remain the same as clean version but with style parameters added to _print calls]
    
    def _print_hierarchy_tree(self, hierarchy: dict) -> None:
        """Print module hierarchy as tree with colors."""
        self._print("\\n🌳 Module Hierarchy:", style="bold green")
        self._print(Config.MINOR_SEP, style="dim")
        
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        self._print(root_name, style="bold cyan")
        
        # Build parent-child mapping
        parent_to_children = {}
        for path in hierarchy:
            if not path:
                continue
            
            parent_path = ""
            path_parts = path.split(".")
            
            for i in range(len(path_parts) - 1, 0, -1):
                potential_parent = ".".join(path_parts[:i])
                if potential_parent in hierarchy:
                    parent_path = potential_parent
                    break
            
            if parent_path not in parent_to_children:
                parent_to_children[parent_path] = []
            parent_to_children[parent_path].append(path)
        
        # Print tree
        def print_tree(path: str, prefix: str = "", line_count: Optional[list] = None):
            if line_count is None:
                line_count = [1]
            
            if line_count[0] >= Config.MODULE_TREE_MAX:
                return
            
            children = parent_to_children.get(path, [])
            children.sort()
            
            for i, child_path in enumerate(children):
                if line_count[0] >= Config.MODULE_TREE_MAX:
                    break
                
                is_last = i == len(children) - 1
                child_info = hierarchy.get(child_path, {})
                class_name = child_info.get("class_name", "Unknown")
                
                branch = prefix + ("└── " if is_last else "├── ")
                self._print(f"{branch}{class_name}: {child_path}", style="bright_white" if "layer." in child_path else "white")
                line_count[0] += 1
                
                child_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(child_path, child_prefix, line_count)
        
        print_tree("")
    
    def get_console_output(self) -> str:
        """Get the captured console output with ANSI codes."""
        return self.string_buffer.getvalue()
    
    # ... [Include all other methods from clean version with style parameters]
'''
    
    # Continue with remaining methods
    code += '''
    # Step 4: ONNX Export
    def onnx_export(self, opset_version: int = 17, do_constant_folding: bool = True,
                   input_names: Optional[list] = None) -> None:
        """Step 4: ONNX export."""
        self.state.onnx_export = {
            "opset_version": opset_version,
            "do_constant_folding": do_constant_folding
        }
        if input_names:
            self.state.input_names = input_names
        
        self._header("📦 STEP 4/8: ONNX EXPORT")
        
        self._print(Config.MESSAGES["target_file"].format(path=self.state.output_path), style="cyan")
        self._print("⚙️ Export config:", style="yellow")
        self._print(f"   • opset_version: {opset_version}")
        self._print(f"   • do_constant_folding: {do_constant_folding}")
        self._print("   • verbose: False")
        if input_names:
            self._print(f"   • input_names: {input_names}")
        
        self._print(Config.MESSAGES["export_complete"], style="green")
    
    # Step 5: Tagger Creation
    def tagger_creation(self, enable_operation_fallback: bool = False) -> None:
        """Step 5: Node tagger creation."""
        self.state.tagger_creation = {"enable_operation_fallback": enable_operation_fallback}
        
        self._header("🏷️ STEP 5/8: NODE TAGGER CREATION")
        
        self._print(Config.MESSAGES["tagger_created"], style="green")
        root_class = self.state.hierarchy.get("", {}).get("class_name", "Model")
        self._print(Config.MESSAGES["model_root_tag"].format(class_name=root_class), style="cyan")
        self._print(Config.MESSAGES["operation_fallback"].format(
            status="enabled" if enable_operation_fallback else "disabled"
        ), style="yellow")
    
    # Step 6: Node Tagging
    def node_tagging(self, total_nodes: int, tagged_nodes: dict, statistics: dict) -> None:
        """Step 6: Node tagging."""
        self.state.total_nodes = total_nodes
        self.state.tagged_nodes = tagged_nodes
        self.state.tagging_stats = statistics
        
        self._header("🔗 STEP 6/8: ONNX NODE TAGGING")
        
        self._print(Config.MESSAGES["tagging_complete"], style="green")
        self._print(Config.MESSAGES["coverage"].format(percent=self.state.coverage), style="cyan")
        self._print(Config.MESSAGES["tagged_nodes"].format(tagged=len(tagged_nodes), total=total_nodes), style="yellow")
        
        # Stats
        if statistics and total_nodes > 0:
            direct = statistics.get("direct_matches", 0)
            parent = statistics.get("parent_matches", 0)
            root = statistics.get("root_fallbacks", 0)
            
            self._print(f"   • Direct matches: {direct} ({direct/total_nodes*100:.1f}%)")
            self._print(f"   • Parent matches: {parent} ({parent/total_nodes*100:.1f}%)")
            self._print(f"   • Root fallbacks: {root} ({root/total_nodes*100:.1f}%)")
        
        empty = statistics.get("empty_tags", 0)
        if empty == 0:
            self._print(Config.MESSAGES["empty_tags_ok"].format(count=empty), style="green")
        else:
            self._print(Config.MESSAGES["empty_tags_error"].format(count=empty), style="red")
        
        # Top nodes by hierarchy
        self._print_top_nodes(tagged_nodes)
        
        # Complete hierarchy with nodes
        self._print_hierarchy_with_nodes()
        
        # Report
        if self.report_buffer:
            self._write_node_tagging_report()
    
    # Step 7: Tag Injection
    def tag_injection(self) -> None:
        """Step 7: Tag injection."""
        self._header("🏷️ STEP 7/8: TAG INJECTION")
        
        if self.state.embed_hierarchy_attributes:
            self._print(Config.MESSAGES["tag_injection_enabled"], style="cyan")
            self._print(Config.MESSAGES["tag_injection_complete"], style="green")
            self._print(Config.MESSAGES["updated_file"].format(path=self.state.output_path), style="yellow")
    
    # Step 8: Metadata Generation
    def metadata_generation(self) -> None:
        """Step 8: Metadata generation."""
        self._header("📄 STEP 8/8: METADATA GENERATION")
        
        # Calculate file size
        if Path(self.state.output_path).exists():
            self.state.onnx_size_mb = Path(self.state.output_path).stat().st_size / (1024 * 1024)
        
        self._print(Config.MESSAGES["metadata_created"], style="green")
        self._print(Config.MESSAGES["metadata_file"].format(path=self.state.metadata_path), style="yellow")
        
        # Write metadata
        self._write_metadata()
    
    # Final summary
    def complete(self, export_time: Optional[float] = None) -> None:
        """Final export summary."""
        if export_time is None:
            export_time = time.time() - self.state.start_time
        self.state.export_time = export_time
        
        self._header("📋 FINAL EXPORT SUMMARY")
        
        self._print(Config.MESSAGES["export_success"].format(time=export_time), style="bold green")
        self._print(Config.MESSAGES["export_stats"], style="cyan")
        self._print(f"   • Export time: {export_time:.2f}s")
        self._print(f"   • Hierarchy modules: {len(self.state.hierarchy)}")
        self._print(f"   • ONNX nodes: {self.state.total_nodes}")
        self._print(f"   • Tagged nodes: {len(self.state.tagged_nodes)}")
        self._print(f"   • Coverage: {self.state.coverage:.1f}%")
        
        empty = self.state.tagging_stats.get("empty_tags", 0)
        if empty == 0:
            self._print("   • Empty tags: 0 ✅", style="green")
        else:
            self._print(f"   • Empty tags: {empty} ❌", style="red")
        
        self._print(f"\\n{Config.MESSAGES['output_files']}", style="bold cyan")
        self._print(f"   • ONNX model: {self.state.output_path}")
        self._print(f"   • Metadata: {self.state.metadata_path}")
        if self.state.report_path:
            self._print(f"   • Report: {self.state.report_path}")
        
        # Write report
        if self.report_buffer:
            self._write_export_summary()
            self._write_report_file()
    
    # Helper methods
    def _print_top_nodes(self, tagged_nodes: dict) -> None:
        """Print top nodes by hierarchy."""
        from collections import Counter
        
        tag_counts = Counter(tagged_nodes.values())
        top_tags = tag_counts.most_common(Config.TOP_NODES_COUNT)
        
        if top_tags:
            self._print("\\n📊 Top 20 Nodes by Hierarchy:", style="bold yellow")
            self._print(Config.SHORT_SEP, style="dim")
            
            for i, (tag, count) in enumerate(top_tags[:Config.TOP_NODES_COUNT], 1):
                self._print(f"{i:2d}. {tag}: {count} nodes", style="white" if i <= 10 else "dim")
    
    def _print_hierarchy_with_nodes(self) -> None:
        """Print hierarchy with ONNX nodes."""
        from collections import defaultdict
        
        self._print("\\n🌳 Complete HF Hierarchy with ONNX Nodes:", style="bold green")
        self._print(Config.MINOR_SEP, style="dim")
        
        # Group nodes by tag and operation
        nodes_by_tag = defaultdict(lambda: defaultdict(list))
        for node_name, tag in self.state.tagged_nodes.items():
            # Extract operation type
            if '/' in node_name:
                parts = node_name.split('/')
                op_type = parts[-1].split('_')[0] if '_' in parts[-1] else parts[-1]
                if parts[-1] in ['LayerNormalization', 'Gemm', 'Tanh', 'Softmax', 'Erf']:
                    op_type = parts[-1]
            else:
                op_type = node_name.split('_')[0] if '_' in node_name else node_name
            
            nodes_by_tag[tag][op_type].append(node_name)
        
        # Print tree
        root_info = self.state.hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        self._print(f"{root_name} ({self.state.total_nodes} ONNX nodes)", style="bold cyan")
        
        # Similar tree building as before but with nodes
        parent_to_children = {}
        for path in self.state.hierarchy:
            if not path:
                continue
            
            parent_path = ""
            path_parts = path.split(".")
            
            for i in range(len(path_parts) - 1, 0, -1):
                potential_parent = ".".join(path_parts[:i])
                if potential_parent in self.state.hierarchy:
                    parent_path = potential_parent
                    break
            
            if parent_path not in parent_to_children:
                parent_to_children[parent_path] = []
            parent_to_children[parent_path].append(path)
        
        def print_module_nodes(path: str, level: int = 1, prefix: str = "", line_count: Optional[list] = None):
            if line_count is None:
                line_count = [1]
            
            if line_count[0] >= Config.NODE_TREE_MAX:
                return
            
            children = parent_to_children.get(path, [])
            children.sort()
            
            for i, child_path in enumerate(children):
                if line_count[0] >= Config.NODE_TREE_MAX:
                    break
                
                is_last = i == len(children) - 1
                child_info = self.state.hierarchy.get(child_path, {})
                class_name = child_info.get("class_name", "Unknown")
                tag = child_info.get("traced_tag", "")
                
                # Count nodes
                node_count = len([n for n, t in self.state.tagged_nodes.items() if t == tag])
                
                branch = prefix + ("└── " if is_last else "├── ")
                style = "bright_white" if node_count > 0 else "dim"
                self._print(f"{branch}{class_name}: {child_path} ({node_count} nodes)", style=style)
                line_count[0] += 1
                
                # Show operations
                if tag in nodes_by_tag and level <= 3 and node_count > 0:
                    ops = nodes_by_tag[tag]
                    sorted_ops = sorted(ops.items(), key=lambda x: len(x[1]), reverse=True)
                    
                    op_prefix = prefix + ("    " if is_last else "│   ")
                    
                    for j, (op_type, op_nodes) in enumerate(sorted_ops[:5]):
                        if line_count[0] >= Config.NODE_TREE_MAX:
                            break
                        
                        is_last_op = j == len(sorted_ops) - 1 or j == 4
                        op_branch = op_prefix + ("└── " if is_last_op else "├── ")
                        
                        if len(op_nodes) > 1:
                            self._print(f"{op_branch}{op_type} ({len(op_nodes)} ops)", style="magenta")
                        else:
                            self._print(f"{op_branch}{op_type}: {op_nodes[0]}", style="magenta")
                        line_count[0] += 1
                
                child_prefix = prefix + ("    " if is_last else "│   ")
                print_module_nodes(child_path, level + 1, child_prefix, line_count)
        
        line_counter = [1]
        print_module_nodes("", line_count=line_counter)
        
        if line_counter[0] >= Config.NODE_TREE_MAX:
            remaining = self.state.total_nodes - line_counter[0] + 1
            self._print(f"... and {remaining} more lines (truncated for console)", style="dim")
        self._print(f"(showing {min(line_counter[0]-1, Config.NODE_TREE_MAX)}/{self.state.total_nodes-1} lines)", style="dim")
    
    def _write_metadata(self) -> None:
        """Write metadata JSON file."""
        metadata = {
            "export_info": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_name": self.state.model_name,
                "model_class": self.state.model_class,
                "export_time": self.state.export_time,
                "strategy": "htp",
                "embed_hierarchy_attributes": self.state.embed_hierarchy_attributes
            },
            "model_info": {
                "total_modules": self.state.total_modules,
                "total_parameters": self.state.total_parameters,
                "execution_steps": self.state.execution_steps
            },
            "input_info": {
                "input_names": self.state.input_names,
                "output_names": self.state.output_names
            },
            "hierarchy": self.state.hierarchy,
            "nodes": self.state.tagged_nodes,
            "report": {
                "node_tagging": {
                    "statistics": {
                        "total_nodes": self.state.total_nodes,
                        "tagged_nodes": len(self.state.tagged_nodes),
                        "coverage": f"{self.state.coverage:.1f}%",
                        "direct_matches": self.state.tagging_stats.get("direct_matches", 0),
                        "parent_matches": self.state.tagging_stats.get("parent_matches", 0),
                        "operation_matches": self.state.tagging_stats.get("operation_matches", 0),
                        "root_fallbacks": self.state.tagging_stats.get("root_fallbacks", 0),
                        "empty_tags": self.state.tagging_stats.get("empty_tags", 0)
                    },
                    "coverage": {
                        "percentage": self.state.coverage,
                        "empty_tags": self.state.tagging_stats.get("empty_tags", 0)
                    }
                }
            },
            "file_info": {
                "onnx_path": self.state.output_path,
                "onnx_size_mb": self.state.onnx_size_mb,
                "metadata_path": self.state.metadata_path
            }
        }
        
        with open(self.state.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _write_node_tagging_report(self) -> None:
        """Write node tagging section to report."""
        if not self.report_buffer:
            return
        
        self.report_buffer.write(f"\\nTotal Modules: {len(self.state.hierarchy)}\\n")
        self.report_buffer.write("\\nNODE TAGGING STATISTICS\\n")
        self.report_buffer.write("-" * 40 + "\\n")
        self.report_buffer.write(f"Total ONNX Nodes: {self.state.total_nodes}\\n")
        self.report_buffer.write(f"Tagged Nodes: {len(self.state.tagged_nodes)}\\n")
        self.report_buffer.write(f"Coverage: {self.state.coverage:.1f}%\\n")
        
        # Stats
        stats = self.state.tagging_stats
        unique_tags = len(set(self.state.tagged_nodes.values()))
        root_nodes = sum(1 for tag in self.state.tagged_nodes.values() if tag.count('/') <= 1)
        scoped_nodes = len(self.state.tagged_nodes) - root_nodes
        
        self.report_buffer.write(f"  Root Nodes: {root_nodes}\\n")
        self.report_buffer.write(f"  Scoped Nodes: {scoped_nodes}\\n")
        self.report_buffer.write(f"  Unique Scopes: {unique_tags}\\n")
        self.report_buffer.write(f"  Direct Matches: {stats.get('direct_matches', 0)}\\n")
        self.report_buffer.write(f"  Parent Matches: {stats.get('parent_matches', 0)}\\n")
        self.report_buffer.write(f"  Operation Matches: {stats.get('operation_matches', 0)}\\n")
        self.report_buffer.write(f"  Root Fallbacks: {stats.get('root_fallbacks', 0)}\\n")
        self.report_buffer.write(f"  Empty Tags: {stats.get('empty_tags', 0)}\\n")
        
        # Node mappings
        self.report_buffer.write("\\nCOMPLETE NODE MAPPINGS\\n")
        self.report_buffer.write("-" * 40 + "\\n")
        
        sorted_nodes = sorted(self.state.tagged_nodes.items())
        for node_name, tag in sorted_nodes:
            self.report_buffer.write(f"{node_name} -> {tag}\\n")
    
    def _write_export_summary(self) -> None:
        """Write export summary to report."""
        if not self.report_buffer:
            return
        
        self.report_buffer.write("\\nEXPORT SUMMARY\\n")
        self.report_buffer.write("-" * 40 + "\\n")
        self.report_buffer.write(f"Total Export Time: {self.state.export_time:.2f}s\\n")
        self.report_buffer.write(f"ONNX File Size: {self.state.onnx_size_mb:.2f}MB\\n")
        self.report_buffer.write(f"Final Coverage: {self.state.coverage:.1f}%\\n")
        
        empty = self.state.tagging_stats.get("empty_tags", 0)
        if empty == 0:
            self.report_buffer.write(f"Empty Tags: {empty} ✅\\n")
        else:
            self.report_buffer.write(f"Empty Tags: {empty} ❌\\n")
        
        self.report_buffer.write("\\n" + Config.MAJOR_SEP + "\\n")
        self.report_buffer.write("Export completed successfully!\\n")
    
    def _write_report_file(self) -> None:
        """Write report to file."""
        if self.report_buffer:
            with open(self.state.report_path, 'w', encoding='utf-8') as f:
                f.write(self.report_buffer.getvalue())
'''
    
    # Save the file
    output_path = Path("/home/zhengte/modelexport_allmodels/experiments/export_monitor/export_monitor_rich.py")
    with open(output_path, 'w') as f:
        f.write(code)
    
    print(f"✅ Created rich export monitor at {output_path}")
    
    # Run ruff
    import subprocess
    result = subprocess.run(
        ["uv", "run", "ruff", "check", str(output_path), "--fix"],
        cwd=output_path.parent,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Ruff check passed!")
    else:
        print(f"⚠️ Ruff found issues:\n{result.stdout}")
    
    return output_path


def main():
    """Create iteration 6 rich monitor."""
    print("🎨 ITERATION 6 - Rich Text Styling")
    print("=" * 60)
    
    print("\n📝 Goals:")
    print("1. Add rich console for colored output")
    print("2. Capture ANSI codes in console output")
    print("3. Match baseline text styling")
    print("4. Keep plain text for report file")
    print("5. Maintain all functionality from clean version")
    
    # Create the rich monitor
    monitor_path = create_rich_export_monitor()
    
    print("\n✅ Rich export monitor created!")
    print("\nNext: Test with actual data to ensure styling matches baseline")


if __name__ == "__main__":
    main()