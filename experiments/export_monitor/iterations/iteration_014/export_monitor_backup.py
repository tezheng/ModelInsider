"""
Export monitoring system for HTP strategy with step-aware writers.
Adapted from experiments/export_monitor for production use.
"""

import io
import json
import time
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.tree import Tree


class HTPExportStep(Enum):
    """HTP export process steps - mapped to the 8-step process."""
    MODEL_PREP = "model_preparation"          # Step 1
    INPUT_GEN = "input_generation"            # Step 2
    HIERARCHY = "hierarchy_building"          # Step 3
    ONNX_EXPORT = "onnx_export"              # Step 4
    TAGGER_CREATION = "tagger_creation"       # Step 5
    NODE_TAGGING = "node_tagging"            # Step 6
    TAG_INJECTION = "tag_injection"          # Step 7
    METADATA_GEN = "metadata_generation"     # Step 8
    COMPLETE = "export_complete"             # Final summary


@dataclass
class HTPExportData:
    """Unified export data for HTP strategy."""
    # Model info
    model_name: str = ""
    model_class: str = ""
    total_modules: int = 0
    total_parameters: int = 0
    
    # Export settings
    output_path: str = ""
    strategy: str = "htp"
    embed_hierarchy_attributes: bool = True
    
    # Timing
    start_time: float = field(default_factory=time.time)
    export_time: float = 0.0
    step_times: dict[str, float] = field(default_factory=dict)
    
    # Structure data
    hierarchy: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution_steps: int = 0
    
    # Tagging data
    total_nodes: int = 0
    tagged_nodes: dict[str, str] = field(default_factory=dict)
    tagging_stats: dict[str, int] = field(default_factory=dict)
    
    # Files
    onnx_size_mb: float = 0.0
    metadata_path: str = ""
    report_path: str = ""
    
    # Step details
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Input/output info
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    
    @property
    def timestamp(self) -> str:
        """Current timestamp in ISO format."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    @property
    def coverage(self) -> float:
        """Node tagging coverage percentage."""
        if self.total_nodes == 0:
            return 0.0
        return len(self.tagged_nodes) / self.total_nodes * 100
    
    @property
    def elapsed_time(self) -> float:
        """Total elapsed time since start."""
        return time.time() - self.start_time


def step(export_step: HTPExportStep):
    """Decorator to mark step-specific handler methods."""
    def decorator(func: Callable) -> Callable:
        func._handles_step = export_step
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class StepAwareWriter(io.IOBase):
    """Base class for step-aware writers with decorator support."""
    
    def __init__(self):
        super().__init__()
        self._step_handlers: dict[HTPExportStep, Callable] = {}
        self._discover_handlers()
        
    def _discover_handlers(self) -> None:
        """Auto-discover step handler methods."""
        for name in dir(self):
            if name.startswith('_'):
                continue
            method = getattr(self, name)
            if hasattr(method, '_handles_step'):
                step_type = method._handles_step
                self._step_handlers[step_type] = method
    
    def write(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write data for a specific step."""
        # Use specific handler or fall back to default
        handler = self._step_handlers.get(export_step, self._write_default)
        return handler(export_step, data)
    
    @abstractmethod
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default handler for steps without specific handlers."""
        pass
    
    def flush(self) -> None:
        """Flush any buffered data."""
        pass
    
    def close(self) -> None:
        """Close the writer and perform cleanup."""
        from contextlib import suppress
        with suppress(Exception):
            self.flush()
        super().close()


class HTPConsoleWriter(StepAwareWriter):
    """Console output writer for HTP export with Rich formatting."""
    
    # Display constants
    MODULE_TREE_MAX_LINES = 100
    NODE_TREE_MAX_LINES = 30
    TOP_NODES_COUNT = 20
    SEPARATOR_LENGTH = 80
    
    def __init__(self, console: Console = None, verbose: bool = True):
        super().__init__()
        self.console = console or Console(width=80)
        self.verbose = verbose
        self._total_steps = 8
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: simple step completion message."""
        if self.verbose:
            print(f"✓ {export_step.value} completed")
        return 1
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 1: Model preparation."""
        if not self.verbose:
            return 0
            
        self._print_header("📋 STEP 1/8: MODEL PREPARATION")
        print(
            f"✅ Model loaded: {data.model_class} "
            f"({data.total_modules} modules, {data.total_parameters/1e6:.1f}M parameters)"
        )
        print(f"🎯 Export target: {data.output_path}")
        print("⚙️ Strategy: HTP (Hierarchy-Preserving)")
        
        if data.embed_hierarchy_attributes:
            print("✅ Hierarchy attributes will be embedded in ONNX")
        else:
            print("⚠️ Hierarchy attributes will NOT be embedded (clean ONNX)")
        
        print("✅ Model set to evaluation mode")
        return 1
    
    @step(HTPExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 2: Input generation."""
        if not self.verbose:
            return 0
            
        self._print_header("🔧 STEP 2/8: INPUT GENERATION & VALIDATION")
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            print(f"🤖 Auto-generating inputs for: {data.model_name}")
            print(f"   • Model type: {step_data.get('model_type', 'unknown')}")
            print(f"   • Task: {step_data.get('task', 'unknown')}")
            
            if "inputs" in step_data:
                print(f"✅ Created onnx export config for {step_data.get('model_type')} with task {step_data.get('task')}")
                print(f"🔧 Generated {len(step_data['inputs'])} input tensors:")
                
                for name, spec in step_data['inputs'].items():
                    shape_str = str(spec.get('shape', []))
                    dtype_str = spec.get('dtype', 'unknown')
                    print(f"   • {name}: {shape_str} ({dtype_str})")
        return 1
    
    @step(HTPExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 3: Hierarchy building."""
        if not self.verbose:
            return 0
            
        self._print_header("🏗️ STEP 3/8: HIERARCHY BUILDING")
        print("✅ Hierarchy building completed with TracingHierarchyBuilder")
        print(f"📈 Traced {len(data.hierarchy)} modules")
        
        if data.execution_steps > 0:
            print(f"🔄 Execution steps: {data.execution_steps}")
        
        if data.hierarchy:
            self._print_hierarchy_tree(data.hierarchy, max_lines=20)  # Show more lines like baseline
        
        return 1
    
    @step(HTPExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 4: ONNX export."""
        if not self.verbose:
            return 0
            
        self._print_header("📦 STEP 4/8: ONNX EXPORT")
        print(f"🎯 Target file: {data.output_path}")
        
        if "onnx_export" in data.steps:
            step_data = data.steps["onnx_export"]
            print("⚙️ Export config:")
            print(f"   • opset_version: {step_data.get('opset_version', 17)}")
            print(f"   • do_constant_folding: {step_data.get('do_constant_folding', True)}")
            print("   • verbose: False")
            
            # Add input names if available
            if data.input_names:
                print(f"   • input_names: {data.input_names}")
        
        print("✅ ONNX export completed successfully")
        return 1
    
    @step(HTPExportStep.TAGGER_CREATION)
    def write_tagger_creation(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 5: Node tagger creation."""
        if not self.verbose:
            return 0
            
        self._print_header("🏷️ STEP 5/8: NODE TAGGER CREATION")
        print("✅ Node tagger created successfully")
        
        # Get model root tag from hierarchy
        root_info = data.hierarchy.get("", {})
        model_class = root_info.get("class_name", "Model")
        print(f"🏷️ Model root tag: /{model_class}")
        
        if "tagger_creation" in data.steps:
            step_data = data.steps["tagger_creation"]
            enable_fallback = step_data.get("enable_operation_fallback", False)
            print(f"🔧 Operation fallback: {'enabled' if enable_fallback else 'disabled'}")
        
        return 1
    
    @step(HTPExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 6: Node tagging."""
        if not self.verbose:
            return 0
            
        self._print_header("🔗 STEP 6/8: ONNX NODE TAGGING")
        print("✅ Node tagging completed successfully")
        
        stats = data.tagging_stats
        total = data.total_nodes
        tagged = len(data.tagged_nodes)
        
        print(f"📈 Coverage: {data.coverage:.1f}%")
        print(f"📊 Tagged nodes: {tagged}/{total}")
        
        if stats:
            direct = stats.get("direct_matches", 0)
            parent = stats.get("parent_matches", 0)
            root = stats.get("root_fallbacks", 0)
            
            if total > 0:
                print(
                    f"   • Direct matches: {direct} "
                    f"({direct/total*100:.1f}%)"
                )
                print(
                    f"   • Parent matches: {parent} "
                    f"({parent/total*100:.1f}%)"
                )
                print(
                    f"   • Root fallbacks: {root} "
                    f"({root/total*100:.1f}%)"
                )
        
        empty = stats.get("empty_tags", 0)
        if empty == 0:
            print("✅ Empty tags: 0")
        else:
            print(f"❌ Empty tags: {empty}")
        
        # Add Top 20 Nodes by Hierarchy
        if data.tagged_nodes:
            self._print_top_nodes_by_hierarchy(data.tagged_nodes)
        
        # Add Complete HF Hierarchy with ONNX Nodes
        if data.hierarchy and data.tagged_nodes:
            self._print_hierarchy_with_nodes(data.hierarchy, data.tagged_nodes, data.total_nodes)
        
        return 1
    
    @step(HTPExportStep.TAG_INJECTION)
    def write_tag_injection(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 7: Tag injection."""
        if not self.verbose:
            return 0
            
        self._print_header("🏷️ STEP 7/8: TAG INJECTION")
        
        if data.embed_hierarchy_attributes:
            print("🏷️ Hierarchy tag attributes: enabled")
            print("✅ Tags injected into ONNX model successfully")
            print(f"📄 Updated ONNX file: {data.output_path}")
        else:
            print("⚠️ Tag injection skipped (clean ONNX mode)")
        
        return 1
    
    @step(HTPExportStep.METADATA_GEN)
    def write_metadata_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 8: Metadata generation."""
        if not self.verbose:
            return 0
            
        self._print_header("📄 STEP 8/8: METADATA GENERATION")
        print("✅ Metadata file created successfully")
        
        if data.metadata_path:
            print(f"📄 Metadata file: {data.metadata_path}")
        
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Export completion summary."""
        if not self.verbose:
            return 0
            
        self._print_header("📋 FINAL EXPORT SUMMARY")
        print(
            f"🎉 HTP Export completed successfully in {data.elapsed_time:.2f}s!"
        )
        print("📊 Export Statistics:")
        print(f"   • Export time: {data.elapsed_time:.2f}s")
        print(f"   • Hierarchy modules: {len(data.hierarchy)}")
        print(f"   • ONNX nodes: {data.total_nodes}")
        print(f"   • Tagged nodes: {len(data.tagged_nodes)}")
        print(f"   • Coverage: {data.coverage:.1f}%")
        
        empty = data.tagging_stats.get("empty_tags", 0)
        if empty == 0:
            print("   • Empty tags: 0 ✅")
        else:
            print(f"   • Empty tags: {empty} ❌")
        
        print("\n📁 Output Files:")
        
        # Print files in the baseline format
        if data.output_path:
            print(f"   • ONNX model: {data.output_path}")
        if data.metadata_path:
            print(f"   • Metadata: {data.metadata_path}")
        if data.report_path:
            print(f"   • Report: {data.report_path}")
        else:
            print("   • Report: disabled")
        
        return 1
    
    def _print_header(self, text: str) -> None:
        """Print section header."""
        print("")
        print("=" * self.SEPARATOR_LENGTH)
        print(text)
        print("=" * self.SEPARATOR_LENGTH)
    
    def _print_hierarchy_tree(self, hierarchy: dict, max_lines: int | None = None) -> None:
        """Print module hierarchy as a tree."""
        if max_lines is None:
            max_lines = self.MODULE_TREE_MAX_LINES
            
        print("\n🌳 Module Hierarchy:")
        print("-" * 60)
        
        # Build tree using Rich
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        tree = Tree(root_name)
        
        # Build a parent-child mapping first to handle components with dots
        parent_to_children = {}
        for path in hierarchy:
            if not path:  # Skip root
                continue
            
            # Find the parent by looking for the longest existing prefix
            parent_path = ""
            path_parts = path.split(".")
            
            # Try to find the longest matching parent
            for i in range(len(path_parts) - 1, 0, -1):
                potential_parent = ".".join(path_parts[:i])
                if potential_parent in hierarchy:
                    parent_path = potential_parent
                    break
            
            # Add to parent's children list
            if parent_path not in parent_to_children:
                parent_to_children[parent_path] = []
            parent_to_children[parent_path].append(path)
        
        def add_children(parent_tree: Tree, parent_path: str, level: int = 0):
            # Get children from our mapping
            children = parent_to_children.get(parent_path, [])
            
            # Sort children
            children.sort()
            
            # Add each child to tree
            for child_path in children:
                info = hierarchy.get(child_path, {})
                class_name = info.get("class_name", "Unknown")
                
                # Get display name - everything after the parent path
                if parent_path:
                    display_name = child_path[len(parent_path) + 1:]
                else:
                    display_name = child_path
                
                # Create tree node
                child_tree = parent_tree.add(f"{class_name}: {display_name}")
                
                # Recursively add this child's children
                add_children(child_tree, child_path, level + 1)
        
        # Start building from root
        add_children(tree, "")
        
        # Render the tree
        import io

        from rich.console import Console
        buffer = io.StringIO()
        temp_console = Console(file=buffer, width=80, force_terminal=True)
        temp_console.print(tree)
        
        # Get the rendered output
        lines = buffer.getvalue().splitlines()
        
        # Print all lines up to max_lines
        for i, line in enumerate(lines):
            if i >= max_lines:
                remaining = len(lines) - i
                if remaining > 0:
                    print(f"... and {remaining} more lines (truncated for console)")
                break
            print(line)
        
        # Show line count summary
        if len(lines) <= max_lines:
            print(f"(showing {len(lines)}/{len(lines)} lines)")
        else:
            print(f"(showing {max_lines}/{len(lines)} lines)")
    
    def _print_top_nodes_by_hierarchy(self, tagged_nodes: dict[str, str]) -> None:
        """Print top nodes grouped by hierarchy tag."""
        from collections import Counter
        
        # Count nodes by tag
        tag_counts = Counter(tagged_nodes.values())
        
        # Get top tags
        top_tags = tag_counts.most_common(self.TOP_NODES_COUNT)
        
        if top_tags:
            print(f"\n📊 Top {min(len(top_tags), self.TOP_NODES_COUNT)} Nodes by Hierarchy:")
            print("-" * 30)
            
            for i, (tag, count) in enumerate(top_tags[:self.TOP_NODES_COUNT], 1):
                print(f"{i:2d}. {tag}: {count} nodes")
    
    def _print_hierarchy_with_nodes(self, hierarchy: dict, tagged_nodes: dict[str, str], total_nodes: int) -> None:
        """Print hierarchy tree with ONNX node operations."""
        from collections import defaultdict
        
        print("\n🌳 Complete HF Hierarchy with ONNX Nodes:")
        print("-" * 60)
        
        # Group nodes by tag and operation type
        nodes_by_tag = defaultdict(lambda: defaultdict(list))
        for node_name, tag in tagged_nodes.items():
            # Extract operation type from node name
            op_type = node_name.split('_')[0] if '_' in node_name else node_name
            nodes_by_tag[tag][op_type].append(node_name)
        
        # Build tree with node counts
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        
        print(f"{root_name} ({total_nodes} ONNX nodes)")
        
        # Build parent-child mapping to handle components with dots
        parent_to_children = {}
        for path in hierarchy:
            if not path:  # Skip root
                continue
            
            # Find the parent by looking for the longest existing prefix
            parent_path = ""
            path_parts = path.split(".")
            
            # Try to find the longest matching parent
            for i in range(len(path_parts) - 1, 0, -1):
                potential_parent = ".".join(path_parts[:i])
                if potential_parent in hierarchy:
                    parent_path = potential_parent
                    break
            
            # Add to parent's children list
            if parent_path not in parent_to_children:
                parent_to_children[parent_path] = []
            parent_to_children[parent_path].append(path)
        
        def print_module_with_nodes(path: str, level: int = 1, line_count: list | None = None):
            if line_count is None:
                line_count = [0]
                
            # Increase limit to show more content  
            if line_count[0] >= 50:  # Increased from 30
                return
                
            if level > 4:  # Increased from 3 to show more depth
                return
                
            # Get children from our mapping
            children = parent_to_children.get(path, [])
            
            # Sort children
            children.sort()
            
            # Print each child with its nodes
            for child_path in children:
                if line_count[0] >= 50:  # Increased limit
                    break
                    
                child_info = hierarchy.get(child_path, {})
                class_name = child_info.get("class_name", "Unknown")
                
                # Get display name - everything after the parent path
                display_name = child_path[len(path) + 1:] if path else child_path
                
                tag = child_info.get("traced_tag", "")
                
                # Count nodes for this tag
                node_count = len([n for n, t in tagged_nodes.items() if t == tag])
                
                indent = "│   " * (level - 1) + "├── "
                print(f"{indent}{class_name}: {display_name} ({node_count} nodes)")
                line_count[0] += 1
                
                # Show operation breakdown for this module
                if tag in nodes_by_tag and level <= 3 and node_count > 0:
                    ops = nodes_by_tag[tag]
                    sorted_ops = sorted(ops.items(), key=lambda x: len(x[1]), reverse=True)
                    
                    for op_type, op_nodes in sorted_ops[:5]:  # Show top 5 operation types
                        if line_count[0] >= 50:  # Increased limit
                            break
                            
                        op_indent = "│   " * level + "├── "
                        count = len(op_nodes)
                        if count > 1:
                            print(f"{op_indent}{op_type} ({count} ops)")
                        else:
                            # For single ops, show the full name
                            print(f"{op_indent}{op_type}: {op_nodes[0]}")
                        line_count[0] += 1
                
                # Recurse for children
                print_module_with_nodes(child_path, level + 1, line_count)
        
        line_counter = [1]  # Start at 1 for the root line
        print_module_with_nodes("", line_count=line_counter)
        
        # Add truncation notice
        lines_shown = line_counter[0]
        if lines_shown < total_nodes:
            print(f"... and {total_nodes - lines_shown} more lines (truncated for console)")
        print(f"(showing {lines_shown}/{total_nodes} lines)")


class HTPMetadataWriter(StepAwareWriter):
    """JSON metadata writer for HTP export."""
    
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.metadata_path = f"{self.output_path}_htp_metadata.json"
        self.metadata = {
            "export_context": {},
            "model": {},
            "modules": {},
            "nodes": {},
            "outputs": {},
            "report": {"steps": {}}
        }
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: record step completion."""
        self.metadata["report"]["steps"][export_step.value] = {
            "completed": True,
            "timestamp": data.timestamp
        }
        return 1
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record model information."""
        self.metadata["export_context"] = {
            "timestamp": data.timestamp,
            "strategy": data.strategy,
            "version": "1.0",
            "exporter": "HTPExporter",
            "embed_hierarchy_attributes": data.embed_hierarchy_attributes
        }
        
        self.metadata["model"] = {
            "name_or_path": data.model_name,
            "class": data.model_class,
            "framework": "transformers",  # Could be detected dynamically
            "total_modules": data.total_modules,
            "total_parameters": data.total_parameters
        }
        
        return 1
    
    @step(HTPExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record input generation details."""
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            
            # Create tracing section if not exists
            if "tracing" not in self.metadata:
                self.metadata["tracing"] = {}
            
            self.metadata["tracing"].update({
                "model_type": step_data.get("model_type", "unknown"),
                "task": step_data.get("task", "unknown"),
                "inputs": step_data.get("inputs", {}),
                "outputs": data.output_names
            })
        
        return 1
    
    @step(HTPExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record hierarchy data."""
        self.metadata["modules"] = data.hierarchy.copy()
        
        # Add to tracing section
        if "tracing" not in self.metadata:
            self.metadata["tracing"] = {}
            
        self.metadata["tracing"].update({
            "builder": "TracingHierarchyBuilder",
            "modules_traced": len(data.hierarchy),
            "execution_steps": data.execution_steps
        })
        
        self.metadata["report"]["steps"]["hierarchy_building"] = {
            "modules_traced": len(data.hierarchy),
            "execution_steps": data.execution_steps,
            "timestamp": data.timestamp
        }
        return 1
    
    @step(HTPExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record tagging results."""
        self.metadata["nodes"] = data.tagged_nodes.copy()
        
        stats = data.tagging_stats
        self.metadata["report"]["node_tagging"] = {
            "statistics": {
                "total_nodes": data.total_nodes,
                "root_nodes": stats.get("root_nodes", 0),
                "scoped_nodes": stats.get("scoped_nodes", 0),
                "unique_scopes": stats.get("unique_scopes", 0),
                "direct_matches": stats.get("direct_matches", 0),
                "parent_matches": stats.get("parent_matches", 0),
                "operation_matches": stats.get("operation_matches", 0),
                "root_fallbacks": stats.get("root_fallbacks", 0)
            },
            "coverage": {
                "total_onnx_nodes": data.total_nodes,
                "tagged_nodes": len(data.tagged_nodes),
                "coverage_percentage": data.coverage,
                "empty_tags": stats.get("empty_tags", 0)
            }
        }
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Finalize metadata."""
        self.metadata["export_context"]["export_time_seconds"] = round(data.elapsed_time, 2)
        
        # Build outputs section
        self.metadata["outputs"] = {}
        
        if data.output_path:
            self.metadata["outputs"]["onnx_model"] = {
                "path": Path(data.output_path).name,
                "size_mb": data.onnx_size_mb
            }
        
        if self.metadata_path:
            self.metadata["outputs"]["metadata"] = {
                "path": Path(self.metadata_path).name
            }
        
        if data.report_path:
            self.metadata["outputs"]["report"] = {
                "path": Path(data.report_path).name
            }
        
        return 1
    
    def flush(self) -> None:
        """Write metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)


class HTPReportWriter(StepAwareWriter):
    """Full text report writer for HTP export."""
    
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.report_path = f"{self.output_path}_full_report.txt"
        self.buffer = io.StringIO()
        self._write_header()
    
    def _write_header(self) -> None:
        """Write report header."""
        self.buffer.write("=" * 80 + "\n")
        self.buffer.write("HTP EXPORT FULL REPORT\n")
        self.buffer.write("=" * 80 + "\n")
        self.buffer.write(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n")
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: record step with timestamp."""
        self.buffer.write(f"\n[{data.timestamp}] {export_step.value}: Completed\n")
        return 1
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write model details."""
        self.buffer.write("\nMODEL INFORMATION\n")
        self.buffer.write("-" * 40 + "\n")
        self.buffer.write(f"Model Name: {data.model_name}\n")
        self.buffer.write(f"Model Class: {data.model_class}\n")
        self.buffer.write(f"Total Modules: {data.total_modules}\n")
        self.buffer.write(f"Total Parameters: {data.total_parameters:,}\n")
        self.buffer.write(f"Export Strategy: {data.strategy.upper()}\n")
        self.buffer.write(f"Output Path: {data.output_path}\n")
        self.buffer.write(f"Embed Hierarchy: {data.embed_hierarchy_attributes}\n")
        return 1
    
    @step(HTPExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write input generation details."""
        self.buffer.write("\nINPUT GENERATION\n")
        self.buffer.write("-" * 40 + "\n")
        
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            self.buffer.write(f"Model Type: {step_data.get('model_type', 'unknown')}\n")
            self.buffer.write(f"Task: {step_data.get('task', 'unknown')}\n")
            self.buffer.write(f"Method: {step_data.get('method', 'auto')}\n")
            
            if "inputs" in step_data:
                self.buffer.write("\nGenerated Inputs:\n")
                for name, spec in step_data["inputs"].items():
                    self.buffer.write(f"  {name}: shape={spec.get('shape')}, dtype={spec.get('dtype')}\n")
            
            if data.output_names:
                self.buffer.write("\nExpected Outputs:\n")
                for name in data.output_names:
                    self.buffer.write(f"  - {name}\n")
        
        return 1
    
    @step(HTPExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write complete hierarchy."""
        self.buffer.write("\nCOMPLETE MODULE HIERARCHY\n")
        self.buffer.write("-" * 40 + "\n")
        
        # Sort paths for consistent output
        sorted_paths = sorted(data.hierarchy.keys(), key=lambda x: (x.count('.'), x))
        
        for path in sorted_paths:
            info = data.hierarchy[path]
            module_path = path or "[ROOT]"
            class_name = info.get("class_name", "Unknown")
            tag = info.get("traced_tag", "")
            
            self.buffer.write(f"\nModule: {module_path}\n")
            self.buffer.write(f"  Class: {class_name}\n")
            self.buffer.write(f"  Tag: {tag}\n")
            
            # Include additional info if available
            if "module_type" in info:
                self.buffer.write(f"  Type: {info['module_type']}\n")
            if "execution_order" in info:
                self.buffer.write(f"  Execution Order: {info['execution_order']}\n")
        
        self.buffer.write(f"\nTotal Modules: {len(data.hierarchy)}\n")
        return 1
    
    @step(HTPExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write tagging statistics and full mappings."""
        stats = data.tagging_stats
        
        self.buffer.write("\nNODE TAGGING STATISTICS\n")
        self.buffer.write("-" * 40 + "\n")
        self.buffer.write(f"Total ONNX Nodes: {data.total_nodes}\n")
        self.buffer.write(f"Tagged Nodes: {len(data.tagged_nodes)}\n")
        self.buffer.write(f"Coverage: {data.coverage:.1f}%\n")
        
        if stats:
            self.buffer.write(f"  Root Nodes: {stats.get('root_nodes', 0)}\n")
            self.buffer.write(f"  Scoped Nodes: {stats.get('scoped_nodes', 0)}\n")
            self.buffer.write(f"  Unique Scopes: {stats.get('unique_scopes', 0)}\n")
            self.buffer.write(f"  Direct Matches: {stats.get('direct_matches', 0)}\n")
            self.buffer.write(f"  Parent Matches: {stats.get('parent_matches', 0)}\n")
            self.buffer.write(f"  Operation Matches: {stats.get('operation_matches', 0)}\n")
            self.buffer.write(f"  Root Fallbacks: {stats.get('root_fallbacks', 0)}\n")
            self.buffer.write(f"  Empty Tags: {stats.get('empty_tags', 0)}\n")
        
        # Write complete node mappings
        self.buffer.write("\nCOMPLETE NODE MAPPINGS\n")
        self.buffer.write("-" * 40 + "\n")
        
        # Sort by node name for consistent output
        sorted_nodes = sorted(data.tagged_nodes.items())
        
        for node_name, tag in sorted_nodes:
            self.buffer.write(f"{node_name} -> {tag}\n")
        
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write final summary."""
        self.buffer.write("\nEXPORT SUMMARY\n")
        self.buffer.write("-" * 40 + "\n")
        self.buffer.write(f"Total Export Time: {data.elapsed_time:.2f}s\n")
        self.buffer.write(f"ONNX File Size: {data.onnx_size_mb:.2f}MB\n")
        self.buffer.write(f"Final Coverage: {data.coverage:.1f}%\n")
        
        empty = data.tagging_stats.get("empty_tags", 0)
        if empty == 0:
            self.buffer.write("Empty Tags: 0 ✅\n")
        else:
            self.buffer.write(f"Empty Tags: {empty} ❌\n")
        
        self.buffer.write("\n" + "=" * 80 + "\n")
        self.buffer.write("Export completed successfully!\n")
        return 1
    
    def flush(self) -> None:
        """Write buffer to file."""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(self.buffer.getvalue())
    
    def close(self) -> None:
        """Close buffer and write file."""
        if not self.buffer.closed:
            self.flush()
            self.buffer.close()


class HTPExportMonitor:
    """Central monitor that coordinates HTP export data and output writers."""
    
    def __init__(self, output_path: str, verbose: bool = True, enable_report: bool = True, 
                 console: Console = None, embed_hierarchy: bool = True):
        self.data = HTPExportData(
            output_path=output_path,
            embed_hierarchy_attributes=embed_hierarchy
        )
        self.writers: list[StepAwareWriter] = []
        
        # Always include metadata writer
        self.writers.append(HTPMetadataWriter(output_path))
        self.data.metadata_path = f"{Path(output_path).with_suffix('').as_posix()}_htp_metadata.json"
        
        # Console writer if verbose
        if verbose:
            self.writers.append(HTPConsoleWriter(console=console, verbose=True))
            
        # Report writer if enabled
        if enable_report:
            self.writers.append(HTPReportWriter(output_path))
            self.data.report_path = f"{Path(output_path).with_suffix('').as_posix()}_full_report.txt"
    
    def update(self, step: HTPExportStep, **kwargs) -> None:
        """Update data and notify all writers."""
        # Update shared data
        for key, value in kwargs.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)
            else:
                # Store in steps for step-specific data
                if step.value not in self.data.steps:
                    self.data.steps[step.value] = {}
                self.data.steps[step.value][key] = value
        
        # Record step timing
        self.data.step_times[step.value] = time.time() - self.data.start_time
        
        # Notify all writers
        for writer in self.writers:
            try:
                writer.write(step, self.data)
            except Exception as e:
                print(f"Error in {writer.__class__.__name__}: {e}")
    
    def finalize(self) -> None:
        """Finalize all writers."""
        self.data.export_time = self.data.elapsed_time
        
        # Notify completion
        self.update(HTPExportStep.COMPLETE)
        
        # Close all writers
        for writer in self.writers:
            writer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto finalize."""
        if exc_type is None:
            self.finalize()
        else:
            # Even on error, try to close writers
            from contextlib import suppress
            for writer in self.writers:
                with suppress(Exception):
                    writer.close()