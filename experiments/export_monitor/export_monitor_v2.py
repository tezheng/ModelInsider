"""
Export monitoring system for HTP strategy with step-aware writers.
Version 2: Refactored with config class and rich console.
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


class HTPExportConfig:
    """Configuration for HTP Export Monitor - no hardcoded values."""
    
    # Display formatting
    SEPARATOR_WIDTH = 80
    CONSOLE_WIDTH = 80
    WIDE_CONSOLE_WIDTH = 120
    
    # Tree display limits
    MODULE_TREE_MAX_LINES = 100  # Full module hierarchy
    NODE_TREE_MAX_LINES = 50     # ONNX nodes with operations (increased from 30)
    TOP_NODES_DISPLAY_COUNT = 20  # Top N nodes by hierarchy
    MAX_OPERATION_TYPES = 5       # Operations to show per module
    
    # Section separators
    MAJOR_SEPARATOR = "=" * SEPARATOR_WIDTH
    MINOR_SEPARATOR = "-" * 60
    SHORT_SEPARATOR = "-" * 30
    
    # Depth limits
    MAX_TREE_DEPTH = 4            # Maximum nesting depth for trees
    NODE_DETAIL_MAX_DEPTH = 3     # Show operation details up to this depth
    
    # File naming
    METADATA_SUFFIX = "_htp_metadata.json"
    REPORT_SUFFIX = "_htp_export_report.txt"
    FULL_REPORT_SUFFIX = "_full_report.txt"
    
    # Export settings
    DEFAULT_OPSET_VERSION = 17
    DEFAULT_CONSTANT_FOLDING = True
    DEFAULT_ONNX_VERBOSE = False
    
    # Step display
    TOTAL_EXPORT_STEPS = 8
    
    # Formatting templates
    STEP_HEADER_TEMPLATE = "📋 STEP {current}/{total}: {title}"
    NODE_COUNT_TEMPLATE = "{class_name}: {name} ({count} nodes)"
    OPERATION_TEMPLATE = "{op_type} ({count} ops)"
    OPERATION_SINGLE_TEMPLATE = "{op_type}: {node_name}"
    TRUNCATION_MESSAGE = "... and {count} more lines (truncated for console)"
    LINE_COUNT_MESSAGE = "(showing {shown}/{total} lines)"
    
    # Console messages
    MESSAGES = {
        "model_loaded": "✅ Model loaded: {model_class} ({modules} modules, {params:.1f}M parameters)",
        "export_target": "🎯 Export target: {path}",
        "strategy": "⚙️ Strategy: HTP (Hierarchy-Preserving)",
        "hierarchy_enabled": "✅ Hierarchy attributes will be embedded in ONNX",
        "hierarchy_disabled": "⚠️ Hierarchy attributes will NOT be embedded (clean ONNX)",
        "eval_mode": "✅ Model set to evaluation mode",
        "auto_inputs": "🤖 Auto-generating inputs for: {model}",
        "export_config": "✅ Created onnx export config for {model_type} with task {task}",
        "generated_tensors": "🔧 Generated {count} input tensors:",
        "hierarchy_complete": "✅ Hierarchy building completed with TracingHierarchyBuilder",
        "traced_modules": "📈 Traced {count} modules",
        "execution_steps": "🔄 Execution steps: {count}",
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
        "tag_injection_skipped": "⚠️ Tag injection skipped (clean ONNX mode)",
        "updated_file": "📄 Updated ONNX file: {path}",
        "metadata_created": "✅ Metadata file created successfully",
        "metadata_file": "📄 Metadata file: {path}",
        "export_success": "🎉 HTP Export completed successfully in {time:.2f}s!",
        "export_stats": "📊 Export Statistics:",
        "output_files": "📁 Output Files:",
        "report_disabled": "   • Report: disabled"
    }
    
    # Step titles
    STEP_TITLES = {
        "MODEL_PREP": "MODEL PREPARATION",
        "INPUT_GEN": "INPUT GENERATION & VALIDATION", 
        "HIERARCHY": "HIERARCHY BUILDING",
        "ONNX_EXPORT": "ONNX EXPORT",
        "TAGGER_CREATION": "NODE TAGGER CREATION",
        "NODE_TAGGING": "ONNX NODE TAGGING",
        "TAG_INJECTION": "TAG INJECTION",
        "METADATA_GEN": "METADATA GENERATION",
        "COMPLETE": "FINAL EXPORT SUMMARY"
    }
    
    # Step icons (matching baseline)
    STEP_ICONS = {
        "MODEL_PREP": "📋",
        "INPUT_GEN": "🔧",
        "HIERARCHY": "🏗️",
        "ONNX_EXPORT": "📦",
        "TAGGER_CREATION": "🏷️",
        "NODE_TAGGING": "🔗",
        "TAG_INJECTION": "🏷️",
        "METADATA_GEN": "📄",
        "COMPLETE": "📋"
    }


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
    
    def __init__(self, console: Console = None, verbose: bool = True):
        super().__init__()
        self.console = console or Console(width=HTPExportConfig.CONSOLE_WIDTH)
        self.verbose = verbose
        self._total_steps = HTPExportConfig.TOTAL_EXPORT_STEPS
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: simple step completion message."""
        if self.verbose:
            self._print(f"✓ {export_step.value} completed")
        return 1
    
    def _print(self, message: str, style: str = None) -> None:
        """Print using rich console instead of print()."""
        self.console.print(message, style=style or "")
    
    def _print_header(self, text: str) -> None:
        """Print section header."""
        self._print("")
        self._print(HTPExportConfig.MAJOR_SEPARATOR)
        self._print(text)
        self._print(HTPExportConfig.MAJOR_SEPARATOR)
    
    def _print_minor_header(self, text: str) -> None:
        """Print minor section header."""
        self._print(f"\n{HTPExportConfig.MESSAGES.get('tree', '🌳')} {text}:")
        self._print(HTPExportConfig.MINOR_SEPARATOR)
    
    def _format_step_header(self, step_num: int, title: str, step_key: str = None) -> str:
        """Format step header using template."""
        # Get icon for this step
        icon = "📋"  # Default
        if step_key and step_key in HTPExportConfig.STEP_ICONS:
            icon = HTPExportConfig.STEP_ICONS[step_key]
        
        return f"{icon} STEP {step_num}/{self._total_steps}: {title}"
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 1: Model preparation."""
        if not self.verbose:
            return 0
            
        self._print_header(self._format_step_header(1, HTPExportConfig.STEP_TITLES["MODEL_PREP"], "MODEL_PREP"))
        
        self._print(
            HTPExportConfig.MESSAGES["model_loaded"].format(
                model_class=data.model_class,
                modules=data.total_modules,
                params=data.total_parameters/1e6
            )
        )
        self._print(HTPExportConfig.MESSAGES["export_target"].format(path=data.output_path))
        self._print(HTPExportConfig.MESSAGES["strategy"])
        
        if data.embed_hierarchy_attributes:
            self._print(HTPExportConfig.MESSAGES["hierarchy_enabled"])
        else:
            self._print(HTPExportConfig.MESSAGES["hierarchy_disabled"])
        
        self._print(HTPExportConfig.MESSAGES["eval_mode"])
        return 1
    
    @step(HTPExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 2: Input generation."""
        if not self.verbose:
            return 0
            
        self._print_header(self._format_step_header(2, HTPExportConfig.STEP_TITLES["INPUT_GEN"], "INPUT_GEN"))
        
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            self._print(HTPExportConfig.MESSAGES["auto_inputs"].format(model=data.model_name))
            self._print(f"   • Model type: {step_data.get('model_type', 'unknown')}")
            self._print(f"   • Task: {step_data.get('task', 'unknown')}")
            
            if "inputs" in step_data:
                self._print(
                    HTPExportConfig.MESSAGES["export_config"].format(
                        model_type=step_data.get('model_type'),
                        task=step_data.get('task')
                    )
                )
                self._print(
                    HTPExportConfig.MESSAGES["generated_tensors"].format(
                        count=len(step_data['inputs'])
                    )
                )
                
                for name, spec in step_data['inputs'].items():
                    shape_str = str(spec.get('shape', []))
                    dtype_str = spec.get('dtype', 'unknown')
                    self._print(f"   • {name}: {shape_str} ({dtype_str})")
        return 1
    
    @step(HTPExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 3: Hierarchy building."""
        if not self.verbose:
            return 0
            
        self._print_header(self._format_step_header(3, HTPExportConfig.STEP_TITLES["HIERARCHY"], "HIERARCHY"))
        self._print(HTPExportConfig.MESSAGES["hierarchy_complete"])
        self._print(HTPExportConfig.MESSAGES["traced_modules"].format(count=len(data.hierarchy)))
        
        if data.execution_steps > 0:
            self._print(HTPExportConfig.MESSAGES["execution_steps"].format(count=data.execution_steps))
        
        if data.hierarchy:
            self._print_hierarchy_tree(data.hierarchy, max_lines=HTPExportConfig.MODULE_TREE_MAX_LINES)
        
        return 1
    
    @step(HTPExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 4: ONNX export."""
        if not self.verbose:
            return 0
            
        self._print_header(self._format_step_header(4, HTPExportConfig.STEP_TITLES["ONNX_EXPORT"], "ONNX_EXPORT"))
        self._print(HTPExportConfig.MESSAGES["export_target"].format(path=data.output_path))
        
        if "onnx_export" in data.steps:
            step_data = data.steps["onnx_export"]
            self._print("⚙️ Export config:")
            self._print(f"   • opset_version: {step_data.get('opset_version', HTPExportConfig.DEFAULT_OPSET_VERSION)}")
            self._print(f"   • do_constant_folding: {step_data.get('do_constant_folding', HTPExportConfig.DEFAULT_CONSTANT_FOLDING)}")
            self._print(f"   • verbose: {HTPExportConfig.DEFAULT_ONNX_VERBOSE}")
            
            # Add input names if available
            if data.input_names:
                self._print(f"   • input_names: {data.input_names}")
        
        self._print(HTPExportConfig.MESSAGES["export_complete"])
        return 1
    
    @step(HTPExportStep.TAGGER_CREATION)
    def write_tagger_creation(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 5: Node tagger creation."""
        if not self.verbose:
            return 0
            
        self._print_header(self._format_step_header(5, HTPExportConfig.STEP_TITLES["TAGGER_CREATION"], "TAGGER_CREATION"))
        self._print(HTPExportConfig.MESSAGES["tagger_created"])
        
        # Get model root tag from hierarchy
        root_info = data.hierarchy.get("", {})
        model_class = root_info.get("class_name", "Model")
        self._print(HTPExportConfig.MESSAGES["model_root_tag"].format(class_name=model_class))
        
        if "tagger_creation" in data.steps:
            step_data = data.steps["tagger_creation"]
            enable_fallback = step_data.get("enable_operation_fallback", False)
            self._print(
                HTPExportConfig.MESSAGES["operation_fallback"].format(
                    status="enabled" if enable_fallback else "disabled"
                )
            )
        
        return 1
    
    @step(HTPExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 6: Node tagging."""
        if not self.verbose:
            return 0
            
        self._print_header(self._format_step_header(6, HTPExportConfig.STEP_TITLES["NODE_TAGGING"], "NODE_TAGGING"))
        self._print(HTPExportConfig.MESSAGES["tagging_complete"])
        
        stats = data.tagging_stats
        total = data.total_nodes
        tagged = len(data.tagged_nodes)
        
        self._print(HTPExportConfig.MESSAGES["coverage"].format(percent=data.coverage))
        self._print(HTPExportConfig.MESSAGES["tagged_nodes"].format(tagged=tagged, total=total))
        
        if stats:
            direct = stats.get("direct_matches", 0)
            parent = stats.get("parent_matches", 0)
            root = stats.get("root_fallbacks", 0)
            
            if total > 0:
                self._print(f"   • Direct matches: {direct} ({direct/total*100:.1f}%)")
                self._print(f"   • Parent matches: {parent} ({parent/total*100:.1f}%)")
                self._print(f"   • Root fallbacks: {root} ({root/total*100:.1f}%)")
        
        empty = stats.get("empty_tags", 0)
        if empty == 0:
            self._print(HTPExportConfig.MESSAGES["empty_tags_ok"].format(count=empty))
        else:
            self._print(HTPExportConfig.MESSAGES["empty_tags_error"].format(count=empty))
        
        # Add Top N Nodes by Hierarchy
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
            
        self._print_header(self._format_step_header(7, HTPExportConfig.STEP_TITLES["TAG_INJECTION"], "TAG_INJECTION"))
        
        if data.embed_hierarchy_attributes:
            self._print(HTPExportConfig.MESSAGES["tag_injection_enabled"])
            self._print(HTPExportConfig.MESSAGES["tag_injection_complete"])
            self._print(HTPExportConfig.MESSAGES["updated_file"].format(path=data.output_path))
        else:
            self._print(HTPExportConfig.MESSAGES["tag_injection_skipped"])
        
        return 1
    
    @step(HTPExportStep.METADATA_GEN)
    def write_metadata_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 8: Metadata generation."""
        if not self.verbose:
            return 0
            
        self._print_header(self._format_step_header(8, HTPExportConfig.STEP_TITLES["METADATA_GEN"], "METADATA_GEN"))
        self._print(HTPExportConfig.MESSAGES["metadata_created"])
        
        if data.metadata_path:
            self._print(HTPExportConfig.MESSAGES["metadata_file"].format(path=data.metadata_path))
        
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Export completion summary."""
        if not self.verbose:
            return 0
            
        self._print_header("📋 " + HTPExportConfig.STEP_TITLES["COMPLETE"])
        self._print(HTPExportConfig.MESSAGES["export_success"].format(time=data.elapsed_time))
        self._print(HTPExportConfig.MESSAGES["export_stats"])
        self._print(f"   • Export time: {data.elapsed_time:.2f}s")
        self._print(f"   • Hierarchy modules: {len(data.hierarchy)}")
        self._print(f"   • ONNX nodes: {data.total_nodes}")
        self._print(f"   • Tagged nodes: {len(data.tagged_nodes)}")
        self._print(f"   • Coverage: {data.coverage:.1f}%")
        
        empty = data.tagging_stats.get("empty_tags", 0)
        if empty == 0:
            self._print(f"   • Empty tags: {empty} ✅")
        else:
            self._print(f"   • Empty tags: {empty} ❌")
        
        self._print(f"\n{HTPExportConfig.MESSAGES['output_files']}")
        
        # Print files in the baseline format
        if data.output_path:
            self._print(f"   • ONNX model: {data.output_path}")
        if data.metadata_path:
            self._print(f"   • Metadata: {data.metadata_path}")
        if data.report_path:
            self._print(f"   • Report: {data.report_path}")
        else:
            self._print(HTPExportConfig.MESSAGES["report_disabled"])
        
        return 1
    
    def _print_hierarchy_tree(self, hierarchy: dict, max_lines: int | None = None) -> None:
        """Print module hierarchy as a tree."""
        if max_lines is None:
            max_lines = HTPExportConfig.MODULE_TREE_MAX_LINES
            
        self._print_minor_header("Module Hierarchy")
        
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
                
                # Use full path as display name to match baseline
                display_name = child_path
                
                # Create tree node
                child_tree = parent_tree.add(
                    HTPExportConfig.NODE_COUNT_TEMPLATE.format(
                        class_name=class_name,
                        name=display_name,
                        count=0
                    ).replace(" (0 nodes)", "")  # Remove node count for hierarchy tree
                )
                
                # Recursively add this child's children
                add_children(child_tree, child_path, level + 1)
        
        # Start building from root
        add_children(tree, "")
        
        # Render the tree
        buffer = io.StringIO()
        temp_console = Console(file=buffer, width=HTPExportConfig.CONSOLE_WIDTH, force_terminal=True)
        temp_console.print(tree)
        
        # Get the rendered output
        lines = buffer.getvalue().splitlines()
        
        # Print all lines up to max_lines
        for i, line in enumerate(lines):
            if i >= max_lines:
                remaining = len(lines) - i
                if remaining > 0:
                    self._print(HTPExportConfig.TRUNCATION_MESSAGE.format(count=remaining))
                break
            self._print(line)
        
        # Show line count summary
        if len(lines) <= max_lines:
            self._print(HTPExportConfig.LINE_COUNT_MESSAGE.format(shown=len(lines), total=len(lines)))
        else:
            self._print(HTPExportConfig.LINE_COUNT_MESSAGE.format(shown=max_lines, total=len(lines)))
    
    def _print_top_nodes_by_hierarchy(self, tagged_nodes: dict[str, str]) -> None:
        """Print top nodes grouped by hierarchy tag."""
        from collections import Counter
        
        # Count nodes by tag
        tag_counts = Counter(tagged_nodes.values())
        
        # Get top tags
        top_tags = tag_counts.most_common(HTPExportConfig.TOP_NODES_DISPLAY_COUNT)
        
        if top_tags:
            self._print(
                f"\n📊 Top 20 Nodes by Hierarchy:"
            )
            self._print(HTPExportConfig.SHORT_SEPARATOR)
            
            for i, (tag, count) in enumerate(top_tags[:HTPExportConfig.TOP_NODES_DISPLAY_COUNT], 1):
                self._print(f"{i:2d}. {tag}: {count} nodes")
    
    def _print_hierarchy_with_nodes(self, hierarchy: dict, tagged_nodes: dict[str, str], total_nodes: int) -> None:
        """Print hierarchy tree with ONNX node operations."""
        from collections import defaultdict
        
        self._print_minor_header("Complete HF Hierarchy with ONNX Nodes")
        
        # Group nodes by tag and operation type
        nodes_by_tag = defaultdict(lambda: defaultdict(list))
        for node_name, tag in tagged_nodes.items():
            # Extract operation type from node name
            op_type = node_name.split('_')[0] if '_' in node_name else node_name
            nodes_by_tag[tag][op_type].append(node_name)
        
        # Build tree with node counts
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        
        self._print(f"{root_name} ({total_nodes} ONNX nodes)")
        
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
            if line_count[0] >= HTPExportConfig.NODE_TREE_MAX_LINES:
                return
                
            if level > HTPExportConfig.MAX_TREE_DEPTH:
                return
                
            # Get children from our mapping
            children = parent_to_children.get(path, [])
            
            # Sort children
            children.sort()
            
            # Print each child with its nodes
            for child_path in children:
                if line_count[0] >= HTPExportConfig.NODE_TREE_MAX_LINES:
                    break
                    
                child_info = hierarchy.get(child_path, {})
                class_name = child_info.get("class_name", "Unknown")
                
                # Use full path as display name to match baseline  
                display_name = child_path
                
                tag = child_info.get("traced_tag", "")
                
                # Count nodes for this tag
                node_count = len([n for n, t in tagged_nodes.items() if t == tag])
                
                indent = "│   " * (level - 1) + "├── "
                self._print(
                    indent + HTPExportConfig.NODE_COUNT_TEMPLATE.format(
                        class_name=class_name,
                        name=display_name,
                        count=node_count
                    )
                )
                line_count[0] += 1
                
                # Show operation breakdown for this module
                if tag in nodes_by_tag and level <= HTPExportConfig.NODE_DETAIL_MAX_DEPTH and node_count > 0:
                    ops = nodes_by_tag[tag]
                    sorted_ops = sorted(ops.items(), key=lambda x: len(x[1]), reverse=True)
                    
                    for op_type, op_nodes in sorted_ops[:HTPExportConfig.MAX_OPERATION_TYPES]:
                        if line_count[0] >= HTPExportConfig.NODE_TREE_MAX_LINES:
                            break
                            
                        op_indent = "│   " * level + "├── "
                        count = len(op_nodes)
                        if count > 1:
                            self._print(
                                op_indent + HTPExportConfig.OPERATION_TEMPLATE.format(
                                    op_type=op_type,
                                    count=count
                                )
                            )
                        else:
                            # For single ops, show the full name
                            self._print(
                                op_indent + HTPExportConfig.OPERATION_SINGLE_TEMPLATE.format(
                                    op_type=op_type,
                                    node_name=op_nodes[0]
                                )
                            )
                        line_count[0] += 1
                
                # Recurse for children
                print_module_with_nodes(child_path, level + 1, line_count)
        
        line_counter = [1]  # Start at 1 for the root line
        print_module_with_nodes("", line_count=line_counter)
        
        # Add truncation notice
        lines_shown = line_counter[0]
        if lines_shown < total_nodes:
            self._print(HTPExportConfig.TRUNCATION_MESSAGE.format(count=total_nodes - lines_shown))
        self._print(HTPExportConfig.LINE_COUNT_MESSAGE.format(shown=lines_shown, total=total_nodes))


class HTPMetadataWriter(StepAwareWriter):
    """Metadata writer for HTP export - writes to JSON."""
    
    def __init__(self, metadata_path: Path):
        super().__init__()
        self.metadata_path = metadata_path
        self.metadata = {}
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: store step data in metadata."""
        step_key = export_step.value
        if step_key not in self.metadata:
            self.metadata[step_key] = {}
        
        self.metadata[step_key]["timestamp"] = data.timestamp
        self.metadata[step_key]["elapsed_time"] = data.elapsed_time
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Generate final metadata file."""
        # Build complete metadata structure
        metadata = {
            "export_info": {
                "timestamp": data.timestamp,
                "model_name": data.model_name,
                "model_class": data.model_class,
                "export_time": data.export_time,
                "strategy": data.strategy,
                "embed_hierarchy_attributes": data.embed_hierarchy_attributes
            },
            "model_info": {
                "total_modules": data.total_modules,
                "total_parameters": data.total_parameters,
                "execution_steps": data.execution_steps
            },
            "input_info": {
                "input_names": data.input_names,
                "output_names": data.output_names
            },
            "hierarchy": data.hierarchy,
            "nodes": data.tagged_nodes,  # Renamed from tagged_nodes to nodes
            "report": {
                "node_tagging": {
                    "statistics": {
                        "total_nodes": data.total_nodes,
                        "tagged_nodes": len(data.tagged_nodes),
                        "coverage": f"{data.coverage:.1f}%",
                        **data.tagging_stats
                    },
                    "coverage": {
                        "percentage": data.coverage,
                        "empty_tags": data.tagging_stats.get("empty_tags", 0)
                    }
                }
            },
            "file_info": {
                "onnx_path": data.output_path,
                "onnx_size_mb": data.onnx_size_mb,
                "metadata_path": str(self.metadata_path)
            }
        }
        
        # Write to file
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return 1
    
    def flush(self) -> None:
        """Flush is handled in write_complete."""
        pass


class HTPReportWriter(StepAwareWriter):
    """Report writer for HTP export - writes human-readable text report."""
    
    def __init__(self, report_path: Path):
        super().__init__()
        self.report_path = report_path
        self.buffer = io.StringIO()
        self._write_header()
    
    def _write_header(self) -> None:
        """Write report header."""
        self.buffer.write(HTPExportConfig.MAJOR_SEPARATOR + "\n")
        self.buffer.write("HTP EXPORT FULL REPORT\n")
        self.buffer.write(HTPExportConfig.MAJOR_SEPARATOR + "\n")
        self.buffer.write(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: log step completion."""
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.buffer.write(f"\n[{timestamp}] {export_step.value}: Completed\n")
        return 1
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write model information section."""
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
        """Write input generation section."""
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            self.buffer.write("\nINPUT GENERATION\n")
            self.buffer.write("-" * 40 + "\n")
            self.buffer.write(f"Model Type: {step_data.get('model_type', 'unknown')}\n")
            self.buffer.write(f"Task: {step_data.get('task', 'unknown')}\n")
            self.buffer.write(f"Method: {step_data.get('method', 'auto_generated')}\n")
            
            if "inputs" in step_data:
                self.buffer.write("\nGenerated Inputs:\n")
                for name, spec in step_data["inputs"].items():
                    shape = spec.get('shape', [])
                    dtype = spec.get('dtype', 'unknown')
                    self.buffer.write(f"  {name}: shape={shape}, dtype={dtype}\n")
        return 1
    
    @step(HTPExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write complete module hierarchy."""
        self.buffer.write("\nCOMPLETE MODULE HIERARCHY\n")
        self.buffer.write("-" * 40 + "\n")
        
        # Sort modules by execution order if available
        sorted_modules = sorted(
            data.hierarchy.items(),
            key=lambda x: x[1].get("execution_order", 0)
        )
        
        for path, info in sorted_modules:
            display_path = path if path else "[ROOT]"
            self.buffer.write(f"\nModule: {display_path}\n")
            self.buffer.write(f"  Class: {info.get('class_name', 'Unknown')}\n")
            self.buffer.write(f"  Tag: {info.get('traced_tag', 'N/A')}\n")
            self.buffer.write(f"  Type: {info.get('module_type', 'unknown')}\n")
            self.buffer.write(f"  Execution Order: {info.get('execution_order', 'N/A')}\n")
        
        self.buffer.write(f"\nTotal Modules: {len(data.hierarchy)}\n")
        return 1
    
    @step(HTPExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write node tagging statistics and complete mappings."""
        self.buffer.write("\nNODE TAGGING STATISTICS\n")
        self.buffer.write("-" * 40 + "\n")
        self.buffer.write(f"Total ONNX Nodes: {data.total_nodes}\n")
        self.buffer.write(f"Tagged Nodes: {len(data.tagged_nodes)}\n")
        self.buffer.write(f"Coverage: {data.coverage:.1f}%\n")
        
        # Detailed statistics
        if data.tagging_stats:
            stats = data.tagging_stats
            unique_tags = len(set(data.tagged_nodes.values()))
            
            # Count root vs scoped nodes
            root_nodes = sum(1 for tag in data.tagged_nodes.values() if tag.count('/') <= 1)
            scoped_nodes = len(data.tagged_nodes) - root_nodes
            
            self.buffer.write(f"  Root Nodes: {root_nodes}\n")
            self.buffer.write(f"  Scoped Nodes: {scoped_nodes}\n")
            self.buffer.write(f"  Unique Scopes: {unique_tags}\n")
            self.buffer.write(f"  Direct Matches: {stats.get('direct_matches', 0)}\n")
            self.buffer.write(f"  Parent Matches: {stats.get('parent_matches', 0)}\n")
            self.buffer.write(f"  Operation Matches: {stats.get('operation_matches', 0)}\n")
            self.buffer.write(f"  Root Fallbacks: {stats.get('root_fallbacks', 0)}\n")
            self.buffer.write(f"  Empty Tags: {stats.get('empty_tags', 0)}\n")
        
        # Write complete node mappings
        self.buffer.write("\nCOMPLETE NODE MAPPINGS\n")
        self.buffer.write("-" * 40 + "\n")
        
        # Sort nodes by name for consistent output
        sorted_nodes = sorted(data.tagged_nodes.items())
        for node_name, tag in sorted_nodes:
            self.buffer.write(f"{node_name} -> {tag}\n")
        
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write export summary."""
        self.buffer.write("\nEXPORT SUMMARY\n")
        self.buffer.write("-" * 40 + "\n")
        self.buffer.write(f"Total Export Time: {data.export_time:.2f}s\n")
        self.buffer.write(f"ONNX File Size: {data.onnx_size_mb:.2f}MB\n")
        self.buffer.write(f"Final Coverage: {data.coverage:.1f}%\n")
        
        empty = data.tagging_stats.get("empty_tags", 0)
        if empty == 0:
            self.buffer.write(f"Empty Tags: {empty} ✅\n")
        else:
            self.buffer.write(f"Empty Tags: {empty} ❌\n")
        
        self.buffer.write("\n" + HTPExportConfig.MAJOR_SEPARATOR + "\n")
        self.buffer.write("Export completed successfully!\n")
        
        # Write buffer to file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(self.buffer.getvalue())
        
        return 1
    
    def flush(self) -> None:
        """Flush is handled in write_complete."""
        pass


class HTPExportMonitor:
    """
    Unified export monitoring system for HTP strategy.
    Coordinates console output, metadata generation, and report writing.
    """
    
    def __init__(
        self,
        output_path: str,
        model_name: str = "",
        verbose: bool = True,
        enable_report: bool = True,
        console: Console = None
    ):
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.verbose = verbose
        self.enable_report = enable_report
        
        # Initialize export data
        self.data = HTPExportData(
            model_name=model_name,
            output_path=str(output_path)
        )
        
        # Initialize writers
        self.console_writer = HTPConsoleWriter(console=console, verbose=verbose)
        
        # Setup file paths
        base_name = self.output_path.stem
        output_dir = self.output_path.parent
        
        metadata_path = output_dir / f"{base_name}{HTPExportConfig.METADATA_SUFFIX}"
        self.metadata_writer = HTPMetadataWriter(metadata_path)
        self.data.metadata_path = str(metadata_path)
        
        if enable_report:
            report_path = output_dir / f"{base_name}{HTPExportConfig.FULL_REPORT_SUFFIX}"
            self.report_writer = HTPReportWriter(report_path)
            self.data.report_path = str(report_path)
        else:
            self.report_writer = None
        
        # Track current step
        self.current_step = None
        
        # Buffer for capturing all console output
        self.text_report_buffer = io.StringIO()
    
    def update(self, step: HTPExportStep, **kwargs) -> None:
        """Update export data and notify all writers."""
        self.current_step = step
        
        # Update data based on step
        if step == HTPExportStep.MODEL_PREP:
            self.data.model_class = kwargs.get("model_class", "")
            self.data.total_modules = kwargs.get("total_modules", 0)
            self.data.total_parameters = kwargs.get("total_parameters", 0)
            self.data.embed_hierarchy_attributes = kwargs.get("embed_hierarchy_attributes", True)
        
        elif step == HTPExportStep.INPUT_GEN:
            self.data.steps["input_generation"] = kwargs
            self.data.input_names = list(kwargs.get("inputs", {}).keys())
        
        elif step == HTPExportStep.HIERARCHY:
            self.data.hierarchy = kwargs.get("hierarchy", {})
            self.data.execution_steps = kwargs.get("execution_steps", 0)
        
        elif step == HTPExportStep.ONNX_EXPORT:
            self.data.steps["onnx_export"] = kwargs
            self.data.output_names = kwargs.get("output_names", [])
        
        elif step == HTPExportStep.TAGGER_CREATION:
            self.data.steps["tagger_creation"] = kwargs
        
        elif step == HTPExportStep.NODE_TAGGING:
            self.data.total_nodes = kwargs.get("total_nodes", 0)
            self.data.tagged_nodes = kwargs.get("tagged_nodes", {})
            self.data.tagging_stats = kwargs.get("statistics", {})
        
        elif step == HTPExportStep.TAG_INJECTION:
            self.data.steps["tag_injection"] = kwargs
        
        elif step == HTPExportStep.METADATA_GEN:
            # Calculate file size
            if self.output_path.exists():
                self.data.onnx_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        
        elif step == HTPExportStep.COMPLETE:
            self.data.export_time = self.data.elapsed_time
        
        # Notify all writers
        self.console_writer.write(step, self.data)
        self.metadata_writer.write(step, self.data)
        
        if self.report_writer:
            self.report_writer.write(step, self.data)
    
    def finalize(self) -> None:
        """Finalize all writers and close resources."""
        # Ensure completion step is processed
        if self.current_step != HTPExportStep.COMPLETE:
            self.update(HTPExportStep.COMPLETE)
        
        # Close all writers
        self.console_writer.close()
        self.metadata_writer.close()
        
        if self.report_writer:
            self.report_writer.close()
        
        # Write console buffer to text report if we have one
        if hasattr(self.console_writer, 'console') and self.console_writer.console.file == self.text_report_buffer:
            report_path = self.output_path.parent / f"{self.output_path.stem}{HTPExportConfig.REPORT_SUFFIX}"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self.text_report_buffer.getvalue())
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure finalization."""
        self.finalize()
        return False