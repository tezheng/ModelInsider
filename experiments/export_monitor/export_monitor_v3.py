#!/usr/bin/env python3
"""
Export monitoring system for HTP strategy - Version 3.
Fixed metadata and report generation to match baseline exactly.
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


class HTPExportConfig:
    """Configuration for HTP Export Monitor - no hardcoded values."""
    
    # Display formatting
    SEPARATOR_WIDTH = 80
    CONSOLE_WIDTH = 80
    
    # Tree display limits
    MODULE_TREE_MAX_LINES = 100
    NODE_TREE_MAX_LINES = 30  # Back to 30 to match baseline
    TOP_NODES_DISPLAY_COUNT = 20
    MAX_OPERATION_TYPES = 5
    
    # Section separators
    MAJOR_SEPARATOR = "=" * SEPARATOR_WIDTH
    MINOR_SEPARATOR = "-" * 60
    SHORT_SEPARATOR = "-" * 30
    
    # Depth limits
    MAX_TREE_DEPTH = 4
    NODE_DETAIL_MAX_DEPTH = 3
    
    # File naming
    METADATA_SUFFIX = "_htp_metadata.json"
    REPORT_SUFFIX = "_htp_export_report.txt"
    FULL_REPORT_SUFFIX = "_full_report.txt"
    
    # Export settings
    DEFAULT_OPSET_VERSION = 17
    DEFAULT_CONSTANT_FOLDING = True
    
    # Step display
    TOTAL_EXPORT_STEPS = 8
    
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
    
    # Step icons
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
    """HTP export process steps."""
    MODEL_PREP = "model_preparation"
    INPUT_GEN = "input_generation"
    HIERARCHY = "hierarchy_building"
    ONNX_EXPORT = "onnx_export"
    TAGGER_CREATION = "tagger_creation"
    NODE_TAGGING = "node_tagging"
    TAG_INJECTION = "tag_injection"
    METADATA_GEN = "metadata_generation"
    COMPLETE = "export_complete"


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
    """Base class for step-aware writers."""
    
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
    """Console output writer for HTP export."""
    
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        self._total_steps = HTPExportConfig.TOTAL_EXPORT_STEPS
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: simple step completion message."""
        if self.verbose:
            print(f"✓ {export_step.value} completed")
        return 1
    
    def _print_header(self, text: str) -> None:
        """Print section header."""
        print("")
        print(HTPExportConfig.MAJOR_SEPARATOR)
        print(text)
        print(HTPExportConfig.MAJOR_SEPARATOR)
    
    def _print_minor_header(self, text: str) -> None:
        """Print minor section header."""
        print(f"\n🌳 {text}:")
        print(HTPExportConfig.MINOR_SEPARATOR)
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 1: Model preparation."""
        if not self.verbose:
            return 0
            
        self._print_header(f"{HTPExportConfig.STEP_ICONS['MODEL_PREP']} STEP 1/{self._total_steps}: {HTPExportConfig.STEP_TITLES['MODEL_PREP']}")
        
        print(f"✅ Model loaded: {data.model_class} ({data.total_modules} modules, {data.total_parameters/1e6:.1f}M parameters)")
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
            
        self._print_header(f"{HTPExportConfig.STEP_ICONS['INPUT_GEN']} STEP 2/{self._total_steps}: {HTPExportConfig.STEP_TITLES['INPUT_GEN']}")
        
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            print(f"🤖 Auto-generating inputs for: {data.model_name}")
            print(f"   • Model type: {step_data.get('model_type', 'unknown')}")
            print(f"   • Auto-detected task: {step_data.get('task', 'unknown')}")
            
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
            
        self._print_header(f"{HTPExportConfig.STEP_ICONS['HIERARCHY']} STEP 3/{self._total_steps}: {HTPExportConfig.STEP_TITLES['HIERARCHY']}")
        print("✅ Hierarchy building completed with TracingHierarchyBuilder")
        print(f"📈 Traced {len(data.hierarchy)} modules")
        
        if data.execution_steps > 0:
            print(f"🔄 Execution steps: {data.execution_steps}")
        
        if data.hierarchy:
            self._print_hierarchy_tree(data.hierarchy, max_lines=HTPExportConfig.MODULE_TREE_MAX_LINES)
        
        return 1
    
    @step(HTPExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 4: ONNX export."""
        if not self.verbose:
            return 0
            
        self._print_header(f"{HTPExportConfig.STEP_ICONS['ONNX_EXPORT']} STEP 4/{self._total_steps}: {HTPExportConfig.STEP_TITLES['ONNX_EXPORT']}")
        print(f"🎯 Target file: {data.output_path}")
        
        if "onnx_export" in data.steps:
            step_data = data.steps["onnx_export"]
            print("⚙️ Export config:")
            print(f"   • opset_version: {step_data.get('opset_version', HTPExportConfig.DEFAULT_OPSET_VERSION)}")
            print(f"   • do_constant_folding: {step_data.get('do_constant_folding', HTPExportConfig.DEFAULT_CONSTANT_FOLDING)}")
            print("   • verbose: False")
            
            if data.input_names:
                print(f"   • input_names: {data.input_names}")
        
        print("✅ ONNX export completed successfully")
        return 1
    
    @step(HTPExportStep.TAGGER_CREATION)
    def write_tagger_creation(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 5: Node tagger creation."""
        if not self.verbose:
            return 0
            
        self._print_header(f"{HTPExportConfig.STEP_ICONS['TAGGER_CREATION']} STEP 5/{self._total_steps}: {HTPExportConfig.STEP_TITLES['TAGGER_CREATION']}")
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
            
        self._print_header(f"{HTPExportConfig.STEP_ICONS['NODE_TAGGING']} STEP 6/{self._total_steps}: {HTPExportConfig.STEP_TITLES['NODE_TAGGING']}")
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
                print(f"   • Direct matches: {direct} ({direct/total*100:.1f}%)")
                print(f"   • Parent matches: {parent} ({parent/total*100:.1f}%)")
                print(f"   • Root fallbacks: {root} ({root/total*100:.1f}%)")
        
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
            
        self._print_header(f"{HTPExportConfig.STEP_ICONS['TAG_INJECTION']} STEP 7/{self._total_steps}: {HTPExportConfig.STEP_TITLES['TAG_INJECTION']}")
        
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
            
        self._print_header(f"{HTPExportConfig.STEP_ICONS['METADATA_GEN']} STEP 8/{self._total_steps}: {HTPExportConfig.STEP_TITLES['METADATA_GEN']}")
        print("✅ Metadata file created successfully")
        
        if data.metadata_path:
            print(f"📄 Metadata file: {data.metadata_path}")
        
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Export completion summary."""
        if not self.verbose:
            return 0
            
        self._print_header("📋 " + HTPExportConfig.STEP_TITLES["COMPLETE"])
        print(f"🎉 HTP Export completed successfully in {data.elapsed_time:.2f}s!")
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
        
        if data.output_path:
            print(f"   • ONNX model: {data.output_path}")
        if data.metadata_path:
            print(f"   • Metadata: {data.metadata_path}")
        if data.report_path:
            print(f"   • Report: {data.report_path}")
        else:
            print("   • Report: disabled")
        
        return 1
    
    def _print_hierarchy_tree(self, hierarchy: dict, max_lines: int | None = None) -> None:
        """Print module hierarchy as a tree."""
        if max_lines is None:
            max_lines = HTPExportConfig.MODULE_TREE_MAX_LINES
            
        self._print_minor_header("Module Hierarchy")
        
        # Build tree structure
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        
        print(root_name)
        
        # Build parent-child mapping
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
        
        def print_tree(path: str, prefix: str = "", is_last: bool = True, line_count: list | None = None):
            if line_count is None:
                line_count = [1]  # Start at 1 for root
                
            if line_count[0] >= max_lines:
                return
                
            # Get children
            children = parent_to_children.get(path, [])
            children.sort()
            
            # Print each child
            for i, child_path in enumerate(children):
                if line_count[0] >= max_lines:
                    break
                    
                is_last_child = i == len(children) - 1
                child_info = hierarchy.get(child_path, {})
                class_name = child_info.get("class_name", "Unknown")
                
                # Print the tree branch
                if path == "":  # Root level
                    branch = "├── " if not is_last_child else "└── "
                else:
                    branch = prefix + ("├── " if not is_last_child else "└── ")
                
                print(f"{branch}{class_name}: {child_path}")
                line_count[0] += 1
                
                # Recurse for children
                if path == "":
                    child_prefix = "│   " if not is_last_child else "    "
                else:
                    child_prefix = prefix + ("│   " if not is_last_child else "    ")
                
                print_tree(child_path, child_prefix, is_last_child, line_count)
        
        # Start printing from root
        line_counter = [1]
        print_tree("", line_count=line_counter)
        
        # Print line count
        if line_counter[0] < len(hierarchy):
            print(f"(showing {line_counter[0]}/{len(hierarchy)} lines)")
    
    def _print_top_nodes_by_hierarchy(self, tagged_nodes: dict[str, str]) -> None:
        """Print top nodes grouped by hierarchy tag."""
        from collections import Counter
        
        # Count nodes by tag
        tag_counts = Counter(tagged_nodes.values())
        
        # Get top tags
        top_tags = tag_counts.most_common(HTPExportConfig.TOP_NODES_DISPLAY_COUNT)
        
        if top_tags:
            print(f"\n📊 Top 20 Nodes by Hierarchy:")
            print(HTPExportConfig.SHORT_SEPARATOR)
            
            for i, (tag, count) in enumerate(top_tags[:HTPExportConfig.TOP_NODES_DISPLAY_COUNT], 1):
                print(f"{i:2d}. {tag}: {count} nodes")
    
    def _print_hierarchy_with_nodes(self, hierarchy: dict, tagged_nodes: dict[str, str], total_nodes: int) -> None:
        """Print hierarchy tree with ONNX node operations."""
        from collections import defaultdict
        
        self._print_minor_header("Complete HF Hierarchy with ONNX Nodes")
        
        # Group nodes by tag and operation type
        nodes_by_tag = defaultdict(lambda: defaultdict(list))
        for node_name, tag in tagged_nodes.items():
            # Extract operation type from node name
            if '/' in node_name:
                parts = node_name.split('/')
                # Find the operation type (usually the last part before parameters)
                op_type = parts[-1].split('_')[0] if '_' in parts[-1] else parts[-1]
                # Handle special cases
                if parts[-1] in ['LayerNormalization', 'Gemm', 'Tanh', 'Softmax', 'Erf']:
                    op_type = parts[-1]
            else:
                op_type = node_name.split('_')[0] if '_' in node_name else node_name
            
            nodes_by_tag[tag][op_type].append(node_name)
        
        # Build tree with node counts
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        
        print(f"{root_name} ({total_nodes} ONNX nodes)")
        
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
        
        def print_module_with_nodes(path: str, level: int = 1, prefix: str = "", line_count: list | None = None):
            if line_count is None:
                line_count = [1]  # Start at 1 for root
                
            if line_count[0] >= HTPExportConfig.NODE_TREE_MAX_LINES:
                return
                
            if level > HTPExportConfig.MAX_TREE_DEPTH:
                return
                
            # Get children
            children = parent_to_children.get(path, [])
            children.sort()
            
            # Print each child with its nodes
            for i, child_path in enumerate(children):
                if line_count[0] >= HTPExportConfig.NODE_TREE_MAX_LINES:
                    break
                    
                is_last = i == len(children) - 1
                child_info = hierarchy.get(child_path, {})
                class_name = child_info.get("class_name", "Unknown")
                tag = child_info.get("traced_tag", "")
                
                # Count nodes for this tag
                node_count = len([n for n, t in tagged_nodes.items() if t == tag])
                
                # Print branch
                if level == 1:
                    branch = "├── " if not is_last else "└── "
                else:
                    branch = prefix + ("├── " if not is_last else "└── ")
                
                print(f"{branch}{class_name}: {child_path} ({node_count} nodes)")
                line_count[0] += 1
                
                # Show operation breakdown for this module
                if tag in nodes_by_tag and level <= HTPExportConfig.NODE_DETAIL_MAX_DEPTH and node_count > 0:
                    ops = nodes_by_tag[tag]
                    sorted_ops = sorted(ops.items(), key=lambda x: len(x[1]), reverse=True)
                    
                    # Determine prefix for operations
                    if level == 1:
                        op_prefix = "│   " if not is_last else "    "
                    else:
                        op_prefix = prefix + ("│   " if not is_last else "    ")
                    
                    for j, (op_type, op_nodes) in enumerate(sorted_ops[:HTPExportConfig.MAX_OPERATION_TYPES]):
                        if line_count[0] >= HTPExportConfig.NODE_TREE_MAX_LINES:
                            break
                            
                        is_last_op = j == len(sorted_ops) - 1 or j == HTPExportConfig.MAX_OPERATION_TYPES - 1
                        op_branch = op_prefix + ("├── " if not is_last_op else "└── ")
                        
                        count = len(op_nodes)
                        if count > 1:
                            print(f"{op_branch}{op_type} ({count} ops)")
                        else:
                            # For single ops, show the full name
                            print(f"{op_branch}{op_type}: {op_nodes[0]}")
                        line_count[0] += 1
                
                # Recurse for children
                if level == 1:
                    child_prefix = "│   " if not is_last else "    "
                else:
                    child_prefix = prefix + ("│   " if not is_last else "    ")
                
                print_module_with_nodes(child_path, level + 1, child_prefix, line_count)
        
        # Start printing
        line_counter = [1]
        print_module_with_nodes("", line_count=line_counter)
        
        # Add truncation notice
        if line_counter[0] >= HTPExportConfig.NODE_TREE_MAX_LINES:
            remaining = total_nodes - line_counter[0] + 1
            print(f"... and {remaining} more lines (truncated for console)")
        print(f"(showing {min(line_counter[0]-1, HTPExportConfig.NODE_TREE_MAX_LINES)}/{total_nodes-1} lines)")


class HTPMetadataWriter(StepAwareWriter):
    """Metadata writer for HTP export - writes to JSON."""
    
    def __init__(self, metadata_path: Path):
        super().__init__()
        self.metadata_path = metadata_path
        self.metadata = {}
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: store step data in metadata."""
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Generate final metadata file."""
        # Build complete metadata structure EXACTLY matching baseline
        metadata = {
            "export_info": {
                "timestamp": data.timestamp,
                "model_name": data.model_name,
                "model_class": data.model_class,
                "export_time": data.export_time,  # Use passed value
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
            "nodes": data.tagged_nodes,
            "report": {
                "node_tagging": {
                    "statistics": {
                        "total_nodes": data.total_nodes,
                        "tagged_nodes": len(data.tagged_nodes),
                        "coverage": f"{data.coverage:.1f}%",
                        "direct_matches": data.tagging_stats.get("direct_matches", 0),
                        "parent_matches": data.tagging_stats.get("parent_matches", 0),
                        "operation_matches": data.tagging_stats.get("operation_matches", 0),
                        "root_fallbacks": data.tagging_stats.get("root_fallbacks", 0),
                        "empty_tags": data.tagging_stats.get("empty_tags", 0)
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


class HTPExportMonitor:
    """
    Unified export monitoring system for HTP strategy.
    Version 3 - Fixed to match baseline exactly.
    """
    
    def __init__(
        self,
        output_path: str,
        model_name: str = "",
        verbose: bool = True,
        enable_report: bool = True
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
        self.console_writer = HTPConsoleWriter(verbose=verbose)
        
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
            # Add input names if provided
            if "input_names" in kwargs:
                self.data.input_names = kwargs["input_names"]
        
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
            self.data.export_time = kwargs.get("export_time", self.data.elapsed_time)
        
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
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure finalization."""
        self.finalize()
        return False
