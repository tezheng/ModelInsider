"""
HTP Export Monitoring System - Comprehensive Fix

This module provides a unified monitoring system for the HTP export process with:
- Proper ANSI text styling matching baseline
- Complete console output capture (no truncation)
- Full text reports (plain text, no ANSI)
- Complete metadata with all console data in JSON format
- Clean design following best practices
"""

import io
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for export monitor."""
    
    # Display limits
    MAX_TREE_DEPTH = 100  # No truncation
    MAX_DISPLAY_NODES = 1000  # No truncation
    TOP_NODES_COUNT = 20
    
    # Formatting
    SEPARATOR_LENGTH = 80
    SEPARATOR_CHAR = "="
    SUBSEPARATOR_CHAR = "-"
    
    # Console settings
    CONSOLE_WIDTH = 80
    FORCE_TERMINAL = True
    COLOR_SYSTEM = "standard"
    
    # File names
    METADATA_SUFFIX = "_htp_metadata.json"
    REPORT_SUFFIX = "_htp_export_report.txt"
    CONSOLE_LOG_SUFFIX = "_console.log"


# ============================================================================
# DATA MODELS
# ============================================================================

class HTPExportStep(Enum):
    """Export process steps."""
    MODEL_PREP = "model_preparation"
    INPUT_GEN = "input_generation"
    HIERARCHY = "hierarchy_building"
    TRACE = "model_tracing"
    ONNX_EXPORT = "onnx_export"
    TAGGER_CREATION = "tagger_creation"
    NODE_TAGGING = "node_tagging"
    SAVE = "model_save"
    COMPLETE = "export_complete"


@dataclass
class HTPExportData:
    """Container for all export-related data."""
    # Model info
    model_name: str = ""
    model_class: str = ""
    total_modules: int = 0
    total_parameters: int = 0
    
    # Export config
    output_path: str = ""
    strategy: str = "htp"
    embed_hierarchy_attributes: bool = True
    
    # Hierarchy data
    hierarchy: dict[str, dict[str, Any]] = None
    execution_steps: int = 0
    
    # ONNX data
    output_names: list[str] = None
    onnx_size_mb: float = 0.0
    
    # Tagging results
    total_nodes: int = 0
    tagged_nodes: dict[str, str] = None
    tagging_stats: dict[str, int] = None
    coverage: float = 0.0
    
    # Timing
    timestamp: str = ""
    elapsed_time: float = 0.0
    export_time: float = 0.0
    
    # Step-specific data
    steps: dict[str, Any] = None
    
    # Output paths
    report_path: str | None = None
    console_log_path: str | None = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.hierarchy is None:
            self.hierarchy = {}
        if self.tagged_nodes is None:
            self.tagged_nodes = {}
        if self.tagging_stats is None:
            self.tagging_stats = {}
        if self.steps is None:
            self.steps = {}
        if self.output_names is None:
            self.output_names = []
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================================
# TEXT STYLING UTILITIES
# ============================================================================

class TextStyler:
    """Utilities for ANSI text styling matching baseline."""
    
    @staticmethod
    def bold(text: str) -> str:
        """Format text as bold."""
        return f"\033[1m{text}\033[0m"
    
    @staticmethod
    def bold_cyan(text: str | int | float) -> str:
        """Format number as bold cyan."""
        return f"\033[1;36m{text}\033[0m"
    
    @staticmethod
    def bold_parens(content: str) -> str:
        """Format with bold parentheses."""
        return f"\033[1m(\033[0m{content}\033[1m)\033[0m"
    
    @staticmethod
    def green_true() -> str:
        """Format True as italic green."""
        return "\033[3;92mTrue\033[0m"
    
    @staticmethod
    def red_false() -> str:
        """Format False as italic red."""
        return "\033[3;91mFalse\033[0m"
    
    @staticmethod
    def green_string(text: str) -> str:
        """Format string as green."""
        return f"\033[32m'{text}'\033[0m"
    
    @staticmethod
    def magenta_path(path: str) -> str:
        """Format path as magenta."""
        return f"\033[35m{path}\033[0m"
    
    @staticmethod
    def bright_magenta(text: str) -> str:
        """Format text as bright magenta."""
        return f"\033[95m{text}\033[0m"
    
    @staticmethod
    def format_bool(value: bool) -> str:
        """Format boolean with color."""
        return TextStyler.green_true() if value else TextStyler.red_false()
    
    @staticmethod
    def format_step_header(step_num: int, total: int, title: str) -> str:
        """Format step header with styled numbers."""
        return (f"📋 STEP {TextStyler.bold_cyan(step_num)}/"
                f"{TextStyler.bold_cyan(total)}: {title}")
    
    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove all ANSI escape codes from text."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


# ============================================================================
# BASE WRITER CLASS
# ============================================================================

class StepAwareWriter:
    """Base class for step-aware writers."""
    
    def __init__(self):
        self._step_handlers = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register step handlers from decorated methods."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_export_step'):
                self._step_handlers[attr._export_step] = attr
    
    def write(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write data for the given export step."""
        handler = self._step_handlers.get(export_step, self._write_default)
        return handler(export_step, data)
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default handler for unregistered steps."""
        return 0
    
    def flush(self) -> None:
        """Flush any buffered data. Override in subclasses."""
        pass


def step(export_step: HTPExportStep):
    """Decorator to register a method as handler for an export step."""
    def decorator(func):
        func._export_step = export_step
        return func
    return decorator


# ============================================================================
# CONSOLE WRITER WITH PROPER STYLING
# ============================================================================

class HTPConsoleWriter(StepAwareWriter):
    """Console output writer with proper ANSI styling."""
    
    def __init__(self, console: Console = None, verbose: bool = True, 
                 capture_buffer: io.StringIO = None):
        super().__init__()
        self.console = console or Console(
            width=Config.CONSOLE_WIDTH,
            force_terminal=Config.FORCE_TERMINAL,
            legacy_windows=False,
            color_system=Config.COLOR_SYSTEM
        )
        self.verbose = verbose
        self.capture_buffer = capture_buffer  # For capturing output
        self._total_steps = 8
    
    def _print(self, text: str, **kwargs):
        """Print to console and capture buffer."""
        if self.verbose:
            # Write directly to console file to preserve ANSI codes
            self.console.file.write(text + "\n")
            
            # Also capture to buffer if provided
            if self.capture_buffer:
                self.capture_buffer.write(text + "\n")
    
    def _print_separator(self):
        """Print section separator."""
        self._print(Config.SEPARATOR_CHAR * Config.SEPARATOR_LENGTH)
    
    def _print_header(self, text: str):
        """Print section header."""
        self._print("")
        self._print_separator()
        self._print(text)
        self._print_separator()
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 1: Model preparation with proper styling."""
        if not self.verbose:
            return 0
        
        # Initial messages
        if data.model_name:
            self._print(f"🔄 Loading model and exporting: {data.model_name}")
        
        # Strategy line with special formatting
        strategy_line = f"🧠 Using HTP {TextStyler.bold_parens('Hierarchical Trace-and-Project')} strategy"
        self._print(strategy_line)
        
        if data.model_name:
            self._print(f"Auto-loading model from: {data.model_name}")
            self._print(f"Successfully loaded {data.model_class}")
            self._print(f"Starting HTP export for {data.model_class}")
        
        # Step header
        self._print_header(TextStyler.format_step_header(1, self._total_steps, "MODEL PREPARATION"))
        
        # Model info with styled numbers
        params_str = f"{data.total_parameters/1e6:.1f}"
        model_line = (f"✅ Model loaded: {data.model_class} "
                     f"{TextStyler.bold_parens(f'{TextStyler.bold_cyan(data.total_modules)} modules, '
                     f'{TextStyler.bold_cyan(params_str)}M parameters')}")
        self._print(model_line)
        
        self._print(f"🎯 Export target: {data.output_path}")
        self._print(f"⚙️ Strategy: HTP {TextStyler.bold_parens('Hierarchy-Preserving')}")
        
        if data.embed_hierarchy_attributes:
            self._print("✅ Hierarchy attributes will be embedded in ONNX")
        else:
            self._print("⚠️ Hierarchy attributes will NOT be embedded (clean ONNX)")
        
        self._print("✅ Model set to evaluation mode")
        return 1
    
    @step(HTPExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 2: Input generation with styled output."""
        if not self.verbose:
            return 0
        
        self._print_header(TextStyler.format_step_header(2, self._total_steps, 
                                                         "INPUT GENERATION & VALIDATION"))
        
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            self._print(f"🤖 Auto-generating inputs for: {data.model_name}")
            self._print(f"   • Model type: {step_data.get('model_type', 'unknown')}")
            self._print(f"   • Auto-detected task: {step_data.get('task', 'unknown')}")
            
            if "model_type" in step_data and "task" in step_data:
                self._print(f"✅ Created onnx export config for {step_data['model_type']} "
                          f"with task {step_data['task']}")
            
            # Input tensors with styled count
            inputs = step_data.get("inputs", {})
            if inputs:
                self._print(f"🔧 Generated {TextStyler.bold_cyan(len(inputs))} input tensors:")
                
                for name, spec in inputs.items():
                    # Format shape with bold brackets and cyan numbers
                    shape = spec.get("shape", [])
                    shape_str = TextStyler.bold("[")
                    shape_str += ", ".join(TextStyler.bold_cyan(dim) for dim in shape)
                    shape_str += TextStyler.bold("]")
                    
                    # Format dtype with bold parentheses
                    dtype_str = TextStyler.bold_parens(spec.get("dtype", "unknown"))
                    
                    self._print(f"   • {name}: {shape_str} {dtype_str}")
        
        return 1
    
    @step(HTPExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 3: Hierarchy building with full tree."""
        if not self.verbose:
            return 0
        
        self._print_header(TextStyler.format_step_header(3, self._total_steps, "HIERARCHY BUILDING"))
        
        self._print("✅ Hierarchy building completed with TracingHierarchyBuilder")
        self._print(f"📈 Traced {TextStyler.bold_cyan(len(data.hierarchy))} modules")
        self._print(f"🔄 Execution steps: {TextStyler.bold_cyan(data.execution_steps)}")
        
        # Print full hierarchy tree
        self._print("\n🌳 Module Hierarchy:")
        self._print("-" * 60)
        self._print_hierarchy_tree(data.hierarchy)
        
        return 1
    
    def _print_hierarchy_tree(self, hierarchy: dict[str, dict[str, Any]]) -> None:
        """Print the complete module hierarchy tree."""
        # Find root
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        self._print(root_name)
        
        def print_module(path: str, prefix: str = "", is_last: bool = True):
            if not path:
                return
            
            info = hierarchy.get(path, {})
            class_name = info.get("class_name", "Unknown")
            
            # Tree characters
            if prefix:
                connector = "└── " if is_last else "├── "
                # Style numbers in path
                styled_path = re.sub(r'\.(\d+)', lambda m: f'.{TextStyler.bold_cyan(m.group(1))}', path)
                self._print(f"{prefix}{connector}{class_name}: {styled_path}")
            
            # Find children
            children = []
            for other_path in hierarchy:
                if other_path.startswith(path + ".") and other_path.count(".") == path.count(".") + 1:
                    children.append(other_path)
            
            # Sort children for consistent output
            children.sort()
            
            # Print children
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                print_module(child, child_prefix, is_last_child)
        
        # Print all root modules
        root_modules = sorted([p for p in hierarchy if p and "." not in p])
        for i, module in enumerate(root_modules):
            is_last = (i == len(root_modules) - 1)
            print_module(module, "", is_last)
    
    @step(HTPExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 4: ONNX export with styled config."""
        if not self.verbose:
            return 0
        
        self._print_header(TextStyler.format_step_header(4, self._total_steps, "ONNX EXPORT"))
        
        self._print(f"🎯 Target file: {data.output_path}")
        
        # Export config with styled values
        if "onnx_export" in data.steps:
            config = data.steps["onnx_export"].get("config", {})
            self._print("⚙️ Export config:")
            self._print(f"   • opset_version: {TextStyler.bold_cyan(config.get('opset_version', 17))}")
            self._print(f"   • do_constant_folding: {TextStyler.format_bool(config.get('do_constant_folding', True))}")
            self._print(f"   • verbose: {TextStyler.format_bool(config.get('verbose', False))}")
            
            # Input names with green strings
            input_names = config.get('input_names', [])
            if input_names:
                names_str = TextStyler.bold("[")
                names_str += ", ".join(TextStyler.green_string(name) for name in input_names)
                names_str += TextStyler.bold("]")
                self._print(f"   • input_names: {names_str}")
        
        self._print("✅ ONNX export completed successfully")
        return 1
    
    @step(HTPExportStep.TAGGER_CREATION)
    def write_tagger_creation(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 5: Node tagger creation."""
        if not self.verbose:
            return 0
        
        self._print_header(TextStyler.format_step_header(5, self._total_steps, "NODE TAGGER CREATION"))
        
        self._print("✅ Node tagger created successfully")
        
        # Model root tag with styled path and class
        root_tag = data.steps.get("tagger_creation", {}).get("root_tag", "/Model")
        if "/" in root_tag:
            path, class_name = root_tag.rsplit("/", 1)
            styled_tag = f"{TextStyler.magenta_path(path + '/')}{TextStyler.bright_magenta(class_name)}"
        else:
            styled_tag = TextStyler.bright_magenta(root_tag)
        
        self._print(f"🏷️ Model root tag: {styled_tag}")
        self._print("🔧 Operation fallback: disabled")
        return 1
    
    @step(HTPExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 6: Node tagging with full statistics."""
        if not self.verbose:
            return 0
        
        self._print_header(TextStyler.format_step_header(6, self._total_steps, "ONNX NODE TAGGING"))
        
        self._print("✅ Node tagging completed successfully")
        self._print(f"📈 Coverage: {TextStyler.bold_cyan(f'{data.coverage:.1f}')}%")
        self._print(f"📊 Tagged nodes: {TextStyler.bold_cyan(len(data.tagged_nodes))}"
                   f"/{TextStyler.bold_cyan(data.total_nodes)}")
        
        # Tagging statistics with styled numbers and percentages
        stats = data.tagging_stats
        if stats and data.total_nodes > 0:
            direct = stats.get("direct_matches", 0)
            parent = stats.get("parent_matches", 0)
            root = stats.get("root_fallbacks", 0)
            
            direct_pct = f"{direct/data.total_nodes*100:.1f}"
            parent_pct = f"{parent/data.total_nodes*100:.1f}"
            root_pct = f"{root/data.total_nodes*100:.1f}"
            
            self._print(f"   • Direct matches: {TextStyler.bold_cyan(direct)} "
                       f"{TextStyler.bold_parens(f'{TextStyler.bold_cyan(direct_pct)}%')}")
            self._print(f"   • Parent matches: {TextStyler.bold_cyan(parent)} "
                       f"{TextStyler.bold_parens(f'{TextStyler.bold_cyan(parent_pct)}%')}")
            self._print(f"   • Root fallbacks: {TextStyler.bold_cyan(root)} "
                       f"{TextStyler.bold_parens(f'{TextStyler.bold_cyan(root_pct)}%')}")
        
        empty_tags = stats.get("empty_tags", 0)
        self._print(f"✅ Empty tags: {TextStyler.bold_cyan(empty_tags)}")
        
        # Top nodes by hierarchy
        self._print_top_nodes(data.tagged_nodes)
        
        # Full hierarchy with ONNX nodes
        self._print_hierarchy_with_nodes(data.hierarchy, data.tagged_nodes)
        
        return 1
    
    def _print_top_nodes(self, tagged_nodes: dict[str, str]) -> None:
        """Print top nodes by hierarchy."""
        tag_counts = Counter(tagged_nodes.values())
        top_tags = tag_counts.most_common(Config.TOP_NODES_COUNT)
        
        if top_tags:
            self._print(f"\n📊 Top {TextStyler.bold_cyan(min(len(top_tags), Config.TOP_NODES_COUNT))} "
                       f"Nodes by Hierarchy:")
            self._print("-" * 30)
            
            for i, (tag, count) in enumerate(top_tags[:Config.TOP_NODES_COUNT], 1):
                # Style the tag path and class
                if "/" in tag:
                    parts = tag.split("/")
                    path = "/".join(parts[:-1])
                    class_name = parts[-1]
                    styled_tag = f"{TextStyler.magenta_path(path + '/')}{TextStyler.bright_magenta(class_name)}"
                else:
                    styled_tag = TextStyler.bright_magenta(tag)
                
                rank_str = f"{i:>2}"
                self._print(f"{TextStyler.bold_cyan(rank_str)}. {styled_tag}: {TextStyler.bold_cyan(count)} nodes")
    
    def _print_hierarchy_with_nodes(self, hierarchy: dict[str, dict[str, Any]], 
                                   tagged_nodes: dict[str, str]) -> None:
        """Print complete hierarchy with ONNX node details."""
        self._print("\n🌳 Complete HF Hierarchy with ONNX Nodes:")
        self._print("-" * 60)
        
        # Group nodes by tag and operation
        nodes_by_tag = defaultdict(lambda: defaultdict(list))
        for node_name, tag in tagged_nodes.items():
            op_type = node_name.split('_')[0] if '_' in node_name else node_name
            nodes_by_tag[tag][op_type].append(node_name)
        
        # Find root
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        root_tag = root_info.get("traced_tag", "")
        
        # Count nodes for root
        root_node_count = len([n for n, t in tagged_nodes.items() if t == root_tag])
        self._print(f"{root_name} {TextStyler.bold_parens(f'{TextStyler.bold_cyan(root_node_count)} ONNX nodes')}")
        
        def print_module_with_nodes(path: str, prefix: str = "", is_last: bool = True):
            if not path:
                return
            
            info = hierarchy.get(path, {})
            class_name = info.get("class_name", "Unknown")
            tag = info.get("traced_tag", "")
            
            # Count nodes
            node_count = len([n for n, t in tagged_nodes.items() if t == tag])
            
            if prefix:
                connector = "└── " if is_last else "├── "
                styled_path = re.sub(r'\.(\d+)', lambda m: f'.{TextStyler.bold_cyan(m.group(1))}', path)
                self._print(f"{prefix}{connector}{class_name}: {styled_path} "
                           f"{TextStyler.bold_parens(f'{TextStyler.bold_cyan(node_count)} nodes')}")
            
            # Print operations for this module
            if tag in nodes_by_tag and node_count > 0:
                ops = nodes_by_tag[tag]
                op_prefix = prefix + ("    " if is_last else "│   ")
                self._print(f"{op_prefix}└── Operations:")
                
                # Group and sort operations
                op_summary = []
                for op_type, op_nodes in sorted(ops.items()):
                    if len(op_nodes) > 1:
                        op_summary.append(f"{op_type} ({len(op_nodes)}x)")
                    else:
                        op_summary.append(op_type)
                
                # Print operation summary
                op_line = f"{op_prefix}    ├── " + ", ".join(op_summary[:10])
                if len(op_summary) > 10:
                    op_line += f" ... and {len(op_summary) - 10} more"
                self._print(op_line)
            
            # Find children
            children = []
            for other_path in hierarchy:
                if other_path.startswith(path + ".") and other_path.count(".") == path.count(".") + 1:
                    children.append(other_path)
            
            children.sort()
            
            # Print children
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                print_module_with_nodes(child, child_prefix, is_last_child)
        
        # Print all root modules
        root_modules = sorted([p for p in hierarchy if p and "." not in p])
        for i, module in enumerate(root_modules):
            is_last = (i == len(root_modules) - 1)
            print_module_with_nodes(module, "", is_last)
    
    @step(HTPExportStep.SAVE)
    def write_save(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 7: Save ONNX model."""
        if not self.verbose:
            return 0
        
        self._print_header(TextStyler.format_step_header(7, self._total_steps, "SAVE ONNX MODEL"))
        
        self._print(f"✅ Model saved to: {data.output_path}")
        self._print(f"⚙️ Hierarchy attributes: {'Embedded in ONNX' if data.embed_hierarchy_attributes else 'Not embedded'}")
        
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 8: Export complete with summary."""
        if not self.verbose:
            return 0
        
        self._print_header(TextStyler.format_step_header(8, self._total_steps, "EXPORT COMPLETE"))
        
        self._print(f"🎉 HTP Export completed successfully in {TextStyler.bold_cyan(f'{data.export_time:.2f}')}s!")
        
        self._print("📊 Export Statistics:")
        self._print(f"   • Export time: {TextStyler.bold_cyan(f'{data.export_time:.2f}')}s")
        self._print(f"   • Hierarchy modules: {TextStyler.bold_cyan(len(data.hierarchy))}")
        self._print(f"   • ONNX nodes: {TextStyler.bold_cyan(data.total_nodes)}")
        self._print(f"   • Tagged nodes: {TextStyler.bold_cyan(len(data.tagged_nodes))}")
        self._print(f"   • Coverage: {TextStyler.bold_cyan(f'{data.coverage:.1f}')}%")
        
        # Output files
        self._print("📁 Output files:")
        self._print(f"   • ONNX model: {Path(data.output_path).name} "
                   f"{TextStyler.bold_parens(f'{data.onnx_size_mb:.1f} MB')}")
        
        if data.report_path:
            self._print(f"   • Metadata: {Path(data.report_path).with_suffix('.json').name}")
            self._print(f"   • Report: {Path(data.report_path).name}")
        
        if data.console_log_path:
            self._print(f"   • Console log: {Path(data.console_log_path).name}")
        
        return 1


# ============================================================================
# METADATA WRITER WITH COMPLETE REPORT SECTION
# ============================================================================

class HTPMetadataWriter(StepAwareWriter):
    """JSON metadata writer with complete report section."""
    
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.metadata_path = f"{self.output_path}{Config.METADATA_SUFFIX}"
        self.metadata = {
            "export_context": {},
            "model": {},
            "modules": {},
            "nodes": {},
            "outputs": {},
            "report": {"steps": {}}
        }
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record detailed model preparation info."""
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
            "framework": "transformers",
            "total_modules": data.total_modules,
            "total_parameters": data.total_parameters
        }
        
        # Complete report section
        self.metadata["report"]["steps"]["model_preparation"] = {
            "model_class": data.model_class,
            "total_modules": data.total_modules,
            "total_parameters": data.total_parameters,
            "parameters_formatted": f"{data.total_parameters/1e6:.1f}M",
            "export_target": data.output_path,
            "strategy": f"{data.strategy.upper()} (Hierarchy-Preserving)",
            "embed_hierarchy_attributes": data.embed_hierarchy_attributes,
            "evaluation_mode": True,
            "timestamp": data.timestamp
        }
        
        return 1
    
    @step(HTPExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record detailed input generation info."""
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            
            if "tracing" not in self.metadata:
                self.metadata["tracing"] = {}
            
            self.metadata["tracing"].update({
                "model_type": step_data.get("model_type", "unknown"),
                "task": step_data.get("task", "unknown"),
                "inputs": step_data.get("inputs", {}),
                "outputs": data.output_names
            })
            
            # Complete report section
            self.metadata["report"]["steps"]["input_generation"] = {
                "model_name": data.model_name,
                "model_type": step_data.get("model_type", "unknown"),
                "task": step_data.get("task", "unknown"),
                "auto_detected_task": step_data.get("task", "unknown"),
                "config_created": True,
                "inputs_generated": {
                    "count": len(step_data.get("inputs", {})),
                    "tensors": step_data.get("inputs", {})
                },
                "output_names": data.output_names,
                "timestamp": data.timestamp
            }
        
        return 1
    
    @step(HTPExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record complete hierarchy data."""
        self.metadata["modules"] = data.hierarchy.copy()
        
        if "tracing" not in self.metadata:
            self.metadata["tracing"] = {}
        
        self.metadata["tracing"].update({
            "builder": "TracingHierarchyBuilder",
            "modules_traced": len(data.hierarchy),
            "execution_steps": data.execution_steps
        })
        
        # Build hierarchy tree structure for report
        tree_structure = self._build_tree_structure(data.hierarchy)
        
        self.metadata["report"]["steps"]["hierarchy_building"] = {
            "builder": "TracingHierarchyBuilder",
            "modules_traced": len(data.hierarchy),
            "execution_steps": data.execution_steps,
            "hierarchy_depth": max(len(p.split('.')) for p in data.hierarchy if p) if data.hierarchy else 0,
            "module_tree": tree_structure,
            "timestamp": data.timestamp
        }
        
        return 1
    
    def _build_tree_structure(self, hierarchy: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Build tree structure for metadata."""
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        
        tree = {
            "root": root_name,
            "total_modules": len(hierarchy),
            "modules": {}
        }
        
        # Build nested structure
        for path, _info in hierarchy.items():
            if not path:
                continue
            
            parts = path.split(".")
            current = tree["modules"]
            
            for i, part in enumerate(parts):
                current_path = ".".join(parts[:i+1])
                if part not in current:
                    current[part] = {
                        "class": hierarchy.get(current_path, {}).get("class_name", "Unknown"),
                        "path": current_path,
                        "children": {}
                    }
                current = current[part]["children"]
        
        return tree
    
    @step(HTPExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record detailed ONNX export info."""
        export_config = data.steps.get("onnx_export", {}).get("config", {})
        
        self.metadata["report"]["steps"]["onnx_export"] = {
            "target_file": data.output_path,
            "opset_version": export_config.get("opset_version", 17),
            "do_constant_folding": export_config.get("do_constant_folding", True),
            "verbose": export_config.get("verbose", False),
            "input_names": export_config.get("input_names", []),
            "output_names": data.output_names,
            "dynamic_axes": export_config.get("dynamic_axes"),
            "export_successful": True,
            "file_size_mb": data.onnx_size_mb,
            "timestamp": data.timestamp
        }
        
        return 1
    
    @step(HTPExportStep.TAGGER_CREATION)
    def write_tagger_creation(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record tagger creation details."""
        tagger_info = data.steps.get("tagger_creation", {})
        
        self.metadata["report"]["steps"]["node_tagger_creation"] = {
            "tagger_created": True,
            "model_root_tag": tagger_info.get("root_tag", "/Model"),
            "operation_fallback": "disabled",
            "timestamp": data.timestamp
        }
        
        return 1
    
    @step(HTPExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record complete tagging results."""
        self.metadata["nodes"] = data.tagged_nodes.copy()
        
        stats = data.tagging_stats
        
        # Build top nodes list
        tag_counts = Counter(data.tagged_nodes.values())
        top_nodes = [
            {
                "rank": i + 1,
                "tag": tag,
                "count": count
            }
            for i, (tag, count) in enumerate(tag_counts.most_common(Config.TOP_NODES_COUNT))
        ]
        
        self.metadata["report"]["steps"]["node_tagging"] = {
            "total_nodes": data.total_nodes,
            "tagged_nodes": len(data.tagged_nodes),
            "coverage_percentage": data.coverage,
            "tagging_statistics": {
                "direct_matches": stats.get("direct_matches", 0),
                "direct_percentage": round(stats.get("direct_matches", 0) / data.total_nodes * 100, 1) if data.total_nodes > 0 else 0,
                "parent_matches": stats.get("parent_matches", 0),
                "parent_percentage": round(stats.get("parent_matches", 0) / data.total_nodes * 100, 1) if data.total_nodes > 0 else 0,
                "root_fallbacks": stats.get("root_fallbacks", 0),
                "root_percentage": round(stats.get("root_fallbacks", 0) / data.total_nodes * 100, 1) if data.total_nodes > 0 else 0,
                "empty_tags": stats.get("empty_tags", 0)
            },
            "top_nodes_by_hierarchy": top_nodes,
            "timestamp": data.timestamp
        }
        
        return 1
    
    @step(HTPExportStep.SAVE)
    def write_save(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record model save details."""
        self.metadata["report"]["steps"]["model_save"] = {
            "output_path": data.output_path,
            "hierarchy_attributes_embedded": data.embed_hierarchy_attributes,
            "file_saved": True,
            "timestamp": data.timestamp
        }
        
        return 1
    
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record export completion with full statistics."""
        self.metadata["export_context"]["export_time_seconds"] = round(data.export_time, 2)
        
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
        
        if data.console_log_path:
            self.metadata["outputs"]["console_log"] = {
                "path": Path(data.console_log_path).name
            }
        
        # Complete report section
        self.metadata["report"]["steps"]["export_complete"] = {
            "export_time_seconds": data.export_time,
            "export_statistics": {
                "hierarchy_modules": len(data.hierarchy),
                "onnx_nodes": data.total_nodes,
                "tagged_nodes": len(data.tagged_nodes),
                "coverage": data.coverage
            },
            "output_files": {
                "onnx_model": Path(data.output_path).name if data.output_path else None,
                "metadata": Path(self.metadata_path).name,
                "report": Path(data.report_path).name if data.report_path else None,
                "console_log": Path(data.console_log_path).name if data.console_log_path else None
            },
            "timestamp": data.timestamp
        }
        
        return 1
    
    def flush(self) -> None:
        """Write metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)


# ============================================================================
# TEXT REPORT WRITER WITH COMPLETE CONSOLE OUTPUT
# ============================================================================

class HTPReportWriter(StepAwareWriter):
    """Full text report writer that captures ALL console output."""
    
    def __init__(self, output_path: str, console_buffer: io.StringIO = None):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.report_path = f"{self.output_path}{Config.REPORT_SUFFIX}"
        self.buffer = io.StringIO()
        self.console_buffer = console_buffer
        self._write_header()
    
    def _write_header(self):
        """Write report header."""
        self.buffer.write("=" * Config.SEPARATOR_LENGTH + "\n")
        self.buffer.write("HTP EXPORT FULL REPORT\n")
        self.buffer.write("=" * Config.SEPARATOR_LENGTH + "\n")
        self.buffer.write(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
    
    def flush(self):
        """Write the complete console output to report file."""
        # If we have console buffer, append its content (stripped of ANSI)
        if self.console_buffer:
            console_output = self.console_buffer.getvalue()
            plain_output = TextStyler.strip_ansi(console_output)
            self.buffer.write(plain_output)
        
        # Write to file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(self.buffer.getvalue())


# ============================================================================
# MAIN EXPORT MONITOR
# ============================================================================

class HTPExportMonitor:
    """Main orchestrator for HTP export monitoring."""
    
    def __init__(self, output_path: str, model_name: str = "", verbose: bool = True):
        self.output_path = output_path
        self.model_name = model_name
        self.verbose = verbose
        
        # Console output buffer for capturing
        self.console_buffer = io.StringIO()
        
        # Initialize writers
        self.console_writer = HTPConsoleWriter(
            verbose=verbose,
            capture_buffer=self.console_buffer
        )
        self.metadata_writer = HTPMetadataWriter(output_path)
        self.report_writer = HTPReportWriter(
            output_path,
            console_buffer=self.console_buffer
        )
        
        # Track timing
        self.start_time = time.time()
    
    def log_step(self, step: HTPExportStep, data: HTPExportData) -> None:
        """Log data for an export step to all writers."""
        # Update timing
        data.elapsed_time = time.time() - self.start_time
        
        # Write to all outputs
        self.console_writer.write(step, data)
        self.metadata_writer.write(step, data)
        self.report_writer.write(step, data)
    
    def get_console_output(self) -> str:
        """Get captured console output."""
        return self.console_buffer.getvalue()
    
    def get_metadata(self) -> dict[str, Any]:
        """Get current metadata."""
        return self.metadata_writer.metadata.copy()
    
    def finalize(self) -> dict[str, str]:
        """Finalize all outputs and return paths."""
        # Flush all writers
        self.console_writer.flush()
        self.metadata_writer.flush()
        self.report_writer.flush()
        
        # Save console log with ANSI codes
        console_log_path = f"{Path(self.output_path).with_suffix('').as_posix()}{Config.CONSOLE_LOG_SUFFIX}"
        with open(console_log_path, 'w', encoding='utf-8') as f:
            f.write(self.console_buffer.getvalue())
        
        return {
            "metadata_path": self.metadata_writer.metadata_path,
            "report_path": self.report_writer.report_path,
            "console_log_path": console_log_path
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    monitor = HTPExportMonitor("model.onnx", "bert-base", verbose=True)
    
    # Create sample data
    data = HTPExportData(
        model_name="bert-base",
        model_class="BertModel",
        total_modules=48,
        total_parameters=4385536,
        output_path="model.onnx",
        hierarchy={"": {"class_name": "BertModel"}},
        execution_steps=36,
        total_nodes=136,
        tagged_nodes={f"node_{i}": "/BertModel" for i in range(136)},
        tagging_stats={
            "direct_matches": 83,
            "parent_matches": 34,
            "root_fallbacks": 19
        },
        coverage=100.0,
        export_time=2.35,
        onnx_size_mb=17.5
    )
    
    # Log all steps
    for step in HTPExportStep:
        monitor.log_step(step, data)
    
    # Finalize
    paths = monitor.finalize()
    print(f"\nOutputs saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")