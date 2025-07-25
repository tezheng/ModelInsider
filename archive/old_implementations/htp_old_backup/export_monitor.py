"""
HTP Export Monitor - Simplified with Rich Library

This module provides monitoring for the HTP (Hierarchical Tracing and Projection)
export process using Rich library for all console output.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

from rich.console import Console
from rich.text import Text
from rich.tree import Tree

# ============================================================================
# CONFIGURATION
# ============================================================================


class HTPExportMonitorConfig:
    """Configuration constants for HTP Export Monitor.

    TODO: Current config implementation is too aggressive. Many strings that are
    used only once might not need to be configured. Consider refactoring to only
    extract frequently used strings, strings that might change, or strings that
    would benefit from centralization (e.g., for internationalization).
    """

    # Display settings
    TOP_NODES_COUNT = 20
    SEPARATOR_LENGTH = 80
    MAX_HIERARCHY_LINES = 30  # Maximum lines to display in hierarchy trees
    TOP_NODES_SEPARATOR_LENGTH = 30
    SECTION_SEPARATOR_LENGTH = 60

    # Console settings
    CONSOLE_WIDTH = 120
    HEADER_SEPARATOR = "="
    SECTION_SEPARATOR = "-"
    TOTAL_STEPS = 6

    # File suffixes
    METADATA_SUFFIX = "_htp_metadata.json"
    REPORT_SUFFIX = "_htp_export_report.txt"

    # Numeric constants
    MILLION = 1e6
    PERCENT = 100.0
    DEFAULT_OPSET_VERSION = 17

    # Report settings
    TOP_OPERATIONS_COUNT = 10

    # Formatting settings
    INDENT_SPACES = 3

    # Emojis for steps
    EMOJI_MODEL_PREP = "📋"
    EMOJI_INPUT_GEN = "🔧"
    EMOJI_HIERARCHY = "🏗️"
    EMOJI_ONNX_EXPORT = "📦"
    EMOJI_NODE_TAGGING = "🔗"
    EMOJI_TAG_INJECTION = "🏷️"

    # Other emojis
    EMOJI_LAUNCH = "🚀"
    EMOJI_CALENDAR = "📅"
    EMOJI_RELOAD = "🔄"
    EMOJI_TARGET = "🎯"
    EMOJI_SUCCESS = "✅"
    EMOJI_WARNING = "⚠️"
    EMOJI_INFO = "📝"
    EMOJI_ROBOT = "🤖"
    EMOJI_SEARCH = "🔍"
    EMOJI_TREE = "🌳"
    EMOJI_CHART = "📊"
    EMOJI_CHART_UP = "📈"
    EMOJI_CONFIG = "🔧"
    EMOJI_INBOX = "📥"
    EMOJI_OUTBOX = "📤"
    EMOJI_SAVE = "💾"
    EMOJI_FILE = "📁"

    # Step titles
    TITLE_MODEL_PREP = "MODEL PREPARATION"
    TITLE_INPUT_GEN = "INPUT GENERATION"
    TITLE_HIERARCHY = "HIERARCHY BUILDING"
    TITLE_ONNX_EXPORT = "ONNX EXPORT"
    TITLE_NODE_TAGGING = "ONNX NODE TAGGING"
    TITLE_TAG_INJECTION = "TAG INJECTION"

    # Text messages
    MSG_HTP_EXPORT_PROCESS = "HTP ONNX EXPORT PROCESS"
    MSG_EXPORT_TIME = "Export Time"
    MSG_LOADING_MODEL = "Loading model and exporting"
    MSG_STRATEGY_HTP = "HTP"
    MSG_STRATEGY_DESC = "(Hierarchical Tracing and Projection)"
    MSG_STRATEGY_DISABLED = "DISABLED"
    MSG_CLEAN_ONNX = "(--clean-onnx)"
    MSG_MODEL_LOADED = "Model loaded"
    MSG_MODULES = "modules"
    MSG_PARAMETERS = "parameters"
    MSG_EXPORT_TARGET = "Export target"
    MSG_EVAL_MODE = "Model set to evaluation mode"
    MSG_PROVIDED_INPUTS = "Using provided input specifications"
    MSG_AUTO_GEN_INPUTS = "Auto-generating inputs for"
    MSG_MODEL_TYPE = "Model type"
    MSG_DETECTED_TASK = "Detected task"
    MSG_GENERATED_INPUTS = "Generated inputs"
    MSG_SHAPE = "shape"
    MSG_DTYPE = "dtype"
    MSG_TRACING_EXECUTION = "Tracing module execution with dummy inputs..."
    MSG_CAPTURED_MODULES = "Captured"
    MSG_MODULES_IN_HIERARCHY = "modules in hierarchy"
    MSG_TOTAL_EXEC_STEPS = "Total execution steps"
    MSG_MODULE_HIERARCHY = "Module Hierarchy"
    MSG_EXPORT_CONFIG = "Export configuration"
    MSG_OPSET_VERSION = "Opset version"
    MSG_CONSTANT_FOLDING = "Constant folding"
    MSG_INPUT_NAMES = "Input names"
    MSG_OUTPUT_NAMES = "Output names"
    MSG_OUTPUT_NAMES_WARNING = "Not detected"
    MSG_OUTPUT_NAMES_NOTE = "(model may not have named outputs)"
    MSG_ONNX_EXPORTED = "ONNX model exported successfully"
    MSG_MODEL_SIZE = "Model size"
    MSG_NODE_TAGGING_COMPLETE = "Node tagging completed successfully"
    MSG_COVERAGE = "Coverage"
    MSG_TAGGED_NODES = "Tagged nodes"
    MSG_EMPTY_TAGS = "Empty tags"
    MSG_TOP_NODES = "Top"
    MSG_NODES_BY_HIERARCHY = "Nodes by Hierarchy"
    MSG_NODES = "nodes"
    MSG_DIRECT_MATCHES = "Direct matches"
    MSG_PARENT_MATCHES = "Parent matches"
    MSG_ROOT_FALLBACKS = "Root fallbacks"
    MSG_COMPLETE_HIERARCHY = "Complete HF Hierarchy with ONNX Nodes"
    MSG_ONNX_NODES = "ONNX nodes"
    MSG_OPS = "ops"
    MSG_INJECTING_TAGS = "Injecting hierarchy tags into ONNX model..."
    MSG_TAGS_EMBEDDED = "Tags successfully embedded as node attributes"
    MSG_TAG_INJECTION_SKIPPED = "Hierarchy tag injection skipped (--clean-onnx mode)"
    MSG_MODEL_SAVED = "Model saved to"
    MSG_EXPORT_COMPLETE = "EXPORT COMPLETE"
    MSG_EXPORT_SUMMARY = "Export Summary"
    MSG_TOTAL_TIME = "Total time"
    MSG_HIERARCHY_MODULES = "Hierarchy modules"
    MSG_OUTPUT_FILES = "Output files"
    MSG_ONNX_MODEL = "ONNX model"
    MSG_METADATA = "Metadata"
    MSG_REPORT = "Report"
    MSG_LINES_TRUNCATED = "... showing first"
    MSG_LINES = "lines"
    MSG_TRUNCATED_NOTE = "truncated for console"
    MSG_TRUE = "True"
    MSG_FALSE = "False"

    # Report messages
    MSG_REPORT_HEADER = "HTP ONNX EXPORT REPORT"
    MSG_TIMESTAMP = "Timestamp"
    MSG_MODEL = "Model"
    MSG_OUTPUT = "Output"
    MSG_STEP = "STEP"
    MSG_MODEL_CLASS = "Model Class"
    MSG_TOTAL_MODULES = "Total Modules"
    MSG_TOTAL_PARAMETERS = "Total Parameters"
    MSG_CAPTURED_MODULES_REPORT = "Captured Modules"
    MSG_EXECUTION_STEPS = "Execution Steps"
    MSG_TOTAL_ONNX_NODES = "Total ONNX Nodes"
    MSG_TOP_OPERATIONS = "Top Operations"
    MSG_EXPORT_TIME_REPORT = "Export Time"
    MSG_EMBED_HIERARCHY = "Embed Hierarchy"
    MSG_FAILED_WRITE_REPORT = "Failed to write report"
    MSG_FAILED_WRITE_METADATA = "Failed to write metadata"


# ============================================================================
# DATA MODELS
# ============================================================================


class HTPExportStep(Enum):
    """6-step export process (removed TAGGER_CREATION)."""

    MODEL_PREP = "model_preparation"  # Step 1
    INPUT_GEN = "input_generation"  # Step 2
    HIERARCHY = "hierarchy_building"  # Step 3
    ONNX_EXPORT = "onnx_export"  # Step 4
    NODE_TAGGING = "node_tagging"  # Step 5
    TAG_INJECTION = "tag_injection"  # Step 6


@dataclass
class HTPExportData:
    """Container for export data."""

    # Model info
    model_name: str = ""
    model_class: str = ""
    total_modules: int = 0
    total_parameters: int = 0

    # Export config
    output_path: str = ""
    embed_hierarchy: bool = True

    # Step data
    step_data: dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: float = field(default_factory=time.time)
    export_time: float = 0.0


# ============================================================================
# EXPORT MONITOR
# ============================================================================


class HTPExportMonitor:
    """Simplified HTP export monitor using Rich library."""

    # Step metadata
    STEP_INFO: ClassVar[dict[HTPExportStep, tuple[str, str]]] = {
        HTPExportStep.MODEL_PREP: (
            HTPExportMonitorConfig.EMOJI_MODEL_PREP,
            HTPExportMonitorConfig.TITLE_MODEL_PREP,
        ),
        HTPExportStep.INPUT_GEN: (
            HTPExportMonitorConfig.EMOJI_INPUT_GEN,
            HTPExportMonitorConfig.TITLE_INPUT_GEN,
        ),
        HTPExportStep.HIERARCHY: (
            HTPExportMonitorConfig.EMOJI_HIERARCHY,
            HTPExportMonitorConfig.TITLE_HIERARCHY,
        ),
        HTPExportStep.ONNX_EXPORT: (
            HTPExportMonitorConfig.EMOJI_ONNX_EXPORT,
            HTPExportMonitorConfig.TITLE_ONNX_EXPORT,
        ),
        HTPExportStep.NODE_TAGGING: (
            HTPExportMonitorConfig.EMOJI_NODE_TAGGING,
            HTPExportMonitorConfig.TITLE_NODE_TAGGING,
        ),
        HTPExportStep.TAG_INJECTION: (
            HTPExportMonitorConfig.EMOJI_TAG_INJECTION,
            HTPExportMonitorConfig.TITLE_TAG_INJECTION,
        ),
    }

    # ========================================================================
    # STYLING UTILITIES
    # ========================================================================

    @staticmethod
    def _bright_cyan(text: str) -> str:
        """Format text in bright cyan."""
        return f"[bold cyan]{text}[/bold cyan]"

    @staticmethod
    def _bright_green(text: str) -> str:
        """Format text in bright green."""
        return f"[bold green]{text}[/bold green]"

    @staticmethod
    def _bright_red(text: str) -> str:
        """Format text in bright red."""
        return f"[bold red]{text}[/bold red]"

    @staticmethod
    def _bright_magenta(text: str) -> str:
        """Format text in bright magenta."""
        return f"[bold magenta]{text}[/bold magenta]"

    @staticmethod
    def _bright_yellow(text: str) -> str:
        """Format text in bright yellow."""
        return f"[bold yellow]{text}[/bold yellow]"

    @staticmethod
    def _dim(text: str) -> str:
        """Format text in dim style."""
        return f"[dim]{text}[/dim]"

    @staticmethod
    def _bold(text: str) -> str:
        """Format text in bold."""
        return f"[bold]{text}[/bold]"

    def __init__(
        self,
        output_path: str,
        model_name: str = "",
        verbose: bool = True,
        enable_report: bool = True,
        embed_hierarchy: bool = True,
    ):
        """Initialize monitor."""
        self.output_path = output_path
        self.model_name = model_name
        self.verbose = verbose
        self.enable_report = enable_report
        self.embed_hierarchy = embed_hierarchy

        # Console setup - use wider width to avoid line wrapping in tree
        # Disable highlight to prevent automatic path coloring
        self.console = Console(
            width=HTPExportMonitorConfig.CONSOLE_WIDTH,
            force_terminal=True,
            legacy_windows=False,
            highlight=False,
        )

        # Data storage
        self.data = HTPExportData(
            model_name=model_name,
            output_path=output_path,
            embed_hierarchy=embed_hierarchy,
        )

        # Metadata for JSON output
        self.metadata = {
            "export_context": {},
            "steps": {},
            "hierarchy": {},
            "tagging": {},
        }

    def __enter__(self):
        """Context manager entry."""
        if self.verbose:
            self._print_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            # Success - always write metadata
            self._write_metadata()

            # Write report only if enabled
            if self.enable_report:
                self._write_report()

        # Always add empty line at the end for visual separation
        self.console.print()

    def update(self, step: HTPExportStep, **kwargs):
        """Update monitoring with step data."""
        # Store step data
        self.data.step_data[step.value] = kwargs

        # Update model data for specific steps
        if step == HTPExportStep.MODEL_PREP:
            self.data.model_class = kwargs.get("model_class", "")
            self.data.total_modules = kwargs.get("total_modules", 0)
            self.data.total_parameters = kwargs.get("total_parameters", 0)

        # Display step output
        if self.verbose:
            self._display_step(step, kwargs)

        # Update metadata
        self._update_metadata(step, kwargs)

    def finalize_export(self, export_time: float, output_path: str, **kwargs):
        """Finalize export with summary."""
        self.data.export_time = export_time

        if self.verbose:
            self._print_summary()

    # ========================================================================
    # DISPLAY METHODS
    # ========================================================================

    def _print_header(self):
        """Print export header."""
        self.console.print(
            "\n"
            + HTPExportMonitorConfig.HEADER_SEPARATOR
            * HTPExportMonitorConfig.SEPARATOR_LENGTH
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_LAUNCH} {self._bright_cyan(HTPExportMonitorConfig.MSG_HTP_EXPORT_PROCESS)}"
        )
        self.console.print(
            HTPExportMonitorConfig.HEADER_SEPARATOR
            * HTPExportMonitorConfig.SEPARATOR_LENGTH
        )

        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_CALENDAR} {HTPExportMonitorConfig.MSG_EXPORT_TIME}: {self._bright_green(timestamp)}"
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_RELOAD} {HTPExportMonitorConfig.MSG_LOADING_MODEL}: {self._bright_magenta(self.model_name)}"
        )

        # Strategy info
        if self.embed_hierarchy:
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_TARGET} Strategy: {self._bright_cyan(HTPExportMonitorConfig.MSG_STRATEGY_HTP)} "
                f"{HTPExportMonitorConfig.MSG_STRATEGY_DESC}"
            )
        else:
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_TARGET} Strategy: {self._bright_cyan(HTPExportMonitorConfig.MSG_STRATEGY_HTP)} "
                f"{HTPExportMonitorConfig.MSG_STRATEGY_DESC} - "
                f"{self._bright_red(HTPExportMonitorConfig.MSG_STRATEGY_DISABLED)} {HTPExportMonitorConfig.MSG_CLEAN_ONNX}"
            )
        self.console.print(
            HTPExportMonitorConfig.HEADER_SEPARATOR
            * HTPExportMonitorConfig.SEPARATOR_LENGTH
        )

    def _display_step(self, step: HTPExportStep, data: dict):
        """Display step output."""
        icon, title = self.STEP_INFO[step]
        step_num = list(self.STEP_INFO.keys()).index(step) + 1

        # Step header
        self.console.print(
            f"\n{icon} {self._bold(f'{HTPExportMonitorConfig.MSG_STEP} {step_num}/{HTPExportMonitorConfig.TOTAL_STEPS}: {title}')}"
        )
        self.console.print(
            HTPExportMonitorConfig.SECTION_SEPARATOR
            * HTPExportMonitorConfig.SEPARATOR_LENGTH
        )

        # Step-specific display
        if step == HTPExportStep.MODEL_PREP:
            self._display_model_prep(data)
        elif step == HTPExportStep.INPUT_GEN:
            self._display_input_gen(data)
        elif step == HTPExportStep.HIERARCHY:
            self._display_hierarchy(data)
        elif step == HTPExportStep.ONNX_EXPORT:
            self._display_onnx_export(data)
        elif step == HTPExportStep.NODE_TAGGING:
            self._display_node_tagging(data)
        elif step == HTPExportStep.TAG_INJECTION:
            self._display_tag_injection(data)

    def _display_model_prep(self, data: dict):
        """Display model preparation step."""
        # Display
        # Format parameters to match baseline exactly (4.4M not 4.4M)
        params_m = self.data.total_parameters / HTPExportMonitorConfig.MILLION
        if params_m == int(params_m):
            params_str = f"{int(params_m)}"
        else:
            params_str = f"{params_m:.1f}"

        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {HTPExportMonitorConfig.MSG_MODEL_LOADED}: {self.data.model_class} "
            f"({self._bright_cyan(str(self.data.total_modules))} {HTPExportMonitorConfig.MSG_MODULES}, "
            f"{self._bright_cyan(params_str + 'M')} {HTPExportMonitorConfig.MSG_PARAMETERS})"
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_TARGET} {HTPExportMonitorConfig.MSG_EXPORT_TARGET}: {self._bright_magenta(self.output_path)}"
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {HTPExportMonitorConfig.MSG_EVAL_MODE}"
        )

    def _display_input_gen(self, data: dict):
        """Display input generation step."""
        method = data.get("method", "auto_generated")

        if method == "provided":
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_INFO} {HTPExportMonitorConfig.MSG_PROVIDED_INPUTS}"
            )
        else:
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_ROBOT} {HTPExportMonitorConfig.MSG_AUTO_GEN_INPUTS}: {self.model_name}"
            )
            if "model_type" in data:
                self.console.print(
                    f"   • {HTPExportMonitorConfig.MSG_MODEL_TYPE}: {self._bright_green(data['model_type'])}"
                )  # Green for config
            if "task" in data:
                self.console.print(
                    f"   • {HTPExportMonitorConfig.MSG_DETECTED_TASK}: {self._bright_green(data['task'])}"
                )  # Green for config

        # Display input details
        if "inputs" in data:
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {HTPExportMonitorConfig.MSG_GENERATED_INPUTS}:"
            )
            for name, info in data["inputs"].items():
                shape_str = str(info["shape"])
                dtype_str = info["dtype"]
                self.console.print(
                    f"   • {name}: {HTPExportMonitorConfig.MSG_SHAPE}={self._bright_green(shape_str)}, {HTPExportMonitorConfig.MSG_DTYPE}={self._bright_green(dtype_str)}"
                )  # Green for values

    def _display_hierarchy(self, data: dict):
        """Display hierarchy building step using Rich Tree."""
        hierarchy = data.get("hierarchy", {})
        execution_steps = data.get("execution_steps", 0)

        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SEARCH} {HTPExportMonitorConfig.MSG_TRACING_EXECUTION}"
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {HTPExportMonitorConfig.MSG_CAPTURED_MODULES} {self._bright_cyan(str(len(hierarchy)))} {HTPExportMonitorConfig.MSG_MODULES_IN_HIERARCHY}"
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_CHART} {HTPExportMonitorConfig.MSG_TOTAL_EXEC_STEPS}: {self._bright_cyan(str(execution_steps))}"
        )

        # Build and display hierarchy tree
        if hierarchy:
            self.console.print(
                f"\n{HTPExportMonitorConfig.EMOJI_TREE} {HTPExportMonitorConfig.MSG_MODULE_HIERARCHY}:"
            )
            tree = self._build_hierarchy_tree(hierarchy)
            self._display_truncated_tree(tree)

    def _build_hierarchy_tree(self, hierarchy: dict) -> Tree:
        """Build Rich Tree from hierarchy data."""
        # Find root
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")

        # Create tree
        tree = Tree(self._bold(root_name))

        # Build tree recursively
        def add_children(parent_node, parent_path: str):
            children = self._find_immediate_children(parent_path, hierarchy)

            for child_path in children:
                child_info = hierarchy.get(child_path, {})
                class_name = child_info.get("class_name", "Unknown")

                # Create node text with simplified styling
                # Instead of inline markup, use Text object styling
                node_text = Text()

                # Display format: ClassName: path
                node_text.append(class_name, style="bold")
                node_text.append(": ", style="white")
                node_text.append(child_path, style="dim")

                # Add node to tree
                child_node = parent_node.add(node_text)

                # Recursively add children
                add_children(child_node, child_path)

        # Start building from root
        add_children(tree, "")

        return tree

    def _find_immediate_children(self, parent_path: str, hierarchy: dict) -> list:
        """Find immediate children of a path."""
        if parent_path == "":
            # Root case
            return sorted([p for p in hierarchy if p and "." not in p])

        # Non-root case
        prefix = parent_path + "."
        immediate = []

        for path in hierarchy:
            if not path.startswith(prefix) or path == parent_path:
                continue

            suffix = path[len(prefix) :]

            # Check if immediate child
            if "." not in suffix:
                immediate.append(path)
            elif suffix.count(".") == 1 and suffix.split(".")[1].isdigit():
                # Compound pattern like layer.0
                immediate.append(path)

        # Custom sort
        def sort_key(path):
            parts = path.split(".")
            result = []
            for part in parts:
                if part.isdigit():
                    result.append((0, int(part)))
                else:
                    result.append((1, part))
            return result

        return sorted(immediate, key=sort_key)

    def _build_truncated_tree(
        self, source_tree: Tree, target_tree: Tree, max_lines: int
    ) -> int:
        """Build a truncated version of the tree that fits within max_lines."""
        line_count = 1  # Start with root

        # Helper to add nodes up to limit
        def add_nodes_to_limit(source_children, target_parent, current_count):
            count = current_count
            for child in source_children:
                if count >= max_lines:
                    break
                # Add this child
                target_child = target_parent.add(child.label)
                count += 1

                # Try to add its children
                if hasattr(child, "children") and child.children and count < max_lines:
                    count = add_nodes_to_limit(child.children, target_child, count)
            return count

        # Add nodes from source to target
        if hasattr(source_tree, "children") and source_tree.children:
            line_count = add_nodes_to_limit(
                source_tree.children, target_tree, line_count
            )

        return line_count

    def _display_onnx_export(self, data: dict):
        """Display ONNX export step."""
        opset = data.get("opset_version", HTPExportMonitorConfig.DEFAULT_OPSET_VERSION)
        folding = data.get("do_constant_folding", True)
        size_mb = data.get("onnx_size_mb", 0)

        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_CONFIG} {HTPExportMonitorConfig.MSG_EXPORT_CONFIG}:"
        )
        self.console.print(
            f"   • {HTPExportMonitorConfig.MSG_OPSET_VERSION}: {self._bright_green(str(opset))}"
        )  # Green for config value
        self.console.print(
            f"   • {HTPExportMonitorConfig.MSG_CONSTANT_FOLDING}: {self._format_bool(folding)}"
        )

        if "input_names" in data:
            # Format array items with green color
            input_names = data["input_names"]
            formatted_inputs = (
                "["
                + ", ".join(
                    f"{self._bright_green(f"'{name}'")}" for name in input_names
                )
                + "]"
            )
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_INBOX} {HTPExportMonitorConfig.MSG_INPUT_NAMES}: {formatted_inputs}"
            )

        if "output_names" in data:
            output_names = data["output_names"]
            if output_names:
                # Format array items with green color
                formatted_outputs = (
                    "["
                    + ", ".join(
                        f"{self._bright_green(f"'{name}'")}" for name in output_names
                    )
                    + "]"
                )
                self.console.print(
                    f"{HTPExportMonitorConfig.EMOJI_OUTBOX} {HTPExportMonitorConfig.MSG_OUTPUT_NAMES}: {formatted_outputs}"
                )
            else:
                # Log warning when output names are not available
                self.console.print(
                    f"{HTPExportMonitorConfig.EMOJI_WARNING}  {HTPExportMonitorConfig.MSG_OUTPUT_NAMES}: {self._bright_yellow(HTPExportMonitorConfig.MSG_OUTPUT_NAMES_WARNING)} {HTPExportMonitorConfig.MSG_OUTPUT_NAMES_NOTE}"
                )

        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {HTPExportMonitorConfig.MSG_ONNX_EXPORTED}"
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_ONNX_EXPORT} {HTPExportMonitorConfig.MSG_MODEL_SIZE}: {self._bright_cyan(f'{size_mb:.2f}MB')}"
        )

    def _display_node_tagging(self, data: dict):
        """Display node tagging step."""
        total_nodes = data.get("total_nodes", 0)
        tagged_nodes = data.get("tagged_nodes", {})
        coverage = data.get("coverage", 0.0)
        tagging_stats = data.get("tagging_stats", {})

        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {HTPExportMonitorConfig.MSG_NODE_TAGGING_COMPLETE}"
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_CHART_UP} {HTPExportMonitorConfig.MSG_COVERAGE}: {self._bright_cyan(f'{coverage:.1f}%')}"
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_CHART} {HTPExportMonitorConfig.MSG_TAGGED_NODES}: {self._bright_cyan(str(len(tagged_nodes)))}/{self._bright_cyan(str(total_nodes))}"
        )

        # Display tagging statistics
        self._display_tagging_statistics(tagging_stats, total_nodes)

        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {HTPExportMonitorConfig.MSG_EMPTY_TAGS}: {self._bright_cyan('0')}"
        )

        # Display nodes by hierarchy
        self._display_nodes_by_hierarchy(tagged_nodes)

        # Display hierarchy tree with node counts
        hierarchy = data.get("hierarchy")
        if hierarchy and tagged_nodes:
            self.console.print(
                f"\n{HTPExportMonitorConfig.EMOJI_TREE} {HTPExportMonitorConfig.MSG_COMPLETE_HIERARCHY}:"
            )
            self.console.print(
                HTPExportMonitorConfig.SECTION_SEPARATOR
                * HTPExportMonitorConfig.SECTION_SEPARATOR_LENGTH
            )

            # Build a tree that includes node counts
            tree = self._build_hierarchy_tree_with_counts(hierarchy, tagged_nodes)
            self._display_truncated_tree(tree)

    def _display_tagging_statistics(self, tagging_stats: dict, total_nodes: int):
        """Display tagging statistics with percentages."""
        if not tagging_stats:
            return

        direct = tagging_stats.get("direct_matches", 0)
        parent = tagging_stats.get("parent_matches", 0)
        root = tagging_stats.get("root_fallbacks", 0)

        if total_nodes > 0:
            direct_pct = self._calculate_percentage(direct, total_nodes)
            parent_pct = self._calculate_percentage(parent, total_nodes)
            root_pct = self._calculate_percentage(root, total_nodes)
            self.console.print(
                f"   • {HTPExportMonitorConfig.MSG_DIRECT_MATCHES}: {self._bright_cyan(str(direct))} ({self._bright_cyan(f'{direct_pct:.1f}%')})"
            )
            self.console.print(
                f"   • {HTPExportMonitorConfig.MSG_PARENT_MATCHES}: {self._bright_cyan(str(parent))} ({self._bright_cyan(f'{parent_pct:.1f}%')})"
            )
            self.console.print(
                f"   • {HTPExportMonitorConfig.MSG_ROOT_FALLBACKS}: {self._bright_cyan(str(root))} ({self._bright_cyan(f'{root_pct:.1f}%')})"
            )

    def _display_nodes_by_hierarchy(self, tagged_nodes: dict):
        """Display top nodes grouped by hierarchy."""
        if not tagged_nodes:
            return

        from collections import Counter

        # Count nodes by hierarchy tag
        tag_counts = Counter(tagged_nodes.values())

        self.console.print(
            f"\n{HTPExportMonitorConfig.EMOJI_CHART} {HTPExportMonitorConfig.MSG_TOP_NODES} {self._bright_cyan(str(min(len(tag_counts), HTPExportMonitorConfig.TOP_NODES_COUNT)))} {HTPExportMonitorConfig.MSG_NODES_BY_HIERARCHY}:"
        )
        self.console.print(
            HTPExportMonitorConfig.SECTION_SEPARATOR
            * HTPExportMonitorConfig.TOP_NODES_SEPARATOR_LENGTH
        )

        sorted_tags = tag_counts.most_common(HTPExportMonitorConfig.TOP_NODES_COUNT)
        for i, (tag, count) in enumerate(sorted_tags):
            # Simple display without path/class splitting
            self.console.print(
                f" {i + 1:2d}. {tag}: {self._bright_cyan(str(count))} {HTPExportMonitorConfig.MSG_NODES}"
            )

    def _build_hierarchy_tree_with_counts(
        self, hierarchy: dict, tagged_nodes: dict
    ) -> Tree:
        """Build Rich Tree with node counts from tagged nodes."""

        # Count nodes per hierarchy path
        node_counts, nodes_by_module = self._count_nodes_by_hierarchy(tagged_nodes)

        # Find root
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        root_tag = root_info.get("traced_tag", "/Model")
        root_count = node_counts.get(root_tag, 0)

        # Create tree with count
        tree = Tree(
            f"{self._bold(root_name)} ({self._bright_cyan(str(root_count))} {HTPExportMonitorConfig.MSG_ONNX_NODES})"
        )

        # Build tree recursively with counts
        self._add_hierarchy_children_with_counts(
            tree, "", root_tag, hierarchy, node_counts, nodes_by_module
        )

        return tree

    def _count_nodes_by_hierarchy(self, tagged_nodes: dict) -> tuple[dict, dict]:
        """Count nodes per hierarchy path and group by module."""
        from collections import defaultdict

        node_counts = defaultdict(int)
        nodes_by_module = defaultdict(list)  # module_tag -> [(node_name, simple_name)]

        for node_name, tag in tagged_nodes.items():
            # Count nodes for each level of the hierarchy
            parts = tag.split("/")
            for i in range(1, len(parts) + 1):
                prefix = "/".join(parts[:i])
                if prefix:
                    node_counts[prefix] += 1

            # Store node with its module
            nodes_by_module[tag].append(node_name)

        return node_counts, nodes_by_module

    def _add_hierarchy_children_with_counts(
        self,
        parent_node,
        parent_path: str,
        parent_tag: str,
        hierarchy: dict,
        node_counts: dict,
        nodes_by_module: dict,
    ):
        """Add children to hierarchy tree with node counts."""
        children = self._find_immediate_children(parent_path, hierarchy)

        for child_path in children:
            child_info = hierarchy.get(child_path, {})
            class_name = child_info.get("class_name", "Unknown")
            child_tag = child_info.get("traced_tag", "")
            child_count = node_counts.get(child_tag, 0)

            # Create node text with count and proper styling
            node_text = self._create_hierarchy_node_text(
                class_name, child_path, child_count
            )

            # Add node to tree
            child_node = parent_node.add(node_text)

            # Add ONNX operation nodes under this module
            if child_tag in nodes_by_module:
                self._add_onnx_operations_to_node(
                    child_node, nodes_by_module[child_tag]
                )

            # Recursively add children
            self._add_hierarchy_children_with_counts(
                child_node,
                child_path,
                child_tag,
                hierarchy,
                node_counts,
                nodes_by_module,
            )

    def _create_hierarchy_node_text(
        self, class_name: str, path: str, count: int
    ) -> Text:
        """Create styled text for hierarchy node."""
        node_text = Text()
        node_text.append(f"{class_name}", style="bold")
        node_text.append(": ", style="")
        node_text.append(path, style="dim")  # Gray/dim style for path
        node_text.append(" (", style="")
        node_text.append(str(count), style="bold cyan")  # Cyan for count
        node_text.append(f" {HTPExportMonitorConfig.MSG_NODES})", style="")
        return node_text

    def _add_onnx_operations_to_node(self, parent_node, node_names: list):
        """Add ONNX operations under a module node."""
        from collections import defaultdict

        # Group nodes by their base operation type
        ops_grouped = defaultdict(list)
        for node_name in sorted(node_names):
            op_type = self._extract_operation_type(node_name)
            ops_grouped[op_type].append(node_name)

        # Display grouped operations
        for op_type in sorted(ops_grouped.keys()):
            op_nodes = ops_grouped[op_type]
            if len(op_nodes) > 1:
                # Multiple operations of same type - show count
                op_text = Text()
                op_text.append(op_type, style="bold")
                op_text.append(" (", style="")
                op_text.append(str(len(op_nodes)), style="bold cyan")
                op_text.append(f" {HTPExportMonitorConfig.MSG_OPS})", style="")
                parent_node.add(op_text)
            else:
                # Single operation - show the simple name
                node_name = op_nodes[0]
                simple_name = node_name.split("/")[-1]  # Just the last part
                op_text = Text()
                op_text.append(simple_name, style="bold")
                op_text.append(": ", style="")
                op_text.append(node_name, style="dim")  # Full path in dim
                parent_node.add(op_text)

    def _extract_operation_type(self, node_name: str) -> str:
        """Extract operation type from node name."""
        # Extract base operation name (e.g., "/embeddings/Add_0" -> "Add")
        # Handle both simple names like "Add_0" and paths like "/embeddings/Add_0"
        base_name = node_name.split("/")[-1]  # Get last part of path
        if "_" in base_name:
            return base_name.split("_")[0]
        else:
            # For names without underscore, try to extract operation type
            # e.g., "LayerNormalization" -> "LayerNormalization"
            return base_name

    def _display_tag_injection(self, data: dict):
        """Display tag injection step."""
        if self.embed_hierarchy:
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_TAG_INJECTION} {HTPExportMonitorConfig.MSG_INJECTING_TAGS}"
            )
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {HTPExportMonitorConfig.MSG_TAGS_EMBEDDED}"
            )
        else:
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_WARNING} {HTPExportMonitorConfig.MSG_TAG_INJECTION_SKIPPED}"
            )

        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SAVE} {HTPExportMonitorConfig.MSG_MODEL_SAVED}: {self._bright_magenta(self.output_path)}"
        )

    def _print_summary(self):
        """Print export summary."""
        self.console.print(
            "\n"
            + HTPExportMonitorConfig.HEADER_SEPARATOR
            * HTPExportMonitorConfig.SEPARATOR_LENGTH
        )
        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_SUCCESS} {self._bright_green(HTPExportMonitorConfig.MSG_EXPORT_COMPLETE)}"
        )
        self.console.print(
            HTPExportMonitorConfig.HEADER_SEPARATOR
            * HTPExportMonitorConfig.SEPARATOR_LENGTH
        )

        # Summary stats
        total_time = self.data.export_time
        hierarchy_data = self.data.step_data.get(HTPExportStep.HIERARCHY.value, {})
        tagging_data = self.data.step_data.get(HTPExportStep.NODE_TAGGING.value, {})

        modules = len(hierarchy_data.get("hierarchy", {}))
        nodes = tagging_data.get("total_nodes", 0)
        tagged = len(tagging_data.get("tagged_nodes", {}))
        coverage = tagging_data.get("coverage", 0.0)

        self.console.print(
            f"{HTPExportMonitorConfig.EMOJI_CHART} {HTPExportMonitorConfig.MSG_EXPORT_SUMMARY}:"
        )
        self.console.print(
            f"   • {HTPExportMonitorConfig.MSG_TOTAL_TIME}: {self._bright_cyan(f'{total_time:.2f}s')}"
        )
        self.console.print(
            f"   • {HTPExportMonitorConfig.MSG_HIERARCHY_MODULES}: {self._bright_cyan(str(modules))}"
        )
        self.console.print(
            f"   • {HTPExportMonitorConfig.MSG_ONNX_NODES}: {self._bright_cyan(str(nodes))}"
        )
        self.console.print(
            f"   • {HTPExportMonitorConfig.MSG_TAGGED_NODES}: {self._bright_cyan(str(tagged))} "
            f"({self._bright_cyan(f'{coverage:.1f}%')} {HTPExportMonitorConfig.MSG_COVERAGE.lower()})"
        )

        self.console.print(
            f"\n{HTPExportMonitorConfig.EMOJI_FILE} {HTPExportMonitorConfig.MSG_OUTPUT_FILES}:"
        )
        self.console.print(
            f"   • {HTPExportMonitorConfig.MSG_ONNX_MODEL}: {self._bright_magenta(self.output_path)}"
        )

        # Metadata is always written
        base_path = Path(self.output_path).with_suffix("")
        metadata_path = f"{base_path}{HTPExportMonitorConfig.METADATA_SUFFIX}"
        self.console.print(
            f"   • {HTPExportMonitorConfig.MSG_METADATA}: {self._bright_magenta(metadata_path)}"
        )

        if self.enable_report:
            report_path = f"{base_path}{HTPExportMonitorConfig.REPORT_SUFFIX}"
            self.console.print(
                f"   • {HTPExportMonitorConfig.MSG_REPORT}: {self._bright_magenta(report_path)}"
            )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _format_bool(self, value: bool) -> str:
        """Format boolean with color."""
        if value:
            return self._bright_green(HTPExportMonitorConfig.MSG_TRUE)
        else:
            return self._bright_red(HTPExportMonitorConfig.MSG_FALSE)

    def _create_report_console(self, file=None) -> Console:
        """Create a console for report generation."""
        return Console(
            file=file,
            width=HTPExportMonitorConfig.CONSOLE_WIDTH,
            force_terminal=False,  # No ANSI codes in reports
            legacy_windows=False,
            highlight=False,
        )

    def _calculate_percentage(self, part: float, total: float) -> float:
        """Calculate percentage safely."""
        return (part / total * HTPExportMonitorConfig.PERCENT) if total > 0 else 0.0

    def _build_output_path(self, suffix: str) -> str:
        """Build output path with given suffix."""
        base_path = Path(self.output_path).with_suffix("")
        return f"{base_path}{suffix}"

    def _display_truncated_tree(self, tree: Tree, max_lines: int | None = None) -> None:
        """Display a tree with optional truncation."""
        if max_lines is None:
            max_lines = HTPExportMonitorConfig.MAX_HIERARCHY_LINES

        # Create a temporary console to capture the tree output
        string_buffer = StringIO()
        temp_console = self._create_report_console(file=string_buffer)
        temp_console.print(tree)

        # Get lines and apply truncation
        lines = string_buffer.getvalue().strip().split("\n")
        if len(lines) <= max_lines:
            self.console.print(tree)
        else:
            # Create truncated tree
            truncated_tree = Tree(tree.label)
            self._build_truncated_tree(tree, truncated_tree, max_lines - 1)
            self.console.print(truncated_tree)
            self.console.print(
                f"{HTPExportMonitorConfig.MSG_LINES_TRUNCATED} {max_lines} {HTPExportMonitorConfig.MSG_LINES} ({self._bold(HTPExportMonitorConfig.MSG_TRUNCATED_NOTE)})"
            )

    def _update_metadata(self, step: HTPExportStep, data: dict):
        """Update metadata for JSON output."""
        self.metadata["steps"][step.value] = data

    def _write_report(self):
        """Write text report by capturing console output."""
        try:
            # Create a string buffer to capture output
            string_buffer = io.StringIO()
            report_console = self._create_report_console(file=string_buffer)

            # Write header
            report_console.print(
                HTPExportMonitorConfig.HEADER_SEPARATOR
                * HTPExportMonitorConfig.SEPARATOR_LENGTH
            )
            report_console.print(HTPExportMonitorConfig.MSG_REPORT_HEADER)
            report_console.print(
                HTPExportMonitorConfig.HEADER_SEPARATOR
                * HTPExportMonitorConfig.SEPARATOR_LENGTH
            )
            report_console.print(
                f"{HTPExportMonitorConfig.MSG_TIMESTAMP}: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            report_console.print(
                f"{HTPExportMonitorConfig.MSG_MODEL}: {self.data.model_name}"
            )
            report_console.print(
                f"{HTPExportMonitorConfig.MSG_OUTPUT}: {self.output_path}"
            )
            report_console.print(
                HTPExportMonitorConfig.HEADER_SEPARATOR
                * HTPExportMonitorConfig.SEPARATOR_LENGTH
            )

            # Write each step's data
            for step in HTPExportStep:
                if step.value in self.data.step_data:
                    step_data = self.data.step_data[step.value]
                    icon, title = self.STEP_INFO[step]
                    step_num = list(self.STEP_INFO.keys()).index(step) + 1

                    report_console.print(
                        f"\n{HTPExportMonitorConfig.MSG_STEP} {step_num}/{HTPExportMonitorConfig.TOTAL_STEPS}: {title}"
                    )
                    report_console.print(
                        HTPExportMonitorConfig.SECTION_SEPARATOR
                        * HTPExportMonitorConfig.SEPARATOR_LENGTH
                    )

                    # Step-specific content
                    if step == HTPExportStep.MODEL_PREP:
                        report_console.print(
                            f"{HTPExportMonitorConfig.MSG_MODEL_CLASS}: {self.data.model_class}"
                        )
                        report_console.print(
                            f"{HTPExportMonitorConfig.MSG_TOTAL_MODULES}: {self.data.total_modules}"
                        )
                        report_console.print(
                            f"{HTPExportMonitorConfig.MSG_TOTAL_PARAMETERS}: {self.data.total_parameters:,}"
                        )

                    elif step == HTPExportStep.HIERARCHY:
                        hierarchy = step_data.get("hierarchy", {})
                        report_console.print(
                            f"{HTPExportMonitorConfig.MSG_CAPTURED_MODULES_REPORT}: {len(hierarchy)}"
                        )
                        report_console.print(
                            f"{HTPExportMonitorConfig.MSG_EXECUTION_STEPS}: {step_data.get('execution_steps', 0)}"
                        )

                        # Include hierarchy tree
                        if hierarchy:
                            report_console.print(
                                f"\n{HTPExportMonitorConfig.MSG_MODULE_HIERARCHY}:"
                            )
                            tree = self._build_hierarchy_tree(hierarchy)
                            report_console.print(tree)

                    elif step == HTPExportStep.NODE_TAGGING:
                        report_console.print(
                            f"{HTPExportMonitorConfig.MSG_TOTAL_ONNX_NODES}: {step_data.get('total_nodes', 0)}"
                        )
                        report_console.print(
                            f"{HTPExportMonitorConfig.MSG_TAGGED_NODES}: {len(step_data.get('tagged_nodes', {}))}"
                        )
                        report_console.print(
                            f"{HTPExportMonitorConfig.MSG_COVERAGE}: {step_data.get('coverage', 0):.1f}%"
                        )

                        # Include operation counts
                        op_counts = step_data.get("op_counts", {})
                        if op_counts:
                            report_console.print(
                                f"\n{HTPExportMonitorConfig.MSG_TOP_OPERATIONS}:"
                            )
                            sorted_ops = sorted(
                                op_counts.items(), key=lambda x: x[1], reverse=True
                            )
                            for op, count in sorted_ops[
                                : HTPExportMonitorConfig.TOP_OPERATIONS_COUNT
                            ]:
                                report_console.print(
                                    f"  {op}: {count} {HTPExportMonitorConfig.MSG_NODES}"
                                )

            # Write summary
            report_console.print(
                "\n"
                + HTPExportMonitorConfig.HEADER_SEPARATOR
                * HTPExportMonitorConfig.SEPARATOR_LENGTH
            )
            report_console.print(HTPExportMonitorConfig.MSG_EXPORT_SUMMARY.upper())
            report_console.print(
                HTPExportMonitorConfig.HEADER_SEPARATOR
                * HTPExportMonitorConfig.SEPARATOR_LENGTH
            )
            report_console.print(
                f"{HTPExportMonitorConfig.MSG_EXPORT_TIME_REPORT}: {self.data.export_time:.2f}s"
            )
            report_console.print(
                f"{HTPExportMonitorConfig.MSG_EMBED_HIERARCHY}: {self.embed_hierarchy}"
            )

            # Write to file
            report_path = self._build_output_path(HTPExportMonitorConfig.REPORT_SUFFIX)

            with open(report_path, "w") as f:
                f.write(string_buffer.getvalue())
        except Exception as e:
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_WARNING} {HTPExportMonitorConfig.MSG_FAILED_WRITE_REPORT}: {e}"
            )

    def _write_metadata(self):
        """Write JSON metadata."""
        try:
            metadata_path = self._build_output_path(
                HTPExportMonitorConfig.METADATA_SUFFIX
            )

            # Add export context
            self.metadata["export_context"] = {
                "model_name": self.data.model_name,
                "output_path": self.output_path,
                "export_time": self.data.export_time,
                "embed_hierarchy": self.embed_hierarchy,
            }

            # Write JSON
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.console.print(
                f"{HTPExportMonitorConfig.EMOJI_WARNING} {HTPExportMonitorConfig.MSG_FAILED_WRITE_METADATA}: {e}"
            )
