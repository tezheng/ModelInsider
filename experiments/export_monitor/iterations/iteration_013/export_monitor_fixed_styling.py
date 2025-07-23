"""
Export monitoring system for HTP strategy with PROPER text styling.
Fixed in iteration 13 to use Rich console correctly.
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
from rich.text import Text
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
    """Console output writer for HTP export with PROPER Rich formatting."""
    
    # Display constants
    MODULE_TREE_MAX_LINES = 100
    NODE_TREE_MAX_LINES = 30
    TOP_NODES_COUNT = 20
    SEPARATOR_LENGTH = 80
    
    def __init__(self, console: Console = None, verbose: bool = True):
        super().__init__()
        # CRITICAL: Use force_terminal=True to ensure ANSI codes are output
        self.console = console or Console(width=80, force_terminal=True)
        self.verbose = verbose
        self._total_steps = 8
    
    def _write_default(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Default: simple step completion message."""
        if self.verbose:
            self.console.print(f"✓ {export_step.value} completed")
        return 1
    
    def _print_header(self, text: str) -> None:
        """Print section header with proper styling."""
        self.console.print()
        self.console.print("=" * self.SEPARATOR_LENGTH, style="bright_blue")
        # Parse step numbers for special styling
        if "STEP" in text:
            # Match pattern like "STEP 1/8"
            import re
            match = re.search(r'(.*STEP )(\d+)(/)(\d+)(.*)', text)
            if match:
                before, num1, slash, num2, after = match.groups()
                # Build styled text
                styled = Text()
                styled.append(before)
                styled.append(num1, style="bold cyan")
                styled.append(slash)
                styled.append(num2, style="bold cyan")
                styled.append(after)
                self.console.print(styled)
            else:
                self.console.print(text)
        else:
            self.console.print(text)
        self.console.print("=" * self.SEPARATOR_LENGTH, style="bright_blue")
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 1: Model preparation."""
        if not self.verbose:
            return 0
            
        self._print_header("📋 STEP 1/8: MODEL PREPARATION")
        
        # Model loaded message with styled numbers
        text = Text("✅ Model loaded: ")
        text.append(data.model_class)
        text.append(" ")
        text.append("(", style="bold")
        text.append(str(data.total_modules), style="bold cyan")
        text.append(" modules, ", style="bold")
        text.append(f"{data.total_parameters/1e6:.1f}", style="bold cyan")
        text.append("M parameters", style="bold")
        text.append(")", style="bold")
        self.console.print(text)
        
        self.console.print(f"🎯 Export target: {data.output_path}")
        
        # Strategy line with special formatting
        strategy_text = Text("⚙️ Strategy: HTP ")
        strategy_text.append("(", style="bold")
        strategy_text.append("Hierarchy-Preserving", style="normal")
        strategy_text.append(")", style="bold")
        self.console.print(strategy_text)
        
        if data.embed_hierarchy_attributes:
            self.console.print("✅ Hierarchy attributes will be embedded in ONNX")
        else:
            self.console.print("⚠️ Hierarchy attributes will NOT be embedded (clean ONNX)")
        
        self.console.print("✅ Model set to evaluation mode")
        return 1
    
    @step(HTPExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Step 2: Input generation."""
        if not self.verbose:
            return 0
            
        self._print_header("🔧 STEP 2/8: INPUT GENERATION & VALIDATION")
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            self.console.print(f"🤖 Auto-generating inputs for: {data.model_name}")
            self.console.print(f"   • Model type: {step_data.get('model_type', 'unknown')}")
            self.console.print(f"   • Task: {step_data.get('task', 'unknown')}")
            
            if "model_type" in step_data and "task" in step_data:
                self.console.print(
                    f"✅ Created onnx export config for {step_data['model_type']} "
                    f"with task {step_data['task']}"
                )
            
            # Input tensors with styled count
            inputs = step_data.get("inputs", {})
            if inputs:
                text = Text("🔧 Generated ")
                text.append(str(len(inputs)), style="bold cyan")
                text.append(" input tensors:")
                self.console.print(text)
                
                for name, spec in inputs.items():
                    # Format tensor info with styled brackets
                    shape_text = Text(f"   • {name}: ")
                    shape_text.append("[", style="bold")
                    shape_parts = spec.get("shape", "").strip("[]").split(", ")
                    for i, part in enumerate(shape_parts):
                        if i > 0:
                            shape_text.append(", ")
                        shape_text.append(part, style="bold cyan")
                    shape_text.append("]", style="bold")
                    shape_text.append(" ")
                    shape_text.append("(", style="bold")
                    shape_text.append(spec.get("dtype", "unknown"))
                    shape_text.append(")", style="bold")
                    self.console.print(shape_text)
        return 1
    
    # ... Continue with other step methods, all using console.print() with proper styling ...
