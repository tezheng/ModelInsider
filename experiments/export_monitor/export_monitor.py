"""
ExportMonitor system with step-aware writers using decorator pattern.
"""

import contextlib
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


class ExportStep(Enum):
    """Export process steps."""
    MODEL_PREP = "model_preparation"
    INPUT_GEN = "input_generation"
    HIERARCHY = "hierarchy_building"
    STRUCTURE = "structure_analysis"
    CONVERSION = "onnx_conversion"
    NODE_TAGGING = "node_tagging"
    VALIDATION = "model_validation"
    COMPLETE = "export_complete"


@dataclass
class ExportData:
    """Unified export data shared across all writers."""
    # Model info
    model_name: str = ""
    model_class: str = ""
    total_modules: int = 0
    total_parameters: int = 0
    
    # Export settings
    output_path: str = ""
    strategy: str = "htp"
    
    # Timing
    start_time: float = field(default_factory=time.time)
    export_time: float = 0.0
    step_times: dict[str, float] = field(default_factory=dict)
    
    # Structure data
    hierarchy: dict[str, dict[str, Any]] = field(default_factory=dict)
    module_list: list[tuple[str, str]] = field(default_factory=list)
    
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
    
    @property
    def timestamp(self) -> str:
        """Current timestamp."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    @property
    def coverage(self) -> float:
        """Tagging coverage percentage."""
        if self.total_nodes == 0:
            return 0.0
        return len(self.tagged_nodes) / self.total_nodes * 100
    
    @property
    def elapsed_time(self) -> float:
        """Total elapsed time."""
        return time.time() - self.start_time


def step(export_step: ExportStep):
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
        self._step_handlers: dict[ExportStep, Callable] = {}
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
    
    def write(self, export_step: ExportStep, data: ExportData) -> int:
        """Write data for a specific step."""
        # Use specific handler or fall back to default
        handler = self._step_handlers.get(export_step, self._write_default)
        return handler(export_step, data)
    
    @abstractmethod
    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default handler for steps without specific handlers."""
        pass
    
    def flush(self) -> None:
        """Flush any buffered data."""
        pass
    
    def close(self) -> None:
        """Close the writer and perform cleanup."""
        try:
            self.flush()
        except Exception:
            pass  # Ignore flush errors on close
        super().close()


class ConsoleWriter(StepAwareWriter):
    """Real-time console output with Rich formatting."""
    
    def __init__(self, width: int = 80, verbose: bool = True):
        super().__init__()
        self.console = Console(width=width)
        self.verbose = verbose
        self._step_count = 0
        self._total_steps = 8
    
    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default: simple progress message."""
        self._step_count += 1
        self.console.print(f"✓ Step {self._step_count}/{self._total_steps}: {export_step.value}")
        return 1
    
    @step(ExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: ExportStep, data: ExportData) -> int:
        """Model preparation with styled output."""
        self._print_header(f"📋 STEP 1/{self._total_steps}: MODEL PREPARATION")
        self.console.print(
            f"✅ Model loaded: {data.model_class} "
            f"({data.total_modules} modules, {data.total_parameters/1e6:.1f}M parameters)"
        )
        self.console.print(f"🎯 Export target: {data.output_path}")
        self.console.print(f"⚙️ Strategy: {data.strategy.upper()} (Hierarchy-Preserving)")
        return 1
    
    @step(ExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: ExportStep, data: ExportData) -> int:
        """Input generation details."""
        self._print_header(f"🔧 STEP 2/{self._total_steps}: INPUT GENERATION")
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            self.console.print(f"🤖 Auto-generating inputs for: {data.model_name}")
            self.console.print(f"   • Model type: {step_data.get('model_type', 'unknown')}")
            self.console.print(f"   • Task: {step_data.get('task', 'unknown')}")
            if "inputs" in step_data:
                self.console.print(f"🔧 Generated {len(step_data['inputs'])} input tensors")
        return 1
    
    @step(ExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
        """Hierarchy with truncated tree view."""
        self._print_header(f"🏗️ STEP 3/{self._total_steps}: HIERARCHY BUILDING")
        self.console.print("✅ Hierarchy building completed")
        self.console.print(f"📈 Traced {len(data.hierarchy)} modules")
        
        if self.verbose and data.hierarchy:
            self.console.print("\n🌳 Module Hierarchy:")
            self.console.print("-" * 60)
            
            # Build and print tree with truncation
            tree = self._build_hierarchy_tree(data.hierarchy)
            
            # Capture output for truncation
            with self.console.capture() as capture:
                self.console.print(tree)
            
            lines = capture.get().splitlines()
            max_lines = 30
            
            for line in lines[:max_lines]:
                self.console.print(line)
            
            if len(lines) > max_lines:
                self.console.print(f"... and {len(lines) - max_lines} more lines (truncated)")
        
        return 1
    
    @step(ExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: ExportStep, data: ExportData) -> int:
        """Node tagging statistics with colors."""
        self._print_header(f"🔗 STEP 6/{self._total_steps}: NODE TAGGING")
        
        stats = data.tagging_stats
        total = data.total_nodes
        tagged = len(data.tagged_nodes)
        
        self.console.print("✅ Node tagging completed successfully")
        self.console.print(f"📈 Coverage: [green]{data.coverage:.1f}%[/green]")
        self.console.print(f"📊 Tagged nodes: {tagged}/{total}")
        
        if stats:
            direct = stats.get("direct_matches", 0)
            parent = stats.get("parent_matches", 0)
            root = stats.get("root_fallbacks", 0)
            
            self.console.print(
                f"   • Direct matches: {direct} "
                f"([cyan]{direct/total*100:.1f}%[/cyan])"
            )
            self.console.print(
                f"   • Parent matches: {parent} "
                f"([yellow]{parent/total*100:.1f}%[/yellow])"
            )
            self.console.print(
                f"   • Root fallbacks: {root} "
                f"([red]{root/total*100:.1f}%[/red])"
            )
        
        return 1
    
    @step(ExportStep.COMPLETE)
    def write_complete(self, export_step: ExportStep, data: ExportData) -> int:
        """Export completion summary."""
        self._print_header("📋 FINAL EXPORT SUMMARY")
        self.console.print(
            f"🎉 {data.strategy.upper()} Export completed successfully "
            f"in {data.elapsed_time:.2f}s!"
        )
        
        self.console.print("\n📊 Export Statistics:")
        self.console.print(f"   • Export time: {data.elapsed_time:.2f}s")
        self.console.print(f"   • Hierarchy modules: {len(data.hierarchy)}")
        self.console.print(f"   • ONNX nodes: {data.total_nodes}")
        self.console.print(f"   • Tagged nodes: {len(data.tagged_nodes)}")
        self.console.print(f"   • Coverage: {data.coverage:.1f}%")
        
        self.console.print("\n📁 Output Files:")
        self.console.print(f"   • ONNX model: {Path(data.output_path).name}")
        if data.metadata_path:
            self.console.print(f"   • Metadata: {Path(data.metadata_path).name}")
        if data.report_path:
            self.console.print(f"   • Report: {Path(data.report_path).name}")
        
        return 1
    
    def _print_header(self, text: str) -> None:
        """Print section header."""
        self.console.print("")
        self.console.print("=" * 80)
        self.console.print(text)
        self.console.print("=" * 80)
    
    def _build_hierarchy_tree(self, hierarchy: dict) -> Tree:
        """Build Rich tree from hierarchy data."""
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        tree = Tree(Text(root_name, style="bold magenta"))
        
        def add_children(parent_tree: Tree, parent_path: str, level: int = 0):
            if level > 5:  # Limit depth for console
                return
                
            for path, info in hierarchy.items():
                if not path or path == parent_path:
                    continue
                
                # Check if direct child
                if parent_path:
                    if not path.startswith(parent_path + "."):
                        continue
                    suffix = path[len(parent_path) + 1:]
                    if "." in suffix:
                        continue
                else:
                    if "." in path:
                        continue
                
                # Add node
                class_name = info.get("class_name", "Unknown")
                child_name = path.split(".")[-1]
                style = "green" if level < 2 else "cyan" if level < 4 else "white"
                
                child_tree = parent_tree.add(
                    Text(f"{class_name}: {child_name}", style=style)
                )
                add_children(child_tree, path, level + 1)
        
        add_children(tree, "")
        return tree


class MetadataWriter(StepAwareWriter):
    """JSON metadata writer with accumulation."""
    
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.metadata_path = f"{self.output_path}_metadata.json"
        self.metadata = {
            "export_context": {},
            "model": {},
            "modules": {},
            "nodes": {},
            "outputs": {},
            "report": {"steps": {}}
        }
    
    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default: record step completion."""
        self.metadata["report"]["steps"][export_step.value] = {
            "completed": True,
            "timestamp": data.timestamp
        }
        return 1
    
    @step(ExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: ExportStep, data: ExportData) -> int:
        """Record model information."""
        self.metadata["export_context"] = {
            "timestamp": data.timestamp,
            "strategy": data.strategy,
            "version": "1.0"
        }
        
        self.metadata["model"] = {
            "name_or_path": data.model_name,
            "class": data.model_class,
            "total_modules": data.total_modules,
            "total_parameters": data.total_parameters
        }
        
        return 1
    
    @step(ExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
        """Record hierarchy data."""
        self.metadata["modules"] = data.hierarchy.copy()
        self.metadata["report"]["steps"]["hierarchy_building"] = {
            "modules_traced": len(data.hierarchy),
            "timestamp": data.timestamp
        }
        return 1
    
    @step(ExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: ExportStep, data: ExportData) -> int:
        """Record tagging results."""
        self.metadata["nodes"] = data.tagged_nodes.copy()
        
        stats = data.tagging_stats
        self.metadata["report"]["node_tagging"] = {
            "statistics": {
                "total_nodes": data.total_nodes,
                "tagged_nodes": len(data.tagged_nodes),
                "direct_matches": stats.get("direct_matches", 0),
                "parent_matches": stats.get("parent_matches", 0),
                "root_fallbacks": stats.get("root_fallbacks", 0),
                "empty_tags": stats.get("empty_tags", 0)
            },
            "coverage": {
                "percentage": data.coverage,
                "total_onnx_nodes": data.total_nodes,
                "tagged_nodes": len(data.tagged_nodes)
            }
        }
        return 1
    
    @step(ExportStep.COMPLETE)
    def write_complete(self, export_step: ExportStep, data: ExportData) -> int:
        """Finalize metadata."""
        self.metadata["export_context"]["export_time_seconds"] = round(data.elapsed_time, 2)
        
        self.metadata["outputs"] = {
            "onnx_model": {
                "path": Path(data.output_path).name,
                "size_mb": data.onnx_size_mb
            },
            "metadata": {
                "path": Path(self.metadata_path).name
            }
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


class ReportWriter(StepAwareWriter):
    """Full text report writer with buffering."""
    
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.report_path = f"{self.output_path}_report.txt"
        self.buffer = io.StringIO()
        self._write_header()
    
    def _write_header(self) -> None:
        """Write report header."""
        self.buffer.write("=" * 80 + "\n")
        self.buffer.write("HTP EXPORT FULL REPORT\n")
        self.buffer.write("=" * 80 + "\n")
        self.buffer.write(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n")
    
    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default: record step with timestamp."""
        self.buffer.write(f"\n[{data.timestamp}] {export_step.value}: Completed\n")
        return 1
    
    @step(ExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: ExportStep, data: ExportData) -> int:
        """Write model details."""
        self.buffer.write("\nMODEL INFORMATION\n")
        self.buffer.write("-" * 40 + "\n")
        self.buffer.write(f"Model Name: {data.model_name}\n")
        self.buffer.write(f"Model Class: {data.model_class}\n")
        self.buffer.write(f"Total Modules: {data.total_modules}\n")
        self.buffer.write(f"Total Parameters: {data.total_parameters:,}\n")
        self.buffer.write(f"Export Strategy: {data.strategy.upper()}\n")
        self.buffer.write(f"Output Path: {data.output_path}\n")
        return 1
    
    @step(ExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
        """Write complete hierarchy."""
        self.buffer.write("\nCOMPLETE MODULE HIERARCHY\n")
        self.buffer.write("-" * 40 + "\n")
        
        for path, info in sorted(data.hierarchy.items()):
            module_path = path or "[ROOT]"
            class_name = info.get("class_name", "Unknown")
            tag = info.get("traced_tag", "")
            
            self.buffer.write(f"\nModule: {module_path}\n")
            self.buffer.write(f"  Class: {class_name}\n")
            self.buffer.write(f"  Tag: {tag}\n")
        
        self.buffer.write(f"\nTotal Modules: {len(data.hierarchy)}\n")
        return 1
    
    @step(ExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: ExportStep, data: ExportData) -> int:
        """Write tagging statistics and full mappings."""
        stats = data.tagging_stats
        
        self.buffer.write("\nNODE TAGGING STATISTICS\n")
        self.buffer.write("-" * 40 + "\n")
        self.buffer.write(f"Total ONNX Nodes: {data.total_nodes}\n")
        self.buffer.write(f"Tagged Nodes: {len(data.tagged_nodes)}\n")
        self.buffer.write(f"Coverage: {data.coverage:.1f}%\n")
        
        if stats:
            self.buffer.write(f"  Direct Matches: {stats.get('direct_matches', 0)}\n")
            self.buffer.write(f"  Parent Matches: {stats.get('parent_matches', 0)}\n")
            self.buffer.write(f"  Root Fallbacks: {stats.get('root_fallbacks', 0)}\n")
            self.buffer.write(f"  Empty Tags: {stats.get('empty_tags', 0)}\n")
        
        # Write complete node mappings
        self.buffer.write("\nCOMPLETE NODE MAPPINGS\n")
        self.buffer.write("-" * 40 + "\n")
        for node_name, tag in sorted(data.tagged_nodes.items()):
            self.buffer.write(f"{node_name} -> {tag}\n")
        
        return 1
    
    @step(ExportStep.COMPLETE)
    def write_complete(self, export_step: ExportStep, data: ExportData) -> int:
        """Write final summary."""
        self.buffer.write("\nEXPORT SUMMARY\n")
        self.buffer.write("-" * 40 + "\n")
        self.buffer.write(f"Total Export Time: {data.elapsed_time:.2f}s\n")
        self.buffer.write(f"ONNX File Size: {data.onnx_size_mb:.2f}MB\n")
        self.buffer.write(f"Final Coverage: {data.coverage:.1f}%\n")
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
        # Don't call super().close() as it would try to flush again


class ExportMonitor:
    """Central monitor that coordinates data updates and writer dispatch."""
    
    def __init__(self, output_path: str, verbose: bool = True, enable_report: bool = True):
        self.data = ExportData(output_path=output_path)
        self.writers: list[StepAwareWriter] = []
        
        # Always include metadata writer
        self.writers.append(MetadataWriter(output_path))
        self.data.metadata_path = f"{Path(output_path).with_suffix('').as_posix()}_metadata.json"
        
        # Conditional writers
        if verbose:
            self.writers.append(ConsoleWriter())
            
        if enable_report:
            self.writers.append(ReportWriter(output_path))
            self.data.report_path = f"{Path(output_path).with_suffix('').as_posix()}_report.txt"
    
    def update(self, step: ExportStep, **kwargs) -> None:
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
        self.update(ExportStep.COMPLETE)
        
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
            for writer in self.writers:
                with contextlib.suppress(Exception):
                    writer.close()
