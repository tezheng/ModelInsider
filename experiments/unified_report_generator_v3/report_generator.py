"""
Pythonic unified report generator using composition and duck typing.
"""

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import singledispatch
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text
from rich.tree import Tree


class Format(Enum):
    """Report formats."""
    CONSOLE = auto()
    TEXT = auto()
    JSON = auto()


@dataclass
class ExportData:
    """All export data in one place."""
    # Model info
    model_name: str = ""
    model_class: str = ""
    total_modules: int = 0
    total_parameters: int = 0
    
    # Files
    output_path: str = ""
    onnx_size_mb: float = 0.0
    
    # Timing
    export_time: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    
    # Data
    hierarchy: dict[str, dict[str, Any]] = field(default_factory=dict)
    tagged_nodes: dict[str, str] = field(default_factory=dict)
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Statistics
    total_nodes: int = 0
    tagged_count: int = 0
    direct_matches: int = 0
    parent_matches: int = 0
    root_fallbacks: int = 0
    empty_tags: int = 0
    
    @property
    def coverage(self) -> float:
        """Calculate coverage percentage."""
        return (self.tagged_count / self.total_nodes * 100) if self.total_nodes > 0 else 0.0
    
    @property
    def top_tags(self, limit: int = 20) -> list[tuple[str, int]]:
        """Get most common tags."""
        return Counter(self.tagged_nodes.values()).most_common(limit)


class ReportGenerator:
    """Simple report generator that works with multiple formats."""
    
    def __init__(self, data: ExportData):
        self.data = data
        self._formatters = {
            Format.CONSOLE: self._format_console,
            Format.TEXT: self._format_text,
            Format.JSON: self._format_json,
        }
    
    def generate(self, format: Format, **options) -> Any:
        """Generate report in specified format."""
        formatter = self._formatters.get(format)
        if not formatter:
            raise ValueError(f"Unknown format: {format}")
        return formatter(**options)
    
    def save(self, path: str, format: Format, **options):
        """Save report to file."""
        content = self.generate(format, **options)
        
        if format == Format.JSON:
            with open(path, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    # Format-specific methods
    def _format_console(self, truncate: bool = True, width: int = 80) -> str:
        """Generate console output with Rich formatting."""
        console = Console(width=width, record=True)
        
        # Step 1: Model info
        self._print_header(console, "📋 STEP 1/8: MODEL PREPARATION")
        console.print(f"✅ Model loaded: {self.data.model_class} "
                     f"({self.data.total_modules} modules, {self.data.total_parameters/1e6:.1f}M parameters)")
        console.print(f"🎯 Export target: {self.data.output_path}")
        console.print("⚙️ Strategy: HTP (Hierarchy-Preserving)")
        
        # Step 2: Input generation
        if "input_generation" in self.data.steps:
            self._print_header(console, "🔧 STEP 2/8: INPUT GENERATION & VALIDATION")
            step = self.data.steps["input_generation"]
            console.print(f"🤖 Auto-generating inputs for: {self.data.model_name}")
            console.print(f"   • Model type: {step.get('model_type', 'unknown')}")
            console.print(f"   • Auto-detected task: {step.get('task', 'unknown')}")
        
        # Step 6: Node tagging
        self._print_header(console, "🔗 STEP 6/8: ONNX NODE TAGGING")
        console.print("✅ Node tagging completed successfully")
        console.print(f"📈 Coverage: {self.data.coverage:.1f}%")
        console.print(f"📊 Tagged nodes: {self.data.tagged_count}/{self.data.total_nodes}")
        console.print(f"   • Direct matches: {self.data.direct_matches} "
                     f"({self.data.direct_matches/self.data.total_nodes*100:.1f}%)")
        console.print(f"   • Parent matches: {self.data.parent_matches} "
                     f"({self.data.parent_matches/self.data.total_nodes*100:.1f}%)")
        console.print(f"   • Root fallbacks: {self.data.root_fallbacks} "
                     f"({self.data.root_fallbacks/self.data.total_nodes*100:.1f}%)")
        console.print(f"✅ Empty tags: {self.data.empty_tags}")
        
        # Hierarchy tree
        console.print("\n🌳 Module Hierarchy:")
        console.print("-" * 60)
        tree = self._build_tree()
        if truncate:
            # Capture and truncate
            with console.capture() as capture:
                console.print(tree)
            lines = capture.get().splitlines()
            for _, line in enumerate(lines[:30]):
                console.print(line)
            if len(lines) > 30:
                console.print(f"... and {len(lines) - 30} more lines (truncated)")
        else:
            console.print(tree)
        
        # Summary
        self._print_header(console, "📋 FINAL EXPORT SUMMARY")
        console.print(f"🎉 HTP Export completed successfully in {self.data.export_time:.2f}s!")
        
        return console.export_text()
    
    def _format_text(self) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 80,
            "HTP EXPORT FULL REPORT",
            "=" * 80,
            f"Generated: {self.data.timestamp}",
            f"Model: {self.data.model_name}",
            "",
            "MODULE HIERARCHY",
            "-" * 40,
        ]
        
        # Add hierarchy
        for path, info in sorted(self.data.hierarchy.items()):
            lines.append(f"Module: {path or '[ROOT]'}")
            lines.append(f"  Class: {info.get('class_name', 'Unknown')}")
            lines.append(f"  Tag: {info.get('traced_tag', '')}")
            lines.append("")
        
        # Add statistics
        lines.extend([
            "STATISTICS",
            "-" * 40,
            f"Total Modules: {len(self.data.hierarchy)}",
            f"Total ONNX Nodes: {self.data.total_nodes}",
            f"Tagged Nodes: {self.data.tagged_count}",
            f"Coverage: {self.data.coverage:.1f}%",
            f"  Direct Matches: {self.data.direct_matches}",
            f"  Parent Matches: {self.data.parent_matches}",
            f"  Root Fallbacks: {self.data.root_fallbacks}",
            f"Empty Tags: {self.data.empty_tags}",
            "",
            "NODE MAPPINGS",
            "-" * 40,
        ])
        
        # Add node mappings
        for node, tag in sorted(self.data.tagged_nodes.items()):
            lines.append(f"{node} -> {tag}")
        
        return "\n".join(lines)
    
    def _format_json(self) -> dict:
        """Generate JSON metadata."""
        return {
            "export_context": {
                "timestamp": self.data.timestamp,
                "strategy": "htp",
                "version": "1.0",
                "export_time_seconds": round(self.data.export_time, 2)
            },
            "model": {
                "name_or_path": self.data.model_name,
                "class": self.data.model_class,
                "total_modules": self.data.total_modules,
                "total_parameters": self.data.total_parameters
            },
            "modules": self.data.hierarchy,
            "nodes": self.data.tagged_nodes,
            "outputs": {
                "onnx_model": {
                    "path": Path(self.data.output_path).name,
                    "size_mb": self.data.onnx_size_mb
                },
                "metadata": {
                    "path": Path(self.data.output_path).stem + "_metadata.json"
                },
                "report": {
                    "path": Path(self.data.output_path).stem + "_report.txt"
                }
            },
            "report": {
                "steps": self.data.steps,
                "node_tagging": {
                    "statistics": {
                        "total_nodes": self.data.total_nodes,
                        "tagged_nodes": self.data.tagged_count,
                        "direct_matches": self.data.direct_matches,
                        "parent_matches": self.data.parent_matches,
                        "root_fallbacks": self.data.root_fallbacks,
                        "empty_tags": self.data.empty_tags
                    },
                    "coverage": {
                        "percentage": self.data.coverage,
                        "total_onnx_nodes": self.data.total_nodes,
                        "tagged_nodes": self.data.tagged_count
                    }
                }
            }
        }
    
    def _print_header(self, console: Console, text: str):
        """Print section header."""
        console.print("")
        console.print("=" * 80)
        console.print(text)
        console.print("=" * 80)
    
    def _build_tree(self) -> Tree:
        """Build Rich tree from hierarchy data."""
        root = self.data.hierarchy.get("", {})
        tree = Tree(Text(root.get("class_name", "Model"), style="bold magenta"))
        
        def add_children(parent_tree: Tree, parent_path: str):
            for path, info in self.data.hierarchy.items():
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
                
                # Add child
                class_name = info.get("class_name", "Unknown")
                child_name = path.split(".")[-1]
                child_tree = parent_tree.add(f"{class_name}: {child_name}")
                add_children(child_tree, path)
        
        add_children(tree, "")
        return tree


# Alternative functional approach using single dispatch
@singledispatch
def format_statistics(stats: Any, format: Format) -> str:
    """Format statistics based on output format."""
    raise NotImplementedError(f"No formatter for {type(stats)}")


@format_statistics.register
def _(data: ExportData, format: Format) -> str:
    """Format ExportData statistics."""
    if format == Format.CONSOLE:
        return (f"📈 Coverage: {data.coverage:.1f}%\n"
                f"📊 Tagged nodes: {data.tagged_count}/{data.total_nodes}")
    elif format == Format.TEXT:
        return (f"Coverage: {data.coverage:.1f}%\n"
                f"Tagged nodes: {data.tagged_count}/{data.total_nodes}")
    else:
        return ""


# Simple functional helpers
def make_header(text: str, width: int = 80, char: str = "=") -> str:
    """Create a text header."""
    return f"\n{char * width}\n{text}\n{char * width}\n"


def truncate_lines(text: str, max_lines: int = 30) -> str:
    """Truncate text to max lines."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    
    truncated = lines[:max_lines]
    truncated.append(f"... and {len(lines) - max_lines} more lines (truncated)")
    return "\n".join(truncated)


# Context manager for report generation
class ReportContext:
    """Context manager for report generation with automatic saving."""
    
    def __init__(self, data: ExportData, base_path: str):
        self.data = data
        self.base_path = Path(base_path).stem
        self.generator = ReportGenerator(data)
        self.reports = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        # Auto-save all generated reports
        for format, _content in self.reports.items():
            if format == Format.JSON:
                path = f"{self.base_path}_metadata.json"
            elif format == Format.TEXT:
                path = f"{self.base_path}_report.txt"
            else:
                path = f"{self.base_path}_{format.name.lower()}.txt"
            
            self.generator.save(path, format)
    
    def generate(self, format: Format, **options):
        """Generate and track report."""
        report = self.generator.generate(format, **options)
        self.reports[format] = report
        return report