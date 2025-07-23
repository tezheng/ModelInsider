"""
Concrete formatter implementations for different report formats.
"""

from typing import Any, Dict, List, Optional
import json
from io import StringIO

from rich.console import Console
from rich.tree import Tree
from rich.text import Text
from rich.table import Table

from interfaces import IReportFormatter, ReportFormat, StatisticsInfo


class ConsoleFormatter(IReportFormatter):
    """Formatter for Rich console output."""
    
    def __init__(self, width: int = 80, color: bool = True):
        self.width = width
        self.color = color
        self.console = Console(width=width, force_terminal=color)
        
    def format_header(self, text: str, level: int = 1) -> str:
        """Format header with separator lines."""
        buffer = StringIO()
        console = Console(file=buffer, width=self.width, force_terminal=self.color)
        
        console.print("")
        console.print("=" * self.width)
        console.print(text)
        console.print("=" * self.width)
        
        return buffer.getvalue()
    
    def format_list(self, items: List[str], ordered: bool = False) -> str:
        """Format a list with bullets or numbers."""
        buffer = StringIO()
        console = Console(file=buffer, width=self.width, force_terminal=self.color)
        
        for i, item in enumerate(items):
            if ordered:
                console.print(f"{i+1}. {item}")
            else:
                console.print(f"   • {item}")
        
        return buffer.getvalue()
    
    def format_key_value(self, key: str, value: Any) -> str:
        """Format key-value with styling."""
        buffer = StringIO()
        console = Console(file=buffer, width=self.width, force_terminal=self.color)
        
        console.print(f"{key}: {value}")
        
        return buffer.getvalue()
    
    def format_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """Format as Rich table."""
        buffer = StringIO()
        console = Console(file=buffer, width=self.width, force_terminal=self.color)
        
        table = Table()
        for header in headers:
            table.add_column(header)
        
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        
        console.print(table)
        return buffer.getvalue()
    
    def format_tree(self, root: str, children: Dict[str, Any], max_depth: Optional[int] = None) -> str:
        """Format as Rich tree."""
        buffer = StringIO()
        console = Console(file=buffer, width=self.width, force_terminal=self.color)
        
        tree = Tree(Text(root, style="bold magenta"))
        self._build_tree(tree, children, 0, max_depth)
        
        console.print(tree)
        return buffer.getvalue()
    
    def _build_tree(self, parent: Tree, data: Dict[str, Any], depth: int, max_depth: Optional[int]):
        """Recursively build tree."""
        if max_depth and depth >= max_depth:
            return
            
        for key, value in data.items():
            if isinstance(value, dict):
                node = parent.add(Text(key, style="green"))
                self._build_tree(node, value, depth + 1, max_depth)
            else:
                parent.add(f"{key}: {value}")
    
    def format_separator(self, width: int = 80) -> str:
        """Format separator line."""
        return "-" * width + "\n"
    
    def format_statistics(self, stats: StatisticsInfo) -> str:
        """Format statistics with percentages."""
        buffer = StringIO()
        console = Console(file=buffer, width=self.width, force_terminal=self.color)
        
        console.print(f"📈 Coverage: {stats.coverage_percentage:.1f}%")
        console.print(f"📊 Tagged nodes: {stats.tagged_nodes}/{stats.total_onnx_nodes}")
        console.print(f"   • Direct matches: {stats.direct_matches} ({stats.direct_percentage:.1f}%)")
        console.print(f"   • Parent matches: {stats.parent_matches} ({stats.parent_percentage:.1f}%)")
        console.print(f"   • Root fallbacks: {stats.root_fallbacks} ({stats.root_percentage:.1f}%)")
        console.print(f"✅ Empty tags: {stats.empty_tags}")
        
        return buffer.getvalue()
    
    def join_elements(self, elements: List[Any]) -> str:
        """Join formatted elements."""
        return "".join(str(e) for e in elements)


class TextFormatter(IReportFormatter):
    """Formatter for plain text output."""
    
    def __init__(self, width: int = 80):
        self.width = width
    
    def format_header(self, text: str, level: int = 1) -> str:
        """Format header with ASCII decoration."""
        separator = "=" * self.width
        if level == 1:
            return f"\n{separator}\n{text.upper()}\n{separator}\n"
        elif level == 2:
            return f"\n{text}\n{'-' * len(text)}\n"
        else:
            return f"\n{text}\n"
    
    def format_list(self, items: List[str], ordered: bool = False) -> str:
        """Format simple text list."""
        lines = []
        for i, item in enumerate(items):
            if ordered:
                lines.append(f"{i+1}. {item}")
            else:
                lines.append(f"  - {item}")
        return "\n".join(lines) + "\n"
    
    def format_key_value(self, key: str, value: Any) -> str:
        """Format simple key-value."""
        return f"{key}: {value}\n"
    
    def format_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """Format ASCII table."""
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Format table
        lines = []
        
        # Headers
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            lines.append(row_line)
        
        return "\n".join(lines) + "\n"
    
    def format_tree(self, root: str, children: Dict[str, Any], max_depth: Optional[int] = None) -> str:
        """Format ASCII tree."""
        lines = [root]
        self._format_tree_lines(children, "", True, lines, 0, max_depth)
        return "\n".join(lines) + "\n"
    
    def _format_tree_lines(self, data: Dict[str, Any], prefix: str, is_last: bool, 
                          lines: List[str], depth: int, max_depth: Optional[int]):
        """Recursively format tree lines."""
        if max_depth and depth >= max_depth:
            return
            
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            is_last_item = i == len(items) - 1
            
            # Connector
            if is_last:
                connector = "└── "
                extension = "    "
            else:
                connector = "├── "
                extension = "│   "
            
            # Add line
            if isinstance(value, dict):
                lines.append(f"{prefix}{connector}{key}")
                self._format_tree_lines(value, prefix + extension, is_last_item, 
                                      lines, depth + 1, max_depth)
            else:
                lines.append(f"{prefix}{connector}{key}: {value}")
    
    def format_separator(self, width: int = 80) -> str:
        """Format text separator."""
        return "-" * width + "\n"
    
    def format_statistics(self, stats: StatisticsInfo) -> str:
        """Format statistics as text."""
        lines = [
            f"Coverage: {stats.coverage_percentage:.1f}%",
            f"Tagged nodes: {stats.tagged_nodes}/{stats.total_onnx_nodes}",
            f"  Direct matches: {stats.direct_matches} ({stats.direct_percentage:.1f}%)",
            f"  Parent matches: {stats.parent_matches} ({stats.parent_percentage:.1f}%)",
            f"  Root fallbacks: {stats.root_fallbacks} ({stats.root_percentage:.1f}%)",
            f"Empty tags: {stats.empty_tags}"
        ]
        return "\n".join(lines) + "\n"
    
    def join_elements(self, elements: List[Any]) -> str:
        """Join text elements."""
        return "".join(str(e) for e in elements)


class MetadataFormatter(IReportFormatter):
    """Formatter for JSON metadata output."""
    
    def format_header(self, text: str, level: int = 1) -> Dict[str, str]:
        """Headers become dictionary keys in metadata."""
        return {"_section": text}
    
    def format_list(self, items: List[str], ordered: bool = False) -> List[str]:
        """Lists remain as lists."""
        return items
    
    def format_key_value(self, key: str, value: Any) -> Dict[str, Any]:
        """Key-value becomes dict entry."""
        return {key: value}
    
    def format_table(self, headers: List[str], rows: List[List[Any]]) -> List[Dict[str, Any]]:
        """Table becomes list of dicts."""
        return [
            dict(zip(headers, row))
            for row in rows
        ]
    
    def format_tree(self, root: str, children: Dict[str, Any], max_depth: Optional[int] = None) -> Dict[str, Any]:
        """Tree structure remains as nested dict."""
        return {root: children}
    
    def format_separator(self, width: int = 80) -> None:
        """No separators in metadata."""
        return None
    
    def format_statistics(self, stats: StatisticsInfo) -> Dict[str, Any]:
        """Statistics as structured data."""
        return {
            "coverage_percentage": stats.coverage_percentage,
            "total_onnx_nodes": stats.total_onnx_nodes,
            "tagged_nodes": stats.tagged_nodes,
            "direct_matches": stats.direct_matches,
            "parent_matches": stats.parent_matches,
            "operation_matches": stats.operation_matches,
            "root_fallbacks": stats.root_fallbacks,
            "empty_tags": stats.empty_tags,
            "breakdown": {
                "direct_percentage": stats.direct_percentage,
                "parent_percentage": stats.parent_percentage,
                "root_percentage": stats.root_percentage
            }
        }
    
    def join_elements(self, elements: List[Any]) -> Dict[str, Any]:
        """Join metadata elements into single dict."""
        result = {}
        for element in elements:
            if element is None:
                continue
            elif isinstance(element, dict):
                # Merge dicts, handling special _section keys
                if "_section" in element:
                    section_name = element.pop("_section")
                    result[section_name] = element
                else:
                    result.update(element)
            elif isinstance(element, list):
                # Lists need a key
                result["items"] = element
        return result


def get_formatter(format: ReportFormat, **kwargs) -> IReportFormatter:
    """Factory function to get appropriate formatter."""
    # Extract formatter-specific arguments
    if format == ReportFormat.CONSOLE:
        formatter_args = {k: v for k, v in kwargs.items() if k in ['width', 'color']}
        return ConsoleFormatter(**formatter_args)
    elif format == ReportFormat.TEXT:
        formatter_args = {k: v for k, v in kwargs.items() if k in ['width']}
        return TextFormatter(**formatter_args)
    elif format == ReportFormat.METADATA:
        return MetadataFormatter()
    else:
        raise ValueError(f"Unsupported format: {format}")