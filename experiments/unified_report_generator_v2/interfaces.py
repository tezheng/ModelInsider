"""
Interfaces for unified report generation system.

This module defines the abstract base classes and interfaces for 
creating different types of reports with shared logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class ReportFormat(Enum):
    """Supported report formats."""
    CONSOLE = "console"
    METADATA = "metadata"
    TEXT = "text"
    HTML = "html"  # Future extension


@dataclass
class StepInfo:
    """Information about a single export step."""
    name: str
    number: int
    total: int
    title: str
    icon: str
    status: str = "pending"
    details: dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class StatisticsInfo:
    """Unified statistics structure."""
    total_onnx_nodes: int
    tagged_nodes: int
    coverage_percentage: float
    direct_matches: int
    parent_matches: int
    operation_matches: int
    root_fallbacks: int
    empty_tags: int
    
    @property
    def direct_percentage(self) -> float:
        """Calculate direct match percentage."""
        return (self.direct_matches / self.total_onnx_nodes * 100) if self.total_onnx_nodes > 0 else 0.0
    
    @property
    def parent_percentage(self) -> float:
        """Calculate parent match percentage."""
        return (self.parent_matches / self.total_onnx_nodes * 100) if self.total_onnx_nodes > 0 else 0.0
    
    @property
    def root_percentage(self) -> float:
        """Calculate root fallback percentage."""
        return (self.root_fallbacks / self.total_onnx_nodes * 100) if self.total_onnx_nodes > 0 else 0.0


class IReportSection(ABC):
    """Interface for a report section that can be rendered in different formats."""
    
    @abstractmethod
    def render(self, format: ReportFormat, context: dict[str, Any]) -> Any:
        """Render this section in the specified format."""
        pass
    
    @abstractmethod
    def get_data(self) -> dict[str, Any]:
        """Get the raw data for this section."""
        pass


class IReportFormatter(ABC):
    """Interface for formatting report elements."""
    
    @abstractmethod
    def format_header(self, text: str, level: int = 1) -> Any:
        """Format a header."""
        pass
    
    @abstractmethod
    def format_list(self, items: list[str], ordered: bool = False) -> Any:
        """Format a list of items."""
        pass
    
    @abstractmethod
    def format_key_value(self, key: str, value: Any) -> Any:
        """Format a key-value pair."""
        pass
    
    @abstractmethod
    def format_table(self, headers: list[str], rows: list[list[Any]]) -> Any:
        """Format a table."""
        pass
    
    @abstractmethod
    def format_tree(self, root: str, children: dict[str, Any], max_depth: int | None = None) -> Any:
        """Format a tree structure."""
        pass
    
    @abstractmethod
    def format_separator(self, width: int = 80) -> Any:
        """Format a separator line."""
        pass
    
    @abstractmethod
    def format_statistics(self, stats: StatisticsInfo) -> Any:
        """Format statistics information."""
        pass
    
    @abstractmethod
    def join_elements(self, elements: list[Any]) -> Any:
        """Join multiple formatted elements."""
        pass


class IDataProvider(Protocol):
    """Protocol for providing data to report sections."""
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        ...
    
    def get_step_info(self, step_name: str) -> StepInfo | None:
        """Get information about a specific step."""
        ...
    
    def get_hierarchy_data(self) -> dict[str, dict[str, Any]]:
        """Get module hierarchy data."""
        ...
    
    def get_tagged_nodes(self) -> dict[str, str]:
        """Get ONNX node to tag mappings."""
        ...
    
    def get_statistics(self) -> StatisticsInfo:
        """Get tagging statistics."""
        ...
    
    def get_export_config(self) -> dict[str, Any]:
        """Get export configuration."""
        ...
    
    def get_file_info(self) -> dict[str, Any]:
        """Get output file information."""
        ...


class IReportGenerator(ABC):
    """Interface for generating complete reports."""
    
    @abstractmethod
    def add_section(self, section: IReportSection) -> None:
        """Add a section to the report."""
        pass
    
    @abstractmethod
    def generate(self, format: ReportFormat) -> Any:
        """Generate the complete report in the specified format."""
        pass
    
    @abstractmethod
    def save(self, path: str, format: ReportFormat) -> None:
        """Save the report to a file."""
        pass