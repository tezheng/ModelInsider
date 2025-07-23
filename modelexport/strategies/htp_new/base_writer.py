"""
Base writer class for step-aware export monitoring.

This module provides the abstract base class for all export writers,
using Python's IO protocol and decorator pattern for step handling.
"""

from __future__ import annotations

import contextlib
import io
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from .step_data import (
    HierarchyData,
    InputGenData,
    ModelPrepData,
    NodeTaggingData,
    ONNXExportData,
    TagInjectionData,
)


class ExportStep(Enum):
    """HTP export process steps."""
    
    MODEL_PREP = "model_preparation"      # Step 1
    INPUT_GEN = "input_generation"        # Step 2
    HIERARCHY = "hierarchy_building"      # Step 3
    ONNX_EXPORT = "onnx_export"          # Step 4
    NODE_TAGGING = "node_tagging"        # Step 5
    TAG_INJECTION = "tag_injection"      # Step 6


@dataclass
class ExportData:
    """Unified export data shared across all writers."""
    
    # Basic info
    model_name: str = ""
    output_path: str = ""
    strategy: str = "htp"
    embed_hierarchy: bool = True
    
    # Timing
    start_time: float = field(default_factory=time.time)
    export_time: float = 0.0
    
    # Typed step data
    model_prep: ModelPrepData | None = None
    input_gen: InputGenData | None = None
    hierarchy: HierarchyData | None = None
    onnx_export: ONNXExportData | None = None
    node_tagging: NodeTaggingData | None = None
    tag_injection: TagInjectionData | None = None
    
    @property
    def timestamp(self) -> str:
        """Current timestamp in ISO format with millisecond precision."""
        import datetime
        dt = datetime.datetime.now(datetime.UTC)
        # Format with milliseconds (3 digits) instead of microseconds (6 digits)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    
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


class StepAwareWriter(io.IOBase, ABC):
    """Base class for step-aware writers following Python's IO protocol."""
    
    def __init__(self):
        """Initialize the writer and discover step handlers."""
        super().__init__()
        self._step_handlers: dict[ExportStep, Callable] = {}
        self._discover_handlers()
    
    def _discover_handlers(self) -> None:
        """Auto-discover step handler methods decorated with @step."""
        for name in dir(self):
            if name.startswith('_'):
                continue
            method = getattr(self, name)
            if hasattr(method, '_handles_step'):
                step_type = method._handles_step
                self._step_handlers[step_type] = method
    
    def write(self, export_step: ExportStep, data: ExportData) -> int:
        """
        Write data for a specific step.
        
        Args:
            export_step: The current export step
            data: The export data
            
        Returns:
            Number of bytes written (for IO protocol compliance)
        """
        handler = self._step_handlers.get(export_step, self._write_default)
        return handler(export_step, data)
    
    @abstractmethod
    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """
        Default handler for steps without specific handlers.
        
        Args:
            export_step: The current export step
            data: The export data
            
        Returns:
            Number of bytes written
        """
        pass
    
    def flush(self) -> None:
        """Flush any buffered data."""
        pass
    
    def close(self) -> None:
        """Close the writer and perform cleanup."""
        with contextlib.suppress(Exception):
            self.flush()
        super().close()
    
    # Required IO methods for protocol compliance
    def readable(self) -> bool:
        """This is a write-only stream."""
        return False
    
    def writable(self) -> bool:
        """This stream is writable."""
        return True
    
    def seekable(self) -> bool:
        """This stream is not seekable."""
        return False