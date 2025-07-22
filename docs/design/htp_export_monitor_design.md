# HTP Export Monitor Design - IO/ABC Pattern

## Overview

This document describes the restored design for the HTP Export Monitor system using Python's IO classes and Abstract Base Classes (ABC) for a clean, extensible architecture.

## Core Architecture

### 1. Abstract Base Class with IO Protocol

```python
from abc import ABC, abstractmethod
import io
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Union

class StepAwareWriter(io.IOBase, ABC):
    """Base class for step-aware writers following Python's IO protocol."""
    
    def __init__(self):
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
        """Write data for a specific step."""
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
```

### 2. Step Decorator Pattern

```python
def step(export_step: ExportStep):
    """Decorator to mark step-specific handler methods."""
    def decorator(func: Callable) -> Callable:
        func._handles_step = export_step
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## Data Models

### Export Steps Enumeration

```python
class ExportStep(Enum):
    """HTP export process steps."""
    MODEL_PREP = "model_preparation"      # Step 1
    INPUT_GEN = "input_generation"        # Step 2
    HIERARCHY = "hierarchy_building"      # Step 3
    ONNX_EXPORT = "onnx_export"          # Step 4
    NODE_TAGGING = "node_tagging"        # Step 5
    TAG_INJECTION = "tag_injection"      # Step 6
```

### Structured Step Data (Replacing dict[str, Any])

```python
@dataclass
class ModelPrepData:
    """Data for model preparation step."""
    model_class: str
    total_modules: int
    total_parameters: int

@dataclass
class InputGenData:
    """Data for input generation step."""
    method: str  # "provided" or "auto_generated"
    model_type: str | None = None
    task: str | None = None
    inputs: dict[str, dict[str, Union[list[int], str]]] = field(default_factory=dict)
    # inputs format: {"input_ids": {"shape": [1, 128], "dtype": "int64"}}

@dataclass
class HierarchyData:
    """Data for hierarchy building step."""
    hierarchy: dict[str, dict[str, Any]]  # Module path -> info
    execution_steps: int
    module_list: list[tuple[str, str]] = field(default_factory=list)

@dataclass
class ONNXExportData:
    """Data for ONNX export step."""
    opset_version: int = 17
    do_constant_folding: bool = True
    verbose: bool = False
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] | None = None
    onnx_size_mb: float = 0.0

@dataclass
class NodeTaggingData:
    """Data for node tagging step."""
    total_nodes: int
    tagged_nodes: dict[str, str]  # node_name -> hierarchy_tag
    tagging_stats: dict[str, int]  # statistics
    coverage: float
    op_counts: dict[str, int] = field(default_factory=dict)

@dataclass
class TagInjectionData:
    """Data for tag injection step."""
    tags_injected: bool
    tags_stripped: bool = False  # For --clean-onnx mode
```

### Unified Export Data

```python
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
        """Current timestamp in ISO format."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    @property
    def elapsed_time(self) -> float:
        """Total elapsed time."""
        return time.time() - self.start_time
```

## Writer Implementations

### 1. Console Writer

```python
class ConsoleWriter(StepAwareWriter):
    """Real-time console output with Rich formatting."""
    
    def __init__(self, width: int = 120, verbose: bool = True):
        super().__init__()
        self.console = Console(
            width=width,
            force_terminal=True,
            legacy_windows=False,
            highlight=False  # Disable automatic highlighting
        )
        self.verbose = verbose
        self._current_step = 0
        self._total_steps = 6
```

### 2. Metadata Writer

```python
class MetadataWriter(StepAwareWriter):
    """JSON metadata writer using HTPMetadataBuilder."""
    
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.metadata_path = f"{self.output_path}_htp_metadata.json"
        self.builder = HTPMetadataBuilder()
```

### 3. Report Writer

```python
class ReportWriter(StepAwareWriter):
    """Full text report writer with complete console capture."""
    
    def __init__(self, output_path: str, console_buffer: io.StringIO | None = None):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.report_path = f"{self.output_path}_htp_export_report.txt"
        self.buffer = io.StringIO()
        self.console_buffer = console_buffer  # For capturing console output
```

## Central Orchestrator

```python
class HTPExportMonitor:
    """Central monitor that coordinates data updates and writer dispatch."""
    
    def __init__(
        self,
        output_path: str,
        model_name: str = "",
        verbose: bool = True,
        enable_report: bool = True,
        embed_hierarchy: bool = True,
    ):
        self.data = ExportData(
            model_name=model_name,
            output_path=output_path,
            embed_hierarchy=embed_hierarchy,
        )
        self.writers: list[StepAwareWriter] = []
        
        # Console capture for report
        self.console_buffer = io.StringIO() if enable_report else None
        
        # Initialize writers
        if verbose:
            console = Console(file=self.console_buffer) if self.console_buffer else None
            self.writers.append(ConsoleWriter(console=console))
        
        self.writers.append(MetadataWriter(output_path))
        
        if enable_report:
            self.writers.append(ReportWriter(output_path, self.console_buffer))
    
    def update(self, step: ExportStep, **kwargs) -> None:
        """Update data and notify all writers."""
        # Update typed step data
        self._update_step_data(step, kwargs)
        
        # Notify all writers
        for writer in self.writers:
            try:
                writer.write(step, self.data)
            except Exception as e:
                print(f"Error in {writer.__class__.__name__}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finalize all writers."""
        if exc_type is None:
            self.finalize_export()
        # Close all writers
        for writer in self.writers:
            writer.close()
```

## Key Design Benefits

1. **Extensibility**: Easy to add new writers by extending StepAwareWriter
2. **Type Safety**: Structured step data instead of dict[str, Any]
3. **Separation of Concerns**: Each writer handles its format independently
4. **IO Protocol Compliance**: Proper use of Python's io.IOBase
5. **Testability**: Each component can be tested in isolation
6. **Maintainability**: Clean, documented interfaces

## Migration Strategy

1. Create new implementation in `htp_new/` folder
2. Implement writers one by one with comprehensive tests
3. Ensure backward compatibility with existing API
4. Compare outputs with baseline to ensure no regression
5. Replace old implementation once all tests pass

## Output Specifications

The HTP export monitor generates three main outputs:
1. **Console Output**: Real-time progress displayed to terminal
2. **Metadata (JSON)**: Structured data about the export process
3. **Report (TXT)**: Human-readable report with complete information

For detailed specifications of metadata structure and report format, see:
- [HTP Metadata and Report Specification](../HTP_METADATA_REPORT_SPEC.md)

The metadata follows a strict JSON schema for validation and programmatic access, while the report provides a comprehensive human-readable summary including all console output and complete module/node mappings.

## Testing Strategy

### Unit Tests
- Test each writer independently
- Test step handler discovery
- Test data conversion
- Test buffer management

### Integration Tests
- Test full export flow
- Test error handling
- Test different configurations

### E2E Tests
- Compare with baseline outputs
- Verify all files generated correctly
- Ensure console output matches exactly
- Validate metadata against JSON schema
- Verify report contains all required sections