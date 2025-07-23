
HTP Export Monitor - Comprehensive API Documentation

The HTP Export Monitor provides a unified system for monitoring and reporting
on the Hierarchy-preserving Tags Protocol (HTP) export process.

## Overview

The export monitor consists of several key components:

1. **HTPExportMonitor**: Main orchestrator class
2. **StepAwareWriter**: Base class for step-based output writers
3. **HTPConsoleWriter**: Rich console output with text styling
4. **HTPMetadataWriter**: JSON metadata generation
5. **HTPReportWriter**: Human-readable text reports

## Usage

### Basic Usage

```python
from modelexport.strategies.htp.export_monitor import HTPExportMonitor

# Create monitor
monitor = HTPExportMonitor(
    output_path="model.onnx",
    model_name="bert-base",
    verbose=True
)

# Use throughout export process
monitor.log_step(HTPExportStep.MODEL_PREP, data)
monitor.log_step(HTPExportStep.HIERARCHY, data)
# ... more steps ...
monitor.finalize()
```

### Advanced Usage

```python
# Custom configuration
monitor = HTPExportMonitor(
    output_path="model.onnx",
    model_name="custom-model",
    verbose=True,
    config={
        "max_tree_depth": 15,
        "max_display_nodes": 50,
        "style_numbers": True,
        "batch_console": True
    }
)

# Access individual writers
console_output = monitor.console_writer.get_output()
metadata = monitor.metadata_writer.get_metadata()
report = monitor.report_writer.get_report()
```

## Export Steps

The HTP export process consists of 8 steps:

1. **MODEL_PREP**: Model loading and preparation
2. **INPUT_GEN**: Input tensor generation
3. **HIERARCHY**: Module hierarchy extraction
4. **TRACE**: PyTorch tracing
5. **EXPORT**: ONNX export
6. **NODE_TAGGING**: Tagging ONNX nodes with hierarchy
7. **SAVE**: Saving final ONNX model
8. **COMPLETE**: Export completion

## Data Format

### HTPExportData

```python
@dataclass
class HTPExportData:
    # Model information
    model_name: str
    model_class: str
    total_modules: int
    total_parameters: int
    
    # Export configuration
    output_path: str
    embed_hierarchy_attributes: bool
    
    # Hierarchy data
    hierarchy: Dict[str, ModuleInfo]
    execution_steps: int
    
    # Tagging results
    total_nodes: int
    tagged_nodes: Dict[str, str]
    tagging_stats: Dict[str, int]
    coverage: float
    
    # Timing
    export_time: float
    
    # Step-specific data
    steps: Dict[str, Any]
```

## Styling and Formatting

The console writer supports Rich text styling:

- Numbers: Bold cyan for emphasis
- Headers: Bold with separators
- Trees: Indented hierarchy display
- Progress: Step counters with colors

## Performance Considerations

- Use `batch_console=True` for better performance
- Set `max_tree_depth` for large models
- Enable caching with `enable_cache=True`
- Use streaming JSON for large exports

## Error Handling

The monitor handles errors gracefully:

```python
try:
    monitor.log_step(step, data)
except Exception as e:
    monitor.log_error(step, str(e))
    # Monitor continues with partial data
```

## Examples

### Minimal Export

```python
monitor = HTPExportMonitor("model.onnx", verbose=False)
# ... perform export ...
monitor.finalize()
```

### Full Featured Export

```python
monitor = HTPExportMonitor(
    output_path="model.onnx",
    model_name="bert-base-uncased",
    verbose=True,
    config={
        "max_tree_depth": 20,
        "style_numbers": True,
        "batch_console": True,
        "enable_cache": True
    }
)

# Log all steps with rich data
for step in HTPExportStep:
    monitor.log_step(step, export_data)

# Get outputs
console_log = monitor.get_console_output()
metadata = monitor.get_metadata()
report = monitor.get_report()

# Save all outputs
monitor.finalize()
```
