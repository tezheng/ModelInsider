# ExportMonitor System

A Pythonic unified monitoring system for the HTP exporter with step-aware writers using the decorator pattern.

## Key Features

1. **ExportMonitor**: Central coordinator that manages data and dispatches to writers
2. **@step decorator**: Marks methods for handling specific export steps
3. **Three Writers**:
   - **ConsoleWriter**: Real-time output with Rich formatting
   - **MetadataWriter**: Buffered JSON output
   - **ReportWriter**: Buffered text report
4. **Conditional Creation**: Writers created based on flags (verbose, enable_report)
5. **Shared Base Class**: StepAwareWriter extends Python's io.IOBase
6. **Format Awareness**: Each writer knows its format requirements

## Architecture

```python
# Core Components
ExportStep (Enum) - Export process steps
ExportData (dataclass) - Unified data model
StepAwareWriter (ABC) - Base class with @step support
ExportMonitor - Central coordinator

# Writers
ConsoleWriter - Real-time Rich output
MetadataWriter - JSON metadata
ReportWriter - Full text report
```

## Usage

```python
with ExportMonitor(output_path, verbose=True, enable_report=True) as monitor:
    # Update for each step
    monitor.update(
        ExportStep.MODEL_PREP,
        model_name="bert-base",
        model_class="BertModel",
        total_modules=199
    )
    
    # More steps...
    monitor.update(ExportStep.HIERARCHY, hierarchy=hierarchy_data)
    
    # Auto-finalize on exit
```

## Integration with HTP Exporter

```python
def export(self, model, dummy_inputs, output_path):
    with ExportMonitor(output_path, self.verbose, self.enable_report) as monitor:
        # Model preparation
        monitor.update(ExportStep.MODEL_PREP, ...)
        
        # Build hierarchy
        hierarchy = self._build_hierarchy(model)
        monitor.update(ExportStep.HIERARCHY, hierarchy=hierarchy)
        
        # Continue with export steps...
```

## Step-Specific Handling

Writers can override specific steps using the @step decorator:

```python
class CustomWriter(StepAwareWriter):
    @step(ExportStep.HIERARCHY)
    def write_hierarchy(self, step: ExportStep, data: ExportData) -> int:
        # Custom hierarchy handling
        return 1
    
    def _write_default(self, step: ExportStep, data: ExportData) -> int:
        # Default handler for other steps
        return 1
```

## Benefits

1. **Clean Separation**: Data, logic, and presentation are separated
2. **Real-time Feedback**: Console shows progress as it happens
3. **Buffered Output**: Files written atomically at the end
4. **Extensible**: Easy to add new writers or steps
5. **Pythonic**: Uses dataclasses, decorators, and standard IO patterns
6. **Type Safe**: Proper typing throughout