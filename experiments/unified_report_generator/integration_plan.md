# Integration Plan for Unified Report Generator

## Overview
The unified report generator provides a single source of truth for all report formats, ensuring consistency and reducing code duplication.

## Key Changes Required

### 1. Add Import and Initialization
```python
from modelexport.reporting.unified_report_generator import ExportSession, UnifiedReportGenerator

class HTPExporter:
    def __init__(self, ...):
        # Existing init code...
        
        # Initialize export session
        self.export_session = ExportSession(
            strategy="htp",
            verbose=self.verbose,
            enable_reporting=self.enable_reporting,
            embed_hierarchy_attributes=self.embed_hierarchy_attributes
        )
        
        # Remove or deprecate:
        # - self.text_report_buffer (replaced by unified generator)
        # - self._export_report (data goes to export_session)
```

### 2. Update Each Step Method
Instead of updating `self._export_report`, update `self.export_session`:

```python
def _print_model_preparation(self, model: nn.Module, output_path: str) -> None:
    """Print Step 1: Model Preparation."""
    # Calculate data
    module_count = len(list(model.modules()))
    param_count = sum(p.numel() for p in model.parameters())
    
    # Update session
    self.export_session.model_class = type(model).__name__
    self.export_session.total_modules = module_count
    self.export_session.total_parameters = param_count
    self.export_session.output_path = output_path
    
    # Add step
    self.export_session.add_step(
        "model_preparation",
        "completed",
        model_class=type(model).__name__,
        module_count=module_count,
        parameter_count=param_count,
        eval_mode=True
    )
```

### 3. Update Data Collection
```python
# After hierarchy building
self.export_session.hierarchy_data = self._hierarchy_data

# After node tagging
self.export_session.tagged_nodes = self._tagged_nodes
self.export_session.tagging_statistics = self._tagging_stats
self.export_session.onnx_nodes_count = len(onnx_model.graph.node)
# etc.
```

### 4. Replace Report Generation
Replace the current metadata generation and report writing with:

```python
def _generate_all_reports(self, output_path: str) -> dict:
    """Generate all report formats using unified generator."""
    # Update file paths
    self.export_session.onnx_file_path = output_path
    self.export_session.metadata_file_path = str(output_path).replace('.onnx', '_htp_metadata.json')
    self.export_session.report_file_path = str(output_path).replace('.onnx', '_full_report.txt')
    
    # Get file size
    if Path(output_path).exists():
        self.export_session.onnx_file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    
    # Create generator
    generator = UnifiedReportGenerator(self.export_session)
    
    # Generate console output if verbose
    if self.verbose:
        console_output = generator.generate_console_output(truncate_trees=True)
        # Parse and print line by line to maintain current behavior
        for line in console_output.splitlines():
            self._output_message(line)
    
    # Generate and save metadata
    metadata = generator.generate_metadata()
    with open(self.export_session.metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate and save full report
    if self.enable_reporting or True:  # Always generate full report now
        full_report = generator.generate_text_report()
        with open(self.export_session.report_file_path, 'w') as f:
            f.write(full_report)
    
    return {
        "metadata_path": self.export_session.metadata_file_path,
        "report_path": self.export_session.report_file_path
    }
```

### 5. Clean Up
Remove or deprecate:
- `_generate_metadata_file` method (replaced by unified generator)
- `text_report_buffer` and related code
- Manual report writing code
- Duplicate formatting logic

## Benefits

1. **Consistency**: All outputs use the same data source
2. **Completeness**: No missing fields between formats
3. **Maintainability**: Single place to update report structure
4. **Extensibility**: Easy to add new output formats
5. **Testing**: Easier to test report generation in isolation

## Migration Strategy

1. **Phase 1**: Add unified generator alongside existing code
2. **Phase 2**: Gradually migrate each method to update export_session
3. **Phase 3**: Replace report generation with unified approach
4. **Phase 4**: Remove deprecated code

## Testing Plan

1. Ensure all existing tests pass
2. Add tests for unified generator
3. Verify output compatibility
4. Performance testing to ensure no regression