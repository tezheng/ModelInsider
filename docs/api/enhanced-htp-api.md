# Enhanced HTP API Reference

## Overview

This reference documents the API for enhanced auxiliary operations in the HTP (Hierarchical Trace-and-Project) strategy. The enhanced HTP provides 100% operation coverage through intelligent context inheritance and universal fallback strategies.

## Core Classes

### HierarchyExporter

The main class for enhanced auxiliary operations export.

```python
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter
```

#### Constructor

```python
class HierarchyExporter:
    def __init__(
        self,
        strategy: str = "htp",
        builtin_tracking: bool = True,
        enable_performance_monitoring: bool = False,
        verbose: bool = False
    )
```

**Parameters:**
- `strategy` (str): Export strategy identifier. Use `"htp"` for enhanced auxiliary operations
- `builtin_tracking` (bool): Enable PyTorch builtin tracking for enhanced auxiliary operation coverage (default: True)
- `enable_performance_monitoring` (bool): Enable detailed performance monitoring (default: False)
- `verbose` (bool): Enable verbose logging output (default: False)

**Example:**
```python
# Basic usage with enhanced auxiliary operations
exporter = HierarchyExporter(strategy="htp")

# With performance monitoring
exporter = HierarchyExporter(
    strategy="htp",
    enable_performance_monitoring=True,
    verbose=True
)
```

#### Methods

##### export()

Export a PyTorch model to ONNX with enhanced auxiliary operation coverage.

```python
def export(
    self,
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Tuple, Dict],
    output_path: str,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to export
- `example_inputs` (Union[torch.Tensor, Tuple, Dict]): Example inputs for tracing the model
- `output_path` (str): Path where to save the ONNX model
- `**kwargs`: Additional arguments passed to torch.onnx.export

**Returns:**
- `Dict[str, Any]`: Export result dictionary with enhanced metrics

**Example:**
```python
result = exporter.export(
    model=my_model,
    example_inputs=torch.randn(1, 3, 224, 224),
    output_path="enhanced_model.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["output"]
)
```

## Enhanced Result Format

The enhanced HTP strategy returns a comprehensive result dictionary:

### Core Fields

```python
{
    'output_path': str,              # Path to exported ONNX file
    'onnx_path': str,                # Same as output_path
    'strategy': str,                 # Strategy used (e.g., "htp_builtin")
    'total_operations': int,         # Total operations in ONNX graph
    'tagged_operations': int,        # Operations with hierarchy tags (should equal total_operations)
    'operation_trace_length': int,   # Length of operation trace captured
    'native_op_regions': int,        # Number of native PyTorch operation regions identified
    'builtin_tracking_enabled': bool # Whether enhanced tracking was used
}
```

### Enhanced Metrics

When performance monitoring is enabled, additional fields are included:

```python
{
    # Performance metrics
    'export_time': float,                    # Total export time in seconds
    'auxiliary_operations_coverage': float, # Coverage rate for auxiliary operations (0.0-1.0)
    'context_inheritance_success_rate': float, # Success rate of context inheritance
    'fallback_strategy_usage': int,         # Number of operations using fallback tagging
    
    # Auxiliary operation analysis
    'auxiliary_operations_analysis': {
        'total_auxiliary_ops': int,         # Total auxiliary operations found
        'tagged_auxiliary_ops': int,        # Auxiliary operations successfully tagged
        'context_inherited': int,           # Auxiliary ops tagged via context inheritance
        'fallback_tagged': int,             # Auxiliary ops tagged via fallback strategies
        'operation_types': Dict[str, int]   # Count by operation type
    },
    
    # Performance profiling
    'performance_profile': {
        'graph_context_building_time': float,    # Time spent building graph context
        'context_inheritance_time': float,       # Time spent on context inheritance
        'fallback_strategy_time': float,         # Time spent on fallback strategies
        'memory_peak_usage': int                 # Peak memory usage in MB
    }
}
```

### Usage Examples

#### Basic Coverage Validation

```python
result = exporter.export(model, inputs, "model.onnx")

# Validate 100% coverage
coverage_rate = result['tagged_operations'] / result['total_operations']
assert coverage_rate == 1.0, f"Coverage rate {coverage_rate:.1%}, expected 100%"

print(f"✅ Export successful with {coverage_rate:.1%} operation coverage")
print(f"📊 Strategy used: {result['strategy']}")
print(f"📊 Total operations: {result['total_operations']}")
```

#### Performance Analysis

```python
exporter = HierarchyExporter(
    strategy="htp",
    enable_performance_monitoring=True
)

result = exporter.export(model, inputs, "model.onnx")

# Analyze performance metrics
print(f"Export time: {result['export_time']:.2f}s")
print(f"Auxiliary operation coverage: {result['auxiliary_operations_coverage']:.1%}")
print(f"Context inheritance success: {result['context_inheritance_success_rate']:.1%}")

# Detailed auxiliary operation analysis
aux_analysis = result['auxiliary_operations_analysis']
print(f"Total auxiliary operations: {aux_analysis['total_auxiliary_ops']}")
print(f"Context inherited: {aux_analysis['context_inherited']}")
print(f"Fallback tagged: {aux_analysis['fallback_tagged']}")
```

#### Operation Type Analysis

```python
result = exporter.export(model, inputs, "model.onnx")

# Analyze auxiliary operation types
if 'auxiliary_operations_analysis' in result:
    op_types = result['auxiliary_operations_analysis']['operation_types']
    
    print("Auxiliary operation distribution:")
    for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_type}: {count}")
```

## Unified Export Interface Integration

The enhanced HTP integrates seamlessly with the unified export interface:

### UnifiedExporter

```python
from modelexport.unified_export import UnifiedExporter, ExportStrategy

# Use enhanced HTP explicitly
exporter = UnifiedExporter(
    strategy=ExportStrategy.HTP,
    enable_monitoring=True
)

result = exporter.export(model, inputs, "model.onnx")
```

### export_model() Function

```python
from modelexport.unified_export import export_model

# Convenient function interface
result = export_model(
    model=model,
    example_inputs=inputs,
    output_path="model.onnx",
    strategy="htp",  # Use enhanced auxiliary operations
    optimize=True,
    verbose=True
)
```

## Configuration Options

### Advanced Configuration

```python
# Configure enhanced auxiliary operations behavior
exporter = HierarchyExporter(
    strategy="htp",
    builtin_tracking=True,           # Enhanced tracking (recommended)
    enable_performance_monitoring=True,  # Detailed metrics
    verbose=True                     # Debug output
)

# Export with custom ONNX settings
result = exporter.export(
    model=model,
    example_inputs=inputs,
    output_path="model.onnx",
    
    # ONNX export settings
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    
    # Additional metadata
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL
)
```

### Environment Variables

Control enhanced auxiliary operations behavior via environment variables:

```bash
# Enable debug output for auxiliary operations
export MODELEXPORT_AUX_OPS_DEBUG=1

# Control memory usage limits
export MODELEXPORT_MEMORY_LIMIT_MB=2048

# Performance monitoring settings
export MODELEXPORT_ENABLE_PROFILING=1
```

## Error Handling

### Common Exceptions

```python
from modelexport.strategies.htp.exceptions import (
    AuxiliaryOperationError,
    ContextInheritanceError,
    CoverageValidationError
)

try:
    result = exporter.export(model, inputs, "model.onnx")
except AuxiliaryOperationError as e:
    print(f"Auxiliary operation processing failed: {e}")
except ContextInheritanceError as e:
    print(f"Context inheritance failed: {e}")
except CoverageValidationError as e:
    print(f"Coverage validation failed: {e}")
```

### Graceful Error Recovery

```python
def robust_export(model, inputs, output_path):
    """Export with graceful error handling and fallback."""
    
    try:
        # Try enhanced HTP first
        exporter = HierarchyExporter(strategy="htp")
        result = exporter.export(model, inputs, output_path)
        
        # Validate result
        coverage = result['tagged_operations'] / result['total_operations']
        if coverage < 1.0:
            print(f"⚠️ Partial coverage: {coverage:.1%}")
        
        return result
        
    except Exception as e:
        print(f"⚠️ Enhanced HTP failed: {e}")
        print("🔄 Falling back to usage-based strategy...")
        
        # Fallback to simpler strategy
        from modelexport.strategies.usage_based import UsageBasedExporter
        fallback_exporter = UsageBasedExporter()
        return fallback_exporter.export(model, inputs, output_path)
```

## Integration with Tag Utils

Enhanced auxiliary operations work with the tag utilities for analysis:

### Loading Enhanced Tags

```python
from modelexport.core import tag_utils

# Load hierarchy tags (includes auxiliary operations)
hierarchy_data = tag_utils.load_tags_from_sidecar("model.onnx")
node_tags = hierarchy_data['node_tags']

# Get tag statistics (enhanced coverage)
tag_stats = tag_utils.get_tag_statistics("model.onnx")
print(f"Unique hierarchy tags: {len(tag_stats)}")

# Query specific operations (includes auxiliary ops)
attention_ops = tag_utils.query_operations_by_tag("model.onnx", "Attention")
auxiliary_ops = tag_utils.query_operations_by_tag("model.onnx", "auxiliary")
```

### Validation with Enhanced Coverage

```python
# Validate enhanced tag consistency
validation_result = tag_utils.validate_tag_consistency("model.onnx")

if validation_result['consistent']:
    print("✅ Enhanced hierarchy tags are consistent")
else:
    print("❌ Tag inconsistencies found:")
    if 'tag_mismatches' in validation_result:
        for mismatch in validation_result['tag_mismatches'][:5]:
            print(f"  {mismatch['node']}: {mismatch['onnx_tags']} vs {mismatch['sidecar_tags']}")
```

## Performance Optimization

### Memory Management

```python
def memory_efficient_export(model, inputs, output_path):
    """Export large models with memory optimization."""
    
    import gc
    import torch
    
    # Clear cache before export
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Use minimal monitoring for large models
    exporter = HierarchyExporter(
        strategy="htp",
        enable_performance_monitoring=False,  # Reduce memory overhead
        verbose=False
    )
    
    try:
        result = exporter.export(model, inputs, output_path)
        return result
    finally:
        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
```

### Batch Processing

```python
def batch_export_models(model_configs, batch_size=3):
    """Export multiple models efficiently."""
    
    results = []
    
    for i in range(0, len(model_configs), batch_size):
        batch = model_configs[i:i + batch_size]
        
        for config in batch:
            exporter = HierarchyExporter(strategy="htp")
            result = exporter.export(
                model=config['model'],
                example_inputs=config['inputs'],
                output_path=config['output_path']
            )
            results.append(result)
        
        # Cleanup between batches
        import gc
        gc.collect()
    
    return results
```

## CLI Integration

The enhanced HTP is available through the CLI interface:

```bash
# Basic enhanced export
uv run modelexport export model.py output.onnx --strategy htp

# With configuration file
uv run modelexport export model.py output.onnx --strategy htp --config export_config.json

# Verbose output for debugging
uv run modelexport export model.py output.onnx --strategy htp --verbose
```

### Configuration File Format

```json
{
    "opset_version": 14,
    "input_names": ["input"],
    "output_names": ["output"],
    "dynamic_axes": {
        "input": {"0": "batch_size"}
    },
    "do_constant_folding": true,
    "input_specs": {
        "input_ids": {
            "dtype": "int",
            "range": [0, 50257]
        }
    }
}
```

## Migration from Legacy APIs

### From Usage-Based Strategy

```python
# Old usage-based approach
from modelexport.strategies.usage_based import UsageBasedExporter
old_exporter = UsageBasedExporter()
old_result = old_exporter.export(model, inputs, "old_output.onnx")

# New enhanced HTP approach
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter
new_exporter = HierarchyExporter(strategy="htp")
new_result = new_exporter.export(model, inputs, "new_output.onnx")

# Compare results
print(f"Old coverage: {old_result['tagged_operations']}/{old_result['total_operations']}")
print(f"New coverage: {new_result['tagged_operations']}/{new_result['total_operations']}")
```

### API Compatibility Layer

```python
def create_compatibility_wrapper():
    """Create wrapper for legacy API compatibility."""
    
    class LegacyCompatibleExporter:
        def __init__(self):
            self._exporter = HierarchyExporter(strategy="htp")
        
        def export(self, model, inputs, output_path, **kwargs):
            # Use enhanced HTP but return legacy-compatible format
            result = self._exporter.export(model, inputs, output_path, **kwargs)
            
            # Convert to legacy format if needed
            legacy_result = {
                'onnx_path': result['onnx_path'],
                'strategy': result['strategy'],
                'total_operations': result['total_operations'],
                'tagged_operations': result['tagged_operations']
            }
            
            return legacy_result
    
    return LegacyCompatibleExporter()
```

## Best Practices

### 1. **Error Handling and Validation**

```python
def validated_export(model, inputs, output_path):
    """Export with comprehensive validation."""
    
    exporter = HierarchyExporter(strategy="htp")
    result = exporter.export(model, inputs, output_path)
    
    # Validate coverage
    coverage = result['tagged_operations'] / result['total_operations']
    if coverage < 1.0:
        raise ValueError(f"Incomplete coverage: {coverage:.1%}")
    
    # Validate ONNX file
    import onnx
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        raise ValueError(f"Invalid ONNX model: {e}")
    
    return result
```

### 2. **Performance Monitoring**

```python
def monitored_export(model, inputs, output_path):
    """Export with performance monitoring."""
    
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    exporter = HierarchyExporter(
        strategy="htp",
        enable_performance_monitoring=True
    )
    
    result = exporter.export(model, inputs, output_path)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    # Add custom metrics
    result['custom_metrics'] = {
        'wall_clock_time': end_time - start_time,
        'memory_delta_mb': (end_memory - start_memory) / 1024 / 1024,
        'operations_per_second': result['total_operations'] / (end_time - start_time)
    }
    
    return result
```

### 3. **Resource Management**

```python
def resource_managed_export(model, inputs, output_path):
    """Export with automatic resource management."""
    
    import contextlib
    import tempfile
    
    @contextlib.contextmanager
    def managed_export():
        # Setup
        temp_dir = tempfile.mkdtemp()
        
        try:
            exporter = HierarchyExporter(strategy="htp")
            yield exporter
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    with managed_export() as exporter:
        return exporter.export(model, inputs, output_path)
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory Issues**
   ```python
   # Reduce memory usage
   exporter = HierarchyExporter(
       strategy="htp",
       enable_performance_monitoring=False  # Reduces memory overhead
   )
   ```

2. **Performance Issues**
   ```python
   # For development/testing, use simpler strategy
   if os.getenv('DEVELOPMENT_MODE'):
       exporter = UsageBasedExporter()  # Faster
   else:
       exporter = HierarchyExporter(strategy="htp")  # Production quality
   ```

3. **Coverage Issues**
   ```python
   # Debug coverage issues
   result = exporter.export(model, inputs, output_path)
   
   if result['tagged_operations'] < result['total_operations']:
       print(f"⚠️ Incomplete coverage detected")
       print(f"Tagged: {result['tagged_operations']}")
       print(f"Total: {result['total_operations']}")
       
       # Check auxiliary operation analysis
       if 'auxiliary_operations_analysis' in result:
           aux_analysis = result['auxiliary_operations_analysis']
           print(f"Auxiliary ops: {aux_analysis['total_auxiliary_ops']}")
           print(f"Successfully tagged: {aux_analysis['tagged_auxiliary_ops']}")
   ```

## API Reference Summary

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `HierarchyExporter` | Main export class | Enhanced auxiliary operations, 100% coverage |
| `export()` method | Core export function | Flexible input handling, comprehensive results |
| Result dictionary | Export metadata | Detailed metrics, performance data |
| Tag utilities | Analysis and validation | Enhanced coverage support |
| CLI integration | Command-line interface | Transparent enhanced functionality |

For additional examples and advanced usage patterns, see the [Integration Workflows Guide](integration-workflows.md) and [Examples Directory](../../examples/).