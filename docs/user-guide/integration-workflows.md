# Integration Workflows Guide

## Overview

This guide demonstrates how enhanced auxiliary operations integrate with existing ModelExport workflows and external tools. Learn how to adopt enhanced functionality without disrupting your current processes.

## Integration Principles

### 🔄 **Seamless Compatibility**
Enhanced auxiliary operations are designed with backward compatibility as a core principle:
- **Zero breaking changes** to existing APIs
- **Transparent enhancement** of existing functionality
- **Gradual adoption** supported - use when needed
- **Full integration** with strategy ecosystem

### 🎯 **Strategy Ecosystem Integration**

Enhanced auxiliary operations work within ModelExport's unified strategy framework:

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Export Interface                 │
├─────────────────────────────────────────────────────────────┤
│  Strategy Selector  │  Performance Monitor  │  Optimizer   │
├─────────────────────────────────────────────────────────────┤
│ Usage-Based │  FX Graph  │  Enhanced HTP (w/ Aux Ops)     │
└─────────────────────────────────────────────────────────────┘
```

## Workflow Integration Patterns

### Pattern 1: **Transparent Integration** (Recommended)

Use the unified interface and let the system automatically choose enhanced HTP when beneficial:

```python
from modelexport.unified_export import export_model

def export_with_smart_strategy(model, inputs, output_path):
    """Export with automatic strategy selection."""
    result = export_model(
        model=model,
        example_inputs=inputs,
        output_path=output_path,
        strategy="auto"  # Enhanced HTP selected automatically for complex models
    )
    
    strategy_used = result.get('summary', {}).get('final_strategy')
    coverage = get_coverage_from_result(result)
    
    print(f"Strategy used: {strategy_used}")
    print(f"Operation coverage: {coverage:.1f}%")
    
    return result

def get_coverage_from_result(result):
    """Extract coverage percentage from any strategy result."""
    if 'export_result' in result:
        export_result = result['export_result']
        total_ops = export_result.get('total_operations') or export_result.get('hierarchy_nodes', 1)
        tagged_ops = export_result.get('tagged_operations') or export_result.get('hierarchy_nodes', 0)
    else:
        total_ops = result.get('total_operations', 1)
        tagged_ops = result.get('tagged_operations', 0)
    
    return (tagged_ops / max(total_ops, 1)) * 100
```

### Pattern 2: **Explicit Enhanced Strategy**

Directly use enhanced HTP when you need guaranteed 100% coverage:

```python
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter

def export_with_enhanced_coverage(model, inputs, output_path):
    """Export with guaranteed 100% auxiliary operation coverage."""
    exporter = HierarchyExporter(strategy="htp")
    
    result = exporter.export(
        model=model,
        example_inputs=inputs,
        output_path=output_path
    )
    
    # Enhanced HTP guarantees 100% coverage
    assert result['tagged_operations'] == result['total_operations']
    
    return result
```

### Pattern 3: **Fallback Integration**

Use enhanced HTP as a fallback when other strategies fail:

```python
from modelexport.strategies.fx import FXHierarchyExporter
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter

def export_with_fallback(model, inputs, output_path):
    """Try FX first, fall back to enhanced HTP if needed."""
    
    # Try FX first (faster)
    try:
        fx_exporter = FXHierarchyExporter()
        result = fx_exporter.export(model, inputs, output_path)
        print("✅ FX strategy succeeded")
        return result
        
    except Exception as e:
        print(f"⚠️ FX strategy failed: {e}")
        print("🔄 Falling back to enhanced HTP...")
        
        # Fall back to enhanced HTP (more robust)
        htp_exporter = HierarchyExporter(strategy="htp")
        result = htp_exporter.export(model, inputs, output_path)
        print("✅ Enhanced HTP fallback succeeded")
        return result
```

## Integration with External Tools

### Graph Analysis Tools

Enhanced auxiliary operations improve graph analysis by ensuring complete hierarchy information:

```python
import onnx
from modelexport.core import tag_utils

def analyze_model_hierarchy(onnx_path):
    """Analyze model hierarchy with enhanced auxiliary operation support."""
    
    # Load hierarchy tags (now includes auxiliary operations)
    hierarchy_data = tag_utils.load_tags_from_sidecar(onnx_path)
    node_tags = hierarchy_data['node_tags']
    
    # Analyze hierarchy distribution
    tag_stats = tag_utils.get_tag_statistics(onnx_path)
    
    print(f"Total tagged operations: {len(node_tags)}")
    print(f"Unique hierarchy tags: {len(tag_stats)}")
    
    # Filter by specific components (now includes auxiliary ops)
    attention_ops = tag_utils.query_operations_by_tag(onnx_path, "Attention")
    print(f"Attention-related operations: {len(attention_ops)}")
    
    return {
        'total_operations': len(node_tags),
        'hierarchy_coverage': len(tag_stats),
        'attention_operations': len(attention_ops)
    }
```

### Model Optimization Workflows

Enhanced auxiliary operations enable safer model optimization:

```python
def optimize_model_subgraph(onnx_path, target_component):
    """Optimize specific model components safely."""
    
    # Filter operations by hierarchy tag (auxiliary ops included)
    component_ops = tag_utils.query_operations_by_tag(onnx_path, target_component)
    
    if not component_ops:
        raise ValueError(f"No operations found for component: {target_component}")
    
    # Extract subgraph safely (no malformed graphs due to missing auxiliary ops)
    subgraph = extract_subgraph(onnx_path, component_ops)
    
    # Apply optimization
    optimized_subgraph = apply_optimization(subgraph)
    
    # Replace in original model
    optimized_model = replace_subgraph(onnx_path, optimized_subgraph)
    
    return optimized_model

def extract_subgraph(onnx_path, operation_list):
    """Extract subgraph including auxiliary operations."""
    model = onnx.load(onnx_path)
    
    # Thanks to enhanced auxiliary operations, no missing dependencies
    subgraph_nodes = []
    for op in operation_list:
        node = find_node_by_name(model, op)
        if node:
            subgraph_nodes.append(node)
    
    # Create valid subgraph (auxiliary ops ensure completeness)
    return create_subgraph(subgraph_nodes)
```

### Model Validation Pipelines

Integrate enhanced auxiliary operations into validation workflows:

```python
def validate_export_pipeline(model, inputs, validation_checks=None):
    """Complete export validation pipeline."""
    
    if validation_checks is None:
        validation_checks = [
            'coverage_check',
            'graph_integrity_check', 
            'hierarchy_consistency_check',
            'auxiliary_operation_check'
        ]
    
    # Export with enhanced auxiliary operations
    temp_path = "temp_validation_export.onnx"
    exporter = HierarchyExporter(strategy="htp")
    result = exporter.export(model, inputs, temp_path)
    
    validation_results = {}
    
    # Coverage validation
    if 'coverage_check' in validation_checks:
        coverage = (result['tagged_operations'] / result['total_operations']) * 100
        validation_results['coverage'] = {
            'passed': coverage == 100.0,
            'value': coverage,
            'expected': 100.0
        }
    
    # Graph integrity validation
    if 'graph_integrity_check' in validation_checks:
        integrity_check = validate_onnx_graph_integrity(temp_path)
        validation_results['graph_integrity'] = integrity_check
    
    # Hierarchy consistency validation  
    if 'hierarchy_consistency_check' in validation_checks:
        consistency_check = tag_utils.validate_tag_consistency(temp_path)
        validation_results['hierarchy_consistency'] = consistency_check
    
    # Auxiliary operation validation
    if 'auxiliary_operation_check' in validation_checks:
        aux_check = validate_auxiliary_operation_coverage(temp_path)
        validation_results['auxiliary_operations'] = aux_check
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)
    
    return validation_results

def validate_auxiliary_operation_coverage(onnx_path):
    """Validate that auxiliary operations are properly tagged."""
    model = onnx.load(onnx_path)
    hierarchy_data = tag_utils.load_tags_from_sidecar(onnx_path)
    
    # Find auxiliary operations
    auxiliary_ops = []
    for node in model.graph.node:
        if node.op_type in ['Shape', 'Constant', 'Cast', 'Reshape', 'Transpose', 
                           'Unsqueeze', 'Squeeze', 'Where', 'Gather', 'ReduceMean']:
            auxiliary_ops.append(node.name)
    
    # Check coverage
    tagged_aux_ops = []
    for op in auxiliary_ops:
        if op in hierarchy_data['node_tags'] and hierarchy_data['node_tags'][op]:
            tagged_aux_ops.append(op)
    
    coverage_rate = len(tagged_aux_ops) / max(len(auxiliary_ops), 1) * 100
    
    return {
        'passed': coverage_rate == 100.0,
        'total_auxiliary_ops': len(auxiliary_ops),
        'tagged_auxiliary_ops': len(tagged_aux_ops),
        'coverage_rate': coverage_rate
    }
```

## CLI Integration Workflows

### Development Workflow

```bash
# Development: Use faster strategy for iteration
uv run modelexport export my-model.py model_dev.onnx --strategy usage_based

# Testing: Use enhanced HTP to catch auxiliary operation issues
uv run modelexport export my-model.py model_test.onnx --strategy htp

# Validation: Compare results and verify enhanced coverage
uv run modelexport compare model_dev.onnx model_test.onnx

# Analysis: Verify auxiliary operation handling
uv run modelexport analyze model_test.onnx --filter-tag "auxiliary"
```

### Production Workflow

```bash
# Production: Use auto strategy selection for optimal balance
uv run modelexport export production-model.py production.onnx --strategy auto

# Quality assurance: Validate enhanced coverage
uv run modelexport validate production.onnx --check-consistency

# Performance monitoring: Track export metrics
uv run modelexport analyze production.onnx --output-format summary
```

### Batch Processing Workflow

```python
import subprocess
from pathlib import Path

def batch_export_with_enhanced_coverage(model_configs, output_dir):
    """Batch export multiple models with enhanced auxiliary operations."""
    
    results = []
    
    for config in model_configs:
        model_name = config['name']
        model_path = config['path']
        
        output_path = Path(output_dir) / f"{model_name}_enhanced.onnx"
        
        # Use CLI for batch processing
        cmd = [
            'uv', 'run', 'modelexport', 'export',
            model_path, str(output_path),
            '--strategy', 'htp',  # Enhanced auxiliary operations
            '--verbose'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Validate result
            validation_cmd = [
                'uv', 'run', 'modelexport', 'validate', 
                str(output_path), '--check-consistency'
            ]
            validation_result = subprocess.run(validation_cmd, capture_output=True, text=True)
            
            results.append({
                'model': model_name,
                'status': 'success',
                'output_path': str(output_path),
                'validation_passed': validation_result.returncode == 0
            })
            
        except subprocess.CalledProcessError as e:
            results.append({
                'model': model_name,
                'status': 'failed',
                'error': e.stderr
            })
    
    return results
```

## Migration Strategies

### Gradual Migration Approach

1. **Assessment Phase**
   ```python
   # Assess current workflow impact
   def assess_current_workflow(existing_models):
       for model_path in existing_models:
           # Analyze with current approach
           current_result = analyze_current_export(model_path)
           
           # Test with enhanced auxiliary operations
           enhanced_result = test_enhanced_export(model_path)
           
           # Compare results
           comparison = compare_export_results(current_result, enhanced_result)
           print(f"Model: {model_path}")
           print(f"Coverage improvement: {comparison['coverage_improvement']}")
           print(f"Export time impact: {comparison['time_impact']}")
   ```

2. **Pilot Testing Phase**
   ```python
   # Test enhanced auxiliary operations on subset
   pilot_models = select_pilot_models(all_models, criteria=['complex', 'critical'])
   
   for model in pilot_models:
       result = export_with_enhanced_coverage(model, inputs, output_path)
       validate_enhanced_result(result)
   ```

3. **Gradual Rollout Phase**
   ```python
   # Roll out by model complexity
   def rollout_enhanced_coverage(models, rollout_strategy='complexity'):
       if rollout_strategy == 'complexity':
           # Start with most complex models (most benefit)
           sorted_models = sort_by_complexity(models)
       elif rollout_strategy == 'criticality':
           # Start with most critical models (most important)
           sorted_models = sort_by_criticality(models)
       
       for model in sorted_models:
           migrate_model_to_enhanced(model)
   ```

### Performance Optimization During Migration

```python
def optimize_migration_performance():
    """Optimize performance during migration to enhanced auxiliary operations."""
    
    # 1. Batch processing
    def batch_models(model_list, batch_size=5):
        for i in range(0, len(model_list), batch_size):
            yield model_list[i:i + batch_size]
    
    # 2. Resource monitoring
    def monitor_resources():
        import psutil
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 80:
            print("⚠️ High memory usage, reducing batch size")
            return True
        return False
    
    # 3. Progressive enhancement
    def progressive_enhancement(model, start_simple=True):
        if start_simple:
            # Try simpler strategy first
            try:
                return export_with_usage_based(model)
            except:
                # Fall back to enhanced HTP
                return export_with_enhanced_coverage(model)
        else:
            # Use enhanced HTP directly
            return export_with_enhanced_coverage(model)
```

## Troubleshooting Integration Issues

### Common Integration Challenges

1. **Performance Impact**
   ```python
   def handle_performance_concerns(model, target_export_time=10.0):
       """Handle cases where enhanced auxiliary operations impact performance."""
       
       start_time = time.time()
       result = export_with_enhanced_coverage(model, inputs, output_path)
       export_time = time.time() - start_time
       
       if export_time > target_export_time:
           print(f"⚠️ Export time {export_time:.1f}s exceeds target {target_export_time}s")
           print("Consider using enhanced HTP only for final production exports")
           
           # Provide alternative workflow
           return suggest_alternative_workflow(model, target_export_time)
       
       return result
   ```

2. **Memory Usage Issues**
   ```python
   def handle_memory_issues(model):
       """Handle high memory usage during enhanced export."""
       import psutil
       
       initial_memory = psutil.virtual_memory().used
       
       try:
           result = export_with_enhanced_coverage(model, inputs, output_path)
           
           peak_memory = psutil.virtual_memory().used
           memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
           
           if memory_increase > 1000:  # > 1GB increase
               print(f"⚠️ High memory usage: +{memory_increase:.0f}MB")
               print("Consider using enhanced HTP for smaller models or offline processing")
           
           return result
           
       except MemoryError:
           print("❌ Out of memory during enhanced export")
           return fallback_to_simpler_strategy(model)
   ```

3. **Legacy Compatibility**
   ```python
   def ensure_legacy_compatibility(output_path):
       """Ensure enhanced exports work with legacy tools."""
       
       # Validate ONNX compatibility
       try:
           import onnx
           model = onnx.load(output_path)
           onnx.checker.check_model(model)
           print("✅ ONNX model validation passed")
       except Exception as e:
           print(f"❌ ONNX validation failed: {e}")
           return False
       
       # Check hierarchy format compatibility
       try:
           hierarchy_data = tag_utils.load_tags_from_sidecar(output_path)
           assert 'node_tags' in hierarchy_data
           assert 'strategy' in hierarchy_data
           print("✅ Hierarchy format validation passed")
       except Exception as e:
           print(f"❌ Hierarchy format validation failed: {e}")
           return False
       
       return True
   ```

## Best Practices for Integration

### 1. **Incremental Adoption**
- Start with non-critical models for testing
- Validate results against existing workflows
- Monitor performance impact and adjust accordingly

### 2. **Quality Gates**
```python
def implement_quality_gates(export_result):
    """Implement quality gates for enhanced exports."""
    
    quality_checks = {
        'coverage_check': lambda r: (r['tagged_operations'] / r['total_operations']) == 1.0,
        'performance_check': lambda r: r.get('export_time', 0) < 30.0,
        'memory_check': lambda r: True,  # Implement based on your constraints
    }
    
    for check_name, check_func in quality_checks.items():
        if not check_func(export_result):
            raise QualityGateFailure(f"Quality gate failed: {check_name}")
    
    return True
```

### 3. **Monitoring and Alerting**
```python
def setup_enhanced_export_monitoring():
    """Set up monitoring for enhanced auxiliary operations."""
    
    metrics = {
        'export_success_rate': 0,
        'average_coverage_rate': 0,
        'average_export_time': 0,
        'memory_usage_peak': 0
    }
    
    def update_metrics(export_result):
        # Update success rate
        if export_result.get('success', False):
            metrics['export_success_rate'] += 1
        
        # Update coverage rate
        coverage = (export_result['tagged_operations'] / export_result['total_operations']) * 100
        metrics['average_coverage_rate'] = update_average(metrics['average_coverage_rate'], coverage)
        
        # Update timing
        export_time = export_result.get('export_time', 0)
        metrics['average_export_time'] = update_average(metrics['average_export_time'], export_time)
    
    return update_metrics
```

## Summary

Enhanced auxiliary operations integrate seamlessly with existing ModelExport workflows while providing significant improvements in coverage and quality. The key to successful integration is:

1. **Start small**: Test with non-critical models first
2. **Monitor impact**: Track performance and quality metrics
3. **Gradual rollout**: Migrate models systematically based on complexity or criticality
4. **Quality gates**: Implement validation checks for enhanced exports
5. **Fallback strategies**: Maintain ability to use simpler approaches when needed

The enhanced functionality is designed to be transparent and additive - you get better results without changing your workflows.