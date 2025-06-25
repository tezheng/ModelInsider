# TBD (To Be Done) - Future Enhancements

This file contains suggestions for future improvements to the hierarchy-preserving ONNX export system.

## 1. Multi-Architecture Testing

**Current**: Only BERT tiny testing  
**Enhancement**: Test multiple architectures to validate universality

### Test Models
```python
test_models = [
    "google/bert_uncased_L-2_H-128_A-2",  # BERT (encoder-only)
    "gpt2",                              # GPT (decoder-only)  
    "google/flan-t5-small",             # T5 (encoder-decoder)
    "microsoft/DialoGPT-small"          # Different GPT variant
]
```

### Benefits
- Validate universal approach works across architectures
- Identify architecture-specific edge cases
- Build confidence in the extraction method

## 2. Validation/Verification Suite

**Current**: Generate data, hope it's correct  
**Enhancement**: Add comprehensive validation steps

### A. DAG Completeness Validation
```python
def validate_dag_completeness(model, extracted_dags, operation_metadata):
    """Verify we captured all meaningful operations"""
    
    # Count total parameters in model
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count parameters in our extraction
    extracted_params = len([op for op in operation_metadata.values() 
                           if op['op_type'] == 'Initializer'])
    
    # Validate parameter coverage
    param_coverage = extracted_params / len(list(model.named_parameters()))
    assert param_coverage > 0.9, f"Low parameter coverage: {param_coverage}"
    
    # Validate module coverage
    total_modules = len(list(model.named_modules()))
    extracted_modules = len(extracted_dags)
    module_coverage = extracted_modules / total_modules
    
    return {
        'param_coverage': param_coverage,
        'module_coverage': module_coverage,
        'total_operations': len(operation_metadata)
    }
```

### B. Operation Flow Validation
```python
def validate_operation_flow(extracted_dags, operation_metadata):
    """Verify operations connect correctly"""
    
    for module_path, dag in extracted_dags.items():
        nodes = dag['nodes']
        edges = dag['edges']
        
        # Check for orphaned nodes (no inputs/outputs)
        connected_nodes = set()
        for edge in edges:
            connected_nodes.update(edge)
        
        orphaned = set(nodes) - connected_nodes
        
        # Check for cyclic dependencies
        has_cycle = detect_cycle_in_dag(edges)
        
        # Validate input/output consistency
        for edge in edges:
            source, target = edge
            if source in operation_metadata and target in operation_metadata:
                source_outputs = operation_metadata[source]['outputs']
                target_inputs = operation_metadata[target]['inputs']
                
                # Check if they actually connect
                connection_exists = any(out in target_inputs for out in source_outputs)
                
        yield {
            'module': module_path,
            'orphaned_nodes': list(orphaned),
            'has_cycle': has_cycle,
            'edge_count': len(edges),
            'node_count': len(nodes)
        }
```

### C. Cross-Reference with PyTorch Model
```python
def validate_against_pytorch_model(model, operation_metadata):
    """Cross-check extracted data with actual PyTorch model"""
    
    validation_results = {}
    
    # Validate parameter names and shapes
    for param_name, param in model.named_parameters():
        onnx_param_name = param_name.replace('.', '_')
        
        if onnx_param_name in operation_metadata:
            onnx_data = operation_metadata[onnx_param_name]
            
            # Check shape consistency
            pytorch_shape = list(param.shape)
            onnx_shape = onnx_data.get('shape', [])
            
            validation_results[param_name] = {
                'shape_match': pytorch_shape == onnx_shape,
                'found_in_onnx': True,
                'pytorch_shape': pytorch_shape,
                'onnx_shape': onnx_shape
            }
        else:
            validation_results[param_name] = {
                'found_in_onnx': False,
                'pytorch_shape': list(param.shape)
            }
    
    return validation_results
```

## 3. Shared Parameter Detection

**Current**: Basic tagging  
**Enhancement**: Explicit shared parameter analysis

### A. Identify Shared Parameters
```python
def detect_shared_parameters(operation_metadata):
    """Find parameters used by multiple modules"""
    
    # Group by parameter name (excluding module-specific prefixes)
    param_usage = defaultdict(list)
    
    for op_name, op_data in operation_metadata.items():
        if op_data['op_type'] == 'Initializer':
            # Extract base parameter name (weight, bias, etc.)
            base_name = op_name.split('.')[-1]  # e.g., 'weight', 'bias'
            
            for tag in op_data['tags']:
                param_usage[op_name].append(tag)
    
    # Find truly shared parameters (same tensor, multiple users)
    shared_params = {}
    for param_name, modules in param_usage.items():
        if len(modules) > 1:
            shared_params[param_name] = {
                'used_by': modules,
                'share_count': len(modules),
                'param_type': param_name.split('.')[-1]
            }
    
    return shared_params
```

### B. Analyze Sharing Patterns
```python
def analyze_sharing_patterns(shared_params, model_config):
    """Analyze common sharing patterns in the model"""
    
    patterns = {
        'tied_embeddings': [],      # Input/output embeddings shared
        'layer_sharing': [],        # Same weights across layers  
        'attention_sharing': [],    # Shared attention parameters
        'position_sharing': []      # Shared positional embeddings
    }
    
    for param_name, share_info in shared_params.items():
        modules = share_info['used_by']
        
        # Detect tied embeddings (common in language models)
        if ('embedding' in param_name.lower() and 
            any('prediction' in mod.lower() or 'lm_head' in mod.lower() 
                for mod in modules)):
            patterns['tied_embeddings'].append({
                'parameter': param_name,
                'modules': modules
            })
        
        # Detect layer parameter sharing
        if len(set(extract_layer_number(mod) for mod in modules)) > 1:
            patterns['layer_sharing'].append({
                'parameter': param_name,
                'shared_across_layers': modules
            })
        
        # Detect position embedding sharing
        if 'position' in param_name.lower():
            patterns['position_sharing'].append({
                'parameter': param_name,
                'modules': modules
            })
    
    return patterns

def extract_layer_number(module_path):
    """Extract layer number from module path"""
    import re
    match = re.search(r'layer\.(\d+)', module_path)
    return int(match.group(1)) if match else None
```

### C. Enhanced Output Format
```json
{
  "operation_metadata": { /* existing */ },
  "module_dags": { /* existing */ },
  "validation_results": {
    "param_coverage": 0.95,
    "module_coverage": 0.87,
    "total_operations": 290,
    "orphaned_nodes": [],
    "invalid_connections": []
  },
  "shared_parameters": {
    "word_embeddings.weight": {
      "used_by": ["/BertModel/BertEmbeddings", "/BertModel/BertPredictionHead"],
      "share_count": 2,
      "param_type": "weight",
      "sharing_pattern": "tied_embeddings"
    }
  },
  "sharing_analysis": {
    "tied_embeddings": [...],
    "layer_sharing": [...],
    "total_shared_params": 3,
    "sharing_percentage": 0.15
  }
}
```

## 4. Performance Metrics

**Enhancement**: Track extraction performance and quality

### Metrics to Track
- Time to analyze model structure
- Memory usage during extraction
- ONNX export time
- Tag injection time
- Coverage percentage (% of nodes successfully tagged)
- Validation success rate

### Implementation
```python
@dataclass
class ExtractionMetrics:
    analysis_time: float
    export_time: float
    tag_injection_time: float
    memory_peak_mb: float
    coverage_percentage: float
    validation_success_rate: float
    total_nodes: int
    tagged_nodes: int
```

## 5. Error Handling & Edge Cases

**Enhancement**: Better handling of problematic scenarios

### Edge Cases to Handle
- Modules without operations
- Dynamic/conditional operations
- Unsupported operation types
- Models with custom operations
- Very large models (memory constraints)
- Models with shared submodules

### Robust Error Handling
```python
def safe_extract_dag(model, fallback_strategy='skip'):
    """Extract DAG with graceful error handling"""
    try:
        return extract_full_dag(model)
    except UnsupportedOperationError as e:
        if fallback_strategy == 'skip':
            return extract_supported_ops_only(model)
        elif fallback_strategy == 'approximate':
            return extract_approximate_dag(model)
        else:
            raise
```

## 6. Integration & Tooling

**Enhancement**: Better integration with existing tools

### A. Command Line Interface
```bash
# Extract DAG for any model
python -m modelexport.extract --model "bert-base-uncased" --output results/

# Validate extraction
python -m modelexport.validate --input results/ --model "bert-base-uncased"

# Compare models
python -m modelexport.compare --model1 "bert-base" --model2 "distilbert-base"
```

### B. Configuration Files
```yaml
# extraction_config.yaml
extraction:
  include_constants: false
  min_parameter_count: 10
  max_depth: 5
  
validation:
  min_coverage: 0.9
  check_cycles: true
  verify_connections: true

output:
  format: "json"
  include_metadata: true
  compress: false
```

## Priority Ranking

1. **HIGH**: Validation/Verification Suite (2)
2. **HIGH**: Shared Parameter Detection (3)  
3. **MEDIUM**: Multi-Architecture Testing (1)
4. **MEDIUM**: Performance Metrics (4)
5. **LOW**: Error Handling & Edge Cases (5)
6. **LOW**: Integration & Tooling (6)

## Implementation Notes

- Start with validation suite to ensure current implementation is correct
- Shared parameter detection provides immediate value for understanding models
- Multi-architecture testing validates universality claims
- Performance metrics help optimize the system
- Error handling and tooling can be added incrementally