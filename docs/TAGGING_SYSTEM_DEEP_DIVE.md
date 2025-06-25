# How the Model Exporter Tags Models: Complete Deep Dive

The modelexport system implements a sophisticated **hierarchy-preserving tagging system** that captures PyTorch model structure and maps it to ONNX operations. Here's the complete technical explanation:

## Core Architecture: Universal Approach

The system follows a **universal design principle** - it works with ANY PyTorch model without hardcoded logic:

```python
# Key principle: NO HARDCODED LOGIC
# ❌ Never hardcode model names (BERT, GPT, ResNet)
# ❌ Never hardcode operation names or patterns
# ✅ Always use universal PyTorch structures (nn.Module hierarchy)
```

## Three Tagging Strategies

### 1. Usage-Based Strategy (Legacy)
Basic approach that tags operations when they're used during execution.

### 2. HTP (Hierarchical Trace-and-Project)  
Advanced approach that:
- Traces model execution to capture module context
- Projects execution trace onto ONNX operations
- Uses conservative propagation to avoid over-tagging

### 3. HTP with Built-in Tracking (Breakthrough)
Uses PyTorch's built-in `torch.jit._trace._trace_module_map` infrastructure:
- **29% performance improvement**
- **Better layer differentiation** 
- **33-50% reduction in cross-layer contamination**

## Step-by-Step Tagging Process

### Phase 1: Model Preparation & Hook Registration

```python
def export(self, model, example_inputs, output_path, **kwargs):
    # Step 1: Prepare model for tracing
    model.eval()
    self._reset_state()
    self._model = model  # Store reference to root model
    
    # Step 2: Register hooks for execution tracing
    self._register_hooks(model)
```

**Hook Registration Process:**
```python
def _register_hooks(self, model: torch.nn.Module):
    # Initialize with root module tag
    root_tag = f"/{model.__class__.__name__}"
    self._tag_stack.append(root_tag)
    
    # Create incremental hooks for each module
    def create_incremental_pre_hook(module_info):
        def pre_hook(module, inputs):
            # Get parent context from stack
            parent_tag = self._tag_stack[-1]
            
            # Build hierarchical tag incrementally
            current_class_name = module_info['class_name']
            if module_info['is_indexed']:
                current_class_name = f"{current_class_name}.{module_info['module_index']}"
            
            hierarchical_tag = f"{parent_tag}/{current_class_name}"
            self._tag_stack.append(hierarchical_tag)
```

### Phase 2: Operation Patching

The system patches PyTorch operations to capture execution context:

```python
def _patch_torch_operations(self):
    # Get operations from centralized configuration
    operations_to_patch = OperationConfig.get_operations_to_patch()
    
    # Patch core operations: matmul, add, sub, mul, etc.
    for module, op_name in operations_to_patch:
        original_op = getattr(module, op_name)
        traced_op = self._create_traced_operation(op_name, original_op)
        setattr(module, op_name, traced_op)
```

**Operation Registry:**
```python
OPERATION_REGISTRY = {
    'matmul': {
        'patch_targets': [('torch', 'matmul')],
        'onnx_types': ['MatMul', 'Gemm'],
        'priority': 1
    },
    'add': {
        'patch_targets': [('torch', 'add')],
        'onnx_types': ['Add'],
        'priority': 1
    },
    # ... covers 30+ operation types
}
```

**Traced Operation Wrapper:**
```python
def traced_operation(*args, **kwargs):
    # Capture current module context from stack
    current_tag = self.get_current_tag()
    
    # Call original operation
    result = original_op(*args, **kwargs)
    
    # Record operation trace with context
    trace_entry = {
        'op_name': op_name,
        'module_tag': current_tag,
        'tensor_id': id(result),
        'timestamp': len(self._operation_trace),
    }
    self._operation_trace.append(trace_entry)
```

### Phase 3: Built-in Module Tracking (Advanced)

For the HTP built-in strategy, the system uses PyTorch's internal tracking:

```python
def _patch_torch_operations_with_builtin_tracking(self):
    def create_context_capturing_wrapper_builtin(op_name, original_op):
        def traced_operation_builtin(*args, **kwargs):
            # Get current module using PyTorch's built-in tracking
            current_module = self._get_current_executing_module_builtin()
            
            if current_module:
                module_name = self._get_module_name_from_builtin_tracking(current_module)
                current_tag = self._build_tag_from_module_name(module_name)
            
            result = original_op(*args, **kwargs)
            
            # Record with built-in context
            self._operation_trace.append({
                'op_name': op_name,
                'module_tag': current_tag,
                'context_source': 'builtin_tracking'
            })
```

**Tag Building from Module Names:**
```python
def _build_tag_from_module_name(self, module_name: str) -> str:
    # Convert: "encoder.layer.0.attention" -> "/BertEncoder/BertLayer.0/BertAttention"
    components = module_name.split('.')
    
    tag_parts = []
    for component in components:
        if component.isdigit():
            # Append index to previous component
            if tag_parts:
                tag_parts[-1] += f".{component}"
        else:
            # Convert to CamelCase
            tag_parts.append(self._snake_to_camel(component))
    
    return "/" + "/".join(tag_parts)
```

### Phase 4: Model Execution Tracing

```python
def _trace_model_execution(self, model, example_inputs):
    # Execute model with hooks active to capture execution trace
    with torch.no_grad():
        model(*example_inputs)
    
    # Result: self._operation_trace contains ordered execution with module context
```

### Phase 5: ONNX Export & Trace Projection

```python
def _export_htp(self, model, example_inputs, output_path, **kwargs):
    # Export to ONNX with operation tracing active
    self._export_to_onnx(model, example_inputs, output_path, **kwargs)
    
    # Load exported ONNX and project execution trace onto it
    onnx_model = onnx.load(output_path)
    self._project_execution_trace_to_onnx(onnx_model)
```

**Trace Projection Process:**
```python
def _project_execution_trace_to_onnx(self, onnx_model):
    # Get PyTorch -> ONNX operation mapping
    torch_to_onnx_mapping = OperationConfig.get_torch_to_onnx_mapping()
    
    # Initialize tag mapping for all ONNX nodes
    for node in onnx_model.graph.node:
        self._tag_mapping[node.name] = {
            "op_type": node.op_type,
            "tags": [],
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
    
    # Match traced operations to ONNX nodes by type and order
    for trace_entry in self._operation_trace:
        op_type = trace_entry['op_name']  # e.g., 'matmul'
        module_tag = trace_entry['module_tag']  # e.g., '/BertEncoder/BertLayer.0'
        
        # Find corresponding ONNX nodes
        possible_onnx_types = torch_to_onnx_mapping.get(op_type, [])
        for node in onnx_model.graph.node:
            if node.op_type in possible_onnx_types:
                self._tag_mapping[node.name]['tags'].append(module_tag)
```

### Phase 6: Forward Tag Propagation

Operations not directly traced get tags propagated from their inputs:

```python
def _forward_propagate_tags_htp(self, onnx_model, tensor_producers):
    """Conservative forward propagation for HTP to avoid over-tagging."""
    for node in onnx_model.graph.node:
        current_tags = self._tag_mapping[node.name].get('tags', [])
        
        # Skip if already tagged
        if current_tags:
            continue
        
        # Collect tags from input tensors
        input_tags = set()
        for input_tensor in node.input:
            producer_node = tensor_producers.get(input_tensor)
            if producer_node in self._tag_mapping:
                producer_tags = self._tag_mapping[producer_node].get('tags', [])
                input_tags.update(producer_tags)
        
        # Only inherit if all inputs agree (conservative approach)
        if len(input_tags) == 1:
            self._tag_mapping[node.name]['tags'] = list(input_tags)
```

### Phase 7: Native Operation Pattern Handling

```python
def _tag_native_operation_patterns(self, onnx_model):
    """Handle complex operations like scaled_dot_product_attention."""
    for region in self._native_op_regions:
        # Tag entire regions of ONNX operations that correspond to
        # single high-level PyTorch operations
        start_idx = region['start_node_index']
        end_idx = region['end_node_index']
        module_tag = region['module_context']
        
        for i in range(start_idx, end_idx + 1):
            node = onnx_model.graph.node[i]
            self._tag_mapping[node.name]['tags'].append(module_tag)
```

### Phase 8: Tag Injection into ONNX

```python
def _inject_htp_tags_into_onnx(self, onnx_path: str, onnx_model):
    # 1. Inject tags as node doc_strings (ONNX-compliant)
    for node in onnx_model.graph.node:
        if node.name in self._tag_mapping:
            tags = self._tag_mapping[node.name].get("tags", [])
            if tags:
                hierarchy_info = {
                    "hierarchy_tags": tags,
                    "hierarchy_path": tags[0],
                    "hierarchy_method": "htp",
                }
                node.doc_string = json.dumps(hierarchy_info)
    
    # 2. Create JSON sidecar file with complete metadata
    sidecar_path = onnx_path.replace('.onnx', '_hierarchy.json')
    metadata = {
        "version": "1.0",
        "strategy": "htp",
        "node_tags": self._tag_mapping,
        "operation_trace": self._operation_trace,
        "summary": {
            "total_operations": len(onnx_model.graph.node),
            "tagged_operations": len([n for n in self._tag_mapping.values() if n.get('tags')]),
        }
    }
    
    # Save both ONNX model and sidecar
    onnx.save(onnx_model, onnx_path)
    with open(sidecar_path, 'w') as f:
        json.dump(metadata, f, indent=2)
```

## Key Technical Innovations

### 1. Universal Module Filtering
```python
# Filter out generic torch.nn modules, keep model-specific ones
TORCH_NN_HIERARCHY_EXCEPTIONS = {
    "LayerNorm", "Embedding", "BatchNorm1d", # Semantically important
}

def _should_create_hierarchy_level(self, module_name: str, module_class: str) -> bool:
    # Only create hierarchy for model-specific modules
    if module_class.startswith('torch.nn.'):
        return module_class.split('.')[-1] in self.TORCH_NN_HIERARCHY_EXCEPTIONS
    return True  # All model-specific modules create hierarchy
```

### 2. Stack-Based Context Tracking
```python
# Maintains execution hierarchy via stack
self._tag_stack = ["/RootModel"]  # Initialize with root

# Pre-hook: Push context
def pre_hook(module, inputs):
    hierarchical_tag = f"{parent_tag}/{current_class_name}"
    self._tag_stack.append(hierarchical_tag)

# Post-hook: Pop context  
def post_hook(module, inputs, outputs):
    self._tag_stack.pop()
```

### 3. Multi-Level Tag Mapping
```python
# Each ONNX node gets comprehensive metadata
self._tag_mapping[node_name] = {
    "op_type": node.op_type,           # ONNX operation type
    "tags": ["/BertEncoder/Layer.0"],  # Hierarchical module tags
    "inputs": ["input_tensor"],        # Input tensor names
    "outputs": ["output_tensor"],      # Output tensor names
    "trace_source": "direct_match",    # How the tag was determined
}
```

## Result: Hierarchical ONNX Model

The final ONNX model contains:

1. **Node-level tags**: Each operation tagged with its source module
2. **Hierarchy preservation**: Complete PyTorch module structure
3. **Universal compatibility**: Works with any PyTorch model
4. **Rich metadata**: Comprehensive tracing and analysis information

Example tags:
- `/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention` 
- `/ResNet/Sequential.0/BasicBlock.1/Conv2d`
- `/GPT2Model/Transformer/GPT2Block.5/GPT2MLP`

This system achieves **universal hierarchy preservation** without any model-specific hardcoding, making it work seamlessly across all PyTorch architectures.

## Summary

The modelexport tagging system represents a significant technical achievement in bridging the gap between PyTorch's dynamic execution model and ONNX's static graph representation. By leveraging universal PyTorch structures and sophisticated tracing techniques, it preserves the complete model hierarchy without any architecture-specific code, making it truly universal and future-proof.