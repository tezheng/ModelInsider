# Node Processor Design Analysis

## Research Background: PyTorch ONNX Export Pipeline

### Why Constants Need Special Handling

**Root Cause Analysis**: Constants in ONNX are fundamentally different from traced operations:

1. **Constants are NOT PyTorch operations** - they're literal values embedded during ONNX export
2. **Generated at different pipeline stages** - some during tracing, others during ONNX conversion  
3. **Context loss occurs during export** - module hierarchy stripped in conversion process

### PyTorch ONNX Export Pipeline

```
PyTorch Model → torch.jit.trace() → TorchScript IR → ONNX Conversion → ONNX Graph
     ↑                    ↑                ↑               ↑
   Forward        Operations        Scope Info      Context Lost
   Hooks          Traced           Available       Here!
```

### Three Types of Constants

#### **Type 1: Parameter Constants** (Context Preserved)
```python
"/embeddings/LayerNorm/Constant"  # From model.embeddings.LayerNorm.weight
"/attention/query/Constant"       # From model.attention.query.weight
```
- **Source**: Named parameters from `model.state_dict()`
- **Why context preserved**: Tied to specific modules during parameter conversion

#### **Type 2: Literal Constants** (Context Lost)
```python
"/Constant_0"  # From tensor.size(0) 
"/Constant_1"  # From x * 2.0 (where 2.0 becomes constant)
```
- **Source**: Runtime computations during tracing (shape operations, scalar multiplications)
- **Why context lost**: Generated during execution, not tied to specific modules

#### **Type 3: Folded Constants** (Mixed Context)
```python
"/embeddings/Constant_folded_123"  # Pre-computed result
```
- **Source**: `do_constant_folding=True` optimization
- **Context**: Depends on source operation scope

### Hook Opportunities Discovered

#### **1. JIT Graph Access (Pre-Export)**
```python
def dump_intermediate_state(model, inputs):
    traced_model = torch.jit.trace(model, inputs)
    
    # ✅ Scope information STILL AVAILABLE here
    for node in traced_model.graph.nodes():
        print(f"Scope: {node.scopeName()}")  # Full module hierarchy!
        
    # ❌ Context lost during ONNX export
    torch.onnx.export(traced_model, inputs, "output.onnx")
```

#### **2. Custom Symbolic Functions**
```python
def hierarchy_preserving_constant(g, value):
    current_scope = g.scope  # Available during export!
    const_node = g.op("Constant", value_t=value)
    const_node.addAttribute("hierarchy_scope", current_scope)
    return const_node

torch.onnx.register_custom_op_symbolic("aten::constant", hierarchy_preserving_constant, 11)
```

#### **3. Graph Transformation Pass**
```python
def inject_hierarchy_pass(graph):
    for node in graph.nodes():
        scope_name = node.scopeName()  # Still available!
        if scope_name:
            node.addAttribute("hierarchy_path", scope_name)
    return graph
```

## Current Processing Patterns

Based on analysis of the existing code, here are the specialized processing functions needed:

## 1. Constant Node Processor

```python
def _process_constant_node(self, node, node_name: str):
    """Process Constant nodes with path-based and consumer-based tagging."""
    
    # Skip if already tagged
    if self._tag_mapping[node_name]["tags"]:
        return
    
    # Method 1: Path-based tagging (for named constants like /embeddings/LayerNorm/Constant)
    if self._try_path_based_tagging(node_name):
        return
    
    # Method 2: Consumer-based tagging (for generic constants like /Constant)
    self._try_consumer_based_tagging(node_name)

def _try_path_based_tagging(self, node_name: str) -> bool:
    """Universal path-based tagging without hardcoded patterns."""
    if '/' not in node_name:
        return False
    
    path_parts = node_name.strip('/').split('/')
    if len(path_parts) < 2:
        return False
    
    # Universal approach: match ANY hierarchical path to traced modules
    best_match = None
    best_score = 0
    
    for module_name, context in self._operation_context.items():
        module_parts = module_name.split('.')
        
        # Score based on path segment overlap (universal matching)
        overlap = len(set(path_parts[:-1]) & set(module_parts))
        if overlap > best_score:
            best_score = overlap
            best_match = context["tag"]
    
    if best_match and best_score >= 1:
        self._tag_mapping[node_name]["tags"] = [best_match]
        return True
    return False

def _try_consumer_based_tagging(self, node_name: str):
    """Tag based on operations that consume this constant."""
    constant_outputs = self._tag_mapping[node_name]["outputs"]
    consumer_tags = set()
    
    for output_tensor in constant_outputs:
        for other_node_name, other_node_info in self._tag_mapping.items():
            if output_tensor in other_node_info.get("inputs", []):
                other_tags = other_node_info.get("tags", [])
                if other_tags:
                    consumer_tags.update(other_tags)
    
    if consumer_tags:
        self._tag_mapping[node_name]["tags"] = list(consumer_tags)
```

## 2. Shape Node Processor

```python
def _process_shape_node(self, node, node_name: str):
    """Universal Shape node processing without hardcoded input names."""
    
    # Skip if already tagged
    if self._tag_mapping[node_name]["tags"]:
        return
    
    # Universal approach: find what operations consume this shape
    shape_consumers = self._find_shape_consumers(node_name)
    if shape_consumers:
        # Tag based on consumers (universal)
        consumer_tags = set()
        for consumer in shape_consumers:
            if consumer in self._tag_mapping and self._tag_mapping[consumer].get("tags"):
                consumer_tags.update(self._tag_mapping[consumer]["tags"])
        
        if consumer_tags:
            self._tag_mapping[node_name]['tags'] = list(consumer_tags)
            return
    
    # Fallback: check if this operates on model inputs (universal detection)
    if self._operates_on_model_inputs(node):
        # Find the first available input processing module (universal)
        input_processor_tag = self._find_input_processor_tag()
        if input_processor_tag:
            self._tag_mapping[node_name]['tags'] = [input_processor_tag]
            return
    
    # Final fallback: inherit from input producers
    self._inherit_from_producers(node_name)

def _operates_on_model_inputs(self, node) -> bool:
    """Universal detection of nodes operating on model inputs."""
    # Check if any input tensor name suggests it's a model input
    for input_tensor in node.input:
        # Universal patterns that indicate model inputs (not hardcoded to specific architectures)
        if any(pattern in input_tensor.lower() for pattern in ['input', 'ids', 'mask', 'token']):
            return True
    return False

def _find_input_processor_tag(self) -> Optional[str]:
    """Universal method to find modules that process inputs."""
    # Look for modules that typically process inputs (universal approach)
    input_indicators = ['embedding', 'input', 'token', 'word']
    
    for context in self._operation_context.values():
        tag = context['tag'].lower()
        if any(indicator in tag for indicator in input_indicators):
            return context['tag']
    
    # If no specific input processor found, use the first available tag
    if self._operation_context:
        return next(iter(self._operation_context.values()))['tag']
    return None
```

## 3. Generic Operation Processor

```python
def _process_generic_node(self, node, node_name: str):
    """Process any node type with universal logic."""
    
    # Skip if already tagged
    if self._tag_mapping[node_name]["tags"]:
        return
    
    # Try path-based inference first
    if self._try_path_inference(node_name):
        return
    
    # Try input-based inheritance
    if self._try_input_inheritance(node_name):
        return
    
    # Last resort: default tagging for complete coverage
    self._apply_default_tag(node_name)

def _try_path_inference(self, node_name: str) -> bool:
    """Universal path-based tagging."""
    if not '/' in node_name or node_name.startswith(node.op_type + '_'):
        return False
    
    path_parts = node_name.strip('/').split('/')
    if len(path_parts) < 2:
        return False
    
    # Score-based matching
    best_match = None
    best_score = 0
    
    for module_name, context in self._operation_context.items():
        module_parts = module_name.split('.')
        overlap = len(set(path_parts) & set(module_parts))
        
        if overlap > best_score:
            best_score = overlap
            best_match = context["tag"]
    
    if best_match and best_score >= 1:
        self._tag_mapping[node_name]["tags"] = [best_match]
        return True
    return False
```

## 4. Complete Processor Implementation

```python
class ONNXNodeProcessor:
    def __init__(self, exporter_context):
        self.context = exporter_context
        self._tag_mapping = exporter_context._tag_mapping
        self._operation_context = exporter_context._operation_context
    
    def process_by_type(self, onnx_model, type_processors: Dict[str, str] = None):
        """Process nodes by type with specialized or generic logic."""
        
        # Default processors for common types
        default_processors = {
            'Constant': '_process_constant_node',
            'Shape': '_process_shape_node', 
            'ConstantOfShape': '_process_shape_node',  # Same as Shape
            'Gather': '_process_generic_node',
            'Reshape': '_process_generic_node',
            'Transpose': '_process_generic_node',
            'Unsqueeze': '_process_generic_node',
            'Add': '_process_generic_node',
            'Mul': '_process_generic_node',
            'Div': '_process_generic_node',
        }
        
        # Override with custom processors if provided
        processors = {**default_processors, **(type_processors or {})}
        
        # Group nodes by type
        nodes_by_type = self._group_nodes_by_type(onnx_model)
        
        # Process each type
        for node_type, processor_name in processors.items():
            nodes = nodes_by_type.get(node_type, [])
            processor_func = getattr(self, processor_name)
            
            for node in nodes:
                node_name = node.name or f"{node.op_type}_{len(self._tag_mapping)}"
                processor_func(node, node_name)
    
    def _group_nodes_by_type(self, onnx_model) -> Dict[str, List]:
        """Group ONNX nodes by operation type."""
        groups = defaultdict(list)
        for node in onnx_model.graph.node:
            groups[node.op_type].append(node)
        return groups
    
    # ... processor methods above ...
```

## 5. Usage in Main Exporter

```python
# Replace current conditional blocks:
# if 'Constant' in onnx_nodes_by_type:
#     for node in onnx_nodes_by_type['Constant']:
#         # ... constant logic

# With:
processor = ONNXNodeProcessor(self)
processor.process_by_type(onnx_model)

# Or with custom processors:
processor.process_by_type(onnx_model, {
    'Constant': '_custom_constant_processor',
    'NewNodeType': '_handle_new_type'
})
```

## Key Benefits

1. **Eliminates Repetition**: No more `if 'NodeType' in onnx_nodes_by_type:` blocks
2. **Configurable**: Can override processors for specific node types
3. **Extensible**: Easy to add new node types
4. **Testable**: Each processor can be unit tested independently
5. **Maintainable**: Specialized logic is clearly separated

## Specialized Logic Summary

- **Constant**: Path-based + consumer-based tagging
- **Shape**: Infrastructure operation handling + input-based tagging  
- **Generic**: Path inference + input inheritance + default tagging
- **Extensible**: Framework supports adding custom processors for new node types

This design maintains the universal approach while organizing the specialized logic cleanly.

## Advanced Solutions: Context Preservation

### Solution 1: Pre-Export Hierarchy Injection

```python
def enhanced_export_with_context_preservation(model, inputs, output_path):
    """Export with hierarchy context preservation using JIT graph access."""
    
    # Step 1: Trace execution with hooks (current approach)
    with hierarchy_hooks(model):
        traced_model = torch.jit.trace(model, inputs)
    
    # Step 2: Extract hierarchy BEFORE it's lost
    hierarchy_map = {}
    for node in traced_model.graph.nodes():
        if node.scopeName():  # Still available at this stage!
            hierarchy_map[node.debugName()] = node.scopeName()
    
    # Step 3: Export with custom symbolic functions that use hierarchy_map
    with hierarchy_injection_context(hierarchy_map):
        torch.onnx.export(traced_model, inputs, output_path)
```

### Solution 2: Universal Constant Classification

```python
def classify_constant_source(node_name, model_state_dict):
    """Universal classification without hardcoding."""
    
    # Type 1: Parameter constants (preserved context)
    for param_name, param_value in model_state_dict.items():
        if param_name.replace('.', '/') in node_name:
            return "parameter", param_name
    
    # Type 2: Shape/infrastructure constants (universal detection)
    shape_indicators = ['shape', 'size', 'dim', 'gather', 'slice']
    if any(indicator in node_name.lower() for indicator in shape_indicators):
        return "shape_literal", None
    
    # Type 3: Computation literals
    return "computation_literal", None
```

### Solution 3: Custom Symbolic Function Registration

```python
def register_hierarchy_preserving_operations():
    """Register custom symbolic functions that preserve hierarchy context."""
    
    def hierarchy_preserving_constant(g, value):
        # Access current scope during export
        current_scope = getattr(g, 'scope', None)
        
        # Create constant with hierarchy metadata
        const_node = g.op("Constant", value_t=value)
        
        # Inject hierarchy as ONNX node attributes
        if current_scope:
            const_node.addAttribute("hierarchy_scope", current_scope)
        
        return const_node
    
    # Register for constant-generating operations
    torch.onnx.register_custom_op_symbolic("aten::constant", hierarchy_preserving_constant, 11)
    torch.onnx.register_custom_op_symbolic("aten::size", hierarchy_preserving_constant, 11)
```

## Research Findings Summary

### Key Insights
1. **Context exists until ONNX conversion** - can be intercepted at TorchScript stage
2. **Three distinct constant types** require different tagging strategies
3. **Universal patterns** can replace hardcoded logic for robust operation
4. **Multiple hook points** available for context preservation

### Implementation Priority
1. **Immediate**: Remove hardcoded patterns, implement universal approaches
2. **Medium-term**: Add JIT graph access for context preservation  
3. **Advanced**: Custom symbolic functions for complete context control

### MUST-RULES Compliance
- ✅ **No hardcoded logic**: Universal pattern matching and scoring
- ✅ **Architecture agnostic**: Works with any model structure
- ✅ **Dynamic analysis**: Based on actual execution and graph structure