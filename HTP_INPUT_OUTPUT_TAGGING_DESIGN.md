# Input/Output Tagging Design for HTP Approach

## Problem Statement

With operation-wise tagging, we need to design how input and output tensors should be tagged to enable:
1. **Subgraph extraction**: When filtering by module tag, include all necessary tensors
2. **Data lineage**: Track which module produced/consumed each tensor
3. **Dependency analysis**: Understand tensor flow between modules

## Core Design Principles

### Principle 1: Inputs Inherit from Consumers
**Rule**: Input tensors should be tagged with the modules that consume them.

```python
# If operation Op1 (tagged with /ModuleA) consumes tensor T1
# Then T1 should be tagged with /ModuleA

"/some/tensor": {
    "consumers": ["/ModuleA", "/ModuleB"],  # All modules that consume this tensor
    "producer": "/ModuleC"                  # Module that produced this tensor
}
```

### Principle 2: Outputs Inherit from Producers  
**Rule**: Output tensors should be tagged with the module that produced them.

```python
# If operation Op1 (tagged with /ModuleA) produces tensor T1
# Then T1 should be tagged with /ModuleA

"/output/tensor": {
    "producer": "/ModuleA",                 # Module that produced this tensor
    "consumers": ["/ModuleB", "/ModuleC"]   # Modules that will consume this tensor
}
```

### Principle 3: Multi-Consumer Tensors
**Rule**: Tensors consumed by multiple modules should maintain all consumer tags.

```python
# If tensor flows from ModuleA to both ModuleB and ModuleC
"/shared/tensor": {
    "producer": "/ModuleA",
    "consumers": ["/ModuleB", "/ModuleC"],
    "tags": ["/ModuleA", "/ModuleB", "/ModuleC"]  # Union of producer + consumers
}
```

## Implementation Strategy

### Phase 1: Tensor-Operation Mapping
During ONNX projection, build tensor-operation relationships:

```python
class HTDataFlowTracker:
    def __init__(self):
        self.tensor_producers = {}  # tensor_name -> operation_name
        self.tensor_consumers = {}  # tensor_name -> [operation_names]
        self.operation_tags = {}    # operation_name -> module_tag
    
    def build_tensor_mappings(self, onnx_model):
        """Build tensor-operation mappings from ONNX graph."""
        for node in onnx_model.graph.node:
            # Record what this operation produces
            for output_tensor in node.output:
                self.tensor_producers[output_tensor] = node.name
            
            # Record what this operation consumes
            for input_tensor in node.input:
                if input_tensor not in self.tensor_consumers:
                    self.tensor_consumers[input_tensor] = []
                self.tensor_consumers[input_tensor].append(node.name)
```

### Phase 2: Tag Propagation
Propagate operation tags to tensors:

```python
def propagate_tags_to_tensors(self):
    """Propagate operation tags to input/output tensors."""
    
    tensor_tags = {}
    
    for tensor_name in set(self.tensor_producers.keys()) | set(self.tensor_consumers.keys()):
        tags = set()
        
        # Add producer tag
        if tensor_name in self.tensor_producers:
            producer_op = self.tensor_producers[tensor_name]
            if producer_op in self.operation_tags:
                tags.add(self.operation_tags[producer_op])
        
        # Add consumer tags
        if tensor_name in self.tensor_consumers:
            for consumer_op in self.tensor_consumers[tensor_name]:
                if consumer_op in self.operation_tags:
                    tags.add(self.operation_tags[consumer_op])
        
        tensor_tags[tensor_name] = {
            'tags': list(tags),
            'producer': self.tensor_producers.get(tensor_name),
            'consumers': self.tensor_consumers.get(tensor_name, [])
        }
    
    return tensor_tags
```

### Phase 3: Subgraph Filtering Support
Enable precise subgraph extraction:

```python
def extract_subgraph_by_module(self, module_tag, onnx_model):
    """Extract subgraph containing all operations and tensors for a module."""
    
    # Find operations belonging to this module
    module_operations = []
    for op_name, tags in self.operation_tags.items():
        if module_tag in tags:
            module_operations.append(op_name)
    
    # Find all tensors needed for these operations
    required_tensors = set()
    
    for op_name in module_operations:
        # Add input tensors
        for node in onnx_model.graph.node:
            if node.name == op_name:
                required_tensors.update(node.input)
                required_tensors.update(node.output)
    
    # Build subgraph with only required nodes and tensors
    subgraph_nodes = []
    for node in onnx_model.graph.node:
        if node.name in module_operations:
            subgraph_nodes.append(node)
    
    return subgraph_nodes, required_tensors
```

## Special Cases and Edge Conditions

### Case 1: Model Boundaries (Inputs/Outputs)
**Model inputs**: Tag with first consuming module
**Model outputs**: Tag with producing module

```python
def handle_model_boundaries(self, onnx_model):
    """Handle model input/output tensor tagging."""
    
    # Model inputs - tag with first consumers
    for input_spec in onnx_model.graph.input:
        input_name = input_spec.name
        if input_name in self.tensor_consumers:
            first_consumer = self.tensor_consumers[input_name][0]
            if first_consumer in self.operation_tags:
                self.tensor_tags[input_name] = {
                    'tags': [self.operation_tags[first_consumer]],
                    'type': 'model_input',
                    'consumers': self.tensor_consumers[input_name]
                }
    
    # Model outputs - tag with producers
    for output_spec in onnx_model.graph.output:
        output_name = output_spec.name
        if output_name in self.tensor_producers:
            producer = self.tensor_producers[output_name]
            if producer in self.operation_tags:
                self.tensor_tags[output_name] = {
                    'tags': [self.operation_tags[producer]],
                    'type': 'model_output',
                    'producer': producer
                }
```

### Case 2: Parameter Tensors
**Parameters/Initializers**: Tag with consuming operations

```python
def handle_parameters(self, onnx_model):
    """Tag parameter tensors with their consumers."""
    
    parameter_names = {init.name for init in onnx_model.graph.initializer}
    
    for param_name in parameter_names:
        if param_name in self.tensor_consumers:
            consumer_tags = []
            for consumer_op in self.tensor_consumers[param_name]:
                if consumer_op in self.operation_tags:
                    consumer_tags.append(self.operation_tags[consumer_op])
            
            self.tensor_tags[param_name] = {
                'tags': consumer_tags,
                'type': 'parameter',
                'consumers': self.tensor_consumers[param_name]
            }
```

### Case 3: Intermediate Tensors
**Intermediate tensors**: Tag with both producer and consumers

```python
def handle_intermediate_tensors(self):
    """Tag intermediate tensors with full lineage."""
    
    for tensor_name in self.tensor_producers:
        if (tensor_name not in self._model_inputs and 
            tensor_name not in self._model_outputs and
            tensor_name not in self._parameter_names):
            
            tags = set()
            
            # Producer tag
            producer = self.tensor_producers[tensor_name]
            if producer in self.operation_tags:
                tags.add(self.operation_tags[producer])
            
            # Consumer tags
            for consumer in self.tensor_consumers.get(tensor_name, []):
                if consumer in self.operation_tags:
                    tags.add(self.operation_tags[consumer])
            
            self.tensor_tags[tensor_name] = {
                'tags': list(tags),
                'type': 'intermediate',
                'producer': producer,
                'consumers': self.tensor_consumers.get(tensor_name, [])
            }
```

## Output Format

### JSON Hierarchy Format Extension
```json
{
  "node_tags": {
    "/pooler/dense/Gemm": {
      "op_type": "Gemm",
      "tags": ["/BertModel/BertPooler"],
      "inputs": ["last_hidden_state", "pooler.dense.weight", "pooler.dense.bias"],
      "outputs": ["/pooler/dense/Gemm_output_0"]
    }
  },
  "tensor_tags": {
    "last_hidden_state": {
      "tags": ["/BertModel/BertEncoder", "/BertModel/BertPooler"],
      "type": "intermediate",
      "producer": "/encoder/output/Add",
      "consumers": ["/pooler/Gather", "/pooler/dense/Gemm"]
    },
    "pooler.dense.weight": {
      "tags": ["/BertModel/BertPooler"],
      "type": "parameter",
      "consumers": ["/pooler/dense/Gemm"]
    },
    "/pooler/dense/Gemm_output_0": {
      "tags": ["/BertModel/BertPooler"],
      "type": "intermediate", 
      "producer": "/pooler/dense/Gemm",
      "consumers": ["/pooler/activation/Tanh"]
    }
  },
  "subgraph_info": {
    "/BertModel/BertPooler": {
      "operations": ["/pooler/Gather", "/pooler/dense/Gemm", "/pooler/activation/Tanh"],
      "input_tensors": ["last_hidden_state", "pooler.dense.weight", "pooler.dense.bias"],
      "output_tensors": ["379"],
      "internal_tensors": ["/pooler/dense/Gemm_output_0"]
    }
  }
}
```

## Benefits

### 1. Precise Subgraph Extraction
```python
# Extract just the pooler subgraph
pooler_subgraph = extract_subgraph_by_module("/BertModel/BertPooler", onnx_model)
# Includes: operations + all necessary input/output tensors
```

### 2. Data Lineage Tracking
```python
# Trace tensor flow
tensor_lineage = trace_tensor_lineage("last_hidden_state")
# Shows: BertEncoder -> BertPooler data flow
```

### 3. Dependency Analysis
```python
# Find module dependencies
dependencies = analyze_module_dependencies()
# Shows: BertPooler depends on BertEncoder output
```

### 4. Filtering Capabilities
```python
# Filter operations and tensors by module
pooler_only = filter_by_module("/BertModel/BertPooler")
# Returns: {operations: [...], tensors: [...]}
```

## Implementation Priority

**Phase 1**: Basic tensor tagging (producer/consumer mapping)
**Phase 2**: Subgraph extraction support
**Phase 3**: Advanced lineage tracking and filtering
**Phase 4**: CLI integration for subgraph export

This design ensures that with operation-wise tagging, we maintain complete tensor attribution for precise subgraph extraction and analysis.