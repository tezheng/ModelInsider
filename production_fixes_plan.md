# Production Fixes Plan

## 1. TOPOLOGY SORTING ISSUES - DETAILED SOLUTION

### Problem Analysis
Current issue: Extracted subgraphs include nodes that reference tensors not produced within the subgraph.

Example error:
```
input '/layer1/layer1.1/relu_1/Relu_output_0' of node Conv 
is not output of any previous nodes
```

This happens because we're including dependencies across module boundaries without proper input/output interface creation.

### Solution: Smart Boundary Detection

#### Step 1: Implement Module Boundary Analysis
```python
class ModuleBoundaryDetector:
    def find_module_boundaries(self, target_tag, all_nodes):
        """Find true input/output boundaries for a module"""
        
        # 1. Get all operations belonging to this module
        module_operations = self.get_operations_by_tag(target_tag)
        
        # 2. Find true external inputs (tensors coming from outside module)
        external_inputs = []
        for op in module_operations:
            for input_tensor in op.inputs:
                producer = self.tensor_producers.get(input_tensor)
                if producer not in module_operations:
                    external_inputs.append(input_tensor)
        
        # 3. Find true external outputs (tensors used outside module)  
        external_outputs = []
        for op in module_operations:
            for output_tensor in op.outputs:
                consumers = self.tensor_consumers.get(output_tensor, [])
                external_consumers = [c for c in consumers if c not in module_operations]
                if external_consumers:
                    external_outputs.append(output_tensor)
        
        return external_inputs, external_outputs
```

#### Step 2: Create Clean Subgraphs
```python
def create_clean_subgraph(self, target_tag):
    """Create subgraph with proper boundaries"""
    
    # Get module operations only (no external dependencies)
    module_ops = self.nodes_by_tag[target_tag]
    
    # Find boundaries
    ext_inputs, ext_outputs = self.find_module_boundaries(target_tag, module_ops)
    
    # Create graph inputs for external tensors
    graph_inputs = []
    for tensor_name in ext_inputs:
        tensor_info = self.create_tensor_info(tensor_name)
        graph_inputs.append(tensor_info)
    
    # Create graph outputs
    graph_outputs = []
    for tensor_name in ext_outputs:
        tensor_info = self.create_tensor_info(tensor_name)
        graph_outputs.append(tensor_info)
    
    # Include only module operations + required initializers
    subgraph_nodes = [self.node_by_name[op] for op in module_ops]
    required_initializers = self.find_required_initializers(module_ops)
    
    # Create clean graph
    clean_graph = helper.make_graph(
        subgraph_nodes,
        f"module_{target_tag.replace('/', '_')}",
        graph_inputs,
        graph_outputs, 
        required_initializers
    )
    
    return clean_graph
```

#### Step 3: Topological Sorting
```python
def ensure_topological_order(self, nodes):
    """Sort nodes in topological order"""
    
    # Build dependency graph
    dependencies = {}
    for node in nodes:
        dependencies[node.name] = []
        for input_tensor in node.input:
            producer = self.tensor_producers.get(input_tensor)
            if producer and producer in [n.name for n in nodes]:
                dependencies[node.name].append(producer)
    
    # Topological sort using Kahn's algorithm
    sorted_nodes = []
    in_degree = {node.name: len(dependencies[node.name]) for node in nodes}
    queue = [node for node in nodes if in_degree[node.name] == 0]
    
    while queue:
        current = queue.pop(0)
        sorted_nodes.append(current)
        
        # Update dependencies
        for node in nodes:
            if current.name in dependencies[node.name]:
                in_degree[node.name] -= 1
                if in_degree[node.name] == 0:
                    queue.append(node)
    
    return sorted_nodes
```

## 2. CUSTOM ATTRIBUTE COMPATIBILITY - DETAILED SOLUTION

### Problem Analysis
ONNX validator rejects our custom `source_module` and `hierarchy_tags` attributes.

Error:
```
Unrecognized attribute: source_module for operator Conv
```

### Solution: Metadata-Based Approach

#### Step 1: Move Tags to Model Metadata
```python
def inject_hierarchy_metadata(self, onnx_model):
    """Store hierarchy information in model metadata instead of node attributes"""
    
    # Create hierarchy mapping
    hierarchy_mapping = {}
    node_to_modules = {}
    
    for node in onnx_model.graph.node:
        node_name = node.name or f"{node.op_type}_{id(node)}"
        
        # Extract current tags from our internal metadata
        if node_name in self.operation_metadata:
            tags = self.operation_metadata[node_name]['tags']
            node_to_modules[node_name] = tags
    
    # Store in model metadata
    hierarchy_meta = onnx_model.metadata_props.add()
    hierarchy_meta.key = "hierarchy_mapping"
    hierarchy_meta.value = json.dumps(node_to_modules)
    
    # Store module definitions
    module_meta = onnx_model.metadata_props.add()
    module_meta.key = "module_hierarchy"
    module_meta.value = json.dumps(self.module_hierarchy)
    
    # Remove custom attributes from nodes
    for node in onnx_model.graph.node:
        # Remove our custom attributes
        node.attribute[:] = [attr for attr in node.attribute 
                           if attr.name not in ['source_module', 'hierarchy_tags']]
    
    return onnx_model
```

#### Step 2: Create Hierarchy Query Interface
```python
class HierarchyQueryInterface:
    """Query hierarchy information from ONNX model metadata"""
    
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)
        self.hierarchy_mapping = self._load_hierarchy_mapping()
        self.module_hierarchy = self._load_module_hierarchy()
    
    def _load_hierarchy_mapping(self):
        """Load node-to-module mapping from metadata"""
        for prop in self.model.metadata_props:
            if prop.key == "hierarchy_mapping":
                return json.loads(prop.value)
        return {}
    
    def get_nodes_by_module(self, module_path):
        """Get all nodes belonging to a module"""
        matching_nodes = []
        for node_name, modules in self.hierarchy_mapping.items():
            if module_path in modules:
                matching_nodes.append(node_name)
        return matching_nodes
    
    def get_module_for_node(self, node_name):
        """Get module(s) that own a specific node"""
        return self.hierarchy_mapping.get(node_name, [])
    
    def list_all_modules(self):
        """List all available modules"""
        all_modules = set()
        for modules in self.hierarchy_mapping.values():
            all_modules.update(modules)
        return sorted(all_modules)
```

#### Step 3: Update Subgraph Extractor
```python
class ProductionONNXSubgraphExtractor:
    """Production-ready subgraph extractor using metadata approach"""
    
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)
        self.hierarchy_query = HierarchyQueryInterface(onnx_model_path)
        self.boundary_detector = ModuleBoundaryDetector()
    
    def extract_clean_subgraph(self, target_module):
        """Extract clean, valid subgraph"""
        
        # 1. Get nodes for module using metadata
        target_nodes = self.hierarchy_query.get_nodes_by_module(target_module)
        
        # 2. Find clean boundaries
        boundaries = self.boundary_detector.find_module_boundaries(
            target_module, target_nodes
        )
        
        # 3. Create clean subgraph
        clean_graph = self.boundary_detector.create_clean_subgraph(
            target_module, boundaries
        )
        
        # 4. Ensure topological order
        sorted_nodes = self.boundary_detector.ensure_topological_order(
            clean_graph.node
        )
        clean_graph.node[:] = sorted_nodes
        
        # 5. Create model with clean metadata
        model = helper.make_model(clean_graph)
        
        # 6. Add clean hierarchy metadata (no custom attributes)
        self._add_clean_metadata(model, target_module)
        
        return model
```

## 3. IMPLEMENTATION TIMELINE

### Phase 1: Fix Topology Issues (Priority: HIGH)
- **Week 1**: Implement ModuleBoundaryDetector
- **Week 1**: Add topological sorting 
- **Week 1**: Test with simple modules (Conv, Linear)

### Phase 2: Fix Custom Attributes (Priority: HIGH)  
- **Week 2**: Implement metadata-based hierarchy storage
- **Week 2**: Create HierarchyQueryInterface
- **Week 2**: Update subgraph extractor

### Phase 3: Validation & Testing (Priority: MEDIUM)
- **Week 3**: Comprehensive testing across model types
- **Week 3**: Validate extracted models run correctly
- **Week 3**: Performance optimization

## 4. SUCCESS CRITERIA

### Topology Fixes Success:
- ✅ All extracted subgraphs pass ONNX validation
- ✅ No "topological sorting" errors
- ✅ Clean input/output interfaces

### Attribute Fixes Success:
- ✅ No "unrecognized attribute" errors
- ✅ Standard ONNX compliance
- ✅ Hierarchy information preserved in metadata

### Overall Success:
- ✅ 100% of extracted subgraphs are valid ONNX models
- ✅ 90%+ of extracted subgraphs are runnable in ONNX Runtime
- ✅ Extracted subgraphs produce equivalent outputs to original modules

## 5. RISK MITIGATION

### Risk 1: Complex Module Boundaries
**Mitigation**: Start with simple modules (Conv, Linear), then expand to complex ones (Attention)

### Risk 2: Performance Impact
**Mitigation**: Implement lazy loading and caching for large models

### Risk 3: Edge Cases
**Mitigation**: Comprehensive test suite with diverse model architectures

This plan will transform our 95% working solution into a production-ready system that fully realizes your vision!