# Refined ONNX Node Tagging Design

## 🎯 **Design Goals**

1. **Eliminate duplication** between TracingHierarchyBuilder and EnhancedSemanticMapper
2. **Fix architectural mismatch** - EnhancedSemanticMapper should NOT depend on PyTorch model
3. **Unified interface** for all tagging strategies  
4. **Single source of truth** for module hierarchy analysis
5. **Clear separation of concerns** between hierarchy building and ONNX mapping

## 🏗️ **Current Architecture Problems**

### **Problem 1: Massive Duplication**
```python
# TracingHierarchyBuilder 
for name, module in model.named_modules():
    metadata = extract_module_metadata(module, name)
    
# EnhancedSemanticMapper (DUPLICATED!)
for name, module in model.named_modules(): 
    metadata = extract_module_metadata(module, name)  # 60%+ same work!
```

### **Problem 2: Wrong Dependencies**
```python
# Current (WRONG)
EnhancedSemanticMapper(pytorch_model, onnx_model)  # Depends on BOTH!

# Should be (CORRECT)  
EnhancedSemanticMapper(onnx_model, hierarchy_metadata)  # Only ONNX + metadata
```

### **Problem 3: Reverse-Engineering Complexity**
```python
# Current (COMPLEX)
node_name = "/BertModel/BertEncoder/BertLayer.0/BertAttention/MatMul_123"
parts = node_name.strip('/').split('/')  # Error-prone parsing!
scope_path = '.'.join(parts[:-1])

# Should be (SIMPLE)
# Use pre-computed hierarchy tags from TracingHierarchyBuilder
```

## 🔧 **Refined Architecture**

### **Core Principle: Single Source of Truth**
```
TracingHierarchyBuilder = SINGLE SOURCE for all module hierarchy information
All other components consume its output, never re-analyze the model
```

### **New Component Hierarchy**
```
BaseHierarchyBuilder (abstract)
├── TracingHierarchyBuilder (concrete - the source of truth)
├── StaticHierarchyBuilder (alternative implementation)
└── (future hierarchy builders)

SemanticMapper (abstract)  
├── ONNXSemanticMapper (concrete - maps ONNX nodes using hierarchy metadata)
├── DirectSemanticMapper (concrete - for htp/fx strategies)
└── (future semantic mappers)

Strategy (abstract)
├── EnhancedSemanticStrategy
├── HTPStrategy  
├── FXGraphStrategy
└── UsageBasedStrategy
```

## 📋 **Refined Workflow**

### **Universal Workflow for All Strategies:**
```
1. 🔍 HIERARCHY ANALYSIS (once, single source of truth)
   TracingHierarchyBuilder.trace_model_execution(model, inputs)
   → hierarchy_metadata {module_name: {traced_tag, semantic_info, ...}}

2. 📦 ONNX EXPORT (standard PyTorch)
   torch.onnx.export(model, inputs, output_path)
   → onnx_model

3. 🏷️ SEMANTIC MAPPING (strategy-specific, using hierarchy_metadata)
   SemanticMapper(onnx_model, hierarchy_metadata).map_all_nodes()
   → {onnx_node_name: semantic_info}

4. 💾 FINAL OUTPUT
   Enhanced ONNX + semantic metadata files
```

## 🔨 **Implementation Plan**

### **Phase 1: Refactor EnhancedSemanticMapper** 
```python
class ONNXSemanticMapper:
    """Maps ONNX nodes to semantic info using pre-computed hierarchy metadata."""
    
    def __init__(self, onnx_model: onnx.ModelProto, hierarchy_metadata: Dict[str, Any]):
        self.onnx_model = onnx_model
        self.hierarchy_metadata = hierarchy_metadata  # From TracingHierarchyBuilder
        # NO PyTorch model dependency!
    
    def map_node_to_semantic_info(self, onnx_node: onnx.NodeProto) -> Dict[str, Any]:
        """Map ONNX node to semantic info using hierarchy metadata."""
        # Extract scope from ONNX node name
        scope_path = self._extract_scope_from_node_name(onnx_node.name)
        
        # Look up in pre-computed hierarchy metadata (no model re-analysis!)
        if scope_path in self.hierarchy_metadata:
            module_info = self.hierarchy_metadata[scope_path]
            return {
                'hf_module_name': scope_path,
                'hierarchical_tag': module_info['traced_tag'],
                'semantic_info': module_info.get('semantic_info', {}),
                'confidence': 'high',
                'source': 'hierarchy_metadata'
            }
        
        # Fallback for nodes not in hierarchy
        return self._create_fallback_info(onnx_node)
```

### **Phase 2: Update EnhancedSemanticExporter**
```python
class EnhancedSemanticExporter:
    def export(self, model, args, output_path, **kwargs):
        # Step 1: Get hierarchy metadata (single source of truth)
        tracer = OptimizedTracingHierarchyBuilder()
        tracer.trace_model_execution(model, args)
        hierarchy_metadata = tracer.get_execution_summary()['module_hierarchy']
        
        # Step 2: Standard ONNX export
        torch.onnx.export(model, args, output_path, **kwargs)
        
        # Step 3: Semantic mapping using hierarchy metadata (NO model dependency!)
        onnx_model = onnx.load(output_path)
        semantic_mapper = ONNXSemanticMapper(onnx_model, hierarchy_metadata)
        
        # Step 4: Generate semantic tags for all nodes
        semantic_mappings = {}
        for node in onnx_model.graph.node:
            semantic_mappings[node.name] = semantic_mapper.map_node_to_semantic_info(node)
        
        return semantic_mappings
```

### **Phase 3: Unified Strategy Interface**
```python
class BaseExportStrategy(ABC):
    """Base class for all export strategies."""
    
    @abstractmethod
    def build_hierarchy_metadata(self, model, inputs) -> Dict[str, Any]:
        """Build hierarchy metadata (strategy-specific)."""
        pass
    
    @abstractmethod  
    def map_onnx_nodes(self, onnx_model, hierarchy_metadata) -> Dict[str, Any]:
        """Map ONNX nodes to semantic info (strategy-specific)."""
        pass
    
    def export(self, model, inputs, output_path, **kwargs):
        """Standard export workflow (same for all strategies)."""
        # Step 1: Build hierarchy metadata
        hierarchy_metadata = self.build_hierarchy_metadata(model, inputs)
        
        # Step 2: ONNX export  
        torch.onnx.export(model, inputs, output_path, **kwargs)
        
        # Step 3: Semantic mapping
        onnx_model = onnx.load(output_path) 
        semantic_mappings = self.map_onnx_nodes(onnx_model, hierarchy_metadata)
        
        return semantic_mappings

class EnhancedSemanticStrategy(BaseExportStrategy):
    def build_hierarchy_metadata(self, model, inputs):
        tracer = OptimizedTracingHierarchyBuilder()
        tracer.trace_model_execution(model, inputs)
        return tracer.get_execution_summary()['module_hierarchy']
    
    def map_onnx_nodes(self, onnx_model, hierarchy_metadata):
        mapper = ONNXSemanticMapper(onnx_model, hierarchy_metadata)
        return {node.name: mapper.map_node_to_semantic_info(node) 
                for node in onnx_model.graph.node}
```

## 📊 **Expected Benefits**

| Benefit | Current | After Refactoring | Improvement |
|---------|---------|-------------------|-------------|
| **Code Lines** | ~659 lines (EnhancedSemanticMapper) | ~200 lines (ONNXSemanticMapper) | **70% reduction** |
| **Model Analysis** | 2x (TracingHierarchyBuilder + EnhancedSemanticMapper) | 1x (TracingHierarchyBuilder only) | **50% reduction** |
| **Dependencies** | PyTorch model + ONNX model | ONNX model + metadata only | **Cleaner architecture** |
| **Accuracy** | Inference-based parsing | Direct hierarchy lookup | **Higher accuracy** |
| **Maintainability** | Dual systems | Single source of truth | **Much simpler** |

## 🚀 **Implementation Priority**

### **High Priority (Fix Enhanced Semantic Strategy)**
1. ✅ Refactor `EnhancedSemanticMapper` to remove PyTorch model dependency
2. ✅ Make it consume hierarchy metadata from `TracingHierarchyBuilder`
3. ✅ Eliminate duplication in model analysis

### **Medium Priority (Unify Strategies)**  
4. ⏳ Create `BaseExportStrategy` interface
5. ⏳ Migrate all strategies to unified interface
6. ⏳ Extract common components

### **Low Priority (Optimization)**
7. 🔄 Performance optimization
8. 🔄 Advanced semantic features
9. 🔄 Additional export strategies

## 🎯 **Success Criteria**

1. **Zero Duplication**: No component should re-analyze the PyTorch model if another has already done it
2. **Clear Dependencies**: `SemanticMapper` should only depend on ONNX model + hierarchy metadata
3. **Unified Interface**: All strategies follow the same workflow pattern
4. **Performance**: Single-pass model analysis for all strategies
5. **Maintainability**: Clear separation of concerns between hierarchy building and ONNX mapping

This refined design addresses all the architectural issues identified while maintaining backward compatibility and improving performance significantly.