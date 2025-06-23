# Refactor Memo: HierarchyExporter Code Analysis

## Overview

Analysis of the HierarchyExporter codebase for refactoring opportunities, focusing on code simplification, elimination of redundancy, and architectural improvements.

## Major Refactoring Opportunities

### 1. Strategy Pattern Implementation ⭐⭐⭐

**Current Problem**: Strategy logic scattered throughout with `if self.strategy == "htp"` everywhere

**Impact**: 
- Code duplication across strategies
- Difficult to add new strategies
- Mixed responsibilities in single class

**Solution**: Abstract base class with concrete strategies

```python
from abc import ABC, abstractmethod

class HierarchyStrategy(ABC):
    def __init__(self, exporter_context):
        self.context = exporter_context
    
    @abstractmethod
    def export(self, model, example_inputs, output_path, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_operation_config(self) -> 'OperationConfig':
        pass

class HTPStrategy(HierarchyStrategy):
    def export(self, model, example_inputs, output_path, **kwargs):
        # Current _export_htp logic
        
class UsageBasedStrategy(HierarchyStrategy):
    def export(self, model, example_inputs, output_path, **kwargs):
        # Current _export_usage_based logic
```

**Benefits**:
- Clean separation of strategy logic
- Easy to add new strategies
- Single responsibility principle
- Eliminates conditional branching

### 2. Operation Configuration Unification ⭐⭐⭐

**Current Problem**: `operations_to_patch` and `torch_to_onnx_mapping` defined separately but tightly coupled

**Impact**:
- Duplication of operation definitions
- Easy to miss updating one when changing the other
- No single source of truth

**Solution**: Unified operation registry

```python
@dataclass
class OperationConfig:
    """Unified operation configuration."""
    
    # Single source of truth for operation mappings
    OPERATION_REGISTRY = {
        'matmul': {
            'torch_modules': [(torch, 'matmul')],
            'onnx_types': ['MatMul', 'Gemm'],
            'priority': 1
        },
        'embedding': {
            'torch_modules': [(torch, 'embedding'), (F, 'embedding')],
            'onnx_types': ['Gather'],
            'priority': 2
        },
        'layer_norm': {
            'torch_modules': [(F, 'layer_norm')],
            'onnx_types': ['Add', 'Mul', 'Div', 'ReduceMean', 'Sub', 'Sqrt', 'Pow'],
            'priority': 3
        },
        # ... etc
    }
    
    def get_operations_to_patch(self) -> List[Tuple[Any, str]]:
        result = []
        for op_data in self.OPERATION_REGISTRY.values():
            result.extend(op_data['torch_modules'])
        return result
    
    def get_torch_to_onnx_mapping(self) -> Dict[str, List[str]]:
        return {
            op_name: op_data['onnx_types'] 
            for op_name, op_data in self.OPERATION_REGISTRY.items()
        }
```

**Benefits**:
- Single source of truth
- Guaranteed consistency
- Easy to add new operations
- Configuration-driven approach

### 3. Remove Redundant Conditional Checks ⭐⭐

**Current Problem**: 
```python
if 'Constant' in onnx_nodes_by_type:
    for node in onnx_nodes_by_type['Constant']:
        # process constant
        
if 'Shape' in onnx_nodes_by_type:
    for node in onnx_nodes_by_type['Shape']:
        # process shape
```

**Solution**: Generic node processor

```python
class ONNXNodeProcessor:
    def process_by_type(self, onnx_model, type_processors: Dict[str, Callable]):
        nodes_by_type = self._group_nodes_by_type(onnx_model)
        
        for node_type, processor_func in type_processors.items():
            nodes = nodes_by_type.get(node_type, [])
            for node in nodes:
                processor_func(node)
    
    def _group_nodes_by_type(self, onnx_model):
        groups = defaultdict(list)
        for node in onnx_model.graph.node:
            groups[node.op_type].append(node)
        return groups

# Usage:
processor = ONNXNodeProcessor()
processor.process_by_type(onnx_model, {
    'Constant': self._process_constant_node,
    'Shape': self._process_shape_node, 
    'Gather': self._process_gather_node,
})
```

**Benefits**:
- Eliminates repetitive conditional checks
- Configuration-driven processing
- Easier to add new node types

### 4. Eliminate Deprecated/Redundant Functions ⭐⭐

**Functions to consolidate**:

#### Tag Building Methods
- `_build_hierarchical_tag()` and `_build_instance_aware_tag()` → single `_build_tag(use_instances=True)`

#### Propagation Methods  
- `_forward_propagate_tags()` and `_forward_propagate_tags_htp()` → single `_propagate_tags(strategy)`
- `_propagate_backward()`, `_propagate_forward()`, `_propagate_support_operations()` → unified `TagPropagationEngine`

#### Example Unified Tag Building:
```python
def _build_tag(self, module_name: str, module: torch.nn.Module, use_instances: bool = False) -> str:
    """Build hierarchical tag with optional instance preservation."""
    if not self._model:
        return f"/{module.__class__.__name__}"

    path_segments = [self._model.__class__.__name__]

    if module_name:
        current_module = self._model
        name_parts = module_name.split(".")

        for i, part in enumerate(name_parts):
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
                # ... logic
                
                if use_instances:
                    # Instance-aware logic (current _build_instance_aware_tag)
                    path_segments.append(part)
                else:
                    # Class-based logic (current _build_hierarchical_tag)
                    path_segments.append(class_name)

    return "/" + "/".join(path_segments)
```

### 5. Shared Operation Handler ⭐⭐⭐

**Current Problem**: `_patch_torch_operations()` and `_project_execution_trace_to_onnx()` both handle operation mapping but with different logic

**Solution**: Unified operation handler

```python
class OperationHandler:
    def __init__(self, config: OperationConfig):
        self.config = config
        self.patched_operations = {}
        self.operation_trace = []
    
    def patch_operations(self):
        """Patch PyTorch operations for tracing."""
        for module, op_name in self.config.get_operations_to_patch():
            if hasattr(module, op_name):
                original_op = getattr(module, op_name)
                self.patched_operations[(module, op_name)] = original_op
                
                traced_op = self._create_traced_operation(op_name, original_op)
                setattr(module, op_name, traced_op)
    
    def match_trace_to_onnx(self, onnx_model):
        """Match operation trace to ONNX nodes."""
        mapping = self.config.get_torch_to_onnx_mapping()
        # Unified matching logic using the same operation registry
        
    def unpatch_operations(self):
        """Restore original operations."""
        for (module, op_name), original_op in self.patched_operations.items():
            setattr(module, op_name, original_op)
```

**Benefits**:
- Single responsibility for operation handling
- Shared logic between patching and matching
- Guaranteed consistency

### 6. Proposed Refactored Architecture

```python
class HierarchyExporter:
    def __init__(self, strategy: str = "usage_based"):
        self.strategy_impl = self._create_strategy(strategy)
        self.operation_handler = OperationHandler(OperationConfig())
        self.node_processor = ONNXNodeProcessor()
        self.tag_engine = TagPropagationEngine()
        
        # Shared state
        self._tag_mapping: Dict[str, Dict[str, Any]] = {}
        self._model = None
    
    def _create_strategy(self, strategy: str) -> HierarchyStrategy:
        strategies = {
            'usage_based': UsageBasedStrategy,
            'htp': HTPStrategy
        }
        if strategy not in strategies:
            raise ValueError(f"Unsupported strategy: {strategy}")
        return strategies[strategy](self)
    
    def export(self, model, example_inputs, output_path, **kwargs):
        """Main export entry point - delegates to strategy."""
        return self.strategy_impl.export(model, example_inputs, output_path, **kwargs)
```

## Specific Code Improvements

### Remove Strategy Conditionals

**Before**:
```python
if self.strategy == "htp":
    hierarchical_tag = self._build_instance_aware_tag(module_name, module)
else:
    hierarchical_tag = self._build_hierarchical_tag(module_name, module)
```

**After**:
```python
hierarchical_tag = self.strategy_impl.build_tag(module_name, module)
```

### Unify Tag Building

**Before**: Two separate methods with 80% similar logic
**After**: Single method with strategy parameter

### Generic Node Processing

**Before**: Multiple `if node_type in nodes_by_type:` blocks
**After**: Configuration-driven processing with type handlers

## Implementation Priority

### Phase 1: Core Architecture ⭐⭐⭐
1. **Strategy Pattern Implementation** - Eliminates conditional logic, improves extensibility
2. **Operation Unification** - Single source of truth, reduces duplication

### Phase 2: Code Consolidation ⭐⭐  
3. **Node Processor** - Eliminates repetitive conditional checks
4. **Function Consolidation** - Reduces maintenance burden

### Phase 3: Polish ⭐
5. **Code Organization** - Separates concerns, improves readability

## Benefits Summary

- **Maintainability**: Single responsibility, clear separation of concerns
- **Extensibility**: Easy to add new strategies and operations
- **Reliability**: Single source of truth reduces inconsistencies  
- **Testability**: Smaller, focused classes are easier to unit test
- **Readability**: Less conditional logic, clearer intent

## Risk Assessment

- **Low Risk**: Strategy pattern and operation unification are well-established patterns
- **Medium Risk**: Node processor changes require careful testing of edge cases
- **Mitigation**: Implement incrementally with comprehensive test coverage

## Next Steps

1. Create strategy interfaces and base classes
2. Implement unified operation configuration
3. Refactor one strategy at a time
4. Add comprehensive tests for new architecture
5. Migrate remaining functionality

---

*This refactoring maintains the universal approach and MUST-RULES while significantly improving code organization and maintainability.*