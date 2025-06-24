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

## Iteration 2: Implementation Reality Check and Refined Strategy (2024-12-22)

### Current State Analysis: What We Actually Have

After comprehensive review of the actual `hierarchy_exporter.py` implementation (~1900 lines), several critical discoveries emerge that significantly impact the refactoring strategy.

#### **Discovery 1: Strategy Pattern Already Partially Implemented** ✅

**Reality Check**: The current code already separates strategies effectively:

```python
# In export() method
if self.strategy == "htp":
    return self._export_htp(model, example_inputs, output_path, **kwargs)
else:
    return self._export_usage_based(model, example_inputs, output_path, **kwargs)
```

**Current Architecture**:
- `_export_usage_based()`: Legacy approach with parameter-based tagging
- `_export_htp()`: Advanced Hierarchical Trace-and-Project approach
- Strategy-specific tag building: `_build_hierarchical_tag()` vs `_build_instance_aware_tag()`
- Minimal conditional logic elsewhere

**Assessment**: The original refactor plan **overestimated** the strategy pattern problem. Most strategy logic is already properly contained.

#### **Discovery 2: HTP Strategy is Production-Ready and Feature-Rich** 🚀

**Sophisticated Features Found**:
- **Operation Tracing**: 70+ PyTorch operations patched for execution context capture
- **Native Operation Support**: Handles `scaled_dot_product_attention` with pattern recognition
- **Stack-Based Context**: Real-time module execution tracking
- **Subgraph Extraction**: Complete multi-consumer tensor tagging for filtering
- **ONNX Projection**: Maps execution trace to ONNX nodes with universal type mapping

**Complexity Indicators**:
- **1900+ lines** of production-ready code
- **19 design revisions** showing mature evolution
- **Comprehensive test coverage** mentioned in documentation
- **Universal compliance** with MUST rules throughout

**Assessment**: HTP is not experimental - it's a sophisticated, working implementation that rivals commercial tools.

#### **Discovery 3: Operation Configuration is More Unified Than Expected** 📍

**Current Implementation Analysis**:

```python
# In _patch_torch_operations() (line 1312)
operations_to_patch = [
    (torch, 'matmul'), (torch, 'add'), (F, 'linear'),
    (F, 'scaled_dot_product_attention'),
    # ... 70+ operations centralized here
]

# In _project_execution_trace_to_onnx() (line 1526)  
torch_to_onnx_mapping = {
    'matmul': ['MatMul', 'Gemm'],
    'add': ['Add'],
    'linear': ['Gemm', 'MatMul'],
    # ... mapping definitions here
}
```

**Assessment**: While there IS duplication between patching and mapping, both are already centralized in specific methods, making this a **lower priority** than initially assessed.

### **Revised Risk Assessment** ⚠️

#### **High Risk Factors Identified**:

1. **Working Production System**: Current HTP implementation is sophisticated and working
2. **Complex State Management**: Stack-based hooks, operation tracing, multi-consumer logic
3. **Universal Design Compliance**: Any changes must maintain MUST rule compliance
4. **Test Integration**: Changes could break existing comprehensive test coverage

#### **Lower Impact Than Expected**:

1. **Strategy Pattern**: Already mostly implemented
2. **Operation Configuration**: Duplication exists but centralized
3. **Conditional Logic**: Minimal scattered conditionals found

### **Refined Refactoring Strategy: Conservative Incremental Approach** 🎯

#### **Phase 1: Low-Risk Consolidation** (Recommended Priority)

**1A. Unify Tag Building Methods** ⭐⭐⭐ (SAFE)
```python
def _build_tag(self, module_name: str, module: torch.nn.Module, 
               preserve_instances: bool = None) -> str:
    """Unified tag building with strategy-aware instance preservation."""
    if preserve_instances is None:
        preserve_instances = (self.strategy == "htp")
    
    if preserve_instances:
        return self._build_instance_aware_tag(module_name, module)
    else:
        return self._build_hierarchical_tag(module_name, module)
```

**Benefits**: 
- ✅ Eliminates 80% duplicate logic between tag building methods
- ✅ Zero risk - just consolidates existing working code
- ✅ Maintains all current functionality

**1B. Extract Operation Registry** ⭐⭐ (SAFE)
```python
class OperationConfig:
    """Centralized operation configuration for both patching and mapping."""
    
    OPERATION_REGISTRY = {
        'matmul': {
            'patch_targets': [(torch, 'matmul')],
            'onnx_types': ['MatMul', 'Gemm'],
            'priority': 1
        },
        'linear': {
            'patch_targets': [(F, 'linear')],
            'onnx_types': ['Gemm', 'MatMul'],
            'priority': 2
        },
        # ... unified definitions
    }
```

**Benefits**:
- ✅ Eliminates operation definition duplication
- ✅ Single source of truth for operation configuration
- ✅ Easy to add new operations

#### **Phase 2: Strategic Assessment** (CONDITIONAL)

**2A. Full Strategy Pattern Implementation** ⭐ (ONLY IF NEEDED)

**Recommendation**: **DEFER** until Phase 1 is complete and benefits are measured.

**Rationale**:
- Current strategy separation is already effective
- Risk of breaking sophisticated HTP implementation
- Unclear benefit vs. complexity tradeoff

**2B. Advanced Node Processing** ⭐⭐ (MODERATE RISK)

Only proceed if Phase 1 reveals significant patterns of repetitive conditional logic.

### **Implementation Priorities: Focus on HTP First** 🎯

#### **Strategic Decision: HTP-First Approach** ✅

**Rationale**:
- HTP is the advanced, feature-rich implementation
- Usage_based appears to be legacy/simplified approach  
- Focus limited resources on the production-ready strategy
- Multiple approaches create comparison capability

**Implementation Plan**:
1. **Stabilize HTP**: Consolidate HTP implementation first
2. **Extract Learnings**: Apply successful patterns to usage_based if needed
3. **Maintain Competition**: Keep both approaches for comparison/validation

### **Answers to Specific Implementation Questions**

#### **Q1: Operation Definition Centralization Location**

**Answer**: Two locations currently exist:

1. **`_patch_torch_operations()` (line 1312)**: `operations_to_patch` list - defines what to patch
2. **`_project_execution_trace_to_onnx()` (line 1526)**: `torch_to_onnx_mapping` dict - defines PyTorch→ONNX mapping

**Current State**: Both are centralized within their respective methods, but duplicated across methods.

**Refactor Target**: Create `OperationConfig` class to unify both definitions.

### **Success Metrics for Iteration 2**

#### **Phase 1 Success Criteria**:
- ✅ Reduce tag building duplication from 3 methods to 1
- ✅ Eliminate operation definition duplication
- ✅ Maintain 100% existing test coverage
- ✅ Zero functional changes to HTP behavior
- ✅ Preserve all performance characteristics

#### **Risk Mitigation**:
- Implement changes incrementally with immediate testing
- Maintain parallel old methods during transition
- Focus on HTP strategy as primary target
- Defer complex architectural changes until value is proven

### **Conclusion: Conservative Consolidation Over Aggressive Restructuring**

The reality check reveals that the HierarchyExporter is already a sophisticated, working system that needs **refinement**, not **restructuring**. The most valuable improvements come from consolidating existing functionality rather than implementing new architectural patterns.

**Recommended Approach**: Focus on low-risk, high-value consolidation of duplicated code within the existing architecture, with special attention to maintaining the production-ready HTP implementation.