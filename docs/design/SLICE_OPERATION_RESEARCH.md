# Research: Slice Operation Tagging in Hook-Based Implementation

## Problem Statement

The current hook-based implementation cannot intercept Python `__getitem__` operations (`x[1:4]`) that generate ONNX Slice nodes during export. This is because:

1. **Direct tensor indexing** (`x[1:4]`) is implemented at the C++ level in PyTorch's ATen library
2. **No patchable torch function**: Unlike `torch.matmul()` or `torch.add()`, there's no `torch.slice()` function to patch
3. **Hook limitations**: Forward hooks only capture module boundaries, not tensor-level operations
4. **ONNX conversion timing**: Slice nodes are generated during ONNX export, after our hooks have executed

## Current Test Results

From `test_slice_tagging.py`:
- ✅ HF-style model: Slice operations ARE tagged (1/1 tagged)
- ❌ Simple model: Slice operations NOT tagged (0/1 tagged)

This suggests the current implementation works when slice operations occur within tagged module contexts, but fails for standalone slicing.

## Research Findings

### 1. PyTorch Tensor Slicing Internals

**How `x[1:4]` Works:**
- Python `__getitem__` → C++ `Tensor::index()` → ATen dispatch → `aten::slice`
- No intermediate torch function we can patch
- Direct C++ implementation in `aten/src/ATen/TensorIndexing.h`

**During ONNX Export:**
- PyTorch traces operations during `torch.onnx.export()`
- `x[1:4]` becomes ONNX `Slice` node
- Symbolic functions in `torch/onnx/symbolic_opset*.py` handle conversion

### 2. Interception Strategies

#### Strategy A: Tensor Subclass with `__torch_dispatch__`
```python
class InterceptedTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func in [torch.ops.aten.slice.Tensor, torch.ops.aten.index.Tensor]:
            # Capture slice context
            current_tag = get_current_module_context()
            record_slice_operation(args, current_tag)
        return func(*args, **kwargs)
```

**Pros:** Low-level interception, catches all slice operations
**Cons:** Requires replacing all model tensors, significant complexity

#### Strategy B: Direct `__getitem__` Monkey Patching
```python
original_getitem = torch.Tensor.__getitem__

def traced_getitem(self, key):
    current_tag = get_current_module_context()
    if current_tag:
        record_slice_operation(key, current_tag)
    return original_getitem(self, key)

torch.Tensor.__getitem__ = traced_getitem
```

**Pros:** Direct interception, minimal code changes
**Cons:** Global monkey patch, potential conflicts, maintenance issues

#### Strategy C: ONNX Post-Processing (Current Working Approach)
The reason HF-style model works is that slice operations occur within module execution contexts that are already captured by hooks.

**Current Logic:**
1. Forward hooks capture module execution context
2. Operations within those contexts get tagged through propagation
3. Slice operations that occur during tagged module execution inherit tags

#### Strategy D: Custom ONNX Symbolic Functions
```python
def slice_symbolic(g, input, dim, start, end, step):
    # Custom logic to capture context during ONNX export
    current_tag = get_current_module_context()
    result = g.op("Slice", input, start, end, axes_i=[dim], steps_i=[step])
    tag_onnx_node(result, current_tag)
    return result

torch.onnx.register_custom_op_symbolic("aten::slice", slice_symbolic)
```

**Pros:** Works during ONNX export, official PyTorch extension point
**Cons:** Complex integration with existing hook system

### 3. Why HF-Style Model Works

Looking at the test results, the HF-style model succeeds because:

1. **Module Context**: Slice occurs within `forward()` method of a tagged module
2. **Tag Propagation**: The slice operation's inputs come from tagged operations (embeddings)
3. **Consumer Tagging**: The slice output feeds into tagged operations (LayerNorm)

The simple model fails because:
1. **Shallow Context**: Slice occurs at model root level, not within a meaningful module
2. **No Tagged Inputs**: Slice operates on raw model input
3. **Limited Propagation**: No clear path for tag inheritance

## Recommended Solutions

### Solution 1: Enhanced Context Tracking (Minimal Risk)

Extend the current hook system to maintain a global operation context stack:

```python
# Global context for tracking operations outside module boundaries
_global_operation_context = []

def patch_tensor_getitem():
    original_getitem = torch.Tensor.__getitem__
    
    def context_aware_getitem(self, key):
        # Record slice with current context
        if _global_operation_context:
            current_context = _global_operation_context[-1]
            # Store for later ONNX mapping
            _pending_slice_operations.append({
                'tensor_id': id(self),
                'key': key, 
                'context': current_context
            })
        
        return original_getitem(self, key)
    
    torch.Tensor.__getitem__ = context_aware_getitem
```

### Solution 2: ONNX Export Hook Integration (Medium Risk)

Integrate with PyTorch's ONNX export process:

```python
def register_slice_symbolic():
    from torch.onnx import register_custom_op_symbolic
    
    def slice_with_context(g, input, dim, start, end, step):
        # Get context from our tracking system
        context = get_tensor_context(input)
        result = g.op("Slice", input, start, end, axes_i=[dim], steps_i=[step])
        
        # Tag the result node
        if context:
            tag_onnx_node_with_context(result, context)
        
        return result
    
    register_custom_op_symbolic("aten::slice", slice_with_context, 9)
```

### Solution 3: Tensor Wrapper Approach (High Control)

Create a transparent tensor wrapper for context tracking:

```python
class ContextAwareTensor:
    def __init__(self, tensor, context=None):
        self._tensor = tensor
        self._context = context
    
    def __getitem__(self, key):
        result = self._tensor[key]
        # Propagate context to sliced tensor
        return ContextAwareTensor(result, self._context)
    
    def __getattr__(self, name):
        return getattr(self._tensor, name)
```

## Implementation Recommendation

**Phase 1: Quick Fix (Solution 1)**
- Implement minimal `__getitem__` patching with context tracking
- Integrate with existing tag propagation system
- Test with both simple and HF-style models

**Phase 2: Robust Solution (Solution 2)**
- Integrate with ONNX export hooks
- Custom symbolic functions for slice operations
- Full context awareness during export

**Phase 3: Architecture Enhancement (Solution 3)**
- Consider tensor wrapper approach for maximum control
- Evaluate performance impact
- Long-term maintainability

## Code Integration Points

1. **`_patch_torch_operations()`**: Add `__getitem__` patching
2. **`_project_execution_trace_to_onnx()`**: Handle slice operation mapping
3. **`get_current_tag()`**: Extend to work outside module contexts
4. **`OperationConfig.OPERATION_REGISTRY`**: Add slice operation metadata

## Testing Strategy

1. **Unit Tests**: Verify slice operations are intercepted
2. **Integration Tests**: Test with various slice patterns (`x[1:4]`, `x[::2]`, `x[1:-1]`)
3. **Model Tests**: Validate with simple and complex models
4. **Performance Tests**: Measure overhead of interception