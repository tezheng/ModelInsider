# ADR-001: Root Module Hook Strategy

## Status
Accepted

## Context
During the implementation of the hierarchy-preserving ONNX export system, we need to decide whether to register forward hooks on the root model module or handle root context through manual stack initialization.

## Decision
**We will NOT hook the root module** and instead use manual stack initialization.

## Rationale

### Analysis Summary
We evaluated three approaches:
1. **Hook Root Module**: Register pre/post hooks on root like other modules
2. **Don't Hook Root** (Current): Manual stack initialization with root tag
3. **Conditional Root Hooking**: Hook only if root has meaningful operations

### Decision Factors

#### ✅ Advantages of NOT Hooking Root
1. **ONNX Export Compatibility**: Root hooks might interfere with `torch.onnx.export()` tracing process
2. **Transformer Architecture Reality**: Most HuggingFace models have trivial root forwards (delegation only)
3. **Stack Management Predictability**: Manual initialization ensures stack is never empty
4. **Performance**: One less hook call per forward pass
5. **Cleaner Export Process**: Avoids potential hook interference during ONNX export

#### ❌ Disadvantages of NOT Hooking Root
1. **Stack Initialization Inconsistency**: Root doesn't follow same push/pop pattern as other modules
2. **Asymmetric Hook Pattern**: Root is special-cased rather than uniform
3. **Root-Level Operations Not Traced**: Any operations in root forward() won't be attributed
4. **Missing Context for Fallback**: Edge cases might lack root context

### Implementation Decision
```python
def _register_hooks(self, model: torch.nn.Module):
    # Initialize stack with root (manual approach)
    root_tag = f"/{model.__class__.__name__}"
    self._tag_stack.append(root_tag)
    
    # Register hooks on submodules only
    for name, module in model.named_modules():
        if name:  # Skip root module (name="" for root)
            # ... incremental hook registration
```

## Consequences

### Positive
- **Robust ONNX Export**: No hook interference with export process
- **Predictable Stack State**: Stack always has at least root tag
- **Performance**: Minimal overhead for common delegation pattern
- **Simplicity**: Clear separation between root initialization and submodule hooks

### Negative
- **Special Case Handling**: Root module requires different treatment
- **Potential Missing Attribution**: Root-level operations won't be traced (rare in practice)

### Mitigation
- Root-level operations are rare in transformer architectures
- Manual stack initialization provides necessary fallback context
- Clear documentation of the special case handling

## Alternatives Considered

### Alternative 1: Hook Root Module
```python
# Would register hooks on root like other modules
root_pre_hook = model.register_forward_pre_hook(create_root_pre_hook())
```
**Rejected** due to potential ONNX export interference.

### Alternative 2: Conditional Root Hooking
```python
# Hook root only if it has meaningful operations
if self._has_meaningful_root_operations(model):
    # Hook root
else:
    # Manual initialization
```
**Rejected** as too complex for marginal benefit.

## Notes
- This decision aligns with the current implementation
- Future optimization could implement conditional hooking if needed
- Decision may be revisited if root-level operation attribution becomes critical

## References
- Issue: Cross-layer tag contamination and slice operation context
- Related: Stack-based hierarchical tag management
- Implementation: `HierarchyPreservingExporter._register_hooks()`