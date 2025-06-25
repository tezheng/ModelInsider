# MEMO: Custom Symbolic Functions Breakthrough

## 🎯 BREAKTHROUGH DISCOVERY

Research reveals the fundamental flaw in current approach and shows the direct path to perfect hierarchy tagging.

## Key Findings

### 1. Current Hook Approach is FUNDAMENTALLY BROKEN ❌

**Problem**: Hooks get disabled during ONNX export
```python
module.register_forward_hook(hook)  # ← ERROR: "Modules with hooks can't be JIT traced"
```

This explains the execution order mismatch! Hooks are incompatible with ONNX export's JIT tracing process.

### 2. export_modules_as_functions is DEPRECATED ❌

- Removed in PyTorch 2.6+
- Was part of old TorchScript-based export
- Not a viable solution

### 3. SOLUTION: Custom Symbolic Functions ✅

PyTorch provides `register_custom_op_symbolic()` to hook directly into ONNX node creation:

```python
from torch.onnx import register_custom_op_symbolic

def tagged_matmul_symbolic(g, input1, input2):
    # Get current module context during ONNX export
    current_module = get_current_module_context()  # We need to implement this
    
    # Create ONNX node with embedded module metadata
    return g.op("MatMul", input1, input2,
                module_path_s=current_module)  # ← Direct tagging!

# Register for matmul operations
register_custom_op_symbolic('aten::addmm', tagged_matmul_symbolic, 11)
```

## Implementation Strategy

### Phase 1: Module Context Tracker

```python
class ModuleContextTracker:
    def __init__(self):
        self.module_stack = []
    
    def enter_module(self, name):
        self.module_stack.append(name)
    
    def exit_module(self):
        if self.module_stack:
            self.module_stack.pop()
    
    def get_current_context(self):
        return "/" + "/".join(self.module_stack) if self.module_stack else "/root"

# Global context tracker
_context_tracker = ModuleContextTracker()
```

### Phase 2: Custom Symbolic Functions

```python
def create_tagged_symbolic(op_name):
    def symbolic(g, *args, **kwargs):
        # Get current module context during ONNX export
        module_context = _context_tracker.get_current_context()
        
        # Create ONNX node with module metadata
        return g.op(op_name, *args,
                   hierarchy_tag_s=module_context,  # ← DIRECT EMBEDDING
                   **kwargs)
    return symbolic

# Register for key operations
register_custom_op_symbolic('aten::addmm', create_tagged_symbolic('MatMul'), 11)
register_custom_op_symbolic('aten::add', create_tagged_symbolic('Add'), 11)
```

### Phase 3: Context Synchronization - The Core Challenge

**Problem**: We still need to implement `get_current_module_context()` to track module entry/exit during ONNX export.

## Solutions for Module Entry/Exit Tracking

### Option 1: Forward Method Wrapping (Most Promising) ⭐

Instead of hooks, directly wrap the forward methods:

```python
def inject_context_tracking(model):
    """Wrap forward methods to track module context during ONNX export."""
    global _context_tracker
    
    for name, module in model.named_modules():
        if name and should_track_module(module):  # Skip root, focus on HF modules
            original_forward = module.forward
            module_name = name  # Capture in closure
            
            def create_tracked_forward(orig_forward, mod_name):
                def tracked_forward(*args, **kwargs):
                    _context_tracker.enter_module(mod_name)
                    try:
                        result = orig_forward(*args, **kwargs)
                    finally:
                        _context_tracker.exit_module()
                    return result
                return tracked_forward
            
            # Replace forward method
            module.forward = create_tracked_forward(original_forward, module_name)
```

**Pros**: Works during ONNX export, no hook restrictions
**Cons**: Modifies model, need to restore afterward

### Option 2: TorchScript Graph Analysis

Hook into the TorchScript tracing process to extract module scope:

```python
def extract_module_context_from_torchscript():
    """Use TorchScript's internal scope tracking."""
    
    # During torch.onnx.export, the model gets traced to TorchScript
    # TorchScript maintains scope information that we can access
    
    # This would require hooking into:
    # - torch.jit.trace
    # - torch._C._jit_pass_onnx_graph_shape_type_inference
    # - Graph node creation with scope preservation
```

**Pros**: Doesn't modify model
**Cons**: Requires deep PyTorch internals knowledge

### Option 3: Execution Frame Inspection

Use Python's stack frame inspection during symbolic function calls:

```python
import inspect

def get_current_module_context():
    """Extract module context from Python call stack."""
    
    frame = inspect.currentframe()
    try:
        # Walk up the stack to find module forward calls
        while frame:
            frame_info = inspect.getframeinfo(frame)
            locals_dict = frame.f_locals
            
            # Look for 'self' that's a torch.nn.Module
            if 'self' in locals_dict:
                obj = locals_dict['self']
                if isinstance(obj, torch.nn.Module):
                    # Found a module - try to identify which one
                    module_name = find_module_name_in_model(obj)
                    if module_name:
                        return module_name
            
            frame = frame.f_back
    finally:
        del frame
    
    return None
```

**Pros**: No model modification
**Cons**: Fragile, performance overhead, might not work in JIT

### Option 4: Hybrid Approach - Pre-Export Context Map

Build a context map before export, then use it during symbolic functions:

```python
def build_execution_context_map(model, example_inputs):
    """Pre-build a map of operations to module contexts."""
    
    # Step 1: Trace execution with temporary tracking
    context_map = {}
    
    def trace_forward_execution():
        # Temporarily wrap forwards to build operation→context mapping
        operation_counter = {}
        
        for name, module in model.named_modules():
            if name:
                original_forward = module.forward
                
                def create_context_recorder(mod_name):
                    def record_context(*args, **kwargs):
                        # Record that operations in this call belong to mod_name
                        start_op_count = len(get_all_traced_operations())
                        result = original_forward(*args, **kwargs)
                        end_op_count = len(get_all_traced_operations())
                        
                        # Map operations [start_op_count:end_op_count] to mod_name
                        for i in range(start_op_count, end_op_count):
                            context_map[i] = mod_name
                        
                        return result
                    return record_context
                
                module.forward = create_context_recorder(name)
    
    # Step 2: Use context_map during symbolic functions
    # (context_map[operation_index] → module_name)
```

## Recommended Implementation: Option 1 + Cleanup

**Forward Method Wrapping** is most practical:

```python
class HierarchyContextManager:
    def __init__(self):
        self.original_forwards = {}
        self.context_tracker = ModuleContextTracker()
    
    def inject_tracking(self, model):
        """Inject context tracking into model."""
        for name, module in model.named_modules():
            if self.should_track_module(name, module):
                self.original_forwards[module] = module.forward
                module.forward = self.create_tracked_forward(module.forward, name)
    
    def restore_model(self, model):
        """Restore original forward methods."""
        for module, original_forward in self.original_forwards.items():
            module.forward = original_forward
        self.original_forwards.clear()
    
    def create_tracked_forward(self, original_forward, module_name):
        def tracked_forward(*args, **kwargs):
            self.context_tracker.enter_module(module_name)
            try:
                return original_forward(*args, **kwargs)
            finally:
                self.context_tracker.exit_module()
        return tracked_forward

# Usage:
context_manager = HierarchyContextManager()

# Before ONNX export
context_manager.inject_tracking(model)

try:
    torch.onnx.export(model, inputs, output_path)  # Context tracked during export
finally:
    context_manager.restore_model(model)  # Always restore
```

## Benefits of Custom Symbolic Approach

1. **One-Pass Tagging** ✅ - Tags embedded during ONNX node creation
2. **No Order Mismatch** ✅ - Direct context, no trace matching needed
3. **No Hook Issues** ✅ - Works during JIT tracing
4. **Perfect Accuracy** ✅ - Each node gets exact execution context

This completely eliminates the layer.0/layer.1 bug because each ONNX node is tagged with its actual execution context during creation, not matched later!

## Next Steps

1. Implement `ModuleContextTracker` class
2. Create `HierarchyContextManager` for forward method wrapping
3. Implement custom symbolic functions for key operations
4. Test with BERT model to validate zero contamination
5. Benchmark performance vs current approach

This represents a fundamental architectural improvement that should eliminate cross-layer contamination entirely.