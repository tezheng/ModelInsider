"""
Debug test to understand hook registration for different modules.
"""

import torch
import torch.nn as nn
from modelexport.hierarchy_exporter import HierarchyExporter


class DebugModel(nn.Module):
    """Model for debugging hook registration."""
    
    def __init__(self):
        super().__init__()
        self.processor = nn.Linear(10, 10)
        
    def forward(self, x):
        print(f"In DebugModel.forward: input shape = {x.shape}")
        # Slice operation happens in root model context
        sliced = x[1:4]
        print(f"After slice in root context: shape = {sliced.shape}")
        
        # Process through Linear layer (should have hook)
        processed = self.processor(sliced)
        print(f"After processing: shape = {processed.shape}")
        
        return processed


def test_hook_registration():
    """Test hook registration for different module types."""
    
    print("=== Debugging Hook Registration ===")
    
    model = DebugModel()
    print(f"Model structure:")
    for name, module in model.named_modules():
        module_class = module.__class__.__module__
        module_name = module.__class__.__name__
        print(f"  {name or 'ROOT'}: {module_class}.{module_name}")
    
    exporter = HierarchyExporter(strategy="htp")
    
    print("\n--- Testing Module Classification ---")
    for name, module in model.named_modules():
        if name:  # Skip root
            module_class_path = module.__class__.__module__
            should_tag = exporter._should_tag_module(module_class_path)
            creates_hierarchy = exporter._should_create_hierarchy_level(module)
            
            print(f"Module '{name}':")
            print(f"  Class path: {module_class_path}")
            print(f"  Should tag: {should_tag}")
            print(f"  Creates hierarchy: {creates_hierarchy}")
    
    print("\n--- Registering Hooks ---")
    exporter._reset_state()
    exporter._model = model
    exporter._register_hooks(model)
    
    print(f"Pre-hooks registered: {len(exporter._pre_hooks)}")
    print(f"Post-hooks registered: {len(exporter._post_hooks)}")
    print(f"Operation context after hook registration: {dict(exporter._operation_context)}")
    
    print("\n--- Testing Forward Pass ---")
    inputs = torch.randn(5, 10)
    
    print("Before forward pass:")
    print(f"  Tag stack: {exporter._tag_stack}")
    print(f"  Operation context: {dict(exporter._operation_context)}")
    
    with torch.no_grad():
        output = model(inputs)
    
    print("After forward pass:")
    print(f"  Tag stack: {exporter._tag_stack}")
    print(f"  Operation context: {dict(exporter._operation_context)}")
    
    print("\n--- Testing Root Model Context ---")
    # The issue is that slice happens in root model forward(), not in a hooked submodule
    # We need to ensure root model gets context too
    
    # Clean up
    exporter._remove_hooks()


class BetterDebugModel(nn.Module):
    """Model where slice happens within a hooked module."""
    
    def __init__(self):
        super().__init__()
        self.slicer = SlicerModule()
        self.processor = nn.Linear(10, 10)  # Keep 10 features
        
    def forward(self, x):
        sliced = self.slicer(x)  # Slice within a custom module
        processed = self.processor(sliced)
        return processed


class SlicerModule(nn.Module):
    """Custom module that performs slicing."""
    
    def forward(self, x):
        print(f"In SlicerModule.forward: input shape = {x.shape}")
        result = x[1:4]
        print(f"SlicerModule slice result: {result.shape}")
        return result


def test_better_model():
    """Test with a model where slice happens in a custom module."""
    
    print("\n\n=== Testing Model with Custom Slicer Module ===")
    
    model = BetterDebugModel()
    print(f"Model structure:")
    for name, module in model.named_modules():
        module_class = module.__class__.__module__
        module_name = module.__class__.__name__
        print(f"  {name or 'ROOT'}: {module_class}.{module_name}")
    
    exporter = HierarchyExporter(strategy="htp")
    
    print("\n--- Module Classification ---")
    for name, module in model.named_modules():
        if name:  # Skip root
            module_class_path = module.__class__.__module__
            should_tag = exporter._should_tag_module(module_class_path)
            creates_hierarchy = exporter._should_create_hierarchy_level(module)
            
            print(f"Module '{name}':")
            print(f"  Should tag: {should_tag}")
            print(f"  Creates hierarchy: {creates_hierarchy}")
    
    print("\n--- Hook Registration ---")
    exporter._reset_state()
    exporter._model = model
    exporter._register_hooks(model)
    
    print(f"Pre-hooks: {len(exporter._pre_hooks)}")
    print(f"Post-hooks: {len(exporter._post_hooks)}")
    
    print("\n--- Forward Pass with Hooks ---")
    inputs = torch.randn(5, 10)
    
    with torch.no_grad():
        output = model(inputs)
    
    print(f"Operation context after forward: {dict(exporter._operation_context)}")
    
    exporter._remove_hooks()


if __name__ == "__main__":
    test_hook_registration()
    test_better_model()