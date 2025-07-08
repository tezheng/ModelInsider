#!/usr/bin/env python3
"""Trace how __annotations__ are accessed during ONNX export."""

import torch
import torch.nn as nn
import functools
import sys
from types import SimpleNamespace

def trace_annotation_access():
    """Monkey patch to trace when annotations are accessed."""
    
    print("🔍 Tracing __annotations__ access during ONNX export...\n")
    
    # Store original __getattribute__ methods
    original_getattribute = nn.Module.__getattribute__
    
    # Global tracker
    access_log = []
    
    def traced_getattribute(self, name):
        """Trace when __annotations__ is accessed."""
        if name == '__annotations__':
            # Record the access
            module_name = getattr(self, '_traced_name', 'unknown')
            access_info = {
                'module': module_name,
                'class': self.__class__.__name__,
                'stack_depth': len([frame for frame in sys._current_frames().values()]),
                'annotations': dict(self.__dict__.get('__annotations__', {}))
            }
            access_log.append(access_info)
            print(f"📝 Annotation access: {module_name} ({self.__class__.__name__})")
        
        return original_getattribute(self, name)
    
    # Apply the patch
    nn.Module.__getattribute__ = traced_getattribute
    
    try:
        # Create test model with annotations
        class TracedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 5)
                
                # Add custom annotations
                self.layer1.__annotations__['custom_tag'] = str
                self.layer1.__annotations__['hierarchy'] = str
                self.layer1.custom_tag = "layer1_tag"
                self.layer1.hierarchy = "/model/layer1"
                
                self.layer2.__annotations__['custom_tag'] = str  
                self.layer2.__annotations__['hierarchy'] = str
                self.layer2.custom_tag = "layer2_tag"
                self.layer2.hierarchy = "/model/layer2"
                
                # Add names for tracing
                self._traced_name = "root"
                self.layer1._traced_name = "layer1"
                self.layer2._traced_name = "layer2"
                
            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = self.layer2(x)
                return x
        
        model = TracedModel()
        sample_input = torch.randn(2, 10)
        
        print("🚀 Starting ONNX export with annotation tracing...")
        print("="*60)
        
        # Clear log
        access_log.clear()
        
        # Export to ONNX (this will trigger annotation access if it happens)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            torch.onnx.export(
                model, sample_input, f.name,
                verbose=False
            )
        
        print("="*60)
        print("✅ ONNX export completed")
        
        # Analyze access log
        print(f"\n📊 Annotation Access Summary:")
        print(f"  Total accesses: {len(access_log)}")
        
        if access_log:
            print(f"\n📝 Detailed Access Log:")
            for i, access in enumerate(access_log):
                print(f"  {i+1}. Module: {access['module']} ({access['class']})")
                print(f"     Annotations: {list(access['annotations'].keys())}")
        else:
            print("  ❌ No annotation access detected during ONNX export")
        
        return access_log
        
    finally:
        # Restore original method
        nn.Module.__getattribute__ = original_getattribute

def test_export_modules_as_functions_annotation_usage():
    """Test if export_modules_as_functions uses annotations differently."""
    
    print("\n" + "="*80)
    print("TESTING export_modules_as_functions ANNOTATION USAGE")
    print("="*80)
    
    # Create instrumented model
    class InstrumentedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 3)
            
            # Instrument the linear layer
            original_getattribute = self.linear.__getattribute__
            
            def instrumented_getattribute(name):
                if name == '__annotations__':
                    print(f"🔍 LINEAR: __annotations__ accessed!")
                    print(f"   Current annotations: {list(self.linear.__dict__.get('__annotations__', {}).keys())}")
                return original_getattribute(name)
            
            self.linear.__getattribute__ = instrumented_getattribute
            
        def forward(self, x):
            return self.linear(x)
    
    model = InstrumentedModel()
    sample_input = torch.randn(1, 5)
    
    # Test standard export
    print("\n📤 Testing standard export...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            torch.onnx.export(model, sample_input, f.name, 
                             export_modules_as_functions=False, verbose=False)
        print("✅ Standard export completed")
    except Exception as e:
        print(f"❌ Standard export failed: {e}")
    
    # Test functions export
    print("\n📤 Testing functions export...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            torch.onnx.export(model, sample_input, f.name,
                             export_modules_as_functions=True, verbose=False)
        print("✅ Functions export completed")
    except Exception as e:
        print(f"❌ Functions export failed: {e}")

def investigate_onnx_export_internals():
    """Investigate what PyTorch ONNX export actually does with modules."""
    
    print("\n" + "="*80)
    print("INVESTIGATING ONNX EXPORT INTERNALS")
    print("="*80)
    
    # Look at what attributes ONNX export accesses
    class AttributeLogger(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)
            self._attribute_accesses = []
            
            # Override __getattribute__ to log all access
            original_getattribute = self.linear.__getattribute__
            
            def logging_getattribute(name):
                # Log interesting attributes
                if name in ['__annotations__', '__dict__', '__class__', '_get_name', 
                           '_modules', '_parameters', 'training', 'forward']:
                    self._attribute_accesses.append(name)
                    print(f"🔍 Accessed: {name}")
                
                return original_getattribute(name)
            
            self.linear.__getattribute__ = logging_getattribute
            
        def forward(self, x):
            return self.linear(x)
    
    model = AttributeLogger()
    sample_input = torch.randn(1, 3)
    
    print("🚀 Starting attribute access logging...")
    print("-" * 40)
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            torch.onnx.export(model, sample_input, f.name, verbose=False)
        
        print("-" * 40)
        print("✅ Export completed")
        
        # Analyze what was accessed
        accesses = model._attribute_accesses
        print(f"\n📊 Attribute Access Summary:")
        print(f"  Total accesses: {len(accesses)}")
        print(f"  Unique attributes: {set(accesses)}")
        
        if '__annotations__' in accesses:
            print(f"  🎯 __annotations__ was accessed {accesses.count('__annotations__')} times!")
        else:
            print(f"  ❌ __annotations__ was never accessed")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")

def test_annotation_detection_strategies():
    """Test different ways annotations might be used."""
    
    print("\n" + "="*80) 
    print("TESTING ANNOTATION DETECTION STRATEGIES")
    print("="*80)
    
    # Strategy 1: Check if ONNX export uses getattr vs direct access
    print("🧪 Strategy 1: Testing getattr vs direct access...")
    
    linear = nn.Linear(5, 3)
    linear.__annotations__['test'] = str
    linear.test = "value"
    
    # Test different access methods
    print(f"  Direct access: {linear.__annotations__.get('test', 'NOT FOUND')}")
    print(f"  Getattr access: {getattr(linear, '__annotations__', {}).get('test', 'NOT FOUND')}")
    print(f"  Hasattr check: {hasattr(linear, '__annotations__')}")
    
    # Strategy 2: Check annotation introspection
    print(f"\n🧪 Strategy 2: Testing annotation introspection...")
    import inspect
    
    try:
        annotations = inspect.get_annotations(linear)
        print(f"  inspect.get_annotations: {annotations}")
    except Exception as e:
        print(f"  inspect.get_annotations failed: {e}")
    
    # Strategy 3: Test typing module interaction
    print(f"\n🧪 Strategy 3: Testing typing interaction...")
    try:
        from typing import get_type_hints
        hints = get_type_hints(linear.__class__)
        print(f"  get_type_hints: {hints}")
    except Exception as e:
        print(f"  get_type_hints failed: {e}")

if __name__ == "__main__":
    # Run all investigations
    access_log = trace_annotation_access()
    test_export_modules_as_functions_annotation_usage()
    investigate_onnx_export_internals()
    test_annotation_detection_strategies()
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    
    if access_log:
        print("✅ Annotations ARE accessed during ONNX export")
        print("🎯 This means we can potentially leverage them!")
    else:
        print("❌ Annotations are NOT directly accessed during standard ONNX export")
        print("🤔 Need to investigate alternative mechanisms")
    
    print("\n💡 Next steps:")
    print("  1. Check if annotations are used in specific export modes")
    print("  2. Investigate symbolic tracing interaction with annotations") 
    print("  3. Test custom ONNX metadata injection using annotations")
    print("  4. Explore torch.jit interaction with annotations")