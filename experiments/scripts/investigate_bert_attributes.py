#!/usr/bin/env python3
"""Investigate why export_modules_as_functions fails with BERT-tiny."""

import torch
from transformers import AutoModel
import warnings
warnings.filterwarnings("ignore")

def investigate_module_attributes():
    """Deep dive into BERT module attributes."""
    
    print("🔍 Investigating BERT-tiny module attribute inconsistencies...\n")
    
    # Load BERT-tiny
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    
    # 1. Check for duplicate module classes
    module_classes = {}
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name not in module_classes:
            module_classes[class_name] = []
        module_classes[class_name].append((name, module))
    
    print("📊 Module class distribution:")
    for class_name, instances in module_classes.items():
        if len(instances) > 1:
            print(f"\n{class_name}: {len(instances)} instances")
            
            # Check if all instances have same attributes
            if len(instances) >= 2:
                # Get attributes of first instance
                first_attrs = set(dir(instances[0][1]))
                
                # Check other instances
                inconsistent = False
                for name, module in instances[1:]:
                    current_attrs = set(dir(module))
                    if current_attrs != first_attrs:
                        inconsistent = True
                        diff_added = current_attrs - first_attrs
                        diff_removed = first_attrs - current_attrs
                        print(f"  ⚠️  INCONSISTENCY found in {name}:")
                        if diff_added:
                            print(f"     Added attributes: {diff_added}")
                        if diff_removed:
                            print(f"     Missing attributes: {diff_removed}")
                
                if not inconsistent and len(instances) > 1:
                    print(f"  ✅ All {len(instances)} instances have consistent attributes")
    
    print("\n" + "="*60)
    
    # 2. Check for dynamic attributes
    print("\n🔎 Checking for dynamic/special attributes...\n")
    
    # Common problematic attributes in transformers
    special_attrs = ['gradient_checkpointing', '_gradient_checkpointing_func', 
                     'training', '_modules', '_parameters', '_buffers',
                     'forward_hooks', '_forward_hooks', '_forward_pre_hooks',
                     '_state_dict_hooks', '_load_state_dict_pre_hooks']
    
    for name, module in model.named_modules():
        if 'layer' in name.lower() and '.' in name:  # Focus on layer modules
            print(f"\n{name} ({module.__class__.__name__}):")
            for attr in special_attrs:
                if hasattr(module, attr):
                    value = getattr(module, attr)
                    # Check if it's not empty or default
                    if value is not None and value != {} and value != []:
                        print(f"  {attr}: {type(value).__name__} = {str(value)[:50]}...")
    
    # 3. Check shared parameters
    print("\n" + "="*60)
    print("\n🔗 Checking for shared parameters...\n")
    
    param_to_modules = {}
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            param_id = id(param)
            if param_id not in param_to_modules:
                param_to_modules[param_id] = []
            param_to_modules[param_id].append(f"{name}.{param_name}")
    
    for param_id, modules in param_to_modules.items():
        if len(modules) > 1:
            print(f"⚠️  Parameter shared between: {modules}")
    
    # 4. Check for module modifications
    print("\n" + "="*60)
    print("\n🔧 Checking for runtime modifications...\n")
    
    # Get encoder layers specifically
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layers = model.encoder.layer
        print(f"Found {len(layers)} encoder layers")
        
        # Compare first two layers
        if len(layers) >= 2:
            layer0_attrs = set(vars(layers[0]).keys())
            layer1_attrs = set(vars(layers[1]).keys())
            
            if layer0_attrs != layer1_attrs:
                print("\n⚠️  Layer 0 and Layer 1 have different instance attributes!")
                print(f"   Layer 0 only: {layer0_attrs - layer1_attrs}")
                print(f"   Layer 1 only: {layer1_attrs - layer0_attrs}")
            else:
                print("✅ Layer instance attributes are consistent")
            
            # Check attention modules
            if hasattr(layers[0], 'attention'):
                att0_attrs = set(vars(layers[0].attention).keys())
                att1_attrs = set(vars(layers[1].attention).keys())
                
                if att0_attrs != att1_attrs:
                    print("\n⚠️  Attention modules have different attributes!")
                    print(f"   Attention 0 only: {att0_attrs - att1_attrs}")
                    print(f"   Attention 1 only: {att1_attrs - att0_attrs}")

if __name__ == "__main__":
    investigate_module_attributes()