#!/usr/bin/env python3
"""
Research script for HuggingFace to ONNX conversion methods
Focus: Understanding current conversion approaches and their hierarchy preservation capabilities
"""

import json
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def analyze_model_structure(model_name: str = "google/vit-base-patch16-224"):
    """Analyze the hierarchical structure of a HuggingFace model"""
    print(f"=== Analyzing Model Structure: {model_name} ===")
    
    # Load the model
    model = AutoModel.from_pretrained(model_name)
    
    # Extract module hierarchy
    hierarchy = {}
    for name, module in model.named_modules():
        if name:  # Skip root module
            hierarchy[name] = {
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'direct_children': len(list(module.children())),
                'depth': len(name.split('.'))
            }
    
    print(f"Total modules: {len(hierarchy)}")
    print(f"Max depth: {max(h['depth'] for h in hierarchy.values())}")
    
    # Print some sample hierarchy
    print("\nSample hierarchy (first 10):")
    for i, (name, info) in enumerate(hierarchy.items()):
        if i >= 10:
            break
        print(f"  {name}: {info['type']} (params: {info['parameters']}, children: {info['direct_children']})")
    
    return model, hierarchy

def test_torch_onnx_export(model, model_name: str = "google/vit-base-patch16-224"):
    """Test standard PyTorch ONNX export and analyze node naming"""
    print(f"\n=== Testing torch.onnx.export() ===")
    
    # Prepare dummy input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224
    
    # Export with detailed node names
    output_path = "vit_torch_export.onnx"
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )
        
        # Load and analyze the exported model
        onnx_model = onnx.load(output_path)
        
        print(f"Exported ONNX model saved to: {output_path}")
        print(f"Number of nodes: {len(onnx_model.graph.node)}")
        
        # Analyze node names
        node_names = [node.name for node in onnx_model.graph.node if node.name]
        print(f"Nodes with names: {len(node_names)}")
        
        # Show sample node names
        print("\nSample node names (first 10):")
        for _i, name in enumerate(node_names[:10]):
            print(f"  {name}")
        
        # Check for hierarchy information in node names
        hierarchical_names = [name for name in node_names if '.' in name or '_' in name]
        print(f"Potentially hierarchical names: {len(hierarchical_names)}")
        
        return onnx_model, node_names
        
    except Exception as e:
        print(f"Error during torch.onnx.export: {e}")
        return None, []

def test_optimum_export(model_name: str = "google/vit-base-patch16-224"):
    """Test HuggingFace Optimum export"""
    print(f"\n=== Testing Optimum Export ===")
    
    try:
        from optimum.onnxruntime import ORTModelForImageClassification
        
        # Export using Optimum
        ort_model = ORTModelForImageClassification.from_pretrained(
            model_name, 
            from_transformers=True,
            use_cache=False
        )
        
        # Save the model
        output_dir = "./vit_optimum_export"
        ort_model.save_pretrained(output_dir)
        
        # Load and analyze the ONNX file
        onnx_path = Path(output_dir) / "model.onnx"
        onnx_model = onnx.load(str(onnx_path))
        
        print(f"Optimum exported model saved to: {output_dir}")
        print(f"Number of nodes: {len(onnx_model.graph.node)}")
        
        # Analyze node names
        node_names = [node.name for node in onnx_model.graph.node if node.name]
        print(f"Nodes with names: {len(node_names)}")
        
        # Show sample node names
        print("\nSample node names (first 10):")
        for _i, name in enumerate(node_names[:10]):
            print(f"  {name}")
        
        return onnx_model, node_names
        
    except Exception as e:
        print(f"Error during Optimum export: {e}")
        return None, []

def analyze_onnx_metadata_capabilities():
    """Research ONNX metadata and custom attribute capabilities"""
    print(f"\n=== Analyzing ONNX Metadata Capabilities ===")
    
    # Create a simple model to test metadata
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 5)
            self.layer2 = nn.Linear(5, 1)
        
        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            return x
    
    model = SimpleModel()
    dummy_input = torch.randn(1, 10)
    
    # Export with custom naming
    output_path = "simple_model_metadata_test.onnx"
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        
        # Load and modify the model to add custom metadata
        onnx_model = onnx.load(output_path)
        
        # Analyze current metadata
        print("Current graph metadata:")
        for prop in onnx_model.metadata_props:
            print(f"  {prop.key}: {prop.value}")
        
        # Add custom metadata
        custom_metadata = onnx.StringStringEntryProto()
        custom_metadata.key = "module_hierarchy"
        custom_metadata.value = json.dumps({
            "layer1": {"type": "Linear", "path": "layer1"},
            "layer2": {"type": "Linear", "path": "layer2"}
        })
        onnx_model.metadata_props.append(custom_metadata)
        
        # Add custom attributes to nodes
        for i, node in enumerate(onnx_model.graph.node):
            if node.op_type == "MatMul":
                # Add custom attribute for module path
                attr = onnx.AttributeProto()
                attr.name = "module_path"
                attr.type = onnx.AttributeProto.STRING
                attr.s = f"layer{i//2 + 1}".encode()
                node.attribute.append(attr)
        
        # Save modified model
        modified_path = "simple_model_with_metadata.onnx"
        onnx.save(onnx_model, modified_path)
        
        print(f"Modified model saved to: {modified_path}")
        
        # Verify the metadata was added
        loaded_model = onnx.load(modified_path)
        print("\nAfter modification:")
        for prop in loaded_model.metadata_props:
            print(f"  {prop.key}: {prop.value}")
        
        return True
        
    except Exception as e:
        print(f"Error during metadata test: {e}")
        return False

def research_torch_dynamo_onnx():
    """Research torch.dynamo integration with ONNX export"""
    print(f"\n=== Researching Torch Dynamo ONNX Export ===")
    
    try:
        
        # Simple test to see if dynamo is available
        print("Torch Dynamo is available")
        
        # Test with a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        dummy_input = torch.randn(1, 10)
        
        # Try to compile with dynamo
        compiled_model = torch.compile(model, backend="eager")
        
        print("Model compiled with torch.compile")
        
        # Test inference
        output = compiled_model(dummy_input)
        print(f"Compiled model output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"Torch Dynamo not available or error: {e}")
        return False

def main():
    """Main research function"""
    print("=== HuggingFace to ONNX Conversion Research ===\n")
    
    # Phase 1: Analyze model structure
    model, hierarchy = analyze_model_structure()
    
    # Phase 2: Test different export methods
    torch_onnx_model, torch_node_names = test_torch_onnx_export(model)
    optimum_onnx_model, optimum_node_names = test_optimum_export()
    
    # Phase 3: Analyze metadata capabilities
    metadata_success = analyze_onnx_metadata_capabilities()
    
    # Phase 4: Research torch.dynamo
    dynamo_available = research_torch_dynamo_onnx()
    
    # Summary
    print(f"\n=== Research Summary ===")
    print(f"Model hierarchy depth: {max(h['depth'] for h in hierarchy.values()) if hierarchy else 'N/A'}")
    print(f"Total modules: {len(hierarchy)}")
    print(f"Torch ONNX export: {'Success' if torch_onnx_model else 'Failed'}")
    print(f"Optimum export: {'Success' if optimum_onnx_model else 'Failed'}")
    print(f"Metadata manipulation: {'Success' if metadata_success else 'Failed'}")
    print(f"Torch Dynamo available: {'Yes' if dynamo_available else 'No'}")
    
    # Compare node naming between methods
    if torch_node_names and optimum_node_names:
        print(f"\nNode naming comparison:")
        print(f"Torch export nodes: {len(torch_node_names)}")
        print(f"Optimum export nodes: {len(optimum_node_names)}")
        
        # Check for structured naming patterns
        torch_structured = sum(1 for name in torch_node_names if '.' in name or '_' in name)
        optimum_structured = sum(1 for name in optimum_node_names if '.' in name or '_' in name)
        
        print(f"Torch structured names: {torch_structured}")
        print(f"Optimum structured names: {optimum_structured}")

if __name__ == "__main__":
    main()