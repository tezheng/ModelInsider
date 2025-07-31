#!/usr/bin/env python3
"""
Research ONNX Functions capability and implementation
"""


import onnx
import torch
import torch.nn as nn
from onnx import helper
from transformers import AutoModel


def research_onnx_functions():
    """Research ONNX Functions capabilities"""
    print("=== ONNX Functions Research ===")
    
    # Check ONNX version and function support
    print(f"ONNX Version: {onnx.__version__}")
    
    # Create a simple function to test
    def create_simple_function():
        """Create a simple ONNX function for testing"""
        
        # Create a function that does: output = input * weight + bias (Linear layer)
        linear_function = helper.make_function(
            domain="custom.ops",
            fname="LinearFunction",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            nodes=[
                helper.make_node("MatMul", ["X", "W"], ["matmul_out"]),
                helper.make_node("Add", ["matmul_out", "B"], ["Y"])
            ],
            opset_imports=[helper.make_opsetid("", 11)]
        )
        
        return linear_function
    
    # Test function creation
    try:
        func = create_simple_function()
        print("✓ ONNX Function creation successful")
        print(f"  Function name: {func.name}")
        print(f"  Inputs: {list(func.input)}")
        print(f"  Outputs: {list(func.output)}")
        print(f"  Nodes: {len(func.node)}")
    except Exception as e:
        print(f"✗ ONNX Function creation failed: {e}")
        return False
    
    return True

def analyze_bert_architecture():
    """Analyze BERT model architecture for function identification"""
    print("\n=== BERT Architecture Analysis ===")
    
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    model = AutoModel.from_pretrained(model_name)
    
    print(f"Model: {model_name}")
    print(f"Model type: {type(model).__name__}")
    
    # Extract hierarchy
    hierarchy = {}
    for name, module in model.named_modules():
        if name:
            hierarchy[name] = {
                'type': type(module).__name__,
                'depth': len(name.split('.')),
                'has_children': len(list(module.children())) > 0,
                'parameter_count': sum(p.numel() for p in module.parameters(recurse=False))
            }
    
    print(f"Total modules: {len(hierarchy)}")
    
    # Identify function candidates
    function_candidates = {}
    for name, info in hierarchy.items():
        # Good candidates: substantial modules with children and parameters
        if (info['parameter_count'] > 100 and 
            info['has_children'] and 
            any(keyword in name.lower() for keyword in 
                ['embeddings', 'layer', 'attention', 'intermediate', 'output'])):
            function_candidates[name] = info
    
    print(f"\nFunction candidates ({len(function_candidates)}):")
    for name, info in function_candidates.items():
        print(f"  {name}: {info['type']} (params: {info['parameter_count']})")
    
    # Show detailed structure for first few layers
    print(f"\nDetailed structure (first 10):")
    for i, (name, info) in enumerate(hierarchy.items()):
        if i >= 10:
            break
        indent = "  " * info['depth']
        print(f"{indent}{name}: {info['type']} (params: {info['parameter_count']})")
    
    return model, hierarchy, function_candidates

def extract_module_components(model):
    """Extract individual components for piece-by-piece testing"""
    print("\n=== Module Component Extraction ===")
    
    components = {}
    
    # Extract major components
    if hasattr(model, 'embeddings'):
        components['embeddings'] = model.embeddings
        print("✓ Extracted embeddings")
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        for i, layer in enumerate(model.encoder.layer):
            components[f'encoder.layer.{i}'] = layer
            
            # Extract sub-components of each layer
            if hasattr(layer, 'attention'):
                components[f'encoder.layer.{i}.attention'] = layer.attention
                if hasattr(layer.attention, 'self'):
                    components[f'encoder.layer.{i}.attention.self'] = layer.attention.self
                if hasattr(layer.attention, 'output'):
                    components[f'encoder.layer.{i}.attention.output'] = layer.attention.output
            
            if hasattr(layer, 'intermediate'):
                components[f'encoder.layer.{i}.intermediate'] = layer.intermediate
            
            if hasattr(layer, 'output'):
                components[f'encoder.layer.{i}.output'] = layer.output
        
        print(f"✓ Extracted {len(model.encoder.layer)} encoder layers with sub-components")
    
    if hasattr(model, 'pooler'):
        components['pooler'] = model.pooler
        print("✓ Extracted pooler")
    
    print(f"Total components extracted: {len(components)}")
    return components

def create_dummy_inputs_for_bert():
    """Create appropriate dummy inputs for BERT model"""
    # BERT expects input_ids, attention_mask, token_type_ids
    batch_size = 1
    seq_length = 32
    
    dummy_inputs = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'token_type_ids': torch.zeros(batch_size, seq_length, dtype=torch.long)
    }
    
    return dummy_inputs

def test_component_export(component, component_name, dummy_input):
    """Test exporting a single component to ONNX"""
    print(f"\n=== Testing Component Export: {component_name} ===")
    
    try:
        # Create wrapper for the component
        class ComponentWrapper(nn.Module):
            def __init__(self, component):
                super().__init__()
                self.component = component
            
            def forward(self, *args, **kwargs):
                return self.component(*args, **kwargs)
        
        wrapper = ComponentWrapper(component)
        wrapper.eval()
        
        # Determine appropriate input for this component
        component_input = determine_component_input(component, component_name, dummy_input)
        
        if component_input is None:
            print(f"  ✗ Could not determine input for {component_name}")
            return None
        
        # Export to ONNX
        output_path = f"bert_component_{component_name.replace('.', '_')}.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                component_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=False,
                verbose=False
            )
        
        # Analyze exported model
        onnx_model = onnx.load(output_path)
        node_count = len(onnx_model.graph.node)
        
        print(f"  ✓ Exported {component_name}")
        print(f"    Output: {output_path}")
        print(f"    Nodes: {node_count}")
        print(f"    Inputs: {len(onnx_model.graph.input)}")
        print(f"    Outputs: {len(onnx_model.graph.output)}")
        
        return {
            'path': output_path,
            'node_count': node_count,
            'onnx_model': onnx_model
        }
        
    except Exception as e:
        print(f"  ✗ Export failed for {component_name}: {e}")
        return None

def determine_component_input(component, component_name, dummy_inputs):
    """Determine appropriate input for a component based on its type and position"""
    
    # For embeddings - use input_ids
    if 'embeddings' in component_name:
        return (dummy_inputs['input_ids'],)
    
    # For attention layers - use hidden states (simulate)
    if 'attention' in component_name:
        batch_size, seq_length = dummy_inputs['input_ids'].shape
        hidden_size = 128  # From model config
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = dummy_inputs['attention_mask']
        return (hidden_states, attention_mask)
    
    # For intermediate/output layers - use hidden states
    if any(keyword in component_name for keyword in ['intermediate', 'output']):
        batch_size, seq_length = dummy_inputs['input_ids'].shape
        hidden_size = 128
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        return (hidden_states,)
    
    # For encoder layers - use hidden states and attention mask
    if 'layer' in component_name and 'encoder' in component_name:
        batch_size, seq_length = dummy_inputs['input_ids'].shape
        hidden_size = 128
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = dummy_inputs['attention_mask']
        return (hidden_states, attention_mask)
    
    # For pooler - use hidden states
    if 'pooler' in component_name:
        batch_size, seq_length = dummy_inputs['input_ids'].shape
        hidden_size = 128
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        return (hidden_states,)
    
    return None

def main():
    """Main research function"""
    print("=== ONNX Functions + BERT Hierarchy Research ===\n")
    
    # Step 1: Research ONNX functions
    functions_supported = research_onnx_functions()
    
    if not functions_supported:
        print("ONNX functions not fully supported, will use alternative approach")
    
    # Step 2: Analyze BERT architecture
    model, hierarchy, function_candidates = analyze_bert_architecture()
    
    # Step 3: Extract components for testing
    components = extract_module_components(model)
    
    # Step 4: Create dummy inputs
    dummy_inputs = create_dummy_inputs_for_bert()
    
    # Step 5: Test component exports (first few for proof of concept)
    print(f"\n=== Component Export Testing ===")
    component_results = {}
    
    # Test a few key components
    test_components = [
        'embeddings',
        'encoder.layer.0',
        'encoder.layer.0.attention', 
        'encoder.layer.0.attention.self',
        'encoder.layer.0.intermediate'
    ]
    
    for comp_name in test_components:
        if comp_name in components:
            result = test_component_export(components[comp_name], comp_name, dummy_inputs)
            if result:
                component_results[comp_name] = result
    
    # Step 6: Summary
    print(f"\n=== Research Summary ===")
    print(f"ONNX Functions supported: {functions_supported}")
    print(f"BERT hierarchy depth: {max(info['depth'] for info in hierarchy.values())}")
    print(f"Function candidates: {len(function_candidates)}")
    print(f"Extractable components: {len(components)}")
    print(f"Successfully exported components: {len(component_results)}")
    
    # Show component export summary
    if component_results:
        print(f"\nComponent export results:")
        total_nodes = 0
        for name, result in component_results.items():
            nodes = result['node_count']
            total_nodes += nodes
            print(f"  {name}: {nodes} nodes")
        print(f"Total nodes in components: {total_nodes}")
    
    return model, hierarchy, components, component_results

if __name__ == "__main__":
    model, hierarchy, components, component_results = main()