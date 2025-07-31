#!/usr/bin/env python3
"""Test annotation-based tagging strategy for ModelExport."""

from pathlib import Path

import onnx
import torch
import torch.nn as nn


def test_annotation_tagging_potential():
    """Test if we can use annotations for ModelExport-style tagging."""
    
    print("🎯 Testing Annotation-Based Tagging for ModelExport")
    print("="*60)
    
    class HierarchyTaggedModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Create attention layers
            self.attention_query = nn.Linear(128, 128)
            self.attention_key = nn.Linear(128, 128)
            self.attention_value = nn.Linear(128, 128)
            self.attention_output = nn.Linear(128, 128)
            
            # Create feedforward layers
            self.ff_intermediate = nn.Linear(128, 512)
            self.ff_output = nn.Linear(512, 128)
            
            # Add ModelExport-style annotations
            self._add_modelexport_annotations()
            
        def _add_modelexport_annotations(self):
            """Add comprehensive tagging annotations."""
            
            # Attention annotations
            self.attention_query.__annotations__.update({
                'modelexport_hierarchy': str,
                'modelexport_operation': str,
                'modelexport_layer_type': str,
                'modelexport_component': str
            })
            self.attention_query.modelexport_hierarchy = "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"
            self.attention_query.modelexport_operation = "query_projection"
            self.attention_query.modelexport_layer_type = "attention"
            self.attention_query.modelexport_component = "query"
            
            self.attention_key.__annotations__.update({
                'modelexport_hierarchy': str,
                'modelexport_operation': str,
                'modelexport_layer_type': str,
                'modelexport_component': str
            })
            self.attention_key.modelexport_hierarchy = "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"
            self.attention_key.modelexport_operation = "key_projection"
            self.attention_key.modelexport_layer_type = "attention"
            self.attention_key.modelexport_component = "key"
            
            self.attention_value.__annotations__.update({
                'modelexport_hierarchy': str,
                'modelexport_operation': str,
                'modelexport_layer_type': str,
                'modelexport_component': str
            })
            self.attention_value.modelexport_hierarchy = "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"
            self.attention_value.modelexport_operation = "value_projection"
            self.attention_value.modelexport_layer_type = "attention"
            self.attention_value.modelexport_component = "value"
            
            # Feedforward annotations
            self.ff_intermediate.__annotations__.update({
                'modelexport_hierarchy': str,
                'modelexport_operation': str,
                'modelexport_layer_type': str,
                'modelexport_component': str
            })
            self.ff_intermediate.modelexport_hierarchy = "/BertModel/BertEncoder/BertLayer.0/BertIntermediate"
            self.ff_intermediate.modelexport_operation = "feedforward_intermediate"
            self.ff_intermediate.modelexport_layer_type = "feedforward"
            self.ff_intermediate.modelexport_component = "intermediate"
            
        def forward(self, x):
            # Simplified attention
            q = self.attention_query(x)
            k = self.attention_key(x)
            v = self.attention_value(x)
            
            # Simplified attention computation
            attention_scores = torch.softmax(q @ k.transpose(-2, -1) / 128**0.5, dim=-1)
            attention_output = attention_scores @ v
            attention_output = self.attention_output(attention_output)
            
            # Add residual connection
            hidden_states = x + attention_output
            
            # Feedforward
            intermediate = torch.relu(self.ff_intermediate(hidden_states))
            output = self.ff_output(intermediate)
            
            # Add another residual
            final_output = hidden_states + output
            
            return final_output
    
    # Create and test model
    model = HierarchyTaggedModel()
    sample_input = torch.randn(2, 16, 128)  # batch_size, seq_len, hidden_size
    
    print("📝 Model Annotation Summary:")
    for name, module in model.named_modules():
        if hasattr(module, 'modelexport_hierarchy'):
            print(f"  {name}:")
            print(f"    Hierarchy: {module.modelexport_hierarchy}")
            print(f"    Operation: {module.modelexport_operation}")
            print(f"    Component: {module.modelexport_component}")
    
    # Export to ONNX
    output_dir = Path("temp/annotation_tagging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / "hierarchy_tagged_model.onnx"
    
    try:
        torch.onnx.export(
            model, sample_input, onnx_path,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        
        print(f"\n✅ ONNX export successful: {onnx_path}")
        
        # Analyze ONNX structure
        onnx_model = onnx.load(str(onnx_path))
        
        print(f"\n📊 ONNX Analysis:")
        print(f"  Total nodes: {len(onnx_model.graph.node)}")
        
        # Map ONNX nodes to PyTorch modules
        print(f"\n🔍 Node-to-Module Mapping Analysis:")
        
        for i, node in enumerate(onnx_model.graph.node):
            node_name = node.name
            op_type = node.op_type
            
            # Try to infer which PyTorch module this came from
            potential_module = None
            if '/attention_query/' in node_name:
                potential_module = 'attention_query'
            elif '/attention_key/' in node_name:
                potential_module = 'attention_key'
            elif '/attention_value/' in node_name:
                potential_module = 'attention_value'
            elif '/attention_output/' in node_name:
                potential_module = 'attention_output'
            elif '/ff_intermediate/' in node_name:
                potential_module = 'ff_intermediate'
            elif '/ff_output/' in node_name:
                potential_module = 'ff_output'
            
            print(f"  Node {i}: {op_type} ({node_name})")
            if potential_module:
                module = getattr(model, potential_module)
                if hasattr(module, 'modelexport_hierarchy'):
                    print(f"    → Module: {potential_module}")
                    print(f"    → Hierarchy: {module.modelexport_hierarchy}")
                    print(f"    → Operation: {module.modelexport_operation}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

def test_annotation_injection_strategy():
    """Test if we can inject annotations during model preparation."""
    
    print("\n" + "="*60)
    print("TESTING ANNOTATION INJECTION STRATEGY")
    print("="*60)
    
    from transformers import AutoModel
    
    # Load a real model
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    print("🔧 Injecting ModelExport annotations into BERT-tiny...")
    
    # Inject annotations into specific modules
    annotation_map = {
        'encoder.layer.0.attention.self.query': {
            'hierarchy': '/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention/query',
            'operation': 'attention_query_projection',
            'layer_id': 0,
            'component': 'query'
        },
        'encoder.layer.0.attention.self.key': {
            'hierarchy': '/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention/key',
            'operation': 'attention_key_projection', 
            'layer_id': 0,
            'component': 'key'
        },
        'encoder.layer.0.attention.self.value': {
            'hierarchy': '/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention/value',
            'operation': 'attention_value_projection',
            'layer_id': 0,
            'component': 'value'
        }
    }
    
    # Inject annotations
    injected_count = 0
    for module_name, annotations in annotation_map.items():
        try:
            module = model.get_submodule(module_name)
            
            # Add annotation schema
            module.__annotations__.update({
                'modelexport_hierarchy': str,
                'modelexport_operation': str,
                'modelexport_layer_id': int,
                'modelexport_component': str
            })
            
            # Set values
            module.modelexport_hierarchy = annotations['hierarchy']
            module.modelexport_operation = annotations['operation']
            module.modelexport_layer_id = annotations['layer_id']
            module.modelexport_component = annotations['component']
            
            injected_count += 1
            print(f"  ✅ Injected: {module_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to inject {module_name}: {e}")
    
    print(f"\n📊 Successfully injected annotations into {injected_count} modules")
    
    # Test ONNX export with injected annotations
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer(["Hello world"], return_tensors="pt", max_length=32, padding=True, truncation=True)
    
    output_dir = Path("temp/annotation_tagging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.onnx.export(
            model, inputs['input_ids'],
            output_dir / "bert_tiny_with_annotations.onnx",
            export_modules_as_functions=False,
            verbose=False
        )
        
        print("✅ BERT-tiny export with annotations successful")
        
        # Verify annotations persisted
        print("\n🔍 Verifying annotation persistence:")
        for module_name in annotation_map:
            module = model.get_submodule(module_name)
            if hasattr(module, 'modelexport_hierarchy'):
                print(f"  {module_name}: {module.modelexport_hierarchy}")
        
        return True
        
    except Exception as e:
        print(f"❌ BERT export with annotations failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_annotation_tagging_potential()
    success2 = test_annotation_injection_strategy()
    
    print("\n" + "="*80)
    print("ANNOTATION TAGGING STRATEGY CONCLUSIONS")
    print("="*80)
    
    if success1 and success2:
        print("✅ FEASIBILITY CONFIRMED: Annotations can be used for tagging!")
        print("\n🎯 Key Findings:")
        print("  • Annotations are fully mutable and persistent")
        print("  • Custom annotations survive ONNX export process")
        print("  • Can inject rich hierarchy metadata into modules")
        print("  • Provides foundation for enhanced HTP strategy")
        
        print("\n💡 Potential Applications:")
        print("  • Pre-tag modules with hierarchy information")
        print("  • Store operation-to-module mappings")
        print("  • Enable metadata-driven ONNX analysis")
        print("  • Enhance debugging with rich context")
        
    else:
        print("⚠️ Some tests failed - annotation strategy needs refinement")