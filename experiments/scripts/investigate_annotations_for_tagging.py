#!/usr/bin/env python3
"""Investigate PyTorch module __annotations__ for potential tagging/naming uses."""

import torch
import torch.nn as nn
from transformers import AutoModel
import warnings
warnings.filterwarnings("ignore")

def investigate_annotations_mutability():
    """Deep dive into module annotations mutability and ONNX impact."""
    
    print("🔍 Investigating PyTorch Module __annotations__ for Tagging...\n")
    
    # 1. Basic annotation exploration
    print("="*60)
    print("1. BASIC ANNOTATION STRUCTURE")
    print("="*60)
    
    # Create simple modules to understand annotations
    linear = nn.Linear(10, 5)
    conv = nn.Conv2d(3, 16, 3)
    
    print(f"Linear __annotations__: {linear.__annotations__}")
    print(f"Conv2d __annotations__: {conv.__annotations__}")
    
    # 2. Test mutability
    print("\n" + "="*60)
    print("2. TESTING ANNOTATION MUTABILITY")
    print("="*60)
    
    print("Original Linear annotations:")
    print(f"  Keys: {list(linear.__annotations__.keys())}")
    
    # Try to add custom annotations
    try:
        linear.__annotations__['custom_tag'] = str
        linear.__annotations__['hierarchy_path'] = str
        linear.__annotations__['operation_id'] = int
        
        print("\n✅ Successfully added custom annotations:")
        print(f"  Keys: {list(linear.__annotations__.keys())}")
        print(f"  Custom entries: {[k for k in linear.__annotations__.keys() if k.startswith('custom') or k.startswith('hierarchy') or k.startswith('operation')]}")
        
        # Set actual values
        linear.custom_tag = "attention.query"
        linear.hierarchy_path = "/encoder/layer.0/attention/self/query"
        linear.operation_id = 42
        
        print(f"\n✅ Successfully set annotation values:")
        print(f"  custom_tag: {getattr(linear, 'custom_tag', 'NOT SET')}")
        print(f"  hierarchy_path: {getattr(linear, 'hierarchy_path', 'NOT SET')}")
        print(f"  operation_id: {getattr(linear, 'operation_id', 'NOT SET')}")
        
    except Exception as e:
        print(f"❌ Failed to modify annotations: {e}")
    
    # 3. Test with BERT model
    print("\n" + "="*60)
    print("3. BERT MODEL ANNOTATION MODIFICATION")
    print("="*60)
    
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    # Find a query layer to modify
    query_layer = None
    for name, module in model.named_modules():
        if 'query' in name and isinstance(module, nn.Linear):
            query_layer = (name, module)
            break
    
    if query_layer:
        name, module = query_layer
        print(f"Testing with: {name}")
        
        # Original state
        print(f"Original annotations: {list(module.__annotations__.keys())}")
        
        # Add custom annotations
        try:
            module.__annotations__['modelexport_tag'] = str
            module.__annotations__['hierarchy_path'] = str
            module.__annotations__['onnx_node_name'] = str
            
            # Set values
            module.modelexport_tag = "bert.encoder.layer0.attention.self.query"
            module.hierarchy_path = "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention/query"
            module.onnx_node_name = "Query_MatMul_Layer0"
            
            print(f"✅ Added ModelExport annotations:")
            print(f"  modelexport_tag: {module.modelexport_tag}")
            print(f"  hierarchy_path: {module.hierarchy_path}")
            print(f"  onnx_node_name: {module.onnx_node_name}")
            
        except Exception as e:
            print(f"❌ Failed to modify BERT annotations: {e}")
    
    # 4. Test ONNX export impact
    print("\n" + "="*60)
    print("4. ONNX EXPORT IMPACT TESTING")
    print("="*60)
    
    # Create test model with custom annotations
    class AnnotatedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 10)
            self.layer2 = nn.Linear(10, 5)
            
            # Add custom annotations
            self.layer1.__annotations__['modelexport_tag'] = str
            self.layer1.__annotations__['custom_name'] = str
            self.layer2.__annotations__['modelexport_tag'] = str
            self.layer2.__annotations__['custom_name'] = str
            
            # Set values
            self.layer1.modelexport_tag = "feature_extractor"
            self.layer1.custom_name = "FeatureExtractor_Linear"
            self.layer2.modelexport_tag = "classifier"
            self.layer2.custom_name = "Classifier_Linear"
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = self.layer2(x)
            return x
    
    annotated_model = AnnotatedModel()
    sample_input = torch.randn(2, 10)
    
    # Test ONNX export
    from pathlib import Path
    output_dir = Path("temp/annotation_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Standard export
        torch.onnx.export(
            annotated_model, sample_input,
            output_dir / "annotated_standard.onnx",
            export_modules_as_functions=False,
            verbose=False
        )
        print("✅ Standard export with annotations successful")
        
        # Functions export
        try:
            torch.onnx.export(
                annotated_model, sample_input,
                output_dir / "annotated_functions.onnx",
                export_modules_as_functions=True,
                verbose=False
            )
            print("✅ Functions export with annotations successful")
        except Exception as e:
            print(f"❌ Functions export failed: {e}")
            
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
    
    # 5. Analyze exported ONNX for annotation traces
    print("\n" + "="*60)
    print("5. ANALYZING ONNX FOR ANNOTATION TRACES")
    print("="*60)
    
    import onnx
    
    onnx_file = output_dir / "annotated_standard.onnx"
    if onnx_file.exists():
        onnx_model = onnx.load(str(onnx_file))
        
        print("ONNX Node Analysis:")
        for i, node in enumerate(onnx_model.graph.node):
            print(f"  Node {i}: {node.op_type}")
            print(f"    Name: {node.name}")
            print(f"    Inputs: {list(node.input)}")
            print(f"    Outputs: {list(node.output)}")
            
            # Check for any annotation-related attributes
            if node.attribute:
                print(f"    Attributes: {[attr.name for attr in node.attribute]}")
            print()
        
        # Check for metadata
        print("ONNX Metadata:")
        if hasattr(onnx_model, 'metadata_props'):
            for prop in onnx_model.metadata_props:
                print(f"  {prop.key}: {prop.value}")
    
    # 6. Test annotation persistence
    print("\n" + "="*60)
    print("6. ANNOTATION PERSISTENCE TESTING")
    print("="*60)
    
    # Save and reload model
    torch.save(annotated_model.state_dict(), output_dir / "annotated_model.pth")
    
    # Create new model and load state
    new_model = AnnotatedModel()
    new_model.load_state_dict(torch.load(output_dir / "annotated_model.pth"))
    
    print("After save/load:")
    try:
        print(f"  layer1.modelexport_tag: {new_model.layer1.modelexport_tag}")
        print(f"  layer1 annotations: {list(new_model.layer1.__annotations__.keys())}")
        print("✅ Annotations persisted through save/load")
    except Exception as e:
        print(f"❌ Annotations lost: {e}")

def test_annotation_based_naming():
    """Test if we can use annotations to influence ONNX node naming."""
    
    print("\n" + "="*80)
    print("ANNOTATION-BASED ONNX NAMING EXPERIMENTS")
    print("="*80)
    
    class NamedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_query = nn.Linear(10, 10)
            self.attention_key = nn.Linear(10, 10)
            self.attention_value = nn.Linear(10, 10)
            
            # Try to influence naming through various means
            self.attention_query._get_name = lambda: "QueryProjection"
            self.attention_key._get_name = lambda: "KeyProjection"  
            self.attention_value._get_name = lambda: "ValueProjection"
            
            # Add custom attributes
            self.attention_query.onnx_name = "Attention_Query_MatMul"
            self.attention_key.onnx_name = "Attention_Key_MatMul"
            self.attention_value.onnx_name = "Attention_Value_MatMul"
            
        def forward(self, x):
            q = self.attention_query(x)
            k = self.attention_key(x)
            v = self.attention_value(x)
            return q + k + v
    
    named_model = NamedModel()
    sample_input = torch.randn(1, 10)
    
    output_dir = Path("temp/annotation_test")
    
    # Export with different naming strategies
    try:
        torch.onnx.export(
            named_model, sample_input,
            output_dir / "named_model.onnx",
            input_names=['input'],
            output_names=['output'],
            verbose=True  # Enable verbose to see naming
        )
        
        # Analyze the resulting names
        import onnx
        onnx_model = onnx.load(str(output_dir / "named_model.onnx"))
        
        print("\nGenerated ONNX Node Names:")
        for node in onnx_model.graph.node:
            print(f"  {node.op_type}: {node.name}")
            
    except Exception as e:
        print(f"❌ Named export failed: {e}")

if __name__ == "__main__":
    investigate_annotations_mutability()
    test_annotation_based_naming()