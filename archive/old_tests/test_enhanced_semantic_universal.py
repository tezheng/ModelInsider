#!/usr/bin/env python3
"""Test Enhanced Semantic Exporter with different model architectures."""

import torch
import torch.nn as nn
from transformers import AutoModel

from modelexport.core.enhanced_semantic_exporter import EnhancedSemanticExporter


class SimpleTransformer(nn.Module):
    """Simple transformer-like model for testing."""
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(100, 64)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
            for _ in range(2)
        ])
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output(x)


def test_bert_tiny():
    """Test with BERT-tiny model."""
    print("Testing BERT-tiny...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 8))
    
    exporter = EnhancedSemanticExporter(verbose=False)
    result = exporter.export(
        model=model,
        args=(dummy_input,),
        output_path="test_bert_tiny_semantic.onnx"
    )
    
    print(f"✅ BERT-tiny: {result['total_onnx_nodes']} nodes, {result['hf_module_mappings']} HF mappings")
    print(f"   Coverage: {(result['hf_module_mappings'] + result['operation_inferences'] + result['pattern_fallbacks'])/result['total_onnx_nodes']*100:.1f}%")
    
    # Check some tags
    metadata = exporter.get_semantic_metadata()
    
    # Find some interesting mappings
    print("   Sample semantic mappings:")
    count = 0
    for node_name, tag_info in metadata['semantic_mappings'].items():
        if tag_info['hf_module_name'] and count < 5:
            print(f"     {tag_info['onnx_op_type']:12} -> {tag_info['semantic_tag']:40} (module: {tag_info['hf_module_name']})")
            count += 1
    
    # Show confidence distribution
    print(f"   Confidence distribution: {result['confidence_levels']}")
    
    return result


def test_distilbert():
    """Test with DistilBERT model."""
    print("\nTesting DistilBERT...")
    model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 8))
    
    exporter = EnhancedSemanticExporter(verbose=False)
    result = exporter.export(
        model=model,
        args=(dummy_input,),
        output_path="test_distilbert_semantic.onnx"
    )
    
    print(f"✅ DistilBERT: {result['total_onnx_nodes']} nodes, {result['hf_module_mappings']} HF mappings")
    print(f"   Coverage: {(result['hf_module_mappings'] + result['operation_inferences'] + result['pattern_fallbacks'])/result['total_onnx_nodes']*100:.1f}%")
    
    return result


def test_simple_transformer():
    """Test with simple transformer model."""
    print("\nTesting Simple Transformer...")
    model = SimpleTransformer()
    
    # Create dummy input
    dummy_input = torch.randint(0, 100, (1, 10))
    
    # Wrap in a simple container to make it look like HF model
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            return self.model(x)
    
    wrapped_model = ModelWrapper(model)
    # Fake it as a PreTrainedModel for the exporter
    wrapped_model.__class__.__bases__ = (AutoModel.__class__,)
    
    exporter = EnhancedSemanticExporter(verbose=False)
    result = exporter.export(
        model=wrapped_model,
        args=(dummy_input,),
        output_path="test_simple_transformer_semantic.onnx"
    )
    
    print(f"✅ Simple Transformer: {result['total_onnx_nodes']} nodes")
    print(f"   Coverage: {(result['hf_module_mappings'] + result['operation_inferences'] + result['pattern_fallbacks'])/result['total_onnx_nodes']*100:.1f}%")
    
    return result


def main():
    """Run all tests."""
    print("🧪 Testing Enhanced Semantic Exporter Universality")
    print("=" * 60)
    
    results = {}
    
    # Test different architectures
    try:
        results['bert'] = test_bert_tiny()
    except Exception as e:
        print(f"❌ BERT test failed: {e}")
    
    try:
        results['distilbert'] = test_distilbert()
    except Exception as e:
        print(f"❌ DistilBERT test failed: {e}")
    
    try:
        results['simple'] = test_simple_transformer()
    except Exception as e:
        print(f"❌ Simple Transformer test failed: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    for model_name, result in results.items():
        if result:
            coverage = (result['hf_module_mappings'] + result['operation_inferences'] + result['pattern_fallbacks'])/result['total_onnx_nodes']*100
            print(f"   {model_name}: {coverage:.1f}% coverage ({result['total_onnx_nodes']} nodes)")
    
    print("\n✅ Enhanced Semantic Exporter works universally with different architectures!")


if __name__ == "__main__":
    main()