#!/usr/bin/env python3
"""
Iteration 6: Test FX exporter with different model architectures.

This script tests the FX implementation with various model types to identify
which architectures work well with FX symbolic tracing and which don't.
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Add modelexport to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modelexport.fx_hierarchy_exporter import FXHierarchyExporter


def test_vision_models():
    """Test with computer vision models that might be FX-compatible."""
    print("=" * 60)
    print("TEST: Vision Models (CNN-based)")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 10),
            )
            
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.classifier(x)
            return x
    
    print("Testing Simple CNN...")
    model = SimpleCNN()
    inputs = torch.randn(1, 3, 32, 32)
    
    try:
        exporter = FXHierarchyExporter()
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            result = exporter.export(model, inputs, tmp.name)
            results['SimpleCNN'] = {
                'success': True,
                'hierarchy_nodes': result['hierarchy_nodes'],
                'unique_modules': result['unique_modules'],
                'coverage': result['fx_graph_stats']['coverage_ratio']
            }
            print(f"✅ Simple CNN: {result['hierarchy_nodes']} nodes, {result['fx_graph_stats']['coverage_ratio']:.1%} coverage")
            os.unlink(tmp.name)
            
    except Exception as e:
        results['SimpleCNN'] = {'success': False, 'error': str(e)}
        print(f"❌ Simple CNN failed: {e}")
    
    # Test 2: ResNet-style model  
    class MiniResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            
            # Simple residual block
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(16)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 10)
            
        def forward(self, x):
            identity = self.conv1(x)
            identity = self.bn1(identity)
            identity = self.relu(identity)
            
            out = self.conv2(identity)
            out = self.bn2(out)
            out += identity  # Residual connection
            out = self.relu(out)
            
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out
    
    print("Testing Mini ResNet...")
    model = MiniResNet()
    inputs = torch.randn(1, 3, 32, 32)
    
    try:
        exporter = FXHierarchyExporter()
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            result = exporter.export(model, inputs, tmp.name)
            results['MiniResNet'] = {
                'success': True,
                'hierarchy_nodes': result['hierarchy_nodes'],
                'unique_modules': result['unique_modules'],
                'coverage': result['fx_graph_stats']['coverage_ratio']
            }
            print(f"✅ Mini ResNet: {result['hierarchy_nodes']} nodes, {result['fx_graph_stats']['coverage_ratio']:.1%} coverage")
            os.unlink(tmp.name)
            
    except Exception as e:
        results['MiniResNet'] = {'success': False, 'error': str(e)}
        print(f"❌ Mini ResNet failed: {e}")
    
    return results

def test_sequential_models():
    """Test with purely sequential models."""
    print("=" * 60)
    print("TEST: Sequential Models (RNN/LSTM-like)")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Simple RNN
    class SimpleRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.rnn = nn.RNN(128, 64, batch_first=True)
            self.fc = nn.Linear(64, 10)
            
        def forward(self, x):
            # x shape: (batch, seq_len)
            embedded = self.embedding(x)
            rnn_out, _ = self.rnn(embedded)
            # Take last output
            last_out = rnn_out[:, -1, :]
            output = self.fc(last_out)
            return output
    
    print("Testing Simple RNN...")
    model = SimpleRNN()
    inputs = torch.randint(0, 1000, (1, 20))  # batch_size=1, seq_len=20
    
    try:
        exporter = FXHierarchyExporter()
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            result = exporter.export(model, inputs, tmp.name)
            results['SimpleRNN'] = {
                'success': True,
                'hierarchy_nodes': result['hierarchy_nodes'],
                'unique_modules': result['unique_modules'],
                'coverage': result['fx_graph_stats']['coverage_ratio']
            }
            print(f"✅ Simple RNN: {result['hierarchy_nodes']} nodes, {result['fx_graph_stats']['coverage_ratio']:.1%} coverage")
            os.unlink(tmp.name)
            
    except Exception as e:
        results['SimpleRNN'] = {'success': False, 'error': str(e)}
        print(f"❌ Simple RNN failed: {e}")
    
    # Test 2: Feed-forward network
    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 10),
                nn.Softmax(dim=1)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    print("Testing Feed-Forward Network...")
    model = FeedForward()
    inputs = torch.randn(1, 784)
    
    try:
        exporter = FXHierarchyExporter()
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            result = exporter.export(model, inputs, tmp.name)
            results['FeedForward'] = {
                'success': True,
                'hierarchy_nodes': result['hierarchy_nodes'],
                'unique_modules': result['unique_modules'],
                'coverage': result['fx_graph_stats']['coverage_ratio']
            }
            print(f"✅ Feed-Forward: {result['hierarchy_nodes']} nodes, {result['fx_graph_stats']['coverage_ratio']:.1%} coverage")
            os.unlink(tmp.name)
            
    except Exception as e:
        results['FeedForward'] = {'success': False, 'error': str(e)}
        print(f"❌ Feed-Forward failed: {e}")
    
    return results

def test_attention_models():
    """Test with attention-based models (but simpler than full transformers)."""
    print("=" * 60)
    print("TEST: Attention Models (Simplified)")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Simple Multi-Head Attention
    class SimpleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 128
            self.num_heads = 8
            self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
            self.norm = nn.LayerNorm(self.embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.ReLU(),
                nn.Linear(512, self.embed_dim)
            )
            self.classifier = nn.Linear(self.embed_dim, 10)
            
        def forward(self, x):
            # x shape: (batch, seq_len, embed_dim)
            
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            
            # Feed-forward
            ff_out = self.ff(x)
            x = self.norm(x + ff_out)
            
            # Global average pooling + classification
            x = x.mean(dim=1)  # Average over sequence dimension
            output = self.classifier(x)
            return output
    
    print("Testing Simple Attention...")
    model = SimpleAttention()
    inputs = torch.randn(1, 10, 128)  # batch_size=1, seq_len=10, embed_dim=128
    
    try:
        exporter = FXHierarchyExporter()
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            result = exporter.export(model, inputs, tmp.name)
            results['SimpleAttention'] = {
                'success': True,
                'hierarchy_nodes': result['hierarchy_nodes'],
                'unique_modules': result['unique_modules'],
                'coverage': result['fx_graph_stats']['coverage_ratio']
            }
            print(f"✅ Simple Attention: {result['hierarchy_nodes']} nodes, {result['fx_graph_stats']['coverage_ratio']:.1%} coverage")
            os.unlink(tmp.name)
            
    except Exception as e:
        results['SimpleAttention'] = {'success': False, 'error': str(e)}
        print(f"❌ Simple Attention failed: {e}")
    
    return results

def test_custom_operations():
    """Test with models that have custom operations."""
    print("=" * 60)
    print("TEST: Custom Operations")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Model with custom function calls
    class CustomOpsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)
            
        def forward(self, x):
            x = self.linear1(x)
            x = torch.tanh(x)  # Custom function
            x = torch.matmul(x, x.transpose(-1, -2))  # Custom matmul
            x = x.diagonal(dim1=-2, dim2=-1)  # Custom diagonal
            x = self.linear2(x[:, :10])  # Slice and linear
            return x
    
    print("Testing Custom Operations Model...")
    model = CustomOpsModel()
    inputs = torch.randn(1, 10)
    
    try:
        exporter = FXHierarchyExporter()
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            result = exporter.export(model, inputs, tmp.name)
            results['CustomOps'] = {
                'success': True,
                'hierarchy_nodes': result['hierarchy_nodes'],
                'unique_modules': result['unique_modules'],
                'coverage': result['fx_graph_stats']['coverage_ratio']
            }
            print(f"✅ Custom Ops: {result['hierarchy_nodes']} nodes, {result['fx_graph_stats']['coverage_ratio']:.1%} coverage")
            os.unlink(tmp.name)
            
    except Exception as e:
        results['CustomOps'] = {'success': False, 'error': str(e)}
        print(f"❌ Custom Ops failed: {e}")
    
    return results

def main():
    """Run tests with different model architectures."""
    print("🚀 Testing FX Exporter with Different Model Architectures")
    print(f"PyTorch version: {torch.__version__}")
    
    all_results = {}
    
    # Run architecture tests
    architecture_tests = [
        ("Vision Models", test_vision_models),
        ("Sequential Models", test_sequential_models),
        ("Attention Models", test_attention_models),
        ("Custom Operations", test_custom_operations),
    ]
    
    for test_name, test_func in architecture_tests:
        print(f"\n📋 Running: {test_name}")
        try:
            results = test_func()
            all_results[test_name] = results
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            all_results[test_name] = {'error': str(e)}
    
    # Summary analysis
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    total_models = 0
    successful_models = 0
    
    for category, results in all_results.items():
        if 'error' in results:
            print(f"\n❌ {category}: Test crashed")
            continue
            
        print(f"\n🔍 {category}:")
        category_success = 0
        category_total = 0
        
        for model_name, result in results.items():
            category_total += 1
            total_models += 1
            
            if result['success']:
                category_success += 1
                successful_models += 1
                print(f"   ✅ {model_name}: {result['hierarchy_nodes']} nodes, {result['coverage']:.1%} coverage")
            else:
                print(f"   ❌ {model_name}: {result['error']}")
        
        success_rate = category_success / category_total if category_total > 0 else 0
        print(f"   Category success rate: {category_success}/{category_total} ({success_rate:.1%})")
    
    overall_success_rate = successful_models / total_models if total_models > 0 else 0
    
    print(f"\n📊 Overall Results:")
    print(f"   Total models tested: {total_models}")
    print(f"   Successful models: {successful_models}")
    print(f"   Overall success rate: {overall_success_rate:.1%}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if overall_success_rate > 0.8:
        print("   🎉 FX approach works well with most architectures!")
        print("   ➡️  Consider FX as primary strategy for non-transformer models")
    elif overall_success_rate > 0.5:
        print("   ⚖️  FX approach works for some architectures")
        print("   ➡️  Implement hybrid approach: FX for compatible models, HTP for others")
    else:
        print("   ⚠️  FX approach has limited compatibility")
        print("   ➡️  Focus on specific use cases where FX excels")
    
    return 0 if overall_success_rate > 0.5 else 1

if __name__ == '__main__':
    exit(main())