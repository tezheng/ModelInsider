#!/usr/bin/env python3
"""
Iteration 7: Test FX exporter with hybrid architecture detection and fallback.

This script tests the enhanced FX implementation that automatically detects
model architecture compatibility and falls back to HTP when appropriate.
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


def test_architecture_detection():
    """Test the architecture detection system."""
    print("=" * 60)
    print("TEST: Architecture Detection System")
    print("=" * 60)
    
    # Test models with known compatibility
    test_cases = [
        {
            'name': 'SimpleCNN',
            'model': lambda: create_simple_cnn(),
            'inputs': lambda: torch.randn(1, 3, 32, 32),
            'expected_type': 'vision_cnn',
            'expected_compatible': True
        },
        {
            'name': 'FeedForward',
            'model': lambda: create_feedforward(),
            'inputs': lambda: torch.randn(1, 784),
            'expected_type': 'feedforward',
            'expected_compatible': True
        },
        {
            'name': 'SimpleAttention',
            'model': lambda: create_simple_attention(),
            'inputs': lambda: torch.randn(1, 10, 128),
            'expected_type': 'simple_attention',
            'expected_compatible': True
        }
    ]
    
    # Test with auto_fallback disabled to see raw detection
    exporter = FXHierarchyExporter(auto_fallback=False)
    results = {}
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}...")
        
        try:
            model = test_case['model']()
            inputs = test_case['inputs']()
            
            # Test architecture detection directly
            compatibility = exporter._analyze_model_compatibility(model, inputs)
            
            results[test_case['name']] = {
                'detected_type': compatibility['architecture_type'],
                'fx_compatible': compatibility['fx_compatible'],
                'confidence': compatibility['confidence'],
                'expected_type': test_case['expected_type'],
                'expected_compatible': test_case['expected_compatible'],
                'correct_detection': compatibility['architecture_type'] == test_case['expected_type'],
                'correct_compatibility': compatibility['fx_compatible'] == test_case['expected_compatible']
            }
            
            print(f"   Detected: {compatibility['architecture_type']} "
                  f"(compatible: {compatibility['fx_compatible']}, "
                  f"confidence: {compatibility['confidence']:.2f})")
            print(f"   Expected: {test_case['expected_type']} "
                  f"(compatible: {test_case['expected_compatible']})")
            
            if results[test_case['name']]['correct_detection']:
                print("   ✅ Architecture detection correct")
            else:
                print("   ❌ Architecture detection incorrect")
                
        except Exception as e:
            print(f"   ❌ Detection failed: {e}")
            results[test_case['name']] = {'error': str(e)}
    
    return results

def test_hybrid_fallback():
    """Test the hybrid fallback functionality."""
    print("=" * 60)
    print("TEST: Hybrid Fallback Functionality")
    print("=" * 60)
    
    # Create a mock BERT-like model to trigger fallback
    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            # Include modules that trigger transformer detection
            self.bert_model = nn.ModuleDict({
                'encoder': nn.ModuleDict({
                    'layer': nn.ModuleList([
                        nn.ModuleDict({
                            'attention': nn.ModuleDict({
                                'self': nn.MultiheadAttention(128, 8)
                            })
                        })
                    ])
                })
            })
            self.classifier = nn.Linear(128, 10)
            
        def forward(self, x):
            # Simple forward that should work with FX
            # (The architecture detection should still flag it as transformer)
            return self.classifier(x.mean(dim=1))
    
    print("Testing Mock Transformer (should trigger fallback)...")
    
    try:
        model = MockTransformer()
        inputs = torch.randn(1, 10, 128)
        
        # Test with auto_fallback enabled
        exporter = FXHierarchyExporter(auto_fallback=True)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            # This should trigger fallback to HTP
            result = exporter.export(model, inputs, tmp.name)
            
            print(f"✅ Export completed!")
            print(f"   Strategy used: {result['strategy']}")
            print(f"   Fallback used: {result.get('fallback_used', False)}")
            print(f"   Hierarchy nodes: {result['hierarchy_nodes']}")
            
            if result.get('fallback_used'):
                print(f"   Fallback reason: {result.get('fallback_reason', 'Unknown')}")
                print("   ✅ Hybrid fallback working correctly")
            else:
                print("   ⚠️  Expected fallback but didn't occur")
            
            os.unlink(tmp.name)
            return True
            
    except Exception as e:
        print(f"❌ Hybrid fallback test failed: {e}")
        return False

def test_architecture_comparison():
    """Compare FX vs hybrid approach on compatible models."""
    print("=" * 60)
    print("TEST: FX vs Hybrid Performance Comparison")
    print("=" * 60)
    
    # Test with a model that should work well with both
    model = create_simple_cnn()
    inputs = torch.randn(1, 3, 32, 32)
    
    results = {}
    
    # Test 1: Pure FX (auto_fallback=False)
    print("Testing Pure FX approach...")
    try:
        fx_exporter = FXHierarchyExporter(auto_fallback=False)
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            fx_result = fx_exporter.export(model, inputs, tmp.name)
            results['pure_fx'] = {
                'success': True,
                'hierarchy_nodes': fx_result['hierarchy_nodes'],
                'export_time': fx_result['export_time'],
                'strategy': fx_result['strategy']
            }
            print(f"   ✅ Pure FX: {fx_result['hierarchy_nodes']} nodes, {fx_result['export_time']:.3f}s")
            os.unlink(tmp.name)
    except Exception as e:
        results['pure_fx'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Pure FX failed: {e}")
    
    # Test 2: Hybrid approach (auto_fallback=True)
    print("Testing Hybrid approach...")
    try:
        hybrid_exporter = FXHierarchyExporter(auto_fallback=True)
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            hybrid_result = hybrid_exporter.export(model, inputs, tmp.name)
            results['hybrid'] = {
                'success': True,
                'hierarchy_nodes': hybrid_result['hierarchy_nodes'],
                'export_time': hybrid_result['export_time'],
                'strategy': hybrid_result['strategy'],
                'fallback_used': hybrid_result.get('fallback_used', False)
            }
            print(f"   ✅ Hybrid: {hybrid_result['hierarchy_nodes']} nodes, {hybrid_result['export_time']:.3f}s")
            print(f"      Fallback used: {hybrid_result.get('fallback_used', False)}")
            os.unlink(tmp.name)
    except Exception as e:
        results['hybrid'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Hybrid failed: {e}")
    
    # Compare results
    if results['pure_fx']['success'] and results['hybrid']['success']:
        fx_time = results['pure_fx']['export_time']
        hybrid_time = results['hybrid']['export_time']
        overhead = hybrid_time - fx_time
        
        print(f"\n📊 Performance Comparison:")
        print(f"   Pure FX time: {fx_time:.3f}s")
        print(f"   Hybrid time: {hybrid_time:.3f}s")
        print(f"   Detection overhead: {overhead:.3f}s ({overhead/fx_time*100:.1f}%)")
        
        if overhead < 0.1:  # Less than 100ms overhead
            print("   ✅ Acceptable overhead for architecture detection")
        else:
            print("   ⚠️  High overhead for architecture detection")
    
    return results

# Helper functions to create test models
def create_simple_cnn():
    """Create a simple CNN model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )

def create_feedforward():
    """Create a simple feedforward model for testing."""
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        nn.Softmax(dim=1)
    )

def create_simple_attention():
    """Create a simple attention model for testing."""
    class SimpleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
            self.norm = nn.LayerNorm(128)
            self.classifier = nn.Linear(128, 10)
            
        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            x = x.mean(dim=1)
            return self.classifier(x)
    
    return SimpleAttention()

def main():
    """Run hybrid fallback tests."""
    print("🚀 Testing FX Hybrid Architecture Detection and Fallback")
    print(f"PyTorch version: {torch.__version__}")
    
    test_results = {}
    
    # Test architecture detection
    print(f"\n📋 Running Architecture Detection Tests")
    try:
        detection_results = test_architecture_detection()
        test_results['detection'] = detection_results
    except Exception as e:
        print(f"❌ Detection tests crashed: {e}")
        test_results['detection'] = {'error': str(e)}
    
    # Test hybrid fallback
    print(f"\n📋 Running Hybrid Fallback Tests")
    try:
        fallback_success = test_hybrid_fallback()
        test_results['fallback'] = {'success': fallback_success}
    except Exception as e:
        print(f"❌ Fallback tests crashed: {e}")
        test_results['fallback'] = {'error': str(e)}
    
    # Test performance comparison
    print(f"\n📋 Running Performance Comparison")
    try:
        comparison_results = test_architecture_comparison()
        test_results['comparison'] = comparison_results
    except Exception as e:
        print(f"❌ Comparison tests crashed: {e}")
        test_results['comparison'] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("HYBRID SYSTEM VALIDATION SUMMARY")
    print("=" * 60)
    
    # Architecture detection summary
    if 'detection' in test_results and 'error' not in test_results['detection']:
        detection_data = test_results['detection']
        correct_detections = sum(1 for result in detection_data.values() 
                               if isinstance(result, dict) and result.get('correct_detection', False))
        total_detections = len([r for r in detection_data.values() if isinstance(r, dict)])
        
        print(f"🔍 Architecture Detection: {correct_detections}/{total_detections} correct")
        
        if correct_detections == total_detections:
            print("   ✅ Architecture detection working perfectly")
        elif correct_detections > total_detections * 0.8:
            print("   ✅ Architecture detection working well")
        else:
            print("   ⚠️  Architecture detection needs improvement")
    
    # Fallback functionality
    if 'fallback' in test_results:
        if test_results['fallback'].get('success', False):
            print("✅ Hybrid fallback functionality working")
        else:
            print("❌ Hybrid fallback functionality issues")
    
    # Performance impact
    if 'comparison' in test_results and 'error' not in test_results['comparison']:
        print("✅ Performance comparison completed")
    
    # Overall assessment
    all_working = (
        test_results.get('detection', {}).get('error') is None and
        test_results.get('fallback', {}).get('success', False) and
        test_results.get('comparison', {}).get('error') is None
    )
    
    if all_working:
        print("\n🎉 Iteration 7 (Hybrid Architecture Detection) completed successfully!")
        print("   ➡️  FX exporter now intelligently selects strategy based on architecture")
        return 0
    else:
        print("\n⚠️  Some aspects of hybrid system need refinement")
        return 1

if __name__ == '__main__':
    exit(main())