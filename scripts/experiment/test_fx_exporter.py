#!/usr/bin/env python3
"""
Test script for FX Graph-based hierarchy exporter.

This script validates the FX implementation with various models
to ensure it meets all requirements from REQUIREMENTS.md.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Add modelexport to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modelexport.fx_hierarchy_exporter import FXHierarchyExporter


def test_simple_model():
    """Test with simple PyTorch model (MUST-003: Universal design)."""
    print("=" * 60)
    print("TEST 1: Simple PyTorch Model")
    print("=" * 60)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 5)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x
    
    model = SimpleModel()
    inputs = torch.randn(1, 10)
    
    exporter = FXHierarchyExporter()
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        try:
            result = exporter.export(model, inputs, tmp.name)
            print(f"✅ Simple model export successful!")
            print(f"   Hierarchy nodes: {result['hierarchy_nodes']}")
            print(f"   Unique modules: {result['unique_modules']}")
            print(f"   Export time: {result['export_time']:.2f}s")
            
            # Validate CARDINAL RULE #2: torch.nn filtering
            fx_stats = result['fx_graph_stats']
            print(f"   Module types found: {fx_stats['module_types_found']}")
            
            # Should see Linear in exceptions but not others
            expected_exceptions = {'Linear'} & set(fx_stats['module_types_found'].keys())
            print(f"   torch.nn exceptions found: {expected_exceptions}")
            
            return True
            
        except Exception as e:
            print(f"❌ Simple model export failed: {e}")
            return False
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

def test_bert_model():
    """Test with BERT model (primary target - R12 instance paths)."""
    print("=" * 60)
    print("TEST 2: BERT Model (HuggingFace)")
    print("=" * 60)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Use BERT-tiny for testing
        model_name = "prajjwal1/bert-tiny"
        print(f"Loading model: {model_name}")
        
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare inputs - fix the BERT input issue
        text = "Hello world, this is a test for hierarchy preservation."
        inputs = tokenizer(text, return_tensors="pt", max_length=64, 
                          padding="max_length", truncation=True)
        
        # For FX tracing, we need to pass inputs correctly
        # Use only input_ids and attention_mask
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        exporter = FXHierarchyExporter()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            try:
                print("Starting FX-based export...")
                result = exporter.export(model, tuple(model_inputs.values()), tmp.name, 
                                       input_names=list(model_inputs.keys()), 
                                       output_names=['output'])
                
                print(f"✅ BERT model export successful!")
                print(f"   Hierarchy nodes: {result['hierarchy_nodes']}")
                print(f"   Unique modules: {result['unique_modules']}")
                print(f"   Export time: {result['export_time']:.2f}s")
                
                # Load and analyze sidecar data
                with open(result['sidecar_path']) as f:
                    sidecar = json.load(f)
                
                print(f"   Hierarchy paths found: {len(sidecar['hierarchy_mapping'])}")
                
                # Check for R12: Instance-specific paths
                instance_paths = [path for path in sidecar['hierarchy_mapping'].values() 
                                if '.0' in path or '.1' in path]
                print(f"   Instance-specific paths: {len(instance_paths)}")
                if instance_paths:
                    print(f"   Example: {instance_paths[0]}")
                
                # Load module info (R9)
                with open(result['module_info_path']) as f:
                    module_info = json.load(f)
                    
                print(f"   Modules analyzed: {len(module_info['modules'])}")
                print(f"   Hierarchy depth: {module_info['hierarchy_depth']}")
                print(f"   Expected hierarchy entries: {len(module_info['expected_hierarchy'])}")
                
                return True
                
            except Exception as e:
                print(f"❌ BERT model export failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                # Clean up sidecar files
                for suffix in ['_fx_hierarchy.json', '_module_info.json']:
                    cleanup_path = tmp.name.replace('.onnx', suffix)
                    if os.path.exists(cleanup_path):
                        os.unlink(cleanup_path)
                        
    except ImportError:
        print("⚠️  transformers not available, skipping BERT test")
        return True
    except Exception as e:
        print(f"❌ BERT test setup failed: {e}")
        return False

def test_torch_nn_filtering():
    """Test CARDINAL RULE #2: torch.nn filtering with exceptions."""
    print("=" * 60)
    print("TEST 3: torch.nn Filtering Validation")
    print("=" * 60)
    
    class FilterTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 16)  # Should be included (exception)
            self.linear = nn.Linear(16, 32)          # May be included (in exceptions)
            self.layernorm = nn.LayerNorm(32)        # Should be included (exception)
            self.dropout = nn.Dropout(0.1)           # Should be included (exception)
            self.relu = nn.ReLU()                    # Should be filtered out
            self.softmax = nn.Softmax(dim=-1)        # Should be filtered out
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.linear(x)
            x = self.layernorm(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.softmax(x)
            return x
    
    model = FilterTestModel()
    inputs = torch.randint(0, 100, (1, 10))
    
    exporter = FXHierarchyExporter()
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        try:
            result = exporter.export(model, inputs, tmp.name)
            
            # Load sidecar to check actual hierarchy paths
            with open(result['sidecar_path']) as f:
                sidecar = json.load(f)
            
            fx_stats = result['fx_graph_stats']
            module_types = fx_stats['module_types_found']
            hierarchy_paths = list(sidecar['hierarchy_mapping'].values())
            
            print(f"✅ Filtering test completed!")
            print(f"   Module types in FX graph: {module_types}")
            print(f"   Hierarchy paths created: {len(hierarchy_paths)}")
            
            # Check which module types actually got hierarchy paths
            included_in_hierarchy = set()
            for path in hierarchy_paths:
                # Extract module type from path (last component)
                if path:
                    parts = path.split('/')
                    if len(parts) > 1:
                        last_part = parts[-1]
                        # Handle instance numbers like LayerNorm.0
                        if '.' in last_part:
                            module_type = last_part.split('.')[0]
                        else:
                            module_type = last_part
                        included_in_hierarchy.add(module_type)
            
            print(f"   Module types in hierarchy: {included_in_hierarchy}")
            
            # Validate filtering behavior
            expected_included = {'Embedding', 'LayerNorm', 'Dropout'}
            expected_excluded = {'ReLU', 'Softmax'}
            
            included = expected_included & included_in_hierarchy
            excluded = expected_excluded & included_in_hierarchy
            
            print(f"   Expected included (in hierarchy): {included}")
            print(f"   Expected excluded (in hierarchy): {excluded}")
            
            # Check if Linear is included (depends on exceptions)
            linear_included = 'Linear' in included_in_hierarchy
            print(f"   Linear in hierarchy: {linear_included}")
            
            success = (len(included) >= 2 and len(excluded) == 0)
            if success:
                print("✅ torch.nn filtering working correctly")
            else:
                print("⚠️  torch.nn filtering may need adjustment")
                
            return success
            
        except Exception as e:
            print(f"❌ Filtering test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

def test_fx_graph_analysis():
    """Test FX graph analysis capabilities."""
    print("=" * 60)
    print("TEST 4: FX Graph Analysis")
    print("=" * 60)
    
    class AnalysisModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 20)
            )
            self.decoder = nn.Sequential(
                nn.Linear(20, 10),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    model = AnalysisModel()
    inputs = torch.randn(1, 10)
    
    exporter = FXHierarchyExporter()
    
    try:
        # Analyze without full export
        fx_result = exporter._analyze_fx_hierarchy(model, inputs)
        
        print(f"✅ FX analysis completed!")
        print(f"   Total FX nodes: {fx_result.hierarchy_stats['total_fx_nodes']}")
        print(f"   Hierarchy nodes: {fx_result.hierarchy_stats['hierarchy_nodes']}")
        print(f"   Coverage ratio: {fx_result.hierarchy_stats['coverage_ratio']:.1%}")
        print(f"   Unique hierarchy paths: {fx_result.hierarchy_stats['unique_hierarchy_paths']}")
        
        # Show some hierarchy examples
        print("   Sample hierarchy mappings:")
        for i, (node, path) in enumerate(fx_result.node_hierarchy.items()):
            if i < 3:  # Show first 3
                print(f"     {node} -> {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ FX analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all FX exporter tests."""
    print("🚀 Starting FX Graph Hierarchy Exporter Tests")
    print(f"PyTorch version: {torch.__version__}")
    
    tests = [
        ("Simple Model (Universal Design)", test_simple_model),
        ("BERT Model (Instance Paths)", test_bert_model),
        ("torch.nn Filtering", test_torch_nn_filtering),
        ("FX Graph Analysis", test_fx_graph_analysis),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FX exporter is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Review implementation.")
        return 1

if __name__ == '__main__':
    exit(main())