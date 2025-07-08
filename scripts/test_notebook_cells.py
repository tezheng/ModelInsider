#!/usr/bin/env python3
"""
Test script to verify the notebook cells work correctly.
"""

import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Suppress warnings for clarity
warnings.filterwarnings("ignore", category=UserWarning)

def test_setup():
    """Test setup and model creation."""
    
    # Set up paths
    output_dir = Path("temp/onnx_structure_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX version: {onnx.__version__}")
    print(f"Output directory: {output_dir.absolute()}")
    
    return output_dir

def test_model_creation():
    """Test complex model creation."""
    
    class ComplexModel(nn.Module):
        """A more complex model to better demonstrate structural differences."""
        
        def __init__(self, input_dim=10, hidden_dim=20, num_classes=5):
            super().__init__()
            
            # Feature extraction layers
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Attention mechanism
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
        def forward(self, x):
            # Feature extraction
            features = self.feature_extractor(x)
            
            # Self-attention (reshape for attention layer)
            features_seq = features.unsqueeze(1)  # Add sequence dimension
            attn_out, _ = self.attention(features_seq, features_seq, features_seq)
            attn_out = attn_out.squeeze(1)  # Remove sequence dimension
            
            # Classification
            output = self.classifier(attn_out)
            
            return output
    
    # Create model instance
    model = ComplexModel()
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(2, 10)  # batch_size=2, input_dim=10
    
    print("Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, sample_input

def test_export_with_options(output_dir, model, sample_input):
    """Test export with different options."""
    
    def export_model_with_options(model, sample_input, export_modules_as_functions, suffix):
        """Export model with specified options and return path."""
        
        output_path = output_dir / f"model_{suffix}.onnx"
        
        print(f"\nExporting with export_modules_as_functions={export_modules_as_functions}...")
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                export_modules_as_functions=export_modules_as_functions,
                verbose=False
            )
        
        print(f"✓ Exported to: {output_path.name}")
        
        # Verify model loads correctly
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("✓ Model validation passed")
        except Exception as e:
            print(f"✗ Model validation failed: {e}")
        
        return output_path
    
    # Export with different options
    paths = {
        'standard': export_model_with_options(model, sample_input, False, "standard"),
        'all_functions': export_model_with_options(model, sample_input, True, "all_functions"),
    }
    
    return paths

def test_analysis(paths):
    """Test analysis functions."""
    
    def analyze_onnx_structure(onnx_path: Path) -> Dict[str, Any]:
        """Perform comprehensive analysis of ONNX model structure."""
        
        model = onnx.load(str(onnx_path))
        graph = model.graph
        
        analysis = {
            'file_name': onnx_path.name,
            'file_size_mb': onnx_path.stat().st_size / (1024 * 1024),
            'graph': {
                'inputs': len(graph.input),
                'outputs': len(graph.output),
                'nodes': len(graph.node),
                'initializers': len(graph.initializer),
                'value_info': len(graph.value_info)
            },
            'functions': {
                'count': len(model.functions) if hasattr(model, 'functions') else 0,
                'details': []
            },
            'node_types': defaultdict(int),
            'attributes': defaultdict(list),
            'tensor_shapes': {},
            'parameter_count': 0
        }
        
        # Analyze main graph nodes
        for node in graph.node:
            analysis['node_types'][node.op_type] += 1
            
            # Collect attributes
            for attr in node.attribute:
                analysis['attributes'][attr.name].append({
                    'node': node.name or node.op_type,
                    'type': attr.type
                })
        
        # Analyze functions if present
        if hasattr(model, 'functions') and model.functions:
            for func in model.functions:
                func_info = {
                    'name': func.name,
                    'domain': func.domain,
                    'inputs': len(func.input),
                    'outputs': len(func.output),
                    'nodes': len(func.node),
                    'node_types': defaultdict(int),
                    'attributes': len(func.attribute)
                }
                
                # Count node types in function
                for node in func.node:
                    func_info['node_types'][node.op_type] += 1
                
                func_info['node_types'] = dict(func_info['node_types'])
                analysis['functions']['details'].append(func_info)
        
        # Count parameters
        for init in graph.initializer:
            shape = [dim for dim in init.dims]
            analysis['parameter_count'] += np.prod(shape) if shape else 1
            analysis['tensor_shapes'][init.name] = shape
        
        # Convert defaultdicts to regular dicts
        analysis['node_types'] = dict(analysis['node_types'])
        analysis['attributes'] = dict(analysis['attributes'])
        
        return analysis
    
    # Analyze all exported models
    analyses = {}
    for name, path in paths.items():
        print(f"\nAnalyzing {name} export...")
        analyses[name] = analyze_onnx_structure(path)
        print(f"✓ Analysis complete")
    
    return analyses

def test_display_summary(analyses):
    """Test display summary function."""
    
    print("\n" + "="*80)
    print("ONNX STRUCTURE ANALYSIS SUMMARY")
    print("="*80)
    
    for name, analysis in analyses.items():
        print(f"\n{name.upper()} EXPORT:")
        print("-" * 40)
        
        # Basic info
        print(f"File: {analysis['file_name']}")
        print(f"Size: {analysis['file_size_mb']:.2f} MB")
        
        # Graph structure
        graph = analysis['graph']
        print(f"\nMain Graph:")
        print(f"  Inputs: {graph['inputs']}")
        print(f"  Outputs: {graph['outputs']}")
        print(f"  Nodes: {graph['nodes']}")
        print(f"  Initializers: {graph['initializers']}")
        print(f"  Parameters: {analysis['parameter_count']:,}")
        
        # Node types
        print(f"\nNode Types in Main Graph:")
        for op_type, count in sorted(analysis['node_types'].items()):
            print(f"  {op_type}: {count}")
        
        # Functions
        if analysis['functions']['count'] > 0:
            print(f"\nLocal Functions: {analysis['functions']['count']}")
            for func in analysis['functions']['details'][:5]:  # Show first 5
                print(f"  - {func['name']} (domain: {func['domain']})")
                print(f"    Nodes: {func['nodes']}, I/O: {func['inputs']}/{func['outputs']}")
                print(f"    Operations: {dict(func['node_types'])}")

def test_visualizations(analyses):
    """Test visualization creation."""
    
    # Create comparison visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ONNX Export Structure Comparison', fontsize=16)
    
    # 1. Graph complexity comparison
    ax = axes[0, 0]
    metrics = ['Nodes', 'Functions', 'Initializers']
    standard_values = [
        analyses['standard']['graph']['nodes'],
        analyses['standard']['functions']['count'],
        analyses['standard']['graph']['initializers']
    ]
    functions_values = [
        analyses['all_functions']['graph']['nodes'],
        analyses['all_functions']['functions']['count'],
        analyses['all_functions']['graph']['initializers']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, standard_values, width, label='Standard Export', alpha=0.8)
    ax.bar(x + width/2, functions_values, width, label='Functions Export', alpha=0.8)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Count')
    ax.set_title('Graph Complexity Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Node type distribution - Standard
    ax = axes[0, 1]
    if analyses['standard']['node_types']:
        node_types = list(analyses['standard']['node_types'].keys())
        node_counts = list(analyses['standard']['node_types'].values())
        ax.pie(node_counts, labels=node_types, autopct='%1.1f%%', startangle=90)
        ax.set_title('Node Types - Standard Export')
    else:
        ax.text(0.5, 0.5, 'No nodes in main graph', ha='center', va='center')
        ax.set_title('Node Types - Standard Export')
    
    # 3. Node type distribution - Functions
    ax = axes[1, 0]
    if analyses['all_functions']['node_types']:
        node_types = list(analyses['all_functions']['node_types'].keys())
        node_counts = list(analyses['all_functions']['node_types'].values())
        ax.pie(node_counts, labels=node_types, autopct='%1.1f%%', startangle=90)
        ax.set_title('Node Types - Functions Export (Main Graph)')
    else:
        ax.text(0.5, 0.5, 'Most operations\nin functions', ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Node Types - Functions Export (Main Graph)')
    
    # 4. File size comparison
    ax = axes[1, 1]
    export_types = ['Standard', 'Functions']
    file_sizes = [
        analyses['standard']['file_size_mb'],
        analyses['all_functions']['file_size_mb']
    ]
    bars = ax.bar(export_types, file_sizes, alpha=0.8, color=['blue', 'orange'])
    ax.set_ylabel('File Size (MB)')
    ax.set_title('File Size Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, size in zip(bars, file_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.3f} MB', ha='center', va='bottom')
    
    plt.tight_layout()
    # Save instead of show for testing
    plt.savefig('temp/onnx_structure_analysis/comparison_charts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Visualizations created and saved")

def test_metadata_analysis(paths):
    """Test metadata analysis."""
    
    def examine_metadata_and_attributes(onnx_path: Path) -> Dict[str, Any]:
        """Examine metadata and attributes in ONNX model."""
        
        model = onnx.load(str(onnx_path))
        
        metadata = {
            'model_metadata': {},
            'graph_metadata': {},
            'node_attributes': defaultdict(list),
            'function_attributes': defaultdict(list)
        }
        
        # Model-level metadata
        if hasattr(model, 'metadata_props'):
            for prop in model.metadata_props:
                metadata['model_metadata'][prop.key] = prop.value
        
        # Graph-level metadata
        if hasattr(model.graph, 'doc_string'):
            metadata['graph_metadata']['doc_string'] = model.graph.doc_string
        
        # Node attributes
        for node in model.graph.node:
            if node.attribute:
                node_info = {
                    'node_name': node.name or f"{node.op_type}_unnamed",
                    'op_type': node.op_type,
                    'attributes': {}
                }
                
                for attr in node.attribute:
                    # Extract attribute value based on type
                    if attr.type == onnx.AttributeProto.FLOAT:
                        value = attr.f
                    elif attr.type == onnx.AttributeProto.INT:
                        value = attr.i
                    elif attr.type == onnx.AttributeProto.STRING:
                        value = attr.s.decode('utf-8') if attr.s else ''
                    elif attr.type == onnx.AttributeProto.INTS:
                        value = list(attr.ints)
                    elif attr.type == onnx.AttributeProto.STRINGS:
                        value = [s.decode('utf-8') for s in attr.strings]
                    else:
                        value = f"Type: {attr.type}"
                    
                    node_info['attributes'][attr.name] = value
                
                metadata['node_attributes'][node.op_type].append(node_info)
        
        # Function attributes
        if hasattr(model, 'functions'):
            for func in model.functions:
                func_info = {
                    'name': func.name,
                    'domain': func.domain,
                    'attributes': []
                }
                
                if hasattr(func, 'attribute'):
                    for attr in func.attribute:
                        func_info['attributes'].append(attr.name)
                
                metadata['function_attributes'][func.domain].append(func_info)
        
        # Convert defaultdicts
        metadata['node_attributes'] = dict(metadata['node_attributes'])
        metadata['function_attributes'] = dict(metadata['function_attributes'])
        
        return metadata
    
    # Examine metadata for both exports
    metadata_analysis = {}
    for name, path in paths.items():
        print(f"\nExamining metadata for {name} export...")
        metadata_analysis[name] = examine_metadata_and_attributes(path)
    
    print("✓ Metadata analysis complete")
    return metadata_analysis

def main():
    """Run all tests."""
    
    print("Testing notebook cells...")
    
    try:
        # Test setup
        output_dir = test_setup()
        print("✓ Setup test passed")
        
        # Test model creation
        model, sample_input = test_model_creation()
        print("✓ Model creation test passed")
        
        # Test export
        paths = test_export_with_options(output_dir, model, sample_input)
        print("✓ Export test passed")
        
        # Test analysis
        analyses = test_analysis(paths)
        print("✓ Analysis test passed")
        
        # Test display
        test_display_summary(analyses)
        print("✓ Display test passed")
        
        # Test visualizations
        test_visualizations(analyses)
        print("✓ Visualization test passed")
        
        # Test metadata
        metadata_analysis = test_metadata_analysis(paths)
        print("✓ Metadata test passed")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("The notebook should run without errors.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)