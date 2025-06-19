#!/usr/bin/env python3
"""
Generate reference test data for comprehensive testing.

This script creates baseline ONNX exports, expected tag mappings,
and individual module exports for testing the universal exporter.

Usage:
    python generate_test_data.py [--model bert-tiny] [--output-dir data/]
"""

import torch
import torch.onnx
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from transformers import BertModel, BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def generate_expected_tags_from_model(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Generate expected tag mappings from model structure.
    
    This creates the ground truth for what tags we expect based on
    the model's nn.Module hierarchy - completely universal approach.
    """
    expected_tags = {}
    
    # Get all modules in the model
    for name, module in model.named_modules():
        if name:  # Skip the root module (empty name)
            # Create universal tag based on module class
            module_class = f"{module.__class__.__module__}.{module.__class__.__name__}"
            tag = f"/{module_class}"
            
            expected_tags[name] = {
                "tag": tag,
                "class": module.__class__.__name__,
                "module_path": name,
                "parameters": list(module.named_parameters(recurse=False)),
                "children": list(module.named_children())
            }
    
    return {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "modules": expected_tags,
        "hierarchy_depth": max(len(name.split('.')) for name in expected_tags.keys()) if expected_tags else 0
    }


def export_baseline_onnx(model: torch.nn.Module, inputs: Dict, output_path: Path):
    """Export model using standard PyTorch ONNX export (baseline for comparison)."""
    print(f"Exporting baseline ONNX to {output_path}")
    
    # Convert inputs to tuple for torch.onnx.export
    if hasattr(inputs, 'data'):
        # Handle BatchEncoding from transformers
        tensor_inputs = {k: v for k, v in inputs.data.items() if isinstance(v, torch.Tensor)}
        input_args = tuple(tensor_inputs.values())
        input_names = list(tensor_inputs.keys())
    elif isinstance(inputs, dict):
        input_args = tuple(inputs.values())
        input_names = list(inputs.keys())
    else:
        input_args = (inputs,)
        input_names = ['input']
    
    torch.onnx.export(
        model,
        input_args,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes={
            input_names[0]: {0: 'batch_size', 1: 'sequence_length'}
        } if len(input_names) > 0 else {
            'input': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Baseline ONNX exported successfully")


def export_individual_modules(model: torch.nn.Module, model_inputs: Dict, output_dir: Path):
    """
    Export individual modules for comparison with subgraph extraction.
    
    This is universal - works with any model by finding leaf modules
    that can be exported independently.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_modules = {}
    
    # Find exportable modules (modules with parameters but no child modules with parameters)
    exportable_modules = {}
    for name, module in model.named_modules():
        if name and list(module.parameters(recurse=False)):  # Has its own parameters
            # Check if it has child modules with parameters
            has_param_children = any(
                list(child.parameters(recurse=False)) 
                for child_name, child in module.named_children()
            )
            
            if not has_param_children:  # No child modules with parameters - this is a leaf
                exportable_modules[name] = module
    
    print(f"Found {len(exportable_modules)} exportable modules")
    
    for module_name, module in exportable_modules.items():
        try:
            # Create appropriate input for this module
            # This is a simplified approach - in practice, we'd need more sophisticated input generation
            module_input = create_module_input(module, model_inputs)
            
            if module_input is not None:
                output_path = output_dir / f"{module_name.replace('.', '_')}.onnx"
                
                torch.onnx.export(
                    module,
                    module_input,
                    str(output_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
                
                exported_modules[module_name] = {
                    "file": str(output_path),
                    "class": module.__class__.__name__,
                    "parameters": len(list(module.parameters()))
                }
                print(f"✅ Exported {module_name}")
                
        except Exception as e:
            print(f"⚠️  Could not export {module_name}: {e}")
            continue
    
    return exported_modules


def create_module_input(module: torch.nn.Module, model_inputs: Dict):
    """
    Create appropriate input for individual module export.
    
    This is a heuristic approach - tries to create reasonable inputs
    based on the module's expected input shape.
    """
    # Get first parameter to infer input size
    params = list(module.parameters())
    if not params:
        return None
    
    first_param = params[0]
    
    # Heuristics for common module types
    if hasattr(module, 'in_features'):  # Linear layer
        return torch.randn(1, module.in_features)
    elif hasattr(module, 'embedding_dim'):  # Embedding layer
        # Use model inputs if available
        if 'input_ids' in model_inputs:
            return model_inputs['input_ids']
        return torch.randint(0, 100, (1, 10))
    elif len(first_param.shape) >= 2:  # Assume input matches first dimension
        input_shape = [1] + list(first_param.shape[1:])
        return torch.randn(*input_shape)
    else:
        # Default fallback
        return torch.randn(1, 128)  # Common hidden size


def generate_topology_signatures(model: torch.nn.Module, model_inputs: Dict) -> Dict[str, Any]:
    """
    Generate topology signatures for comparison testing.
    
    This captures expected operation counts and types that we can use
    to validate that our extracted subgraphs match expectations.
    """
    signatures = {}
    
    # Export to temporary ONNX to analyze structure
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        try:
            input_args = tuple(model_inputs.values()) if isinstance(model_inputs, dict) else (model_inputs,)
            torch.onnx.export(model, input_args, tmp.name, opset_version=11)
            
            # Analyze ONNX structure
            import onnx
            onnx_model = onnx.load(tmp.name)
            
            # Count operations by type
            op_counts = {}
            for node in onnx_model.graph.node:
                op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
            
            signatures = {
                "total_nodes": len(onnx_model.graph.node),
                "operation_counts": op_counts,
                "total_parameters": len(onnx_model.graph.initializer),
                "input_count": len(onnx_model.graph.input),
                "output_count": len(onnx_model.graph.output)
            }
            
        except Exception as e:
            print(f"⚠️  Could not generate topology signature: {e}")
            signatures = {"error": str(e)}
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp.name)
            except:
                pass
    
    return signatures


def generate_bert_test_data(output_dir: Path):
    """Generate all reference data for BERT testing."""
    if not HAS_TRANSFORMERS:
        print("❌ transformers library not available, skipping BERT test data")
        return
    
    print("🧪 Generating BERT test data...")
    
    # Load model and tokenizer
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    try:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"❌ Could not load BERT model: {e}")
        return
    
    # Create inputs
    text = "Hello world, this is a test for the universal model exporter."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    
    bert_dir = output_dir / "bert_tiny"
    bert_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate baseline ONNX
    baseline_path = bert_dir / "baseline.onnx"
    export_baseline_onnx(model, inputs, baseline_path)
    
    # 2. Generate expected tags from model structure
    expected_tags = generate_expected_tags_from_model(model)
    with open(bert_dir / "expected_tags.json", 'w') as f:
        json.dump(expected_tags, f, indent=2)
    print("✅ Expected tags saved")
    
    # 3. Export individual modules
    modules_dir = bert_dir / "modules"
    exported_modules = export_individual_modules(model, inputs, modules_dir)
    with open(bert_dir / "exported_modules.json", 'w') as f:
        json.dump(exported_modules, f, indent=2)
    
    # 4. Generate topology signatures
    topology = generate_topology_signatures(model, inputs)
    with open(bert_dir / "topology_signatures.json", 'w') as f:
        json.dump(topology, f, indent=2)
    print("✅ Topology signatures saved")
    
    print(f"🎉 BERT test data generated in {bert_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for modelexport")
    parser.add_argument("--model", choices=["bert-tiny"], default="bert-tiny",
                       help="Model to generate test data for")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                       help="Output directory for test data")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model == "bert-tiny":
        generate_bert_test_data(args.output_dir)
    
    print("✅ Test data generation complete!")


if __name__ == "__main__":
    main()