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

from modelexport.core.base import should_tag_module, build_hierarchy_path

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def generate_expected_tags_from_model(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Generate expected tag mappings from model structure.
    
    This creates the ground truth for what tags we expect based on
    the model's nn.Module hierarchy - completely universal approach.
    
    Stores metadata only (JSON serializable), not actual PyTorch objects.
    """
    import inspect
    
    def extract_forward_signature(module):
        """Extract forward method signature metadata."""
        try:
            sig = inspect.signature(module.forward)
            return {
                "forward_args": list(sig.parameters.keys()),
                "forward_defaults": {
                    name: str(param.default) if param.default != param.empty else None 
                    for name, param in sig.parameters.items()
                }
            }
        except Exception:
            return {"forward_args": [], "forward_defaults": {}}
    
    def extract_parameters_metadata(module):
        """Extract parameter names metadata (not actual tensors)."""
        # All parameters (including from children)
        all_params = [name for name, _ in module.named_parameters()]
        # Direct parameters only
        direct_params = [name for name, _ in module.named_parameters(recurse=False)]
        return {
            "parameters": all_params,
            "direct_parameters": direct_params
        }
    
    def extract_children_metadata(module):
        """Extract children metadata (not actual module objects)."""
        return {name: child.__class__.__name__ for name, child in module.named_children()}
    
    # Extract model-level signature
    model_signature = extract_forward_signature(model)
    
    expected_tags = {}
    expected_hierarchy = {}
    
    # Create lookup dict for all modules
    all_modules = dict(model.named_modules())
    
    # Get all modules in the model
    for name, module in model.named_modules():
        if name:  # Skip the root module (empty name)
            # Create universal tag based on module class
            module_class = f"{module.__class__.__module__}.{module.__class__.__name__}"
            
            # Use the new should_tag_module function for filtering
            should_include = should_tag_module(module)
            
            # Extract all metadata (JSON serializable)
            forward_sig = extract_forward_signature(module)
            params_meta = extract_parameters_metadata(module)
            children_meta = extract_children_metadata(module)
            
            if should_include:
                # This module should be tagged - include it
                hierarchy_path = build_hierarchy_path(model, name, all_modules)
                
                expected_tags[name] = {
                    "class": module.__class__.__name__,
                    "module_path": name,
                    "module_class_full": module_class,
                    "forward_args": forward_sig["forward_args"],
                    "forward_defaults": forward_sig["forward_defaults"],
                    "parameters": params_meta["parameters"], 
                    "direct_parameters": params_meta["direct_parameters"],
                    "children": children_meta
                }
                
                # Build expected hierarchy mapping
                if hierarchy_path not in expected_hierarchy:
                    expected_hierarchy[hierarchy_path] = []
                expected_hierarchy[hierarchy_path].append(name)
    
    return {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_signature": model_signature,
        "modules": expected_tags,
        "hierarchy_depth": max(len(name.split('.')) for name in expected_tags.keys()) if expected_tags else 0,
        "expected_hierarchy": expected_hierarchy
    }


def export_baseline_onnx(model: torch.nn.Module, inputs: Dict, output_path: Path, export_config: Dict):
    """Export model using standard PyTorch ONNX export (baseline for comparison)."""
    print(f"Exporting baseline ONNX to {output_path}")
    
    # Convert inputs to tuple for torch.onnx.export
    if hasattr(inputs, 'data'):
        # Handle BatchEncoding from transformers
        tensor_inputs = {k: v for k, v in inputs.data.items() if isinstance(v, torch.Tensor)}
    elif isinstance(inputs, dict):
        tensor_inputs = inputs
    else:
        tensor_inputs = {"input": inputs}
    
    # Use config input_names order to ensure consistent tensor ordering
    if 'input_names' in export_config:
        input_names = export_config['input_names']
        # Reorder tensors to match config input_names order
        input_args = tuple(tensor_inputs[name] for name in input_names if name in tensor_inputs)
    else:
        # Fallback to original order
        input_args = tuple(tensor_inputs.values())
    
    # Filter out non-ONNX export parameters
    onnx_export_config = {k: v for k, v in export_config.items() if k != 'input_specs'}
    
    torch.onnx.export(
        model,
        input_args,
        str(output_path),
        **onnx_export_config
    )
    
    print(f"✅ Baseline ONNX exported successfully")


def export_individual_modules(model: torch.nn.Module, model_inputs: Dict, output_dir: Path, export_config: Dict):
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
                
                # Use shared config for consistency, but adapt for individual modules
                # Filter out non-ONNX parameters and adapt for single tensor input
                module_config = {
                    'export_params': export_config.get('export_params', True),
                    'opset_version': export_config.get('opset_version', 14),
                    'do_constant_folding': export_config.get('do_constant_folding', True),
                    'input_names': ['input'],
                    'output_names': ['output']
                    # No dynamic_axes for individual modules (single tensor input)
                    # No input_specs (filtered out automatically)
                }
                
                torch.onnx.export(
                    module,
                    module_input,
                    str(output_path),
                    **module_config
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


def generate_topology_signatures(model: torch.nn.Module, model_inputs: Dict, export_config: Dict) -> Dict[str, Any]:
    """
    Generate topology signatures for comparison testing.
    
    This captures expected operation counts and types that we can use
    to validate that our extracted subgraphs match expectations.
    """
    signatures = {}
    
    # Export to temporary ONNX to analyze structure
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        try:
            # Convert inputs properly - handle dict inputs universally
            if isinstance(model_inputs, dict):
                # Dictionary inputs - convert to tuple for ONNX export
                input_args = tuple(model_inputs.values())
            else:
                # Single input or other format
                input_args = (model_inputs,)
            
            # Use same opset version as config to ensure compatibility
            opset_version = export_config.get('opset_version', 14)
            torch.onnx.export(model, input_args, tmp.name, opset_version=opset_version)
            
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


def create_sample_inputs_from_config(export_config: Dict):
    """
    Create sample inputs based on the export configuration - completely universal.
    
    Uses the input_names and dynamic_axes from config to generate appropriate tensors.
    No hardcoded logic - everything driven by config.
    
    TODO: This function will be polished later to handle more sophisticated input generation.
    """
    sample_inputs = {}
    
    # Get input names from config
    input_names = export_config.get('input_names', ['input'])
    dynamic_axes = export_config.get('dynamic_axes', {})
    
    for input_name in input_names:
        # Get dynamic axes for this input (if any)
        input_dynamic_axes = dynamic_axes.get(input_name, {})
        
        # Create base shape - start with reasonable defaults
        # Use batch_size=1 and reasonable sequence/feature dimensions
        if input_dynamic_axes:
            # Use the dynamic axes to determine shape
            max_dim = max(input_dynamic_axes.keys()) if input_dynamic_axes else 1
            shape = [1] * (max_dim + 1)  # +1 because dims are 0-indexed
            
            # Set reasonable sizes for dynamic dimensions
            for dim_idx, dim_name in input_dynamic_axes.items():
                if 'sequence' in dim_name.lower() or 'length' in dim_name.lower():
                    shape[dim_idx] = 16  # Reasonable sequence length
                elif 'batch' in dim_name.lower():
                    shape[dim_idx] = 1   # Batch size 1 for testing
                else:
                    shape[dim_idx] = 32  # Default size
        else:
            # No dynamic axes specified - use simple shape
            shape = [1, 16]  # [batch, features]
        
        # Get tensor specs from config (completely config-driven)
        input_specs = export_config.get('input_specs', {})
        spec = input_specs.get(input_name, {"dtype": "float", "range": [0.0, 1.0]})
        
        # Create tensor based on config specification
        if spec.get("dtype") == "int":
            range_min, range_max = spec.get("range", [0, 100])
            sample_inputs[input_name] = torch.randint(range_min, range_max + 1, shape)
        else:
            # Default to float tensor
            sample_inputs[input_name] = torch.randn(shape)
    
    return sample_inputs


def generate_universal_test_data(output_dir: Path, export_config: Dict, model_name: str):
    """Generate all reference data for any HuggingFace model - universal approach."""
    if not HAS_TRANSFORMERS:
        print("❌ transformers library not available, skipping test data generation")
        return
    
    print(f"🧪 Generating universal test data for {model_name}...")
    
    # Load model using AutoModel (universal)
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return
    
    # Create sample inputs based on export config - completely universal
    inputs = create_sample_inputs_from_config(export_config)
    
    # Create model-specific directory based on model name (universal)
    model_dir_name = model_name.replace("/", "_").replace("-", "_")
    model_dir = output_dir / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate baseline ONNX
    baseline_path = model_dir / "baseline.onnx"
    export_baseline_onnx(model, inputs, baseline_path, export_config)
    
    # 2. Generate expected tags from model structure
    expected_tags = generate_expected_tags_from_model(model)
    with open(model_dir / "expected_tags.json", 'w') as f:
        json.dump(expected_tags, f, indent=2)
    print("✅ Expected tags saved")
    
    # 3. Export individual modules
    modules_dir = model_dir / "modules"
    exported_modules = export_individual_modules(model, inputs, modules_dir, export_config)
    with open(model_dir / "exported_modules.json", 'w') as f:
        json.dump(exported_modules, f, indent=2)
    
    # 4. Generate topology signatures
    topology = generate_topology_signatures(model, inputs, export_config)
    with open(model_dir / "topology_signatures.json", 'w') as f:
        json.dump(topology, f, indent=2)
    print("✅ Topology signatures saved")
    
    print(f"🎉 Universal test data generated in {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for modelexport")
    parser.add_argument("--model", "--model-name", dest="model_name", 
                       default="google/bert_uncased_L-2_H-128_A-2",
                       help="HuggingFace model name to use")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                       help="Output directory for test data")
    parser.add_argument("--config", type=Path, default=Path("export_config_bertmodel.json"),
                       help="Export configuration file")
    
    args = parser.parse_args()
    
    # Load export configuration
    with open(args.config, 'r') as f:
        export_config = json.load(f)
    
    # Convert dynamic_axes string keys to integers (JSON limitation workaround)
    if 'dynamic_axes' in export_config:
        fixed_dynamic_axes = {}
        for input_name, axes in export_config['dynamic_axes'].items():
            fixed_dynamic_axes[input_name] = {int(k): v for k, v in axes.items()}
        export_config['dynamic_axes'] = fixed_dynamic_axes
    
    print(f"Using export config: {args.config}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Universal handling - works with any HuggingFace model
    generate_universal_test_data(args.output_dir, export_config, args.model_name)
    
    print("✅ Test data generation complete!")


if __name__ == "__main__":
    main()