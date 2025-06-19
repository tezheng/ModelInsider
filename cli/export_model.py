#!/usr/bin/env python3
"""
Command-line interface for the Universal Hierarchy Exporter.

This CLI provides an easy way to export PyTorch models to ONNX with
hierarchy-preserving tags using the universal approach.

Usage:
    python -m cli.export_model --model-class MyModel --input-shape 1,3,224,224 --output model.onnx
    python -m cli.export_model --hf-model bert-base-uncased --text "Hello world" --output bert.onnx
"""

import argparse
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modelexport import HierarchyExporter


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Universal Hierarchy-Preserving ONNX Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export HuggingFace BERT model
  python -m cli.export_model --hf-model google/bert_uncased_L-2_H-128_A-2 \\
                            --text "Hello world" --output bert_tagged.onnx

  # Export custom PyTorch model (requires manual input generation)
  python -m cli.export_model --model-path my_model.pt \\
                            --input-shape 1,3,224,224 --output model_tagged.onnx
  
  # Export with custom parameters
  python -m cli.export_model --hf-model distilbert-base-uncased \\
                            --text "Sample text for export" \\
                            --max-length 128 --output distilbert.onnx \\
                            --opset-version 14 --save-tags tags.json
        """
    )
    
    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--hf-model", 
        type=str,
        help="HuggingFace model name (e.g., 'bert-base-uncased')"
    )
    model_group.add_argument(
        "--model-path",
        type=Path,
        help="Path to saved PyTorch model (.pt/.pth file)"
    )
    model_group.add_argument(
        "--model-class",
        type=str,
        help="Python class path (e.g., 'mymodule.MyModel')"
    )
    
    # Input specification
    input_group = parser.add_argument_group("Input Configuration")
    input_group.add_argument(
        "--text",
        type=str,
        help="Input text for language models"
    )
    input_group.add_argument(
        "--input-shape",
        type=str,
        help="Input tensor shape (e.g., '1,3,224,224')"
    )
    input_group.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for text inputs (default: 128)"
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output ONNX file path"
    )
    output_group.add_argument(
        "--save-tags",
        type=Path,
        help="Save tag mapping to JSON file"
    )
    
    # ONNX export parameters
    onnx_group = parser.add_argument_group("ONNX Export Parameters")
    onnx_group.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    onnx_group.add_argument(
        "--dynamic-axes",
        type=str,
        help="Dynamic axes specification (JSON format)"
    )
    
    # Other options
    parser.add_argument(
        "--strategy",
        choices=["usage_based"],
        default="usage_based",
        help="Tagging strategy (default: usage_based)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def load_hf_model(model_name: str, verbose: bool = False):
    """Load HuggingFace model and tokenizer."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library required for HuggingFace models. "
            "Install with: pip install transformers"
        )
    
    if verbose:
        print(f"Loading HuggingFace model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        if verbose:
            print(f"✅ Loaded model: {model.__class__.__name__}")
            print(f"✅ Loaded tokenizer: {tokenizer.__class__.__name__}")
        
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace model '{model_name}': {e}")


def load_pytorch_model(model_path: Path, verbose: bool = False):
    """Load PyTorch model from file."""
    if verbose:
        print(f"Loading PyTorch model from: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = torch.load(model_path, map_location='cpu')
        if hasattr(model, 'eval'):
            model.eval()
        
        if verbose:
            print(f"✅ Loaded model: {model.__class__.__name__}")
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch model: {e}")


def create_hf_inputs(tokenizer, text: str, max_length: int, verbose: bool = False):
    """Create inputs for HuggingFace model."""
    if verbose:
        print(f"Tokenizing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    if verbose:
        print(f"✅ Input shapes: {[(k, v.shape) for k, v in inputs.items() if hasattr(v, 'shape')]}")
    
    return inputs


def create_tensor_inputs(shape_str: str, verbose: bool = False):
    """Create tensor inputs from shape string."""
    if verbose:
        print(f"Creating tensor input with shape: {shape_str}")
    
    try:
        shape = [int(x.strip()) for x in shape_str.split(',')]
        tensor = torch.randn(*shape)
        
        if verbose:
            print(f"✅ Created tensor: {tensor.shape}")
        
        return tensor
    except Exception as e:
        raise ValueError(f"Invalid shape specification '{shape_str}': {e}")


def save_tag_mapping(tag_mapping: Dict[str, Any], output_path: Path, verbose: bool = False):
    """Save tag mapping to JSON file."""
    if verbose:
        print(f"Saving tag mapping to: {output_path}")
    
    # Create a serializable version of the tag mapping
    serializable_mapping = {
        "metadata": {
            "total_operations": len(tag_mapping),
            "tagged_operations": len([op for op in tag_mapping.values() if op.get('tags', [])]),
            "unique_tags": len(set(tag for op in tag_mapping.values() for tag in op.get('tags', [])))
        },
        "operations": tag_mapping
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serializable_mapping, f, indent=2)
    
    if verbose:
        print(f"✅ Saved tag mapping with {serializable_mapping['metadata']['tagged_operations']} tagged operations")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Step 1: Load model and create inputs
        if args.hf_model:
            # HuggingFace model
            if not args.text:
                raise ValueError("--text required for HuggingFace models")
            
            model, tokenizer = load_hf_model(args.hf_model, args.verbose)
            inputs = create_hf_inputs(tokenizer, args.text, args.max_length, args.verbose)
            
        elif args.model_path:
            # Saved PyTorch model
            if not args.input_shape:
                raise ValueError("--input-shape required for saved PyTorch models")
            
            model = load_pytorch_model(args.model_path, args.verbose)
            inputs = create_tensor_inputs(args.input_shape, args.verbose)
            
        elif args.model_class:
            # Python class (advanced usage)
            raise NotImplementedError("--model-class not yet implemented. Use --model-path or --hf-model instead.")
        
        # Step 2: Set up export parameters
        export_kwargs = {
            'opset_version': args.opset_version
        }
        
        if args.dynamic_axes:
            try:
                export_kwargs['dynamic_axes'] = json.loads(args.dynamic_axes)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for --dynamic-axes: {e}")
        
        # Step 3: Export with HierarchyExporter
        if args.verbose:
            print(f"\n🚀 Starting universal hierarchy export...")
            print(f"Model: {model.__class__.__name__}")
            print(f"Strategy: {args.strategy}")
            print(f"Output: {args.output}")
        
        exporter = HierarchyExporter(strategy=args.strategy)
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=str(args.output),
            **export_kwargs
        )
        
        # Step 4: Save tag mapping if requested
        if args.save_tags:
            tag_mapping = exporter.get_tag_mapping()
            save_tag_mapping(tag_mapping, args.save_tags, args.verbose)
        
        # Step 5: Print summary
        print(f"\n✅ Export completed successfully!")
        print(f"📁 ONNX model: {args.output}")
        print(f"📊 Total operations: {result['total_operations']}")
        print(f"🏷️  Tagged operations: {result['tagged_operations']}")
        
        if args.save_tags:
            print(f"📋 Tag mapping: {args.save_tags}")
        
        if args.verbose:
            tag_mapping = exporter.get_tag_mapping()
            unique_tags = set(tag for op in tag_mapping.values() for tag in op.get('tags', []))
            print(f"\n🏷️  Unique tags found:")
            for tag in sorted(unique_tags):
                count = sum(1 for op in tag_mapping.values() if tag in op.get('tags', []))
                print(f"   {tag} ({count} operations)")
    
    except KeyboardInterrupt:
        print("\n❌ Export cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()