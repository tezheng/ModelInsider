#!/usr/bin/env python3
"""
Test script for metadata discovery utilities.

This script demonstrates discovering and aggregating metadata from HuggingFace models.
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.tez_153_onnx_infer_2.metadata_utils import (
    MetadataManager,
    MetadataDiscovery,  # Renamed from MetadataExtractor
    ONNXMetadataReader
)


def test_metadata_discovery(model_name: str = "prajjwal1/bert-tiny"):
    """Test metadata discovery from a HuggingFace model."""
    print(f"\n{'='*60}")
    print(f"Testing metadata discovery for: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        import torch
        
        # Load model and processor
        print("Loading model and processor...")
        model = AutoModel.from_pretrained(model_name)
        
        # Try to load appropriate processor
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            print(f"Loaded AutoProcessor")
        except:
            try:
                processor = AutoTokenizer.from_pretrained(model_name)
                print(f"Loaded AutoTokenizer")
            except:
                print("No processor found")
        
        # Create metadata discovery tool
        discovery = MetadataDiscovery()
        
        # Test different embedding levels
        for level in ["minimal", "essential", "full"]:
            print(f"\n{'-'*40}")
            print(f"Testing {level} embedding level:")
            print(f"{'-'*40}")
            
            metadata = discovery.discover_from_hf(model, processor, level)
            
            # Print summary
            print(f"\nExtracted metadata keys ({len(metadata)} total):")
            for key in sorted(metadata.keys()):
                value = metadata[key]
                if isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} fields")
                elif isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                else:
                    print(f"  {key}: {value}")
            
            # Calculate size
            metadata_json = json.dumps(metadata, separators=(',', ':'))
            size_kb = len(metadata_json) / 1024
            print(f"\nMetadata size: {size_kb:.2f} KB")
            
            # Show task detection
            if 'hf_pipeline_task' in metadata:
                print(f"Detected task: {metadata['hf_pipeline_task']}")
            
            # Show feature engineering type
            if 'feature_engineering.type' in metadata:
                print(f"Feature engineering type: {metadata['feature_engineering.type']}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onnx_embedding(model_name: str = "prajjwal1/bert-tiny"):
    """Test embedding metadata into an ONNX model."""
    print(f"\n{'='*60}")
    print(f"Testing ONNX metadata embedding for: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        import onnx
        
        # Create temp directory
        temp_dir = Path("temp/metadata_test")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Export to ONNX (simple export for testing)
        onnx_path = temp_dir / "model.onnx"
        print(f"Exporting to ONNX: {onnx_path}")
        
        # Create dummy input
        dummy_text = "Hello world"
        inputs = tokenizer(dummy_text, return_tensors="pt")
        
        # Export
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            str(onnx_path),
            input_names=list(inputs.keys()),
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "output": {0: "batch", 1: "sequence"}
            },
            opset_version=17
        )
        
        print(f"ONNX model exported to: {onnx_path}")
        
        # Get original file size
        original_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
        print(f"Original ONNX size: {original_size:.2f} MB")
        
        # Test metadata embedding
        manager = MetadataManager()
        
        for level in ["minimal", "essential", "full"]:
            output_path = temp_dir / f"model_{level}.onnx"
            
            print(f"\n{'-'*40}")
            print(f"Embedding {level} metadata:")
            print(f"{'-'*40}")
            
            # Copy original
            import shutil
            shutil.copy(onnx_path, output_path)
            
            # Process model
            metadata = manager.process_model(
                model=model,
                processor=tokenizer,
                onnx_path=output_path,
                embed_level=level,
                compress=True
            )
            
            # Check new size
            new_size = output_path.stat().st_size / (1024 * 1024)  # MB
            overhead = (new_size - original_size) * 1024  # KB
            print(f"New ONNX size: {new_size:.2f} MB")
            print(f"Metadata overhead: {overhead:.2f} KB")
            
            # Read back metadata
            reader = ONNXMetadataReader()
            read_metadata = reader.read(output_path)
            print(f"Successfully read {len(read_metadata)} metadata entries")
            
            # Verify task detection
            if 'hf_pipeline_task' in read_metadata:
                print(f"Task type: {read_metadata['hf_pipeline_task']}")
        
        print(f"\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering_metadata():
    """Test feature engineering metadata for different modalities."""
    print(f"\n{'='*60}")
    print(f"Testing feature engineering metadata")
    print(f"{'='*60}\n")
    
    test_models = [
        ("prajjwal1/bert-tiny", "text"),
        ("google/vit-base-patch16-224", "vision"),
        # ("openai/whisper-tiny", "audio"),  # Optional
    ]
    
    for model_name, modality in test_models:
        print(f"\n{'-'*40}")
        print(f"Testing {modality} model: {model_name}")
        print(f"{'-'*40}")
        
        try:
            from transformers import AutoProcessor, AutoModel
            
            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Extract metadata
            extractor = MetadataExtractor()
            metadata = extractor.extract_from_hf(model, processor, "full")
            
            # Show feature engineering info
            if 'feature_engineering.type' in metadata:
                fe_type = metadata['feature_engineering.type']
                print(f"Feature engineering type: {fe_type}")
                
                if 'feature_engineering.config' in metadata:
                    config = metadata['feature_engineering.config']
                    if isinstance(config, dict):
                        print(f"Config fields ({len(config)}):")
                        for key in list(config.keys())[:5]:  # Show first 5
                            print(f"  - {key}")
                        if len(config) > 5:
                            print(f"  ... and {len(config) - 5} more")
            
            # Show ONNX image metadata if present
            image_keys = [k for k in metadata.keys() if k.startswith('Image.')]
            if image_keys:
                print(f"ONNX Image metadata:")
                for key in image_keys:
                    print(f"  {key}: {metadata[key]}")
            
        except Exception as e:
            print(f"Skipping {model_name}: {e}")


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test metadata extraction utilities")
    parser.add_argument("--model", default="prajjwal1/bert-tiny", help="Model to test")
    parser.add_argument("--test", choices=["extraction", "embedding", "feature", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test in ["extraction", "all"]:
        test_metadata_extraction(args.model)
    
    if args.test in ["embedding", "all"]:
        test_onnx_embedding(args.model)
    
    if args.test in ["feature", "all"]:
        test_feature_engineering_metadata()


if __name__ == "__main__":
    main()