#!/usr/bin/env python
"""
Test script for UniversalOnnxConfig implementation.

This script tests the universal config generation with various model types.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from onnx_config import UniversalOnnxConfig
from transformers import AutoConfig
import torch


def test_model_config(model_name: str):
    """Test UniversalOnnxConfig with a specific model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    try:
        # Load config
        config = AutoConfig.from_pretrained(model_name)
        print(f"✓ Config loaded: {config.model_type}")
        
        # Create UniversalOnnxConfig
        onnx_config = UniversalOnnxConfig(config)
        print(f"✓ Task detected: {onnx_config.task}")
        print(f"✓ Task family: {onnx_config.task_family}")
        
        # Get specifications
        input_names = onnx_config.get_input_names()
        output_names = onnx_config.get_output_names()
        
        print(f"\n📥 Inputs ({len(input_names)}):")
        for name in input_names[:5]:  # Show first 5
            print(f"  - {name}")
        if len(input_names) > 5:
            print(f"  ... and {len(input_names) - 5} more")
        
        print(f"\n📤 Outputs ({len(output_names)}):")
        for name in output_names:
            print(f"  - {name}")
        
        # Dynamic axes
        dynamic_axes = onnx_config.get_dynamic_axes()
        print(f"\n🔄 Dynamic axes:")
        for name, axes in list(dynamic_axes.items())[:3]:
            print(f"  {name}: {axes}")
        
        # Generate dummy inputs
        print(f"\n🎲 Generating dummy inputs...")
        dummy_inputs = onnx_config.generate_dummy_inputs(
            batch_size=2,
            seq_length=64
        )
        
        print(f"✓ Generated {len(dummy_inputs)} inputs:")
        for name, tensor in list(dummy_inputs.items())[:3]:
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        print(f"\n✅ {model_name} - SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} - FAILED: {e}")
        return False


def test_task_detection():
    """Test task detection for various architectures."""
    print("\n" + "="*60)
    print("Testing Task Detection")
    print("="*60)
    
    test_cases = [
        ("bert-base-uncased", "feature-extraction"),
        ("distilbert-base-uncased-finetuned-sst-2-english", "text-classification"),
        ("gpt2", "text-generation"),
        ("t5-small", "text2text-generation"),
        ("facebook/bart-base", "text2text-generation"),
        ("google/vit-base-patch16-224", "image-classification"),
        ("openai/whisper-tiny", "automatic-speech-recognition"),
    ]
    
    for model_name, expected_task_family in test_cases:
        try:
            config = AutoConfig.from_pretrained(model_name)
            onnx_config = UniversalOnnxConfig(config)
            
            # Check if task family matches expectation
            success = expected_task_family in onnx_config.task or \
                     onnx_config.task == expected_task_family
            
            status = "✓" if success else "✗"
            print(f"{status} {model_name:40} -> {onnx_config.task:25} (expected: {expected_task_family})")
            
        except Exception as e:
            print(f"✗ {model_name:40} -> ERROR: {e}")


def test_export_to_onnx():
    """Test actual ONNX export using UniversalOnnxConfig."""
    print("\n" + "="*60)
    print("Testing ONNX Export")
    print("="*60)
    
    model_name = "prajjwal1/bert-tiny"
    
    try:
        from transformers import AutoModel
        import tempfile
        import onnx
        
        print(f"Loading model: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Create UniversalOnnxConfig
        print("Creating UniversalOnnxConfig...")
        onnx_config = UniversalOnnxConfig(config)
        
        # Generate dummy inputs
        print("Generating dummy inputs...")
        dummy_inputs = onnx_config.generate_dummy_inputs(
            batch_size=1,
            seq_length=128
        )
        
        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            print(f"Exporting to ONNX: {tmp.name}")
            
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                tmp.name,
                input_names=onnx_config.get_input_names(),
                output_names=onnx_config.get_output_names(),
                dynamic_axes=onnx_config.get_dynamic_axes(),
                opset_version=onnx_config.DEFAULT_ONNX_OPSET,
                do_constant_folding=True,
            )
            
            # Verify the export
            print("Verifying ONNX model...")
            onnx_model = onnx.load(tmp.name)
            onnx.checker.check_model(onnx_model)
            
            # Get file size
            import os
            file_size = os.path.getsize(tmp.name) / (1024 * 1024)
            print(f"✅ Export successful! File size: {file_size:.2f} MB")
            
            # Clean up
            os.unlink(tmp.name)
            
    except Exception as e:
        print(f"❌ Export failed: {e}")


def main():
    """Run all tests."""
    print("\n🚀 UniversalOnnxConfig Test Suite\n")
    
    # Test task detection
    test_task_detection()
    
    # Test various model types
    print("\n" + "="*60)
    print("Testing Various Model Types")
    print("="*60)
    
    test_models = [
        "prajjwal1/bert-tiny",  # BERT encoder
        "gpt2",  # GPT decoder
        "t5-small",  # T5 encoder-decoder
        # "google/vit-base-patch16-224",  # Vision (may need to skip if large)
        # "openai/whisper-tiny",  # Audio (may need to skip if large)
    ]
    
    results = {}
    for model in test_models:
        results[model] = test_model_config(model)
    
    # Test actual export
    test_export_to_onnx()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total}")
    if passed < total:
        print("\nFailed models:")
        for model, success in results.items():
            if not success:
                print(f"  - {model}")
    
    print("\n✨ UniversalOnnxConfig test suite completed!")


if __name__ == "__main__":
    main()