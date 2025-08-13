#!/usr/bin/env python3
"""
Test using existing bert.onnx with Optimum by adding config files.
This validates our production approach.
"""

import shutil
import tempfile
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np


def test_existing_onnx_with_config():
    """Test adding config files to existing ONNX for Optimum compatibility."""
    
    print("=" * 60)
    print("Testing Existing ONNX + Config Files with Optimum")
    print("=" * 60)
    
    # Check for existing ONNX
    existing_onnx = Path("models/bert.onnx")
    if not existing_onnx.exists():
        print(f"❌ ONNX model not found: {existing_onnx}")
        print("   Please run HTP export first to generate bert.onnx")
        return False
    
    size_mb = existing_onnx.stat().st_size / 1024 / 1024
    print(f"\n1. Found existing ONNX: {existing_onnx}")
    print(f"   Size: {size_mb:.2f} MB (exported with HTP)")
    
    # Create test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "bert-test"
        test_dir.mkdir()
        
        # Copy ONNX model
        test_onnx = test_dir / "model.onnx"
        shutil.copy(existing_onnx, test_onnx)
        print(f"\n2. Copied to test directory: {test_dir}")
        
        # Test WITHOUT config (should fail)
        print(f"\n3. Testing Optimum load WITHOUT config...")
        print(f"   Files: {[f.name for f in test_dir.glob('*')]}")
        
        try:
            model = ORTModelForSequenceClassification.from_pretrained(test_dir)
            print("   ❌ UNEXPECTED: Loaded without config!")
            return False
        except Exception as e:
            print(f"   ✅ EXPECTED: Failed - {type(e).__name__}")
            print(f"      {str(e)[:80]}...")
        
        # Add config files using our approach
        print(f"\n4. Adding config files from prajjwal1/bert-tiny...")
        model_id = "prajjwal1/bert-tiny"
        
        # Config (required)
        config = AutoConfig.from_pretrained(model_id)
        config.save_pretrained(test_dir)
        print(f"   ✅ Added config.json")
        
        # Tokenizer (conditional)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(test_dir)
            print(f"   ✅ Added tokenizer files")
        except Exception as e:
            print(f"   ⚠️  Tokenizer failed: {e}")
            return False
        
        # Show final directory
        files = list(test_dir.glob("*"))
        config_size = sum(f.stat().st_size for f in files if f.name != "model.onnx")
        overhead = (config_size / test_onnx.stat().st_size) * 100
        
        print(f"\n5. Final directory ({len(files)} files):")
        for f in sorted(files):
            size = f.stat().st_size
            if f.name == "model.onnx":
                print(f"   - {f.name}: {size/1024/1024:.2f} MB")
            else:
                print(f"   - {f.name}: {size/1024:.1f} KB")
        print(f"   Config overhead: {overhead:.3f}%")
        
        # Test WITH config (should work)
        print(f"\n6. Testing Optimum load WITH config...")
        try:
            ort_model = ORTModelForSequenceClassification.from_pretrained(test_dir)
            print(f"   ✅ SUCCESS: Loaded with Optimum!")
            print(f"   Model: {type(ort_model).__name__}")
            print(f"   Config: {ort_model.config.model_type}")
            
            # Test inference
            print(f"\n7. Testing inference...")
            test_text = "This is a test sentence for inference."
            inputs = tokenizer(
                test_text,
                return_tensors="np",
                padding=True,
                truncation=True
            )
            
            outputs = ort_model(**inputs)
            print(f"   ✅ Inference successful!")
            print(f"   Input shape: {inputs['input_ids'].shape}")
            print(f"   Output shape: {outputs.logits.shape}")
            
            # Get prediction
            prediction = np.argmax(outputs.logits, axis=-1)[0]
            confidence = np.max(np.softmax(outputs.logits[0])) * 100
            print(f"   Prediction: Class {prediction} ({confidence:.1f}% confidence)")
            
        except Exception as e:
            print(f"   ❌ UNEXPECTED: Failed - {e}")
            return False
    
    print(f"\n" + "=" * 60)
    print("🎉 VALIDATION SUCCESSFUL!")
    print("Our approach works with existing HTP-exported ONNX models!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_existing_onnx_with_config()
    if success:
        print("\n✅ Ready for production implementation!")
        print("\nImplementation pattern:")
        print("1. Export ONNX with HTP: export_onnx_with_hierarchy()")
        print("2. Add configs: AutoConfig.from_pretrained(model_id).save_pretrained()")
        print("3. Add tokenizer: AutoTokenizer.from_pretrained(model_id).save_pretrained()")
        print("4. Result: Optimum-compatible model directory")
    else:
        print("\n❌ Validation failed - check setup")
        exit(1)