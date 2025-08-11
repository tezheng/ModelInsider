#!/usr/bin/env python3
"""
Demonstrates that Optimum REQUIRES config.json to be present locally.
This validates our "Always Copy Configuration" approach.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np


def test_optimum_config_requirement():
    """Test that Optimum requires config.json locally."""
    
    print("=" * 60)
    print("Testing Optimum Configuration Requirements")
    print("=" * 60)
    
    # Use temp directory for clean test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        model_name = "prajjwal1/bert-tiny"
        
        # Step 1: Export ONNX model
        print(f"\n1. Exporting {model_name} to ONNX...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        # Create dummy input
        dummy_input = tokenizer(
            "Test sentence",
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True
        )
        
        # Export to ONNX
        onnx_path = temp_path / "model.onnx"
        torch.onnx.export(
            model,
            tuple(dummy_input.values()),
            onnx_path,
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['logits'],
            opset_version=17,
            do_constant_folding=True
        )
        print(f"   ✅ ONNX exported: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Step 2: Test loading WITHOUT config.json
        print("\n2. Testing Optimum load WITHOUT config.json...")
        print(f"   Files present: {[f.name for f in temp_path.glob('*')]}")
        
        try:
            ort_model = ORTModelForSequenceClassification.from_pretrained(temp_path)
            print("   ❌ UNEXPECTED: Model loaded without config!")
            return False
        except Exception as e:
            print(f"   ✅ EXPECTED: Loading failed - {type(e).__name__}")
            print(f"      Error: {str(e)[:100]}...")
        
        # Step 3: Copy config files
        print("\n3. Copying configuration files...")
        config.save_pretrained(temp_path)
        tokenizer.save_pretrained(temp_path)
        
        files = list(temp_path.glob("*"))
        print(f"   Files now: {[f.name for f in files]}")
        
        # Calculate overhead
        onnx_size = onnx_path.stat().st_size
        config_size = sum(f.stat().st_size for f in files if f.name != "model.onnx")
        overhead = (config_size / onnx_size) * 100
        print(f"   Config overhead: {config_size / 1024:.1f} KB ({overhead:.4f}% of model)")
        
        # Step 4: Test loading WITH config.json
        print("\n4. Testing Optimum load WITH config.json...")
        try:
            ort_model = ORTModelForSequenceClassification.from_pretrained(temp_path)
            print(f"   ✅ SUCCESS: Model loaded with Optimum!")
            print(f"      Model type: {type(ort_model).__name__}")
            
            # Test inference (use same padding as export)
            inputs = tokenizer(
                "Test inference", 
                return_tensors="np",
                padding="max_length",
                max_length=128,
                truncation=True
            )
            outputs = ort_model(**inputs)
            print(f"      Inference works: {outputs.logits.shape}")
            
        except Exception as e:
            print(f"   ❌ UNEXPECTED: Loading failed - {e}")
            return False
    
    print("\n" + "=" * 60)
    print("CONCLUSION: Optimum REQUIRES config.json locally!")
    print("Our 'Always Copy Configuration' approach is validated.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_optimum_config_requirement()
    sys.exit(0 if success else 1)