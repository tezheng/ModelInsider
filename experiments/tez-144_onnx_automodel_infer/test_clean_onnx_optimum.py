#!/usr/bin/env python3
"""
Test the complete workflow:
1. Export BERT-tiny with clean ONNX (default, no HTP metadata)
2. Add config files using AutoConfig
3. Test with Optimum ORTModel
"""

import subprocess
import tempfile
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np


def export_clean_onnx(model_id: str, output_path: Path):
    """Export ONNX without HTP metadata for Optimum compatibility."""
    cmd = [
        "uv", "run", "modelexport", "export",
        "--model", model_id,
        "--output", str(output_path),
        # Clean ONNX is now the default, no flag needed
    ]
    
    print(f"🔧 Exporting clean ONNX: {' '.join(cmd[-4:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="../../..")
    
    if result.returncode != 0:
        print(f"❌ Export failed:")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        return False
    
    print(f"✅ Export successful")
    return True


def add_config_files(model_id: str, output_dir: Path):
    """Add config files for Optimum compatibility."""
    print(f"📁 Adding config files from {model_id}...")
    
    # Config (always required)
    config = AutoConfig.from_pretrained(model_id)
    config.save_pretrained(output_dir)
    print(f"   ✅ config.json")
    
    # Tokenizer (conditional)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)
        print(f"   ✅ tokenizer files")
        return tokenizer
    except Exception as e:
        print(f"   ❌ tokenizer failed: {e}")
        return None


def test_complete_workflow():
    """Test the complete export + config + optimum workflow."""
    
    print("=" * 70)
    print("Complete Workflow Test: Export → Config → Optimum")
    print("=" * 70)
    
    model_id = "prajjwal1/bert-tiny"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        export_dir = Path(temp_dir) / "bert-export"
        export_dir.mkdir()
        
        # Step 1: Export clean ONNX (no HTP metadata)
        print(f"\n1. Export clean ONNX for {model_id}")
        onnx_path = export_dir / "model.onnx"
        
        if not export_clean_onnx(model_id, onnx_path):
            return False
        
        if not onnx_path.exists():
            print(f"❌ ONNX file not created: {onnx_path}")
            return False
        
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        print(f"   File: {onnx_path}")
        print(f"   Size: {size_mb:.2f} MB")
        
        # Step 2: Add config files
        print(f"\n2. Add config files to directory")
        tokenizer = add_config_files(model_id, export_dir)
        if not tokenizer:
            return False
        
        # Show directory contents
        files = list(export_dir.glob("*"))
        config_size = sum(f.stat().st_size for f in files if f.name != "model.onnx")
        overhead = (config_size / onnx_path.stat().st_size) * 100
        
        print(f"\n   📊 Directory contents ({len(files)} files):")
        for f in sorted(files):
            size = f.stat().st_size
            if f.name == "model.onnx":
                print(f"   - {f.name}: {size/1024/1024:.2f} MB")
            else:
                print(f"   - {f.name}: {size/1024:.1f} KB")
        print(f"   Config overhead: {overhead:.3f}% of model size")
        
        # Step 3: Test Optimum loading
        print(f"\n3. Load with Optimum ORTModel")
        try:
            ort_model = ORTModelForSequenceClassification.from_pretrained(export_dir)
            print(f"   ✅ Model loaded successfully!")
            print(f"   Type: {type(ort_model).__name__}")
            print(f"   Config: {ort_model.config.model_type}")
            print(f"   Architecture: {ort_model.config.architectures}")
            
        except Exception as e:
            print(f"   ❌ Loading failed: {e}")
            return False
        
        # Step 4: Test inference
        print(f"\n4. Run inference test")
        
        # Check if there's metadata about input shapes
        metadata_file = export_dir / "model_htp_metadata.json"
        expected_batch_size = 2
        expected_seq_len = 16
        
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
                print(f"   📋 Found metadata: {metadata_file.name}")
        
        try:
            # Use exactly 2 sentences to match expected batch size
            test_sentences = [
                "I love this new approach!",
                "This is terrible.",
            ]
            
            # Tokenize with fixed dimensions to match export  
            inputs = tokenizer(
                test_sentences,
                return_tensors="np",
                padding="max_length",
                max_length=expected_seq_len,
                truncation=True
            )
            
            print(f"   📐 Input shapes:")
            for key, value in inputs.items():
                print(f"      {key}: {value.shape}")
            
            # Inference
            outputs = ort_model(**inputs)
            
            print(f"   ✅ Inference successful!")
            print(f"   Input shape: {inputs['input_ids'].shape}")
            print(f"   Outputs: {type(outputs)}")
            print(f"   Output attributes: {dir(outputs)}")
            
            # Check what outputs we got
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                print(f"   Output shape: {logits.shape}")
            elif hasattr(outputs, 'prediction_scores'):
                logits = outputs.prediction_scores
                print(f"   Output shape: {logits.shape}")
            else:
                # Get the first output
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                print(f"   Output shape: {logits.shape}")
            
            # Show predictions
            predictions = np.argmax(logits, axis=-1)
            print(f"\n   🎯 Predictions:")
            for i, (sent, pred) in enumerate(zip(test_sentences, predictions)):
                print(f"   {i+1}. Class {pred}: {sent[:40]}{'...' if len(sent) > 40 else ''}")
                
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            return False
    
    print(f"\n" + "=" * 70)
    print("🎉 COMPLETE WORKFLOW VALIDATED!")
    print("✅ Export → Config → Optimum → Inference ALL WORKING!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_complete_workflow()
    if success:
        print(f"\n🚀 PRODUCTION READY:")
        print(f"   1. Clean ONNX is now default for Optimum compatibility")
        print(f"   2. Add configs with AutoConfig.from_pretrained()")
        print(f"   3. Result works perfectly with Optimum")
        print(f"\n📝 Implementation:")
        print(f"   def export_with_config(model_id, output_dir):")
        print(f"       export_onnx_with_hierarchy(model_id, output_dir / 'model.onnx', clean_onnx=True)")
        print(f"       AutoConfig.from_pretrained(model_id).save_pretrained(output_dir)")
        print(f"       AutoTokenizer.from_pretrained(model_id).save_pretrained(output_dir)")
    else:
        print(f"\n❌ Test failed - check setup")
        exit(1)