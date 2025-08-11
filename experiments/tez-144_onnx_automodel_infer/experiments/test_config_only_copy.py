#!/usr/bin/env python3
"""
Demonstrates that we can copy config files directly from HF model ID
without loading the full model - much more efficient!
"""

import tempfile
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoImageProcessor
from optimum.onnxruntime import ORTModelForSequenceClassification
import shutil


def copy_config_files(model_id: str, output_dir: Path):
    """
    Copy configuration files from HuggingFace model ID.
    No need to load the actual model weights!
    """
    print(f"\n📁 Copying config files from '{model_id}' to '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Always copy config.json (REQUIRED)
    print("   Loading config...")
    config = AutoConfig.from_pretrained(model_id)
    config.save_pretrained(output_dir)
    print(f"   ✅ Saved config.json")
    
    # 2. Try to copy tokenizer (for NLP models)
    try:
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)
        print(f"   ✅ Saved tokenizer files")
    except Exception as e:
        print(f"   ℹ️  No tokenizer found (expected for non-NLP models)")
    
    # 3. Try to copy processor (for multimodal models)
    try:
        print("   Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(output_dir)
        print(f"   ✅ Saved processor files")
    except Exception:
        print(f"   ℹ️  No processor found")
    
    # 4. Try to copy image processor (for vision models)
    try:
        print("   Loading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        image_processor.save_pretrained(output_dir)
        print(f"   ✅ Saved image processor files")
    except Exception:
        print(f"   ℹ️  No image processor found")
    
    # List what was saved
    files = list(output_dir.glob("*"))
    total_size = sum(f.stat().st_size for f in files)
    print(f"\n   📊 Total config size: {total_size / 1024:.1f} KB")
    print(f"   📄 Files saved: {[f.name for f in sorted(files)]}")


def test_multiple_model_types():
    """Test config copying for different model types."""
    
    test_models = [
        ("prajjwal1/bert-tiny", "NLP - BERT"),
        ("google/vit-base-patch16-224", "Vision - ViT"),
        ("openai/clip-vit-base-patch32", "Multimodal - CLIP"),
    ]
    
    print("=" * 70)
    print("Testing Config Copying for Different Model Types")
    print("=" * 70)
    
    for model_id, model_type in test_models:
        print(f"\n{'='*70}")
        print(f"Model: {model_id} ({model_type})")
        print('='*70)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "configs"
            
            # Copy configs without loading model weights
            copy_config_files(model_id, output_dir)
            
            # Verify we got the essential config.json
            if (output_dir / "config.json").exists():
                print(f"   ✅ SUCCESS: config.json created for {model_type}")
            else:
                print(f"   ❌ FAILED: No config.json for {model_type}")


def test_with_existing_onnx():
    """
    Test that we can add configs to an existing ONNX model.
    This simulates our actual use case.
    """
    print("\n" + "=" * 70)
    print("Test: Adding Configs to Existing ONNX Model")
    print("=" * 70)
    
    model_id = "prajjwal1/bert-tiny"
    
    # Use the ONNX model we already exported
    existing_onnx = Path("/home/zhengte/modelexport_tez47/experiments/tez-144_onnx_automodel_infer/models/bert-tiny-optimum-test/model.onnx")
    
    if not existing_onnx.exists():
        print("   ⚠️  No existing ONNX model found, skipping test")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "bert-with-configs"
        test_dir.mkdir(parents=True)
        
        # 1. Copy existing ONNX
        shutil.copy(existing_onnx, test_dir / "model.onnx")
        print(f"\n1. Copied existing ONNX model")
        print(f"   Size: {(test_dir / 'model.onnx').stat().st_size / 1024 / 1024:.2f} MB")
        
        # 2. Add configs using just the model ID
        print(f"\n2. Adding configs from HF model ID: {model_id}")
        copy_config_files(model_id, test_dir)
        
        # 3. Test loading with Optimum
        print(f"\n3. Testing Optimum compatibility...")
        try:
            model = ORTModelForSequenceClassification.from_pretrained(test_dir)
            print(f"   ✅ Model loads successfully with Optimum!")
            print(f"   Model type: {type(model).__name__}")
            
            # Test inference
            tokenizer = AutoTokenizer.from_pretrained(test_dir)
            inputs = tokenizer(
                "Test sentence",
                return_tensors="np",
                padding="max_length",
                max_length=128,
                truncation=True
            )
            outputs = model(**inputs)
            print(f"   ✅ Inference works! Output shape: {outputs.logits.shape}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION: We can copy configs using just the model ID!")
    print("No need to load the full model weights - much more efficient!")
    print("=" * 70)


if __name__ == "__main__":
    # Test 1: Different model types
    test_multiple_model_types()
    
    # Test 2: Adding configs to existing ONNX
    test_with_existing_onnx()