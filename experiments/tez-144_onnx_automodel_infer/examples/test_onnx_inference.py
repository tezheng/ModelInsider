#!/usr/bin/env python3
"""
Test ONNX Inference Example
Verifies the onnx_inference_example.py script configuration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all required imports work"""
    try:
        from enhanced_pipeline import pipeline
        print("✓ enhanced_pipeline import successful")
    except ImportError as e:
        print(f"✗ enhanced_pipeline import failed: {e}")
        return False
    
    try:
        from onnx_auto_processor import ONNXAutoProcessor
        print("✓ ONNXAutoProcessor import successful")
    except ImportError as e:
        print(f"✗ ONNXAutoProcessor import failed: {e}")
        return False
    
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        print("✓ Optimum import successful")
    except ImportError as e:
        print(f"✗ Optimum import failed: {e}")
        return False
    
    return True

def test_paths():
    """Test that required paths exist"""
    model_dir = Path(__file__).parent.parent / "models" / "bert-tiny-optimum"
    
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return False
    print(f"✓ Model directory exists: {model_dir}")
    
    onnx_file = model_dir / "model.onnx"
    if not onnx_file.exists():
        print(f"✗ ONNX file not found: {onnx_file}")
        return False
    print(f"✓ ONNX file exists: {onnx_file}")
    
    # Check for config files
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print(f"✗ Config file not found: {config_file}")
        return False
    print(f"✓ Config file exists: {config_file}")
    
    return True

def test_processor_creation():
    """Test that we can create the processor"""
    from onnx_auto_processor import ONNXAutoProcessor
    
    model_dir = Path(__file__).parent.parent / "models" / "bert-tiny-optimum"
    
    try:
        processor = ONNXAutoProcessor.from_model(
            onnx_model_path=model_dir / "model.onnx",
            hf_model_path=model_dir
        )
        print("✓ ONNXAutoProcessor created successfully")
        
        # Test the processor can process text
        result = processor("Test text")
        print(f"✓ Processor can process text, output keys: {list(result.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create processor: {e}")
        return False

def test_model_loading():
    """Test that we can load the ONNX model"""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    
    model_dir = Path(__file__).parent.parent / "models" / "bert-tiny-optimum"
    
    try:
        model = ORTModelForFeatureExtraction.from_pretrained(model_dir)
        print("✓ ONNX model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

def main():
    print("=" * 60)
    print("ONNX Inference Example Verification")
    print("=" * 60)
    
    all_tests_passed = True
    
    print("\n1. Testing imports...")
    if not test_imports():
        all_tests_passed = False
    
    print("\n2. Testing paths...")
    if not test_paths():
        all_tests_passed = False
    
    print("\n3. Testing processor creation...")
    if not test_processor_creation():
        all_tests_passed = False
    
    print("\n4. Testing model loading...")
    if not test_model_loading():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - Script should work correctly!")
    else:
        print("❌ SOME TESTS FAILED - Script may have issues")
    print("=" * 60)
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)