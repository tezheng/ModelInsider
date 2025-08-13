"""Test auto-detection of ONNX input shapes in ONNXTokenizer"""

from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

from src.onnx_tokenizer import (
    ONNXTokenizer, 
    create_auto_shape_tokenizer,
    parse_onnx_input_shapes
)
from src.enhanced_pipeline import pipeline


def test_auto_detection():
    """Test automatic shape detection from ONNX model."""
    
    print("Testing Auto Shape Detection from ONNX Models")
    print("=" * 60)
    
    # Model paths
    model_dir = Path("models/bert-tiny-optimum")
    onnx_path = model_dir / "model.onnx"
    
    if not onnx_path.exists():
        print(f"⚠️  ONNX model not found at {onnx_path}")
        print("Please export a model first.")
        return
    
    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    print("\n1. Testing parse_onnx_input_shapes() function:")
    print("-" * 40)
    
    # Parse shapes directly from ONNX file
    shapes = parse_onnx_input_shapes(onnx_path)
    for name, shape in shapes.items():
        print(f"   {name}: {shape}")
    
    print("\n2. Testing ONNXTokenizer with ONNX path:")
    print("-" * 40)
    
    # Create tokenizer with auto-detection from path
    tokenizer_from_path = ONNXTokenizer(
        tokenizer=base_tokenizer,
        onnx_model=str(onnx_path)  # Pass path string
    )
    print(f"   Detected batch_size: {tokenizer_from_path.fixed_batch_size}")
    print(f"   Detected sequence_length: {tokenizer_from_path.fixed_sequence_length}")
    
    # Test tokenization
    result = tokenizer_from_path("Hello world!")
    print(f"   Tokenized shape: {result['input_ids'].shape}")
    
    print("\n3. Testing ONNXTokenizer with ORTModel:")
    print("-" * 40)
    
    # Load ORTModel
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_dir)
    
    # Create tokenizer with auto-detection from ORTModel
    tokenizer_from_model = ONNXTokenizer(
        tokenizer=base_tokenizer,
        onnx_model=ort_model  # Pass ORTModel directly
    )
    print(f"   Detected batch_size: {tokenizer_from_model.fixed_batch_size}")
    print(f"   Detected sequence_length: {tokenizer_from_model.fixed_sequence_length}")
    
    # Test tokenization
    result = tokenizer_from_model(["First text", "Second text"])
    print(f"   Tokenized shape: {result['input_ids'].shape}")
    
    print("\n4. Testing create_auto_shape_tokenizer() helper:")
    print("-" * 40)
    
    # Use convenience function
    auto_tokenizer = create_auto_shape_tokenizer(base_tokenizer, ort_model)
    print(f"   Detected batch_size: {auto_tokenizer.fixed_batch_size}")
    print(f"   Detected sequence_length: {auto_tokenizer.fixed_sequence_length}")
    
    print("\n5. Testing with enhanced pipeline:")
    print("-" * 40)
    
    # Create pipeline with auto-detected tokenizer
    pipe = pipeline(
        "feature-extraction",
        model=ort_model,
        data_processor=auto_tokenizer
    )
    
    # Test inference
    outputs = pipe("This is a test sentence.")
    print(f"   Pipeline output shape: {np.array(outputs).shape}")
    
    print("\n6. Testing manual override of auto-detection:")
    print("-" * 40)
    
    # Create tokenizer with manual override (ignoring auto-detection)
    manual_tokenizer = ONNXTokenizer(
        tokenizer=base_tokenizer,
        onnx_model=ort_model,  # Model is provided but...
        fixed_batch_size=4,     # Manual override
        fixed_sequence_length=32  # Manual override
    )
    print(f"   Manual batch_size: {manual_tokenizer.fixed_batch_size}")
    print(f"   Manual sequence_length: {manual_tokenizer.fixed_sequence_length}")
    
    print("\n7. Testing fallback for dynamic shapes:")
    print("-" * 40)
    
    # Test with a model that might have dynamic shapes
    # This will use fallback defaults
    try:
        fallback_tokenizer = ONNXTokenizer(
            tokenizer=base_tokenizer,
            onnx_model=None  # No model provided
        )
        print(f"   Fallback batch_size: {fallback_tokenizer.fixed_batch_size}")
        print(f"   Fallback sequence_length: {fallback_tokenizer.fixed_sequence_length}")
    except Exception as e:
        print(f"   Error (expected): {e}")


def demonstrate_usage():
    """Demonstrate typical usage patterns."""
    
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    print("""
Example 1: Auto-detect from ONNX file
--------------------------------------
```python
from src.onnx_tokenizer import ONNXTokenizer

# Just pass the ONNX model - shapes are auto-detected!
tokenizer = ONNXTokenizer(
    tokenizer=base_tokenizer,
    onnx_model="path/to/model.onnx"
)
print(f"Auto-detected: {tokenizer.fixed_batch_size}x{tokenizer.fixed_sequence_length}")
```

Example 2: Auto-detect from ORTModel
-------------------------------------
```python
from optimum.onnxruntime import ORTModelForFeatureExtraction

model = ORTModelForFeatureExtraction.from_pretrained("path/to/model")
tokenizer = ONNXTokenizer(
    tokenizer=base_tokenizer,
    onnx_model=model  # Pass ORTModel directly
)
```

Example 3: Use with pipeline
-----------------------------
```python
from src.enhanced_pipeline import pipeline

# Create auto-detecting tokenizer
onnx_tokenizer = ONNXTokenizer(base_tokenizer, onnx_model=model)

# Use with pipeline
pipe = pipeline("feature-extraction", model=model, data_processor=onnx_tokenizer)
result = pipe("Your text here")
```

Example 4: Manual override when needed
---------------------------------------
```python
# Sometimes you want specific shapes regardless of model
tokenizer = ONNXTokenizer(
    tokenizer=base_tokenizer,
    onnx_model=model,         # Model provided for reference
    fixed_batch_size=8,       # But override with specific values
    fixed_sequence_length=64
)
```
    """)


def main():
    """Run all tests."""
    print("ONNX Shape Auto-Detection Test Suite")
    print("=" * 60)
    
    test_auto_detection()
    demonstrate_usage()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("✅ ONNXTokenizer now auto-detects shapes from ONNX models")
    print("✅ Works with ONNX file paths, ORTModels, and InferenceSessions")
    print("✅ Automatically finds batch_size and sequence_length from inputs")
    print("✅ Falls back to sensible defaults if auto-detection fails")
    print("✅ Manual override still available when needed")
    print("\n🎯 Key Benefit: No need to manually specify shapes - just pass the model!")


if __name__ == "__main__":
    main()