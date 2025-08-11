"""Simple test of auto-detecting FixedShapeTokenizer"""

from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

from src.onnx_tokenizer import ONNXTokenizer
from src.enhanced_pipeline import pipeline


def main():
    """Simple test of the auto-detecting tokenizer."""
    
    print("Simple Auto-Detecting ONNXTokenizer Test")
    print("=" * 50)
    
    # Setup
    model_dir = Path("models/bert-tiny-optimum")
    if not (model_dir / "model.onnx").exists():
        print("⚠️  Model not found. Please export a model first.")
        return
    
    # Load base tokenizer and model
    base_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_dir)
    
    print("\n1. Create auto-detecting tokenizer:")
    print("-" * 30)
    
    # Just pass the ONNX model - shapes auto-detected!
    onnx_tokenizer = ONNXTokenizer(
        tokenizer=base_tokenizer,
        onnx_model=onnx_model
    )
    
    print(f"✅ Auto-detected: {onnx_tokenizer.fixed_batch_size}x{onnx_tokenizer.fixed_sequence_length}")
    
    print("\n2. Use with enhanced pipeline:")
    print("-" * 30)
    
    # Create pipeline with the auto-detecting tokenizer
    pipe = pipeline(
        "feature-extraction",
        model=onnx_model,
        data_processor=onnx_tokenizer  # Just pass it as data_processor!
    )
    
    # Test inference
    results = pipe(["Hello world!", "This is a test."])
    import numpy as np
    results_array = np.array(results)
    print(f"✅ Pipeline works! Output shape: {results_array.shape}")
    
    print("\n3. Direct tokenizer usage:")
    print("-" * 30)
    
    # Use tokenizer directly
    encoded = onnx_tokenizer("Another test sentence")
    print(f"✅ Direct tokenization shape: {encoded['input_ids'].shape}")
    
    print("\n" + "=" * 50)
    print("Summary: Super simple usage!")
    print("✅ Auto-detects shapes from ONNX model")
    print("✅ Works directly with enhanced pipeline")
    print("✅ No manual shape specification needed")
    print("\n🎯 Just create and use - it's that simple!")


if __name__ == "__main__":
    main()