"""Test if we can pass tokenizer as processor parameter"""

from pathlib import Path
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForFeatureExtraction
from src.fixed_shape_tokenizer import FixedShapeTokenizer
import numpy as np


def test_processor_parameter():
    """Test if processor parameter works for tokenizer."""
    
    print("Testing processor parameter with tokenizer")
    print("=" * 60)
    
    # Setup
    model_dir = Path("models/bert-tiny-optimum")
    if not model_dir.exists() or not (model_dir / "model.onnx").exists():
        print("Model not found, creating dummy test...")
        # Just test with base tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        # Add config if needed
        config_path = model_dir / "config.json"
        if not config_path.exists():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
            config.save_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
            tokenizer.save_pretrained(model_dir)
        
        base_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Create fixed shape tokenizer
    fixed_tokenizer = FixedShapeTokenizer(
        tokenizer=base_tokenizer,
        fixed_batch_size=2,
        fixed_sequence_length=16
    )
    
    print("\n1. Testing tokenizer parameter (standard):")
    try:
        if model_dir.exists():
            model = ORTModelForFeatureExtraction.from_pretrained(model_dir, provider="CPUExecutionProvider")
            pipe1 = pipeline(
                "feature-extraction",
                model=model,
                tokenizer=fixed_tokenizer  # Standard approach
            )
            result1 = pipe1("Test text")
            print(f"   ✅ tokenizer parameter works: {np.array(result1).shape}")
        else:
            print("   ⚠️ Cannot test without model")
    except Exception as e:
        print(f"   ❌ Error with tokenizer parameter: {e}")
    
    print("\n2. Testing processor parameter (experimental):")
    try:
        if model_dir.exists():
            model = ORTModelForFeatureExtraction.from_pretrained(model_dir, provider="CPUExecutionProvider")
            
            # Try passing tokenizer as processor
            pipe2 = pipeline(
                "feature-extraction",
                model=model,
                processor=fixed_tokenizer  # Will this work?
            )
            result2 = pipe2("Test text")
            print(f"   ✅ processor parameter works: {np.array(result2).shape}")
        else:
            print("   ⚠️ Cannot test without model")
    except Exception as e:
        print(f"   ❌ Error with processor parameter: {e}")
    
    print("\n3. Testing both parameters (what happens?):")
    try:
        if model_dir.exists():
            model = ORTModelForFeatureExtraction.from_pretrained(model_dir, provider="CPUExecutionProvider")
            
            # Try passing both
            pipe3 = pipeline(
                "feature-extraction", 
                model=model,
                tokenizer=None,  # Explicitly None
                processor=fixed_tokenizer  # Use processor instead
            )
            
            # Check what the pipeline actually uses
            print(f"   Pipeline tokenizer: {type(pipe3.tokenizer).__name__}")
            print(f"   Pipeline processor: {type(pipe3.processor).__name__}")
            
            result3 = pipe3("Test text")
            print(f"   ✅ Result shape: {np.array(result3).shape}")
        else:
            print("   ⚠️ Cannot test without model")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Key Findings:")
    print("Based on transformers/pipelines/base.py lines 1106-1111:")
    print("If processor is provided and tokenizer is None,")
    print("the pipeline extracts tokenizer from processor.tokenizer attribute!")
    print("\nThis means:")
    print("1. You CAN pass tokenizer as processor IF it has a .tokenizer attribute")
    print("2. The pipeline will do: self.tokenizer = processor.tokenizer")
    print("3. Our FixedShapeTokenizer needs a small modification to work as processor")


def test_enhanced_fixed_tokenizer():
    """Test FixedShapeTokenizer that can work as both tokenizer and processor."""
    
    print("\n" + "=" * 60)
    print("Testing Enhanced FixedShapeTokenizer as Processor")
    print("=" * 60)
    
    # Create an enhanced version that exposes itself as .tokenizer
    class FixedShapeTokenizerAsProcessor(FixedShapeTokenizer):
        """Enhanced tokenizer that can be used as processor parameter."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Override the tokenizer attribute with a property after init
            
        def __getattr__(self, name):
            # When pipeline asks for .tokenizer, return self
            if name == "tokenizer":
                return self
            # Otherwise delegate to parent
            return super().__getattr__(name)
    
    # Test it
    base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enhanced_tokenizer = FixedShapeTokenizerAsProcessor(
        tokenizer=base_tokenizer,
        fixed_batch_size=2,
        fixed_sequence_length=16
    )
    
    print("\nEnhanced tokenizer has .tokenizer property:", hasattr(enhanced_tokenizer, "tokenizer"))
    print("enhanced_tokenizer.tokenizer is self:", enhanced_tokenizer.tokenizer is enhanced_tokenizer)
    
    model_dir = Path("models/bert-tiny-optimum")
    if model_dir.exists() and (model_dir / "model.onnx").exists():
        try:
            model = ORTModelForFeatureExtraction.from_pretrained(model_dir, provider="CPUExecutionProvider")
            
            # Now use as processor
            pipe = pipeline(
                "feature-extraction",
                model=model,
                processor=enhanced_tokenizer  # Pass as processor!
            )
            
            print(f"\nPipeline tokenizer type: {type(pipe.tokenizer).__name__}")
            print(f"Pipeline processor type: {type(pipe.processor).__name__}")
            
            result = pipe("This works as processor!")
            print(f"✅ Success! Result shape: {np.array(result).shape}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("⚠️ Model not available for testing")


if __name__ == "__main__":
    test_processor_parameter()
    test_enhanced_fixed_tokenizer()