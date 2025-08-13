"""
Example: Complete ONNX Export with Enhanced Processor Support.

This example demonstrates how to export different types of models
(text, vision, multimodal, audio) with proper preprocessor support
for full Optimum compatibility.
"""

import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from modelexport.strategies.htp import HTPExporter
from experiments.optimum_processor_support.enhanced_config_builder import HTPConfigBuilder


def export_text_model():
    """Export a text model with tokenizer."""
    print("=" * 60)
    print("Exporting Text Model (BERT-tiny)")
    print("=" * 60)
    
    model_name = "prajjwal1/bert-tiny"
    output_dir = Path("temp/optimum-export/bert-tiny")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export ONNX model
    print("Step 1: Exporting ONNX model...")
    onnx_path = output_dir / "model.onnx"
    
    exporter = HTPExporter(verbose=False)
    result = exporter.export(
        model_name_or_path=model_name,
        output_path=str(onnx_path),
    )
    print(f"  ✅ ONNX model exported: {onnx_path}")
    print(f"  - Nodes: {result.get('onnx_nodes', 0)}")
    print(f"  - Tagged: {result.get('tagged_nodes', 0)}")
    
    # Step 2: Generate Optimum-compatible config and tokenizer
    print("\nStep 2: Generating Optimum config and tokenizer...")
    builder = HTPConfigBuilder(model_name)
    config_results = builder.generate_optimum_config(
        output_dir=output_dir,
        save_preprocessor=True,
    )
    
    for key, value in config_results.items():
        if key == "preprocessor_type":
            print(f"  - Detected preprocessor: {value}")
        else:
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
    
    # Step 3: List all files created
    print("\nStep 3: Files created for Optimum compatibility:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            size = file.stat().st_size
            print(f"  - {file.name} ({size:,} bytes)")
    
    # Step 4: Show how to load with Optimum
    print("\nStep 4: Loading with Optimum (example code):")
    print(f"""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    
    # Load the model and tokenizer
    model = ORTModelForSequenceClassification.from_pretrained("{output_dir}")
    tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
    
    # Use for inference
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs)
    """)
    
    return output_dir


def export_vision_model():
    """Export a vision model with image processor."""
    print("\n" + "=" * 60)
    print("Exporting Vision Model (ViT)")
    print("=" * 60)
    
    model_name = "google/vit-base-patch16-224"
    output_dir = Path("temp/optimum-export/vit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export ONNX model
    print("Step 1: Exporting ONNX model...")
    onnx_path = output_dir / "model.onnx"
    
    # Note: Vision models need specific input generation
    print("  ⚠️ Note: Vision model export requires image inputs")
    print("  This is a demonstration of the config generation workflow")
    
    # Step 2: Generate Optimum-compatible config and image processor
    print("\nStep 2: Generating Optimum config and image processor...")
    builder = HTPConfigBuilder(model_name)
    config_results = builder.generate_optimum_config(
        output_dir=output_dir,
        save_preprocessor=True,
    )
    
    for key, value in config_results.items():
        if key == "preprocessor_type":
            print(f"  - Detected preprocessor: {value}")
        else:
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
    
    # Step 3: Show how to load with Optimum
    print("\nStep 3: Loading with Optimum (example code):")
    print(f"""
    from optimum.onnxruntime import ORTModelForImageClassification
    from transformers import AutoImageProcessor
    from PIL import Image
    
    # Load the model and image processor
    model = ORTModelForImageClassification.from_pretrained("{output_dir}")
    processor = AutoImageProcessor.from_pretrained("{output_dir}")
    
    # Use for inference
    image = Image.open("image.jpg")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    """)
    
    return output_dir


def export_multimodal_model():
    """Export a multimodal model with processor."""
    print("\n" + "=" * 60)
    print("Exporting Multimodal Model (CLIP)")
    print("=" * 60)
    
    model_name = "openai/clip-vit-base-patch32"
    output_dir = Path("temp/optimum-export/clip")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Generating Optimum config and processor...")
    builder = HTPConfigBuilder(model_name)
    config_results = builder.generate_optimum_config(
        output_dir=output_dir,
        save_preprocessor=True,
    )
    
    for key, value in config_results.items():
        if key == "preprocessor_type":
            print(f"  - Detected preprocessor: {value}")
        else:
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
    
    # Show how to load with Optimum
    print("\nStep 2: Loading with Optimum (example code):")
    print(f"""
    from optimum.onnxruntime import ORTModel
    from transformers import AutoProcessor
    from PIL import Image
    
    # Load the model and processor
    model = ORTModel.from_pretrained("{output_dir}")
    processor = AutoProcessor.from_pretrained("{output_dir}")
    
    # Use for inference (text + image)
    image = Image.open("image.jpg")
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=image,
        return_tensors="pt",
        padding=True
    )
    outputs = model(**inputs)
    """)
    
    return output_dir


def test_optimum_loading():
    """Test actual loading with Optimum (if available)."""
    print("\n" + "=" * 60)
    print("Testing Optimum Loading (if available)")
    print("=" * 60)
    
    try:
        from optimum.onnxruntime import ORTModel
        from transformers import AutoTokenizer
        
        print("✅ Optimum is installed and available")
        
        # Try to load a previously exported model
        bert_dir = Path("temp/optimum-export/bert-tiny")
        if bert_dir.exists() and (bert_dir / "model.onnx").exists():
            print(f"\nTrying to load BERT-tiny from {bert_dir}...")
            try:
                # Note: We use ORTModel as a generic loader
                model = ORTModel.from_pretrained(bert_dir)
                tokenizer = AutoTokenizer.from_pretrained(bert_dir)
                print("  ✅ Successfully loaded model and tokenizer!")
                print(f"  - Model type: {type(model).__name__}")
                print(f"  - Tokenizer type: {type(tokenizer).__name__}")
            except Exception as e:
                print(f"  ❌ Failed to load: {e}")
        else:
            print(f"  ⚠️ No exported model found at {bert_dir}")
            print("  Run export_text_model() first to create it")
            
    except ImportError:
        print("⚠️ Optimum is not installed")
        print("  Install with: pip install optimum[onnxruntime]")


def main():
    """Run the complete example."""
    print("\n" + "🚀 Enhanced Processor Support Example 🚀".center(60))
    print()
    
    # Export text model (BERT)
    bert_dir = export_text_model()
    
    # Export vision model (ViT) - config only
    vit_dir = export_vision_model()
    
    # Export multimodal model (CLIP) - config only  
    clip_dir = export_multimodal_model()
    
    # Test Optimum loading if available
    test_optimum_loading()
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. HTPConfigBuilder intelligently detects preprocessor type")
    print("2. Text models use tokenizers")
    print("3. Vision models use image processors")
    print("4. Multimodal models use processors (combining text+image)")
    print("5. Audio models use feature extractors")
    print("\nAll these are now supported for Optimum compatibility!")


if __name__ == "__main__":
    main()