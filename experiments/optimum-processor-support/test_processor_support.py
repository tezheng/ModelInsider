"""
Test script to demonstrate enhanced processor support for multimodal models.

This script tests the HTPConfigBuilder's ability to detect and save different
types of preprocessors (processor, tokenizer, image_processor, feature_extractor).
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path to import from experiments
sys.path.append(str(Path(__file__).parent))

from enhanced_config_builder import HTPConfigBuilder


def test_processor_detection():
    """Test detection of different preprocessor types."""
    print("=" * 60)
    print("Testing Preprocessor Detection")
    print("=" * 60)
    
    test_models = [
        # Text models (use tokenizer)
        ("prajjwal1/bert-tiny", "tokenizer"),
        ("gpt2", "tokenizer"),
        ("roberta-base", "tokenizer"),
        
        # Vision models (use image_processor)
        ("google/vit-base-patch16-224", "image_processor"),
        ("microsoft/resnet-50", "image_processor"),
        
        # Multimodal models (use processor)
        ("openai/clip-vit-base-patch32", "processor"),
        ("microsoft/layoutlmv2-base-uncased", "processor"),
        
        # Audio models (use feature_extractor)
        ("facebook/wav2vec2-base", "feature_extractor"),
        ("openai/whisper-tiny", "feature_extractor"),
    ]
    
    for model_name, expected_type in test_models:
        try:
            builder = HTPConfigBuilder(model_name)
            detected_type = builder.detect_preprocessor_type()
            
            status = "✅" if detected_type == expected_type else "❌"
            print(f"{status} {model_name:40} -> {detected_type:20} (expected: {expected_type})")
            
        except Exception as e:
            print(f"❌ {model_name:40} -> Error: {e}")
    
    print()


def test_bert_tiny_export():
    """Test complete export with bert-tiny including config and tokenizer."""
    print("=" * 60)
    print("Testing BERT-tiny Export with Config/Tokenizer")
    print("=" * 60)
    
    model_name = "prajjwal1/bert-tiny"
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "bert-tiny-export"
        
        try:
            # Initialize builder
            builder = HTPConfigBuilder(model_name)
            
            # Generate complete Optimum config
            results = builder.generate_optimum_config(
                output_dir=output_dir,
                save_preprocessor=True,
                additional_config={
                    "transformers_version": "4.36.0",
                    "export_framework": "htp",
                }
            )
            
            print(f"Model: {model_name}")
            print(f"Output directory: {output_dir}")
            print(f"Results:")
            for key, value in results.items():
                if key == "preprocessor_type":
                    print(f"  - Preprocessor type: {value}")
                else:
                    status = "✅" if value else "❌"
                    print(f"  {status} {key}: {value}")
            
            # Check what files were created
            print("\nFiles created:")
            for file in sorted(output_dir.glob("*")):
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")
            
        except Exception as e:
            print(f"Error during export: {e}")
            import traceback
            traceback.print_exc()


def test_multimodal_export():
    """Test export with a multimodal model (CLIP)."""
    print("=" * 60)
    print("Testing Multimodal Model Export (CLIP)")
    print("=" * 60)
    
    model_name = "openai/clip-vit-base-patch32"
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "clip-export"
        
        try:
            # Initialize builder
            builder = HTPConfigBuilder(model_name)
            
            # Generate complete Optimum config
            results = builder.generate_optimum_config(
                output_dir=output_dir,
                save_preprocessor=True,
            )
            
            print(f"Model: {model_name}")
            print(f"Output directory: {output_dir}")
            print(f"Results:")
            for key, value in results.items():
                if key == "preprocessor_type":
                    print(f"  - Preprocessor type: {value}")
                else:
                    status = "✅" if value else "❌"
                    print(f"  {status} {key}: {value}")
            
            # Check what files were created
            print("\nFiles created:")
            for file in sorted(output_dir.glob("*")):
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")
            
        except Exception as e:
            print(f"Error during export: {e}")
            import traceback
            traceback.print_exc()


def test_vision_model_export():
    """Test export with a vision model (ViT)."""
    print("=" * 60)
    print("Testing Vision Model Export (ViT)")
    print("=" * 60)
    
    model_name = "google/vit-base-patch16-224"
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "vit-export"
        
        try:
            # Initialize builder
            builder = HTPConfigBuilder(model_name)
            
            # Generate complete Optimum config
            results = builder.generate_optimum_config(
                output_dir=output_dir,
                save_preprocessor=True,
            )
            
            print(f"Model: {model_name}")
            print(f"Output directory: {output_dir}")
            print(f"Results:")
            for key, value in results.items():
                if key == "preprocessor_type":
                    print(f"  - Preprocessor type: {value}")
                else:
                    status = "✅" if value else "❌"
                    print(f"  {status} {key}: {value}")
            
            # Check what files were created
            print("\nFiles created:")
            for file in sorted(output_dir.glob("*")):
                size = file.stat().st_size
                print(f"  - {file.name} ({size:,} bytes)")
            
        except Exception as e:
            print(f"Error during export: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "🧪 Testing Enhanced Processor Support 🧪".center(60))
    print()
    
    # Test preprocessor detection
    test_processor_detection()
    
    # Test BERT-tiny (text model with tokenizer)
    test_bert_tiny_export()
    print()
    
    # Test CLIP (multimodal model with processor)
    test_multimodal_export()
    print()
    
    # Test ViT (vision model with image processor)
    test_vision_model_export()
    print()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()