"""
Complete example showing export and inference using UniversalOnnxConfig + AutoModelForONNX.

This demonstrates the full workflow from any HuggingFace model to ONNX inference.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import our implementations
from onnx_config import UniversalOnnxConfig
from auto_model_loader import AutoModelForONNX
from inference_utils import load_preprocessor


def export_model_with_universal_config(
    model_name: str,
    output_path: str,
    task: str = None
) -> Dict[str, Any]:
    """
    Export any HuggingFace model to ONNX using UniversalOnnxConfig.
    
    Args:
        model_name: HuggingFace model identifier
        output_path: Path to save ONNX model
        task: Optional task override
        
    Returns:
        Export information dictionary
    """
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        import torch
        import onnx
        
        print(f"🚀 Exporting {model_name} to ONNX...")
        
        # Step 1: Load model and config
        print("1️⃣ Loading model and config...")
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Step 2: Create UniversalOnnxConfig
        print("2️⃣ Creating UniversalOnnxConfig...")
        onnx_config = UniversalOnnxConfig(config, task=task)
        
        print(f"   ✓ Detected task: {onnx_config.task}")
        print(f"   ✓ Task family: {onnx_config.task_family}")
        print(f"   ✓ Input names: {onnx_config.get_input_names()}")
        print(f"   ✓ Output names: {onnx_config.get_output_names()}")
        
        # Step 3: Load tokenizer/preprocessor for better dummy inputs
        print("3️⃣ Loading preprocessor...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"   ✓ Loaded tokenizer: {type(tokenizer).__name__}")
        except:
            tokenizer = None
            print("   ⚠️ Could not load tokenizer, using generated dummy inputs")
        
        # Step 4: Generate dummy inputs
        print("4️⃣ Generating dummy inputs...")
        dummy_inputs = onnx_config.generate_dummy_inputs(
            preprocessor=tokenizer,
            batch_size=1,
            seq_length=128
        )
        
        print(f"   ✓ Generated {len(dummy_inputs)} inputs:")
        for name, tensor in dummy_inputs.items():
            print(f"     {name}: {tensor.shape} ({tensor.dtype})")
        
        # Step 5: Export to ONNX
        print("5️⃣ Exporting to ONNX...")
        
        # Create output directory
        os.makedirs(Path(output_path).parent, exist_ok=True)
        
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            output_path,
            input_names=onnx_config.get_input_names(),
            output_names=onnx_config.get_output_names(),
            dynamic_axes=onnx_config.get_dynamic_axes(),
            opset_version=onnx_config.DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        # Step 6: Validate export
        print("6️⃣ Validating ONNX export...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ Export successful! File size: {file_size:.2f} MB")
        
        # Step 7: Save preprocessor if available
        if tokenizer:
            tokenizer_path = Path(output_path).parent
            tokenizer.save_pretrained(tokenizer_path)
            print(f"   ✓ Saved tokenizer to {tokenizer_path}")
        
        return {
            "model_name": model_name,
            "output_path": output_path,
            "task": onnx_config.task,
            "task_family": onnx_config.task_family,
            "input_names": onnx_config.get_input_names(),
            "output_names": onnx_config.get_output_names(),
            "file_size_mb": file_size,
            "opset_version": onnx_config.DEFAULT_ONNX_OPSET
        }
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return {"error": str(e)}


def test_inference_with_automodel(model_path: str, test_texts: list = None):
    """
    Test inference using AutoModelForONNX.
    
    Args:
        model_path: Path to exported ONNX model directory
        test_texts: Optional test texts for inference
    """
    try:
        print(f"\n🔍 Testing inference with AutoModelForONNX...")
        print(f"Model path: {model_path}")
        
        # Step 1: Load ONNX model
        print("1️⃣ Loading ONNX model...")
        model = AutoModelForONNX.from_pretrained(model_path)
        print(f"   ✓ Model loaded: {type(model).__name__}")
        print(f"   ✓ Detected task: {model.task}")
        
        # Step 2: Load preprocessor
        print("2️⃣ Loading preprocessor...")
        preprocessor = load_preprocessor(model_path)
        print(f"   ✓ Loaded: {type(preprocessor).__name__}")
        
        # Step 3: Prepare test data
        if test_texts is None:
            test_texts = [
                "ONNX models are efficient for inference.",
                "AutoModelForONNX makes loading simple.",
                "Universal configuration works with any model!"
            ]
        
        print(f"3️⃣ Running inference on {len(test_texts)} texts...")
        
        # Step 4: Run inference
        for i, text in enumerate(test_texts, 1):
            print(f"\n   Test {i}: '{text[:50]}...'")
            
            # Tokenize
            inputs = preprocessor(text, return_tensors="pt", padding=True, truncation=True)
            
            # Inference
            outputs = model(**inputs)
            
            # Display results
            if hasattr(outputs, "logits"):
                print(f"     ✓ Logits shape: {outputs.logits.shape}")
                print(f"     📊 First logit values: {outputs.logits[0][:3].tolist()}")
            elif hasattr(outputs, "last_hidden_state"):
                print(f"     ✓ Hidden state shape: {outputs.last_hidden_state.shape}")
                print(f"     📊 Hidden state mean: {outputs.last_hidden_state.mean().item():.4f}")
            else:
                print(f"     ✓ Output type: {type(outputs)}")
        
        print(f"\n   ✅ Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n   ❌ Inference failed: {e}")
        return False


def complete_workflow_example():
    """
    Demonstrate the complete workflow: export + inference.
    """
    print("=" * 80)
    print("🎯 Complete Workflow: Universal Export + AutoModel Inference")
    print("=" * 80)
    
    # Configuration
    model_name = "prajjwal1/bert-tiny"  # Small model for demo
    
    # Create temporary directory for export
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "exported_model"
        onnx_path = model_dir / "model.onnx"
        
        # Step 1: Export model
        print(f"\n📤 PHASE 1: Export {model_name}")
        print("-" * 40)
        
        export_info = export_model_with_universal_config(
            model_name=model_name,
            output_path=str(onnx_path)
        )
        
        if "error" in export_info:
            print(f"Export failed, cannot continue: {export_info['error']}")
            return
        
        # Step 2: Test inference
        print(f"\n📥 PHASE 2: Test Inference")
        print("-" * 40)
        
        success = test_inference_with_automodel(
            model_path=str(model_dir),
            test_texts=[
                "This is a positive sentence.",
                "I love using ONNX models.",
                "Universal configs are amazing!"
            ]
        )
        
        # Step 3: Summary
        print(f"\n📋 PHASE 3: Summary")
        print("-" * 40)
        
        print(f"Model: {export_info['model_name']}")
        print(f"Task: {export_info['task']} ({export_info['task_family']})")
        print(f"File size: {export_info['file_size_mb']:.2f} MB")
        print(f"ONNX opset: {export_info['opset_version']}")
        print(f"Inputs: {', '.join(export_info['input_names'])}")
        print(f"Outputs: {', '.join(export_info['output_names'])}")
        
        if success:
            print(f"\n🎉 Complete workflow successful!")
            print("✅ Universal export works")
            print("✅ AutoModel inference works")
            print("✅ Integration successful")
        else:
            print(f"\n⚠️ Workflow partially successful")
            print("✅ Export works")
            print("❌ Inference had issues")


def batch_export_examples():
    """
    Export multiple models to show universal coverage.
    """
    print("\n" + "=" * 80)
    print("📦 Batch Export Examples")
    print("=" * 80)
    
    # Test models (small ones to avoid large downloads)
    test_models = [
        ("prajjwal1/bert-tiny", "feature-extraction"),
        ("distilgpt2", "text-generation"),
        # Add more when dependencies are available
    ]
    
    results = []
    
    for model_name, expected_task in test_models:
        print(f"\n🔄 Exporting {model_name}...")
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                info = export_model_with_universal_config(
                    model_name=model_name,
                    output_path=tmp.name
                )
                
                if "error" not in info:
                    results.append({
                        "model": model_name,
                        "task": info["task"],
                        "expected": expected_task,
                        "success": True,
                        "size_mb": info["file_size_mb"]
                    })
                    print(f"   ✅ Success: {info['task']}")
                else:
                    results.append({
                        "model": model_name,
                        "error": info["error"],
                        "success": False
                    })
                    print(f"   ❌ Failed: {info['error']}")
                
                # Clean up
                os.unlink(tmp.name)
                
        except Exception as e:
            results.append({
                "model": model_name,
                "error": str(e),
                "success": False
            })
            print(f"   ❌ Exception: {e}")
    
    # Summary
    print(f"\n📊 Batch Export Summary:")
    print("-" * 40)
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for result in results:
        if result["success"]:
            print(f"✅ {result['model']:30} -> {result['task']:20} ({result['size_mb']:.1f}MB)")
        else:
            print(f"❌ {result['model']:30} -> {result['error'][:30]}...")


def main():
    """Run all examples."""
    print("🚀 UniversalOnnxConfig + AutoModelForONNX Integration Examples\n")
    
    # Check if dependencies are available
    try:
        import torch
        import transformers
        import onnx
        has_deps = True
        print("✅ All dependencies available")
    except ImportError as e:
        has_deps = False
        print(f"⚠️ Missing dependencies: {e}")
        print("📝 This example requires: torch, transformers, onnx")
        print("📋 Showing conceptual workflow instead...\n")
    
    if has_deps:
        # Full workflow
        complete_workflow_example()
        
        # Batch examples
        batch_export_examples()
    else:
        # Show the concept without execution
        print("🎯 Conceptual Workflow:")
        print("1. Load any HuggingFace model")
        print("2. Create UniversalOnnxConfig (auto-detects everything)")
        print("3. Export to ONNX with proper configuration")
        print("4. Load with AutoModelForONNX for inference")
        print("5. Run inference with original API compatibility")
        
        print(f"\n💡 Key Innovation:")
        print("- Universal: Works with ANY model")
        print("- Automatic: Zero manual configuration")
        print("- Compatible: Drop-in replacement for Optimum")
        print("- Future-proof: New models work automatically")
    
    print(f"\n" + "=" * 80)
    print("✨ Integration Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()