"""
Basic ONNX model inference example.

This script demonstrates the simplest way to load and use an ONNX model
exported by ModelExport with Optimum.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.auto_model_loader import AutoModelForONNX
from src.inference_utils import load_preprocessor


def basic_inference_example():
    """Run a basic inference example with BERT-tiny."""
    
    print("=" * 60)
    print("Basic ONNX Inference Example")
    print("=" * 60)
    
    # For this example, we'll use a model that should be exported
    # You would replace this with your actual exported model path
    model_path = Path("temp/bert-tiny-exported")
    
    # Check if model exists (for demo purposes)
    if not model_path.exists():
        print(f"\n⚠️  Model not found at {model_path}")
        print("\nTo run this example, first export a model using:")
        print("  modelexport export --model prajjwal1/bert-tiny --output temp/bert-tiny-exported/model.onnx")
        print("\nThen update the model_path in this script.")
        return
    
    print(f"\n📂 Loading model from: {model_path}")
    
    # Step 1: Load the ONNX model with AutoModel-like interface
    print("\n1️⃣ Loading ONNX model...")
    model = AutoModelForONNX.from_pretrained(model_path)
    print(f"   ✅ Model loaded successfully")
    print(f"   📋 Detected task: {model.task}")
    
    # Step 2: Load the preprocessor (tokenizer, processor, etc.)
    print("\n2️⃣ Loading preprocessor...")
    preprocessor = load_preprocessor(model_path)
    print(f"   ✅ Preprocessor loaded: {type(preprocessor).__name__}")
    
    # Step 3: Prepare inputs
    print("\n3️⃣ Preparing inputs...")
    test_texts = [
        "ONNX models are fast and efficient!",
        "I love using Optimum for inference.",
        "This is a test sentence.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"   Text {i}: {text}")
    
    # Process inputs
    inputs = preprocessor(test_texts[0], return_tensors="pt")
    print(f"\n   📊 Input shape: {inputs['input_ids'].shape}")
    
    # Step 4: Run inference
    print("\n4️⃣ Running inference...")
    outputs = model(**inputs)
    
    # Display results based on task
    if hasattr(outputs, "logits"):
        print(f"   ✅ Output shape: {outputs.logits.shape}")
        print(f"   📈 First few logits: {outputs.logits[0][:5].tolist()}")
    elif hasattr(outputs, "last_hidden_state"):
        print(f"   ✅ Output shape: {outputs.last_hidden_state.shape}")
        print(f"   📈 First hidden state mean: {outputs.last_hidden_state.mean().item():.4f}")
    else:
        print(f"   ✅ Output type: {type(outputs)}")
    
    print("\n" + "=" * 60)
    print("✅ Basic inference completed successfully!")
    print("=" * 60)


def batch_inference_example():
    """Demonstrate batch inference with multiple inputs."""
    
    print("\n" + "=" * 60)
    print("Batch Inference Example")
    print("=" * 60)
    
    model_path = Path("temp/bert-tiny-exported")
    
    if not model_path.exists():
        print(f"\n⚠️  Model not found. Please export a model first.")
        return
    
    # Load model and preprocessor
    print("\n📂 Loading model and preprocessor...")
    model = AutoModelForONNX.from_pretrained(model_path)
    tokenizer = load_preprocessor(model_path)
    
    # Prepare batch inputs
    print("\n📋 Preparing batch inputs...")
    texts = [
        "The weather is beautiful today.",
        "Machine learning is fascinating.",
        "ONNX makes models portable.",
        "Optimum provides great performance.",
    ]
    
    # Tokenize all texts at once
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    print(f"   Batch size: {len(texts)}")
    print(f"   Input shape: {inputs['input_ids'].shape}")
    
    # Run batch inference
    print("\n🚀 Running batch inference...")
    outputs = model(**inputs)
    
    if hasattr(outputs, "logits"):
        print(f"   ✅ Output shape: {outputs.logits.shape}")
        print(f"   📊 Batch processing successful!")
        
        # Show individual results
        print("\n📈 Individual results:")
        for i, text in enumerate(texts):
            if outputs.logits.shape[-1] == 2:  # Binary classification
                # Get prediction
                import torch
                probs = torch.softmax(outputs.logits[i], dim=-1)
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
                
                print(f"   {i+1}. '{text[:30]}...'")
                print(f"      Class: {pred_class}, Confidence: {confidence:.2%}")
    
    print("\n✅ Batch inference completed!")


def performance_comparison():
    """Compare inference performance with different settings."""
    
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    model_path = Path("temp/bert-tiny-exported")
    
    if not model_path.exists():
        print(f"\n⚠️  Model not found. Please export a model first.")
        return
    
    from src.inference_utils import benchmark_inference
    import time
    
    # Load model
    print("\n📂 Loading model...")
    model = AutoModelForONNX.from_pretrained(model_path)
    tokenizer = load_preprocessor(model_path)
    
    # Prepare test input
    text = "This is a test sentence for benchmarking inference speed."
    inputs = tokenizer(text, return_tensors="pt")
    
    print("\n📊 Benchmarking inference performance...")
    print(f"   Input: '{text}'")
    print(f"   Token count: {inputs['input_ids'].shape[-1]}")
    
    # Run benchmark
    metrics = benchmark_inference(
        model=model,
        inputs=inputs,
        num_runs=100,
        warmup_runs=10
    )
    
    print("\n📈 Performance Metrics:")
    print(f"   Mean latency: {metrics['mean_latency']:.2f} ms")
    print(f"   Std deviation: {metrics['std_latency']:.2f} ms")
    print(f"   Min latency: {metrics['min_latency']:.2f} ms")
    print(f"   Max latency: {metrics['max_latency']:.2f} ms")
    print(f"   Throughput: {metrics['throughput']:.1f} inferences/second")
    
    # Try different batch sizes
    print("\n📊 Testing different batch sizes...")
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        # Create batched input
        batched_inputs = {
            key: tensor.repeat(batch_size, 1) if len(tensor.shape) == 2 else tensor.repeat(batch_size)
            for key, tensor in inputs.items()
        }
        
        # Measure time for batch
        start = time.perf_counter()
        _ = model(**batched_inputs)
        end = time.perf_counter()
        
        batch_time = (end - start) * 1000
        per_sample_time = batch_time / batch_size
        
        print(f"   Batch size {batch_size:2d}: {batch_time:6.2f} ms total, {per_sample_time:5.2f} ms/sample")
    
    print("\n✅ Performance comparison completed!")


def main():
    """Run all examples."""
    print("\n🚀 ONNX AutoModel Inference Examples\n")
    
    # Run basic example
    basic_inference_example()
    
    # Run batch example
    batch_inference_example()
    
    # Run performance comparison
    performance_comparison()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()