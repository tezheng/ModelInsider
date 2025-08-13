#!/usr/bin/env python3
"""
Performance Benchmark Example

Demonstrates performance comparison between PyTorch and ONNX inference
with the enhanced pipeline system.
"""

import time
from pathlib import Path
from typing import Dict, List

import click
import numpy as np

from modelexport.inference.pipeline import pipeline, create_pipeline
from modelexport.inference.onnx_auto_processor import ONNXAutoProcessor
from modelexport.inference.processors.text import ONNXTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification


def benchmark_text_inference(
    texts: List[str], 
    model_path: Path = None,
    iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark text inference comparing PyTorch vs ONNX.
    
    Args:
        texts: List of test texts
        model_path: Path to ONNX model directory (if available)
        iterations: Number of benchmark iterations
        
    Returns:
        Dictionary with performance metrics
    """
    results = {}
    
    # Use a lightweight model for demonstration
    model_name = "prajjwal1/bert-tiny"
    
    print(f"\nBenchmarking with {len(texts)} texts, {iterations} iterations")
    print(f"Model: {model_name}")
    print("-" * 60)
    
    # Benchmark 1: Standard PyTorch/Transformers Pipeline
    print("🐌 Testing PyTorch baseline...")
    try:
        hf_pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1  # CPU
        )
        
        start_time = time.time()
        for _ in range(iterations):
            _ = hf_pipe(texts)
        pytorch_time = (time.time() - start_time) / iterations
        
        results["pytorch_time"] = pytorch_time
        print(f"  ✓ PyTorch time: {pytorch_time:.4f}s per batch")
        
    except Exception as e:
        print(f"  ✗ PyTorch benchmark failed: {e}")
        results["pytorch_time"] = None
    
    # Benchmark 2: ONNX with Enhanced Pipeline (if model available)
    if model_path and model_path.exists():
        print("🚀 Testing ONNX enhanced pipeline...")
        try:
            # Load ONNX model
            onnx_model = ORTModelForSequenceClassification.from_pretrained(model_path)
            
            # Create processor
            processor = ONNXAutoProcessor.from_model(
                onnx_model_path=model_path / "model.onnx",
                hf_model_path=model_path
            )
            
            # Create pipeline
            onnx_pipe = pipeline(
                "text-classification",
                model=onnx_model,
                data_processor=processor
            )
            
            start_time = time.time()
            for _ in range(iterations):
                _ = onnx_pipe(texts)
            onnx_time = (time.time() - start_time) / iterations
            
            results["onnx_time"] = onnx_time
            print(f"  ✓ ONNX time: {onnx_time:.4f}s per batch")
            
        except Exception as e:
            print(f"  ✗ ONNX benchmark failed: {e}")
            results["onnx_time"] = None
    
    # Benchmark 3: Direct ONNXTokenizer Performance
    print("⚡ Testing ONNXTokenizer performance...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        onnx_tokenizer = ONNXTokenizer(
            tokenizer=tokenizer,
            batch_size=len(texts),
            sequence_length=128
        )
        
        start_time = time.time()
        for _ in range(iterations):
            _ = onnx_tokenizer(texts)
        tokenizer_time = (time.time() - start_time) / iterations
        
        results["tokenizer_time"] = tokenizer_time
        print(f"  ✓ Tokenizer time: {tokenizer_time:.4f}s per batch")
        
    except Exception as e:
        print(f"  ✗ Tokenizer benchmark failed: {e}")
        results["tokenizer_time"] = None
    
    return results


def print_performance_summary(results: Dict[str, float], batch_size: int):
    """Print benchmark results summary."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if results.get("pytorch_time") and results.get("onnx_time"):
        speedup = results["pytorch_time"] / results["onnx_time"]
        print(f"🚀 ONNX Speedup: {speedup:.1f}x faster than PyTorch!")
    
    print(f"\nPer-batch timings ({batch_size} samples):")
    for name, time_val in results.items():
        if time_val:
            throughput = batch_size / time_val
            print(f"  {name:15}: {time_val:.4f}s ({throughput:.1f} samples/sec)")
    
    print(f"\nOptimization benefits:")
    print(f"  • Fixed-shape processing eliminates dynamic overhead")
    print(f"  • ONNX runtime optimization for inference")
    print(f"  • Batch processing amortizes tokenization costs")
    print(f"  • Memory layout optimization")
    print("=" * 60)


@click.command()
@click.option("--model-path", type=click.Path(path_type=Path), 
              help="Path to ONNX model directory")
@click.option("--batch-size", default=8, help="Batch size for testing")
@click.option("--iterations", default=10, help="Number of benchmark iterations")
@click.option("--custom-text", help="Custom text for testing")
def main(model_path: Path, batch_size: int, iterations: int, custom_text: str):
    """
    Run performance benchmarks comparing PyTorch and ONNX inference.
    
    Examples:
        python performance_benchmark.py --batch-size 16
        python performance_benchmark.py --model-path ../models/bert-tiny-onnx
        python performance_benchmark.py --custom-text "Your custom text here"
    """
    print("🔥 ONNX Performance Benchmark")
    print("Testing enhanced pipeline performance vs PyTorch baseline")
    
    # Create test data
    if custom_text:
        texts = [custom_text] * batch_size
    else:
        sample_texts = [
            "This is a great example of fast ONNX inference!",
            "The performance improvement is really impressive.",
            "ONNX optimization makes a significant difference.",
            "Batch processing works efficiently with fixed shapes.",
            "The enhanced pipeline provides excellent compatibility.",
        ]
        texts = (sample_texts * ((batch_size // len(sample_texts)) + 1))[:batch_size]
    
    # Run benchmarks
    results = benchmark_text_inference(texts, model_path, iterations)
    
    # Print summary
    print_performance_summary(results, batch_size)
    
    # Tips for optimization
    print("\n💡 Performance Tips:")
    print("  1. Use larger batch sizes for better throughput")
    print("  2. Fixed sequence lengths optimize memory usage")
    print("  3. ONNX models eliminate PyTorch overhead")
    print("  4. Warm up the pipeline with a few test runs")
    
    if not model_path or not model_path.exists():
        print("\n📝 Note: Provide --model-path for full ONNX comparison")
        print("     Export a model with: modelexport export model_name output.onnx")


if __name__ == "__main__":
    main()