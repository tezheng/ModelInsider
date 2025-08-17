#!/usr/bin/env python3
"""
ONNX Inference Performance Monitor

Integrates ETW monitoring with ONNX model inference to provide real-time
CPU and memory profiling during model execution.

TEZ-165: ETW Integration for Windows
Author: SuperClaude
Date: 2025-08-15
"""

import os
import sys
import time
import json
import threading
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import monitoring components
try:
    from monitors.etw_monitor import ETWMonitor, ETWMonitorConfig, MetricSample
except ImportError:
    from etw_monitor import ETWMonitor, ETWMonitorConfig, MetricSample

# Import ONNX inference components
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available. Install with: pip install optimum[onnxruntime]")


@dataclass
class InferenceMetrics:
    """Metrics collected during inference"""
    inference_time_ms: float
    tokens_processed: int
    throughput_tokens_per_sec: float
    cpu_usage_percent: float
    memory_usage_mb: float
    peak_cpu_percent: float
    peak_memory_mb: float
    samples_collected: int
    effective_sampling_rate_hz: float


class ONNXInferenceMonitor:
    """
    Monitor ONNX model inference with high-frequency process metrics
    """
    
    def __init__(
        self,
        model_path: Path,
        sampling_rate_hz: float = 100.0,
        verbose: bool = False
    ):
        """
        Initialize ONNX inference monitor
        
        Args:
            model_path: Path to ONNX model directory
            sampling_rate_hz: Monitoring frequency (10-100Hz)
            verbose: Enable verbose output
        """
        self.model_path = Path(model_path)
        self.sampling_rate = sampling_rate_hz
        self.verbose = verbose
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.monitor = None
        
        # Metrics storage
        self.cpu_samples = []
        self.memory_samples = []
        self.inference_start = None
        self.inference_end = None
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self) -> None:
        """Load ONNX model and tokenizer"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        
        print(f"📦 Loading model from: {self.model_path}")
        
        # Load ONNX model
        self.model = ORTModelForFeatureExtraction.from_pretrained(self.model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        print(f"✅ Model loaded successfully")
    
    def _setup_monitor(self) -> None:
        """Setup ETW monitor for process tracking"""
        config = ETWMonitorConfig(
            sampling_rate_hz=self.sampling_rate,
            buffer_size=10000,
            target_process_names={"python", "python3", "onnxruntime"},
            include_system_processes=False,
            max_processes=10,
            cpu_sample_interval=0.01  # 10ms for accurate CPU sampling
        )
        
        self.monitor = ETWMonitor(config)
        
        # Register callback for metrics
        self.monitor.register_callback(self._on_metric)
        
        if self.verbose:
            print(f"🔧 Monitor configured at {self.sampling_rate}Hz")
    
    def _on_metric(self, sample: MetricSample) -> None:
        """Callback for metric collection"""
        # Only collect during inference
        if self.inference_start and not self.inference_end:
            if "cpu" in sample.metric_name:
                self.cpu_samples.append(sample.value)
            elif "memory_rss" in sample.metric_name:
                self.memory_samples.append(sample.value)
    
    def run_inference(
        self,
        texts: List[str],
        batch_size: int = 1,
        warmup_runs: int = 2
    ) -> Tuple[Any, InferenceMetrics]:
        """
        Run inference with monitoring
        
        Args:
            texts: Input texts for inference
            batch_size: Batch size for processing
            warmup_runs: Number of warmup runs before monitoring
        
        Returns:
            Tuple of (inference_results, metrics)
        """
        # Setup monitor
        self._setup_monitor()
        
        # Adjust batch size to match model's expected batch size (2)
        # The model was exported with batch_size=2, sequence_length=16
        model_batch_size = 2
        
        # Warmup runs
        if warmup_runs > 0:
            print(f"🔥 Running {warmup_runs} warmup iterations...")
            warmup_texts = texts[:model_batch_size] if len(texts) >= model_batch_size else texts + texts[:model_batch_size - len(texts)]
            for _ in range(warmup_runs):
                inputs = self.tokenizer(
                    warmup_texts,
                    padding='max_length',
                    max_length=16,  # Model expects sequence length of 16
                    truncation=True,
                    return_tensors="np"
                )
                _ = self.model(**inputs)
        
        # Clear metrics
        self.cpu_samples.clear()
        self.memory_samples.clear()
        
        # Start monitoring
        print(f"📊 Starting monitored inference at {self.sampling_rate}Hz...")
        self.monitor.start_monitoring()
        
        # Wait for monitoring to stabilize
        time.sleep(0.1)
        
        # Mark inference start
        self.inference_start = time.perf_counter()
        
        # Process texts in batches (model expects batch_size=2)
        all_results = []
        total_tokens = 0
        model_batch_size = 2
        
        # Process in groups of 2 (model's expected batch size)
        for i in range(0, len(texts), model_batch_size):
            batch = texts[i:i+model_batch_size]
            
            # Pad batch if needed
            if len(batch) < model_batch_size:
                batch = batch + batch[:model_batch_size - len(batch)]
            
            # Tokenize with fixed dimensions
            inputs = self.tokenizer(
                batch,
                padding='max_length',
                max_length=16,  # Model expects sequence length of 16
                truncation=True,
                return_tensors="np"
            )
            
            # Count tokens
            total_tokens += inputs["input_ids"].size
            
            # Run inference
            outputs = self.model(**inputs)
            all_results.append(outputs.last_hidden_state)
            
            if self.verbose:
                print(f"  Batch {i//batch_size + 1}: {len(batch)} texts processed")
        
        # Mark inference end
        self.inference_end = time.perf_counter()
        inference_time = (self.inference_end - self.inference_start) * 1000  # ms
        
        # Stop monitoring
        time.sleep(0.1)  # Collect final samples
        self.monitor.stop_monitoring()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            inference_time,
            total_tokens,
            len(texts)
        )
        
        # Cleanup
        self.monitor.cleanup()
        
        return all_results, metrics
    
    def _calculate_metrics(
        self,
        inference_time_ms: float,
        total_tokens: int,
        num_texts: int
    ) -> InferenceMetrics:
        """Calculate inference metrics"""
        
        # CPU metrics
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0
        
        # Memory metrics (in MB)
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        peak_memory = max(self.memory_samples) if self.memory_samples else 0
        
        # Throughput
        throughput = (total_tokens / inference_time_ms) * 1000 if inference_time_ms > 0 else 0
        
        # Effective sampling rate
        monitoring_duration = (self.inference_end - self.inference_start) if self.inference_start else 0
        effective_rate = len(self.cpu_samples) / monitoring_duration if monitoring_duration > 0 else 0
        
        return InferenceMetrics(
            inference_time_ms=inference_time_ms,
            tokens_processed=total_tokens,
            throughput_tokens_per_sec=throughput,
            cpu_usage_percent=avg_cpu,
            memory_usage_mb=avg_memory,
            peak_cpu_percent=peak_cpu,
            peak_memory_mb=peak_memory,
            samples_collected=len(self.cpu_samples),
            effective_sampling_rate_hz=effective_rate
        )
    
    def benchmark(
        self,
        test_sizes: List[int] = [1, 8, 16, 32],
        text_length: str = "short"
    ) -> Dict[str, List[InferenceMetrics]]:
        """
        Run benchmark with different batch sizes
        
        Args:
            test_sizes: List of batch sizes to test
            text_length: Text length ("short", "medium", "long")
        
        Returns:
            Dictionary of results by batch size
        """
        # Generate test texts
        test_texts = self._generate_test_texts(text_length)
        
        results = {}
        
        print(f"\n🎯 Running benchmark with {len(test_sizes)} configurations")
        print("=" * 60)
        
        for size in test_sizes:
            print(f"\n📈 Testing batch size: {size}")
            print("-" * 40)
            
            # Prepare texts
            texts = test_texts[:size]
            
            # Run inference with monitoring
            _, metrics = self.run_inference(texts, batch_size=min(size, 8))
            
            # Store results
            results[f"batch_{size}"] = metrics
            
            # Print results
            self._print_metrics(metrics)
            
            # Brief pause between tests
            time.sleep(1)
        
        return results
    
    def _generate_test_texts(self, length: str = "short") -> List[str]:
        """Generate test texts of different lengths"""
        base_texts = {
            "short": [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models are powerful tools.",
                "ONNX enables cross-platform model deployment.",
                "Performance monitoring is crucial for optimization.",
            ],
            "medium": [
                "The advancement in artificial intelligence has revolutionized how we approach complex problems in various domains.",
                "Natural language processing models have shown remarkable capabilities in understanding and generating human-like text.",
                "Edge computing brings machine learning inference closer to the data source, reducing latency and improving privacy.",
                "Optimizing model performance requires careful monitoring of resource utilization and inference metrics.",
            ],
            "long": [
                "The evolution of transformer-based architectures has fundamentally changed the landscape of natural language processing, enabling models to capture long-range dependencies and contextual relationships with unprecedented accuracy. These models have demonstrated remarkable versatility across diverse tasks.",
                "High-performance computing infrastructure plays a crucial role in training and deploying large-scale machine learning models. The ability to efficiently utilize hardware accelerators such as GPUs and NPUs has become essential for achieving optimal inference performance in production environments.",
                "Real-time monitoring and profiling of machine learning inference pipelines provides valuable insights into system behavior, resource utilization patterns, and performance bottlenecks. This information is critical for optimizing deployment strategies and ensuring efficient resource allocation.",
                "The standardization of model formats through initiatives like ONNX has greatly simplified the deployment of machine learning models across different platforms and frameworks. This interoperability enables organizations to leverage the best tools for their specific use cases.",
            ]
        }
        
        texts = base_texts.get(length, base_texts["short"])
        
        # Replicate texts to create larger datasets
        extended_texts = []
        for _ in range(100):  # Create up to 400 texts
            extended_texts.extend(texts)
        
        return extended_texts
    
    def _print_metrics(self, metrics: InferenceMetrics) -> None:
        """Print metrics in formatted output"""
        print(f"""
  ⏱️  Inference Time: {metrics.inference_time_ms:.2f}ms
  📝 Tokens Processed: {metrics.tokens_processed}
  🚀 Throughput: {metrics.throughput_tokens_per_sec:.1f} tokens/sec
  
  💻 CPU Usage:
     Average: {metrics.cpu_usage_percent:.1f}%
     Peak: {metrics.peak_cpu_percent:.1f}%
  
  🧠 Memory Usage:
     Average: {metrics.memory_usage_mb:.1f}MB
     Peak: {metrics.peak_memory_mb:.1f}MB
  
  📊 Monitoring:
     Samples: {metrics.samples_collected}
     Effective Rate: {metrics.effective_sampling_rate_hz:.1f}Hz
        """)
    
    def save_results(self, results: Dict, output_path: Path) -> None:
        """Save benchmark results to JSON"""
        output_path = Path(output_path)
        
        # Convert dataclasses to dict
        serializable_results = {}
        for key, metrics in results.items():
            serializable_results[key] = asdict(metrics)
        
        # Add metadata
        output = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "sampling_rate_hz": self.sampling_rate,
            "results": serializable_results
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ONNX Inference Performance Monitor with ETW Integration"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("../models/bert-tiny-optimum"),
        help="Path to ONNX model directory"
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=100.0,
        help="Monitoring sampling rate in Hz (10-100)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--text-length",
        choices=["short", "medium", "long"],
        default="short",
        help="Test text length"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_metrics.json"),
        help="Output file for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not args.model_dir.exists():
        print(f"❌ Model directory not found: {args.model_dir}")
        print(f"Please export a model first using:")
        print(f"  uv run modelexport export --model prajjwal1/bert-tiny --output {args.model_dir}")
        return 1
    
    # Create monitor
    monitor = ONNXInferenceMonitor(
        model_path=args.model_dir,
        sampling_rate_hz=args.sampling_rate,
        verbose=args.verbose
    )
    
    # Run benchmark
    results = monitor.benchmark(
        test_sizes=args.batch_sizes,
        text_length=args.text_length
    )
    
    # Save results
    monitor.save_results(results, args.output)
    
    print("\n✅ Benchmark complete!")
    return 0


if __name__ == "__main__":
    exit(main())