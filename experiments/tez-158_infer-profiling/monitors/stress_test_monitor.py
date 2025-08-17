#!/usr/bin/env python3
"""
Stress Test Monitor for ONNX Inference

Runs continuous ONNX inference while monitoring CPU and memory usage
to demonstrate the ETW monitoring capabilities at high sampling rates.

TEZ-165: ETW Integration for Windows
Author: SuperClaude
Date: 2025-08-15
"""

import os
import sys
import time
import threading
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import monitoring
try:
    from monitors.etw_monitor import ETWMonitor, ETWMonitorConfig
except ImportError:
    from etw_monitor import ETWMonitor, ETWMonitorConfig

# Import ONNX components
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available")


class StressTestMonitor:
    """
    Stress test ONNX inference with continuous monitoring
    """
    
    def __init__(
        self,
        model_path: Path,
        sampling_rate_hz: float = 100.0,
        duration_seconds: float = 10.0
    ):
        """
        Initialize stress test monitor
        
        Args:
            model_path: Path to ONNX model
            sampling_rate_hz: Monitoring frequency
            duration_seconds: Test duration
        """
        self.model_path = Path(model_path)
        self.sampling_rate = sampling_rate_hz
        self.duration = duration_seconds
        
        # Components
        self.model = None
        self.tokenizer = None
        self.monitor = None
        
        # Metrics
        self.inference_count = 0
        self.total_tokens = 0
        self.running = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and tokenizer"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        
        print(f"📦 Loading model: {self.model_path}")
        self.model = ORTModelForFeatureExtraction.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print("✅ Model loaded")
    
    def _inference_worker(self):
        """Worker thread for continuous inference"""
        # Test texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning enables powerful applications."
        ]
        
        # Tokenize once
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            max_length=16,
            truncation=True,
            return_tensors="np"
        )
        
        print("🚀 Starting continuous inference...")
        
        # Run continuous inference
        while self.running:
            try:
                # Run inference
                outputs = self.model(**inputs)
                
                # Update counters
                self.inference_count += 1
                self.total_tokens += inputs["input_ids"].size
                
                # Small delay to prevent CPU saturation
                time.sleep(0.001)  # 1ms between inferences
                
            except Exception as e:
                print(f"❌ Inference error: {e}")
                break
    
    def run_stress_test(self):
        """Run stress test with monitoring"""
        print(f"\n{'='*60}")
        print(f"Stress Test: {self.duration}s at {self.sampling_rate}Hz")
        print(f"{'='*60}\n")
        
        # Setup monitor
        config = ETWMonitorConfig(
            sampling_rate_hz=self.sampling_rate,
            buffer_size=int(self.sampling_rate * self.duration * 1.5),
            target_process_names={"python", "python3"},
            include_system_processes=False,
            max_processes=5,
            cpu_sample_interval=0.01
        )
        
        self.monitor = ETWMonitor(config)
        
        # Start monitoring
        print(f"📊 Starting monitor at {self.sampling_rate}Hz...")
        self.monitor.start_monitoring()
        
        # Start inference worker
        self.running = True
        inference_thread = threading.Thread(target=self._inference_worker)
        inference_thread.start()
        
        # Monitor for specified duration
        start_time = time.time()
        last_print = start_time
        
        while time.time() - start_time < self.duration:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Print status every second
            if current_time - last_print >= 1.0:
                samples = len(self.monitor.samples)
                rate = samples / elapsed if elapsed > 0 else 0
                
                print(f"  [{elapsed:.1f}s] Inferences: {self.inference_count}, "
                      f"Samples: {samples}, Rate: {rate:.1f}Hz")
                last_print = current_time
            
            time.sleep(0.1)
        
        # Stop inference
        print("\n⏹️ Stopping inference...")
        self.running = False
        inference_thread.join()
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Get results
        stats = self.monitor.get_statistics()
        summary = self.monitor.get_process_summary()
        
        # Print results
        print(f"\n{'='*60}")
        print("📈 Test Results")
        print(f"{'='*60}\n")
        
        print(f"🔬 Inference Statistics:")
        print(f"  Total inferences: {self.inference_count}")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Inference rate: {self.inference_count/self.duration:.1f} inferences/sec")
        print(f"  Token throughput: {self.total_tokens/self.duration:.1f} tokens/sec")
        
        print(f"\n📊 Monitoring Statistics:")
        print(f"  Samples collected: {stats.get('total_samples', len(self.monitor.samples))}")
        print(f"  Effective rate: {stats.get('effective_rate_hz', 0):.1f}Hz")
        print(f"  Collection overhead: {stats.get('overhead_percentage', 0):.2f}%")
        print(f"  Avg collection time: {stats.get('avg_collection_time_ms', 0):.2f}ms")
        print(f"  Max collection time: {stats.get('max_collection_time_ms', 0):.2f}ms")
        
        if summary.get("processes"):
            print(f"\n💻 Process Metrics:")
            for proc in summary["processes"][:3]:
                print(f"  {proc['name']} (PID {proc['pid']}):")
                print(f"    CPU: {proc['cpu_percent']:.1f}%")
                print(f"    Memory: {proc['memory_rss_mb']:.1f}MB")
        
        # Cleanup
        self.monitor.cleanup()
        
        return {
            "inference_count": self.inference_count,
            "total_tokens": self.total_tokens,
            "monitoring_samples": stats.get('total_samples', len(self.monitor.samples)),
            "effective_rate_hz": stats.get('effective_rate_hz', 0)
        }


def compare_sampling_rates():
    """Compare different sampling rates"""
    model_path = Path("../models/bert-tiny-optimum")
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Test different configurations
    configs = [
        (10, 5),   # 10Hz for 5 seconds
        (50, 5),   # 50Hz for 5 seconds
        (100, 5),  # 100Hz for 5 seconds
    ]
    
    results = []
    
    for rate, duration in configs:
        print(f"\n{'#'*60}")
        print(f"Testing {rate}Hz sampling for {duration} seconds")
        print(f"{'#'*60}")
        
        tester = StressTestMonitor(
            model_path=model_path,
            sampling_rate_hz=rate,
            duration_seconds=duration
        )
        
        result = tester.run_stress_test()
        results.append({
            "rate_hz": rate,
            "duration_s": duration,
            **result
        })
        
        # Pause between tests
        time.sleep(2)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("📊 Sampling Rate Comparison")
    print(f"{'='*60}\n")
    
    print(f"{'Rate':<10} {'Samples':<10} {'Effective':<12} {'Overhead':<10}")
    print(f"{'-'*42}")
    
    for r in results:
        overhead = (1 - r['effective_rate_hz']/r['rate_hz']) * 100 if r['rate_hz'] > 0 else 0
        print(f"{r['rate_hz']:>4}Hz     {r['monitoring_samples']:>6}     "
              f"{r['effective_rate_hz']:>6.1f}Hz     {overhead:>6.1f}%")
    
    print(f"\n✅ Comparison complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Stress Test Monitor for ONNX Inference"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("../models/bert-tiny-optimum"),
        help="Path to ONNX model directory"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Sampling rate in Hz"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Test duration in seconds"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different sampling rates"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_sampling_rates()
    else:
        if not args.model_dir.exists():
            print(f"❌ Model not found: {args.model_dir}")
            return 1
        
        tester = StressTestMonitor(
            model_path=args.model_dir,
            sampling_rate_hz=args.rate,
            duration_seconds=args.duration
        )
        
        tester.run_stress_test()
    
    return 0


if __name__ == "__main__":
    exit(main())