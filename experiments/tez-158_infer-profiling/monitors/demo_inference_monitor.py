#!/usr/bin/env python3
"""
Demo: High-Performance Process Monitoring for ONNX Inference

Demonstrates ETW monitor integration for profiling ONNX model inference
with per-process CPU and memory tracking at up to 100Hz.

Author: TEZ-165 Implementation
Date: 2025-08-15
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from monitors.etw_monitor import ETWMonitor, ETWMonitorConfig, MetricSample
except ImportError:
    # Fallback for direct execution
    from etw_monitor import ETWMonitor, ETWMonitorConfig, MetricSample


class InferenceProfiler:
    """
    Profile ONNX model inference with system monitoring
    """
    
    def __init__(self, sampling_rate_hz: float = 100.0):
        """
        Initialize inference profiler
        
        Args:
            sampling_rate_hz: Monitoring frequency (10-100Hz recommended)
        """
        self.sampling_rate = sampling_rate_hz
        self.monitor = None
        self.inference_pid = None
        self.metrics_buffer = []
        
    def start_monitoring(self, target_process: Optional[str] = "python") -> None:
        """
        Start system monitoring
        
        Args:
            target_process: Process name to monitor (default: python)
        """
        print(f"🚀 Starting process monitoring at {self.sampling_rate}Hz")
        
        # Configure monitor
        config = ETWMonitorConfig(
            sampling_rate_hz=self.sampling_rate,
            buffer_size=10000,  # Store up to 10K samples
            target_process_names={target_process} if target_process else None,
            include_system_processes=False,
            max_processes=20,  # Limit to reduce overhead at 100Hz
            cpu_sample_interval=0.01  # 10ms CPU sampling for accuracy
        )
        
        # Create monitor
        self.monitor = ETWMonitor(config)
        
        # Register callback to collect metrics
        self.monitor.register_callback(self._on_metric)
        
        # Start monitoring
        self.monitor.start_monitoring()
        print(f"✅ Monitoring started for process: {target_process}")
    
    def _on_metric(self, sample: MetricSample) -> None:
        """Callback for metric collection"""
        # Filter for inference-related metrics
        if sample.metadata.get("process_name") in ["python", "python3", "modelexport"]:
            self.metrics_buffer.append(sample)
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return statistics"""
        if self.monitor:
            self.monitor.stop_monitoring()
            stats = self.monitor.get_statistics()
            summary = self.monitor.get_process_summary()
            
            # Calculate profiling metrics
            result = {
                "monitoring_stats": stats,
                "process_summary": summary,
                "samples_collected": len(self.metrics_buffer),
                "sampling_rate_hz": self.sampling_rate
            }
            
            # Analyze CPU usage for inference processes
            cpu_samples = [s for s in self.metrics_buffer if "cpu" in s.metric_name]
            if cpu_samples:
                cpu_values = [s.value for s in cpu_samples]
                result["cpu_analysis"] = {
                    "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                    "max_cpu_percent": max(cpu_values),
                    "min_cpu_percent": min(cpu_values),
                    "samples": len(cpu_samples)
                }
            
            # Analyze memory usage
            mem_samples = [s for s in self.metrics_buffer if "memory_rss" in s.metric_name]
            if mem_samples:
                mem_values = [s.value for s in mem_samples]
                result["memory_analysis"] = {
                    "avg_memory_mb": sum(mem_values) / len(mem_values),
                    "max_memory_mb": max(mem_values),
                    "min_memory_mb": min(mem_values),
                    "samples": len(mem_samples)
                }
            
            self.monitor.cleanup()
            return result
        
        return {}
    
    def profile_inference(self, duration: float = 5.0) -> None:
        """
        Profile inference for specified duration
        
        Args:
            duration: Monitoring duration in seconds
        """
        print(f"\n📊 Profiling for {duration} seconds...")
        
        # Monitor for specified duration
        start_time = time.time()
        last_print = start_time
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Print status every second
            if current_time - last_print >= 1.0:
                elapsed = current_time - start_time
                samples = len(self.metrics_buffer)
                effective_rate = samples / elapsed if elapsed > 0 else 0
                
                print(f"  [{elapsed:.1f}s] Samples: {samples}, "
                      f"Effective rate: {effective_rate:.1f}Hz")
                last_print = current_time
            
            time.sleep(0.01)  # Small sleep to reduce CPU usage
        
        print(f"✅ Profiling complete!")


def demo_100hz_monitoring():
    """Demonstrate 100Hz monitoring capabilities"""
    print("=" * 60)
    print("100Hz Process Monitoring Demo")
    print("=" * 60)
    
    # Test different sampling rates
    rates = [10, 50, 100]
    
    for rate in rates:
        print(f"\n🔬 Testing {rate}Hz sampling rate")
        print("-" * 40)
        
        profiler = InferenceProfiler(sampling_rate_hz=rate)
        profiler.start_monitoring(target_process="python")
        
        # Profile for 3 seconds
        profiler.profile_inference(duration=3.0)
        
        # Get results
        results = profiler.stop_monitoring()
        
        # Print analysis
        print(f"\n📈 Results at {rate}Hz:")
        print(f"  Samples collected: {results.get('samples_collected', 0)}")
        
        if "monitoring_stats" in results:
            stats = results["monitoring_stats"]
            print(f"  Avg collection time: {stats.get('avg_collection_time_ms', 0):.2f}ms")
            print(f"  Max collection time: {stats.get('max_collection_time_ms', 0):.2f}ms")
            print(f"  Overhead: {stats.get('overhead_percentage', 0):.1f}%")
        
        if "cpu_analysis" in results:
            cpu = results["cpu_analysis"]
            print(f"  CPU Usage: avg={cpu['avg_cpu_percent']:.1f}%, "
                  f"max={cpu['max_cpu_percent']:.1f}%")
        
        if "memory_analysis" in results:
            mem = results["memory_analysis"]
            print(f"  Memory: avg={mem['avg_memory_mb']:.1f}MB, "
                  f"max={mem['max_memory_mb']:.1f}MB")
        
        print()
        time.sleep(1)  # Brief pause between tests
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)


def monitor_onnx_export():
    """Monitor an actual ONNX model export process"""
    print("=" * 60)
    print("ONNX Export Monitoring Demo")
    print("=" * 60)
    
    # Check if modelexport is available
    try:
        import subprocess
        result = subprocess.run(
            ["uv", "run", "modelexport", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("⚠️ modelexport not available, using mock process")
            target_process = "python"
        else:
            target_process = "modelexport"
    except:
        target_process = "python"
    
    # Start monitoring at 100Hz
    profiler = InferenceProfiler(sampling_rate_hz=100.0)
    profiler.start_monitoring(target_process=target_process)
    
    print("\n🎯 Monitoring active. Run ONNX export in another terminal:")
    print("  uv run modelexport export --model prajjwal1/bert-tiny --output test.onnx")
    print("\nPress Ctrl+C to stop monitoring...")
    
    try:
        # Monitor until interrupted
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            samples = len(profiler.metrics_buffer)
            
            # Update status line
            print(f"\r  [{elapsed:.1f}s] Samples: {samples}, Rate: {samples/elapsed if elapsed > 0 else 0:.1f}Hz", end="")
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping monitor...")
    
    # Get final results
    results = profiler.stop_monitoring()
    
    # Print detailed analysis
    print("\n📊 Monitoring Results:")
    print("-" * 40)
    
    if results.get("samples_collected", 0) > 0:
        print(f"Total samples: {results['samples_collected']}")
        
        if "monitoring_stats" in results:
            stats = results["monitoring_stats"]
            print(f"Collection overhead: {stats.get('overhead_percentage', 0):.1f}%")
        
        if "process_summary" in results:
            summary = results["process_summary"]
            print(f"\nMonitored {summary['monitored_processes']} processes")
            
            # Show top processes by CPU
            if summary.get("processes"):
                print("\nTop processes by CPU:")
                for proc in summary["processes"][:5]:
                    print(f"  {proc['name']} (PID {proc['pid']}): "
                          f"CPU={proc['cpu_percent']:.1f}%, "
                          f"Memory={proc['memory_rss_mb']:.1f}MB")
    else:
        print("No samples collected")
    
    print("\n✅ Monitoring complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ETW Process Monitoring Demo for ONNX Inference"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "monitor"],
        default="test",
        help="Demo mode: 'test' for sampling rate comparison, 'monitor' for live monitoring"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Sampling rate in Hz (10-100 recommended)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        demo_100hz_monitoring()
    else:
        monitor_onnx_export()


if __name__ == "__main__":
    main()