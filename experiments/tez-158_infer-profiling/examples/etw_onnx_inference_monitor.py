"""
ETW Monitor with ONNX Inference Example

This example demonstrates using ETW monitoring to profile ONNX model inference
with kernel-level precision and <2% overhead.

Usage:
    python etw_onnx_inference_monitor.py --model-path MODEL.onnx [OPTIONS]

Author: TEZ-165 Implementation
Date: 2025-08-16
"""

import sys
import os
import time
import platform
import argparse
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import monitors
try:
    from monitors.etw_monitor import ETWMonitor, ETWMonitorConfig
    ETW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ETW monitor not available: {e}")
    ETW_AVAILABLE = False

try:
    from monitors.psutil_monitor import PSUtilMonitor, PSUtilMonitorConfig
    PSUTIL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PSUtil monitor not available: {e}")
    PSUTIL_AVAILABLE = False

# Import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime")
    ONNX_AVAILABLE = False


@dataclass
class InferenceMetrics:
    """Metrics for a single inference"""
    inference_id: int
    start_time: float
    end_time: float
    latency_ms: float
    
    # System metrics at inference time
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    context_switches: int = 0
    page_faults: int = 0
    
    # ETW-specific metrics
    cpu_cycles: Optional[int] = None
    kernel_time_ms: Optional[float] = None
    user_time_ms: Optional[float] = None


class ONNXInferenceProfiler:
    """Profile ONNX inference with ETW monitoring"""
    
    def __init__(self, model_path: str, use_etw: bool = True):
        """
        Initialize profiler
        
        Args:
            model_path: Path to ONNX model
            use_etw: Use ETW monitor if available (Windows only)
        """
        self.model_path = model_path
        self.use_etw = use_etw and ETW_AVAILABLE and platform.system() == "Windows"
        
        # Initialize ONNX session
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        
        # Initialize monitor
        self.monitor = None
        
        # Metrics storage
        self.inference_metrics: List[InferenceMetrics] = []
        self.monitor_samples = deque(maxlen=10000)
        
        # Threading
        self.monitoring_active = False
        self.monitor_thread = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize ONNX session and monitor"""
        # Load ONNX model
        if ONNX_AVAILABLE and os.path.exists(self.model_path):
            print(f"Loading ONNX model: {self.model_path}")
            
            # Create session with CPU provider
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"Model loaded successfully")
            print(f"  Input: {self.input_name} {self.input_shape}")
            print(f"  Output: {self.output_name}")
        else:
            raise ValueError(f"ONNX model not found: {self.model_path}")
        
        # Initialize monitor
        if self.use_etw:
            print("Initializing ETW monitor (kernel-level, <2% overhead)")
            config = ETWMonitorConfig(
                sampling_rate_hz=100.0,  # 100Hz sampling
                enable_process_events=True,
                enable_cpu_sampling=True,
                enable_memory_events=True,
                enable_context_switch=True,
                use_existing_session=True,
                exclude_system_processes=True
            )
            self.monitor = ETWMonitor(config)
        else:
            print("Initializing PSUtil monitor (polling-based)")
            config = PSUtilMonitorConfig(
                sampling_rate_hz=100.0,
                max_processes=10,
                include_system_processes=False
            )
            self.monitor = PSUtilMonitor(config)
        
        # Register callback to collect samples
        self.monitor.register_callback(self._process_monitor_sample)
    
    def _process_monitor_sample(self, sample):
        """Process monitoring samples"""
        # Store relevant samples
        if "python" in str(sample.metadata.get('process_name', '')).lower():
            self.monitor_samples.append({
                'timestamp': sample.timestamp,
                'metric': sample.metric_name,
                'value': sample.value,
                'pid': sample.metadata.get('pid')
            })
    
    def _generate_input(self) -> np.ndarray:
        """Generate random input for the model"""
        # Handle dynamic batch size
        shape = list(self.input_shape)
        for i, dim in enumerate(shape):
            if dim is None or dim == 'batch':
                shape[i] = 1  # Use batch size of 1
            elif isinstance(dim, str):
                shape[i] = 128  # Default sequence length
        
        # Generate random input
        return np.random.randn(*shape).astype(np.float32)
    
    def run_inference(self, num_inferences: int = 100, warmup: int = 10):
        """
        Run inference with monitoring
        
        Args:
            num_inferences: Number of inferences to run
            warmup: Number of warmup inferences
        """
        print(f"\n{'=' * 60}")
        print(f"Running {num_inferences} inferences with monitoring")
        print(f"Monitor Type: {'ETW (Kernel-Level)' if self.use_etw else 'PSUtil (Polling)'}")
        print(f"{'=' * 60}\n")
        
        # Start monitoring
        print("Starting system monitoring...")
        self.monitor.start_monitoring()
        self.monitoring_active = True
        
        # Warmup
        if warmup > 0:
            print(f"Running {warmup} warmup inferences...")
            for _ in range(warmup):
                input_data = self._generate_input()
                _ = self.session.run([self.output_name], {self.input_name: input_data})
        
        # Get current process metrics before inference
        import os
        current_pid = os.getpid()
        
        # Run actual inferences
        print(f"Running {num_inferences} monitored inferences...")
        start_time = time.time()
        
        for i in range(num_inferences):
            # Prepare input
            input_data = self._generate_input()
            
            # Record pre-inference metrics
            pre_samples = list(self.monitor_samples)
            
            # Run inference
            inference_start = time.perf_counter()
            output = self.session.run([self.output_name], {self.input_name: input_data})
            inference_end = time.perf_counter()
            
            # Calculate latency
            latency_ms = (inference_end - inference_start) * 1000
            
            # Get post-inference metrics
            post_samples = list(self.monitor_samples)
            
            # Extract metrics during inference
            inference_samples = [s for s in post_samples if s not in pre_samples]
            
            # Create inference metric
            metric = InferenceMetrics(
                inference_id=i,
                start_time=inference_start,
                end_time=inference_end,
                latency_ms=latency_ms
            )
            
            # Extract system metrics
            for sample in inference_samples:
                if sample['pid'] == current_pid:
                    if 'cpu_percent' in sample['metric']:
                        metric.cpu_percent = max(metric.cpu_percent, sample['value'])
                    elif 'memory' in sample['metric'] and 'rss' in sample['metric']:
                        metric.memory_mb = max(metric.memory_mb, sample['value'])
                    elif 'context_switches' in sample['metric']:
                        metric.context_switches += int(sample['value'])
                    elif 'page_faults' in sample['metric']:
                        metric.page_faults += int(sample['value'])
                    elif 'cpu_cycles' in sample['metric']:
                        metric.cpu_cycles = sample['value']
                    elif 'kernel_time' in sample['metric']:
                        metric.kernel_time_ms = sample['value']
                    elif 'user_time' in sample['metric']:
                        metric.user_time_ms = sample['value']
            
            self.inference_metrics.append(metric)
            
            # Print progress
            if (i + 1) % 20 == 0:
                avg_latency = np.mean([m.latency_ms for m in self.inference_metrics])
                print(f"  Progress: {i+1}/{num_inferences} - Avg latency: {avg_latency:.2f}ms")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stop monitoring
        print("\nStopping monitoring...")
        self.monitor.stop_monitoring()
        self.monitoring_active = False
        
        # Print results
        self._print_results(total_time, num_inferences)
    
    def _print_results(self, total_time: float, num_inferences: int):
        """Print profiling results"""
        print(f"\n{'=' * 60}")
        print("Profiling Results")
        print(f"{'=' * 60}\n")
        
        # Inference performance
        latencies = [m.latency_ms for m in self.inference_metrics]
        print("Inference Performance:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {num_inferences / total_time:.1f} inferences/sec")
        print(f"  Latency Statistics:")
        print(f"    Mean: {np.mean(latencies):.2f}ms")
        print(f"    Median: {np.median(latencies):.2f}ms")
        print(f"    Min: {np.min(latencies):.2f}ms")
        print(f"    Max: {np.max(latencies):.2f}ms")
        print(f"    P95: {np.percentile(latencies, 95):.2f}ms")
        print(f"    P99: {np.percentile(latencies, 99):.2f}ms")
        
        # System metrics
        cpu_percents = [m.cpu_percent for m in self.inference_metrics if m.cpu_percent > 0]
        memory_mbs = [m.memory_mb for m in self.inference_metrics if m.memory_mb > 0]
        
        if cpu_percents:
            print(f"\nCPU Usage During Inference:")
            print(f"    Mean: {np.mean(cpu_percents):.1f}%")
            print(f"    Max: {np.max(cpu_percents):.1f}%")
        
        if memory_mbs:
            print(f"\nMemory Usage During Inference:")
            print(f"    Mean: {np.mean(memory_mbs):.1f}MB")
            print(f"    Max: {np.max(memory_mbs):.1f}MB")
        
        # ETW-specific metrics
        if self.use_etw:
            cpu_cycles = [m.cpu_cycles for m in self.inference_metrics if m.cpu_cycles]
            kernel_times = [m.kernel_time_ms for m in self.inference_metrics if m.kernel_time_ms]
            user_times = [m.user_time_ms for m in self.inference_metrics if m.user_time_ms]
            context_switches = [m.context_switches for m in self.inference_metrics if m.context_switches]
            
            print(f"\nETW Kernel-Level Metrics:")
            if cpu_cycles:
                print(f"  CPU Cycles per Inference: {np.mean(cpu_cycles):,.0f}")
            if kernel_times:
                print(f"  Kernel Time: {np.mean(kernel_times):.2f}ms")
            if user_times:
                print(f"  User Time: {np.mean(user_times):.2f}ms")
            if context_switches:
                print(f"  Context Switches: {np.mean(context_switches):.1f}")
        
        # Monitor statistics
        monitor_stats = self.monitor.get_statistics()
        print(f"\nMonitoring Statistics:")
        print(f"  Monitor Type: {'ETW' if self.use_etw else 'PSUtil'}")
        print(f"  Total Samples: {monitor_stats.get('total_samples', 0)}")
        if 'collection_time_ms' in monitor_stats:
            print(f"  Collection Overhead: {monitor_stats['collection_time_ms']['mean']:.2f}ms")
        
        # Save results to file
        self._save_results()
    
    def _save_results(self):
        """Save results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"inference_profile_{timestamp}.json"
        
        results = {
            'model_path': self.model_path,
            'monitor_type': 'ETW' if self.use_etw else 'PSUtil',
            'num_inferences': len(self.inference_metrics),
            'metrics': [
                {
                    'inference_id': m.inference_id,
                    'latency_ms': m.latency_ms,
                    'cpu_percent': m.cpu_percent,
                    'memory_mb': m.memory_mb,
                    'context_switches': m.context_switches,
                    'page_faults': m.page_faults,
                    'cpu_cycles': m.cpu_cycles,
                    'kernel_time_ms': m.kernel_time_ms,
                    'user_time_ms': m.user_time_ms
                }
                for m in self.inference_metrics
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.monitor:
            self.monitor.cleanup()
        if self.session:
            del self.session


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Profile ONNX inference with ETW monitoring"
    )
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--num-inferences', '-n',
        type=int,
        default=100,
        help='Number of inferences to run (default: 100)'
    )
    parser.add_argument(
        '--warmup', '-w',
        type=int,
        default=10,
        help='Number of warmup inferences (default: 10)'
    )
    parser.add_argument(
        '--use-psutil',
        action='store_true',
        help='Use PSUtil monitor instead of ETW'
    )
    
    args = parser.parse_args()
    
    # Check requirements
    if not ONNX_AVAILABLE:
        print("Error: ONNX Runtime not available")
        print("Install with: pip install onnxruntime")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    # Check monitor availability
    use_etw = not args.use_psutil
    if use_etw and not ETW_AVAILABLE:
        print("Warning: ETW not available, falling back to PSUtil")
        use_etw = False
    
    if not use_etw and not PSUTIL_AVAILABLE:
        print("Error: No monitor available")
        return 1
    
    try:
        # Create profiler
        profiler = ONNXInferenceProfiler(
            model_path=args.model_path,
            use_etw=use_etw
        )
        
        # Run profiling
        profiler.run_inference(
            num_inferences=args.num_inferences,
            warmup=args.warmup
        )
        
        # Cleanup
        profiler.cleanup()
        
    except KeyboardInterrupt:
        print("\n\nProfiling stopped by user")
        return 0
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())